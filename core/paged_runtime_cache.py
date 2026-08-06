"""
core.paged_runtime_cache -- a live, persistent paged Cache for Qwen3.5.

Phase 05's `PagedHybridCache` (core.paged_cache) proves the page table is a
lossless one-shot round trip; it is never wired into an actual model forward.
`PagedRuntimeCache` closes that gap for the single-sequence path: it is a
drop-in replacement for `Qwen3_5DynamicCache` that backs full-attention K/V
with a shared `PagedKVCache` pool instead of a per-request growing tensor,
across an arbitrary number of live decode steps.

Scope (this pass): single sequence (batch=1) only -- the object this class
threads through `ModelRunner.forward(tokens, kv)` for one request's own
prefill/decode calls. Wiring it through continuous batching's
`stack_caches`/`split_caches`/`decode_batch` (which build a temporary N-row
shallow-copy cache per step) is deliberately out of scope: that requires
`split_caches` to extract just the new token's delta from that temporary
object and write it into the pool, with nothing verifying the untouched rest
of the row still matches pool contents -- a design that needs to be worked
out on its own, not folded in here.

Linear-attention (GatedDeltaNet) layers are untouched: `conv_states`/
`recurrent_states` are fixed-size, not per-position KV, and continue to be
managed exactly as `Qwen3_5DynamicCache` and `core.qwen35_patch` already do.

`BatchedPagedCache` extends this to continuous batching's decode step: N
`PagedRuntimeCache` rows sharing one pool are combined into a single,
one-step container for the batched forward. The naive bridge -- slicing the
*padded, concatenated* output back apart per row and writing that into the
pool -- was rejected: nothing would verify a slice of padded output still
matches what the model actually attended to, and the existing (logits-only)
equivalence gate can't catch that class of drift. Instead `update()` writes
each row's write directly from its raw, unpadded per-token
`key_states[i:i+1]`/`value_states[i:i+1]` slice -- exactly what that row
contributed this step, before any padding/concatenation happens -- so
there is no reconstruction step to get wrong.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from core.paged_cache import PagedKVCache


class PagedRuntimeCache(Qwen3_5DynamicCache):
    """
    Subclasses `Qwen3_5DynamicCache` so `get_seq_length`, `get_mask_sizes`,
    `has_previous_state`, and `reorder_cache` are inherited unchanged -- they
    already read `self.key_cache`/`self.conv_states`, which `update()` below
    keeps populated in the same shape/layout the base class would produce.
    Only `update()` (full-attention K/V) is overridden; linear-attention
    layers keep using `conv_states`/`recurrent_states` exactly as before.
    """

    def __init__(self, config, *, pool: PagedKVCache, seq_id: int) -> None:
        super().__init__(config)
        if not self.transformer_layers:
            raise ValueError("PagedRuntimeCache requires at least one full_attention layer")

        self._pool = pool
        self._seq_id = seq_id
        self._cache_layer_for = {
            original: cache_layer for cache_layer, original in enumerate(self.transformer_layers)
        }
        self._first_full_attention_layer = self.transformer_layers[0]
        self._step_start_pos: int | None = None
        pool.create_sequence(seq_id)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._cache_layer_for:
            raise ValueError(f"layer {layer_idx} is not a known full_attention layer")
        if key_states.shape[0] != 1:
            raise ValueError(
                "PagedRuntimeCache only supports batch=1 (one sequence per instance); "
                f"got batch={key_states.shape[0]}"
            )

        n_tokens = int(key_states.shape[2])
        if layer_idx == self._first_full_attention_layer:
            self._step_start_pos = self._pool.reserve(self._seq_id, n_tokens)
        if self._step_start_pos is None:
            raise RuntimeError(
                "update() called for a full_attention layer before this step's first "
                "full_attention layer reserved space -- layers must be visited in order"
            )

        cache_layer = self._cache_layer_for[layer_idx]
        k = key_states[0].transpose(0, 1).contiguous()  # [n_tokens, kv_heads, head_dim]
        v = value_states[0].transpose(0, 1).contiguous()
        self._pool.write_layer(self._seq_id, cache_layer, k, v, self._step_start_pos)

        gathered_k, gathered_v = self._pool.gather_layer(self._seq_id, cache_layer)
        self.key_cache[layer_idx] = gathered_k.transpose(0, 1).unsqueeze(0).contiguous()
        self.value_cache[layer_idx] = gathered_v.transpose(0, 1).unsqueeze(0).contiguous()
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def free_sequence(self) -> None:
        """Release this sequence's blocks back to the pool. Call on evict/cancel."""
        self._pool.free_sequence(self._seq_id)


class BatchedPagedCache(Qwen3_5DynamicCache):
    """
    Temporary, one-step container batching N `PagedRuntimeCache` rows for a
    single batched decode forward. Discarded after each `decode_batch` call
    -- the row objects (each request's own `kv` handle) persist across
    scheduler steps, not this container.

    Does not call `Qwen3_5DynamicCache.__init__` (no single `config` object
    applies to a multi-row batch); attributes are copied directly from
    `rows[0]` instead, since they are identical across rows by construction
    (same model). Subclassing still buys the inherited `get_seq_length`/
    `get_mask_sizes`/`has_previous_state`/`reorder_cache` methods -- this
    object's shape (initially-`None` key_cache/value_cache, populated by
    `update()`) is exactly what `core.batched_cache.stack_caches`'s output
    already is in production today, so those inherited methods are already
    proven safe for it.
    """

    def __init__(self, rows: list[PagedRuntimeCache]) -> None:
        if not rows:
            raise ValueError("BatchedPagedCache requires at least one row")
        pool = rows[0]._pool
        if any(row._pool is not pool for row in rows):
            raise ValueError("all rows must share the same PagedKVCache pool")

        self.layer_types = list(rows[0].layer_types)
        self.transformer_layers = list(rows[0].transformer_layers)
        self.last_linear_layer = rows[0].last_linear_layer
        n_layers = len(self.layer_types)
        self.key_cache: list[torch.Tensor | None] = [None] * n_layers
        self.value_cache: list[torch.Tensor | None] = [None] * n_layers
        self.conv_states: list[torch.Tensor | None] = [None] * n_layers
        self.recurrent_states: list[torch.Tensor | None] = [None] * n_layers
        for layer in range(n_layers):
            if self.layer_types[layer] != "full_attention":
                self.conv_states[layer] = torch.cat([r.conv_states[layer] for r in rows], dim=0)
                self.recurrent_states[layer] = torch.cat(
                    [r.recurrent_states[layer] for r in rows], dim=0
                )

        self._rows = rows
        self._pool = pool
        self._seq_ids = [row._seq_id for row in rows]
        self.lengths = [pool.sequence_length(seq_id) for seq_id in self._seq_ids]
        self._cache_layer_for = rows[0]._cache_layer_for
        self._first_full_attention_layer = rows[0]._first_full_attention_layer
        self._step_start_pos: list[int] | None = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._cache_layer_for:
            raise ValueError(f"layer {layer_idx} is not a known full_attention layer")
        if key_states.shape[0] != len(self._rows):
            raise ValueError(
                f"expected batch={len(self._rows)} (one row per sequence), "
                f"got {key_states.shape[0]}"
            )
        if key_states.shape[2] != 1:
            raise ValueError(
                "BatchedPagedCache.update() only supports single-token decode steps "
                f"(one new token per row); got {key_states.shape[2]} tokens"
            )

        if layer_idx == self._first_full_attention_layer:
            self._step_start_pos = [self._pool.reserve(seq_id, 1) for seq_id in self._seq_ids]
        if self._step_start_pos is None:
            raise RuntimeError(
                "update() called for a full_attention layer before this step's first "
                "full_attention layer reserved space -- layers must be visited in order"
            )

        cache_layer = self._cache_layer_for[layer_idx]
        for i, seq_id in enumerate(self._seq_ids):
            k = key_states[i].transpose(0, 1).contiguous()  # [1, kv_heads, head_dim]
            v = value_states[i].transpose(0, 1).contiguous()
            self._pool.write_layer(seq_id, cache_layer, k, v, self._step_start_pos[i])

        max_len = max(self.lengths) + 1
        rows_k, rows_v = [], []
        for seq_id in self._seq_ids:
            gk, gv = self._pool.gather_layer(seq_id, cache_layer)  # [seq_len, kv_heads, head_dim]
            gk = gk.transpose(0, 1).unsqueeze(0)  # [1, kv_heads, seq_len, head_dim]
            gv = gv.transpose(0, 1).unsqueeze(0)
            pad = max_len - gk.shape[2]
            if pad:
                gk = F.pad(gk, (0, 0, pad, 0))
                gv = F.pad(gv, (0, 0, pad, 0))
            rows_k.append(gk)
            rows_v.append(gv)
        self.key_cache[layer_idx] = torch.cat(rows_k, dim=0).contiguous()
        self.value_cache[layer_idx] = torch.cat(rows_v, dim=0).contiguous()
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def scatter_linear_state_back(self) -> None:
        """
        Copy this step's batched conv/recurrent state back into each row's
        own `PagedRuntimeCache`. Mirrors `core.batched_cache.split_caches`'s
        slicing for exactly these two fields -- full-attention K/V needs no
        equivalent step since it was already written to the pool, from
        source, inside `update()`.
        """
        for layer in range(len(self.layer_types)):
            if self.layer_types[layer] == "full_attention":
                continue
            for i, row in enumerate(self._rows):
                row.conv_states[layer] = self.conv_states[layer][i:i + 1].clone()
                row.recurrent_states[layer] = self.recurrent_states[layer][i:i + 1].clone()
