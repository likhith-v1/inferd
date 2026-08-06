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
"""

from __future__ import annotations

import torch
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
