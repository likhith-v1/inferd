"""
core.batched_cache — stack / split per-sequence Qwen3.5 hybrid caches along the
batch dim so the continuous-batching scheduler (phase 06) can decode the whole
running set in ONE model forward instead of one forward per sequence.

Why this is what makes batching a *win*
---------------------------------------
Decode at 9B is memory-bandwidth bound: each step reloads the full weight set.
Running N sequences through N separate forwards reloads the weights N times.
Stacking them into a single [N, 1] forward amortizes that one weight load across
all N sequences — that is the throughput-vs-concurrency win the phase exists to
show. The extra cost here (padding + a couple of cat/slice ops on the KV) is tiny
next to a 9B weight sweep.

The mechanics, given Qwen3.5's hybrid cache (Qwen3_5DynamicCache):
  - full_attention layers: a growing K/V tensor [1, H, L_i, D] per sequence. Rows
    have *different* L_i, so we LEFT-PAD to a common length and cat on dim 0. Left
    (not right) padding keeps every sequence's newest token at the same final
    column, so one causal decode step is correct for all rows given a per-row
    attention mask + position_ids.
  - linear_attention layers: fixed-size conv_states / recurrent_states that do NOT
    depend on sequence length, so they cat on dim 0 directly.

`has_previous_state` is a computed property (reads conv_states), and the cache has
no stored scalar flags, so a shallow copy with four fresh state lists is a valid,
empty-shell clone of the right type — no model config needed.
"""

from __future__ import annotations

import copy

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _full_attn_layers(cache) -> list[int]:
    """Indices whose layer keeps a K/V tensor (full attention); the rest are linear."""
    return [i for i in range(len(cache.key_cache)) if cache.key_cache[i] is not None]


def _empty_like(cache):
    """A shallow clone of `cache` with brand-new (empty) state lists."""
    n = len(cache.key_cache)
    shell = copy.copy(cache)  # copies layer_types / transformer_layers / etc. by ref
    shell.key_cache = [None] * n
    shell.value_cache = [None] * n
    shell.conv_states = [None] * n
    shell.recurrent_states = [None] * n
    return shell


def stack_caches(caches: list):
    """
    Stack single-row caches into one batched cache.

    Returns (batched_cache, lengths) where lengths[i] is row i's true
    full-attention KV length (== its current sequence length). The batched
    full-attention K/V is LEFT-PADDED to max(lengths).
    """
    if not caches:
        raise ValueError("stack_caches requires at least one cache")
    lengths = [int(c.get_seq_length()) for c in caches]
    max_len = max(lengths)
    n_layers = len(caches[0].key_cache)
    batched = _empty_like(caches[0])

    for layer in range(n_layers):
        if caches[0].key_cache[layer] is not None:  # full attention
            ks, vs = [], []
            for c, length in zip(caches, lengths):
                k, v = c.key_cache[layer], c.value_cache[layer]
                pad = max_len - length
                if pad:
                    # pad the seq dim (dim=-2) on the LEFT
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                ks.append(k)
                vs.append(v)
            batched.key_cache[layer] = torch.cat(ks, dim=0)
            batched.value_cache[layer] = torch.cat(vs, dim=0)
        else:  # linear attention — length-independent, cat directly
            batched.conv_states[layer] = torch.cat(
                [c.conv_states[layer] for c in caches], dim=0
            )
            batched.recurrent_states[layer] = torch.cat(
                [c.recurrent_states[layer] for c in caches], dim=0
            )
    return batched, lengths


def split_caches(batched, new_lengths: list[int]) -> list:
    """
    Split a batched cache (after a decode forward) back into single-row caches.

    `new_lengths[i]` is row i's true full-attention KV length AFTER the forward
    (its pre-decode length + the tokens just appended). Each row is trimmed to
    drop its left padding; tensors are cloned so the large batched tensor is freed.
    """
    full_layers = _full_attn_layers(batched)
    padded_len = batched.key_cache[full_layers[0]].shape[-2] if full_layers else 0
    n_layers = len(batched.key_cache)
    out = []
    for i, length in enumerate(new_lengths):
        row = _empty_like(batched)
        start = padded_len - length
        for layer in range(n_layers):
            if batched.key_cache[layer] is not None:
                row.key_cache[layer] = batched.key_cache[layer][i:i + 1, :, start:, :].clone()
                row.value_cache[layer] = batched.value_cache[layer][i:i + 1, :, start:, :].clone()
            else:
                row.conv_states[layer] = batched.conv_states[layer][i:i + 1].clone()
                row.recurrent_states[layer] = batched.recurrent_states[layer][i:i + 1].clone()
        out.append(row)
    return out
