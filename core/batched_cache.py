"""
Stack / split per-sequence Qwen3.5 hybrid caches for batched decode.

Full-attention K/V rows have different lengths, so we left-pad to a common length
and cat on batch dim. Linear-attention conv/recurrent states are fixed-size and
cat directly. One batched forward amortizes weight loads across N sequences.
"""

from __future__ import annotations

import copy

import inferd.env  # noqa: F401

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _full_attn_layers(cache) -> list[int]:
    return [i for i in range(len(cache.key_cache)) if cache.key_cache[i] is not None]


def _empty_like(cache):
    n = len(cache.key_cache)
    shell = copy.copy(cache)
    shell.key_cache = [None] * n
    shell.value_cache = [None] * n
    shell.conv_states = [None] * n
    shell.recurrent_states = [None] * n
    return shell


def stack_caches(caches: list):
    """Stack single-row caches; returns (batched_cache, per-row true KV lengths)."""
    if not caches:
        raise ValueError("stack_caches requires at least one cache")
    lengths = [int(c.get_seq_length()) for c in caches]
    max_len = max(lengths)
    n_layers = len(caches[0].key_cache)
    batched = _empty_like(caches[0])

    for layer in range(n_layers):
        if caches[0].key_cache[layer] is not None:
            ks, vs = [], []
            for c, length in zip(caches, lengths):
                k, v = c.key_cache[layer], c.value_cache[layer]
                pad = max_len - length
                if pad:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                ks.append(k)
                vs.append(v)
            batched.key_cache[layer] = torch.cat(ks, dim=0)
            batched.value_cache[layer] = torch.cat(vs, dim=0)
        else:
            batched.conv_states[layer] = torch.cat(
                [c.conv_states[layer] for c in caches], dim=0
            )
            batched.recurrent_states[layer] = torch.cat(
                [c.recurrent_states[layer] for c in caches], dim=0
            )
    return batched, lengths


def split_caches(batched, new_lengths: list[int]) -> list:
    """Split batched cache back into per-row caches after a decode forward."""
    full_layers = _full_attn_layers(batched)
    padded_len = batched.key_cache[full_layers[0]].shape[-2] if full_layers else None
    n_layers = len(batched.key_cache)
    out = []
    for i, length in enumerate(new_lengths):
        row = _empty_like(batched)
        start = (padded_len - length) if padded_len is not None else None
        if start is not None and start < 0:
            raise ValueError(
                f"split_caches: length {length} exceeds padded KV length {padded_len}"
            )
        for layer in range(n_layers):
            if batched.key_cache[layer] is not None:
                row.key_cache[layer] = batched.key_cache[layer][i:i + 1, :, start:, :].clone()
                row.value_cache[layer] = batched.value_cache[layer][i:i + 1, :, start:, :].clone()
            else:
                row.conv_states[layer] = batched.conv_states[layer][i:i + 1].clone()
                row.recurrent_states[layer] = batched.recurrent_states[layer][i:i + 1].clone()
        out.append(row)
    return out
