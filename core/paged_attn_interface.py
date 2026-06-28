"""
core.paged_attn_interface -- route the model's full-attention through the paged
gather-and-attend path, for model-level numerical-equivalence validation.

Phase 05's equivalence test (`bench/paged_equiv.py`) otherwise only proves the
page table is lossless *storage*. To prove the paged *compute* matches the real
model, we register a custom attention implementation that, for a single-query
decode step, stores the layer's K/V into a `PagedKVCache` and computes attention
via `core.paged_attn.paged_attention` (gather from blocks + attend). Prefill and
batched steps delegate to the stock eager path so the model still runs.

Enable with `install()` (patches the eager global). Validation seam, not production runtime.
"""

from __future__ import annotations

import torch

from core.paged_attn import paged_attention
from core.paged_cache import PagedKVCache

_BLOCK_SIZE = 16


def paged_ref_attention_forward(
    module,
    query: torch.Tensor,   # [b, num_heads, q_len, head_dim]
    key: torch.Tensor,     # [b, num_kv_heads, kv_len, head_dim]
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    b, num_heads, q_len, head_dim = query.shape
    # Only the single-sequence decode step goes through the paged path; prefill
    # (q_len > 1, needs a causal mask) and batched runs delegate to eager.
    if b != 1 or q_len != 1:
        from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )

    num_kv_heads, kv_len = key.shape[1], key.shape[2]
    k = key[0].transpose(0, 1).contiguous()    # [kv_len, num_kv_heads, head_dim]
    v = value[0].transpose(0, 1).contiguous()

    num_blocks = (kv_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    cache = PagedKVCache(
        num_layers=1, num_blocks=num_blocks, block_size=_BLOCK_SIZE,
        num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=k.dtype, device=k.device,
    )
    cache.create_sequence(0)
    cache.append_tokens(0, [k], [v])

    q = query[0, :, 0, :]                       # [num_heads, head_dim]
    out = paged_attention(
        q, cache.key_cache[0], cache.value_cache[0], cache.block_table(0),
        kv_len, scale=scaling,
    )                                           # [num_heads, head_dim]
    attn_output = out.unsqueeze(0).unsqueeze(0).to(query.dtype)  # [b, q_len, heads, dim]
    return attn_output, None


_real_eager = None


def _wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    # Decode step (batch=1, single query) → paged path; everything else (incl.
    # causal prefill, which needs the mask HF prepared for "eager") → real eager.
    if query.shape[0] == 1 and query.shape[2] == 1:
        return paged_ref_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
    return _real_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)


def install() -> None:
    """
    Route the model's full-attention DECODE step through the paged path by
    rebinding the module-level `eager_attention_forward` symbol.

    We patch the global (rather than registering a custom `_attn_implementation`
    name) on purpose: HF prepares the attention mask based on the impl name, and
    a custom name gets sdpa-style (mask=None) prep that breaks causal prefill.
    Keeping `_attn_implementation="eager"` preserves correct causal masking;
    prefill delegates to the real eager, decode uses paged.
    """
    global _real_eager
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    if _real_eager is None:
        _real_eager = m.eager_attention_forward
    m.eager_attention_forward = _wrapper


def uninstall() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    if _real_eager is not None:
        m.eager_attention_forward = _real_eager


def set_impl(runner, impl: str = "eager") -> str:
    """Force `_attn_implementation="eager"` across the backbone so mask prep is
    causal-correct (the paged decode routing happens via the patched global)."""
    prev = getattr(runner.lm.config, "_attn_implementation", "eager")
    for module in runner.lm.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl
    if hasattr(runner.lm, "config"):
        runner.lm.config._attn_implementation = impl
    return prev
