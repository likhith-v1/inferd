"""
core.paged_attn -- paged-attention reference path for phase 05.

The production goal is a Triton gather-and-attend kernel with a FlashAttention
varlen fallback. This module starts with a numerically transparent reference
implementation over the same paged layout as core.paged_cache. The public
`paged_attention` function is the stable interface; a Triton kernel can replace
the internals without changing tests or callers.
"""

from __future__ import annotations

import argparse
import math
import time

import torch


def gather_paged_kv(
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather one sequence's K/V pages into contiguous tensors.

    Inputs:
      key_blocks/value_blocks: [num_blocks, kv_heads, block_size, head_dim]
      block_table:            [num_pages]
      seq_len:                logical sequence length

    Returns:
      key/value: [seq_len, kv_heads, head_dim]
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if key_blocks.shape != value_blocks.shape:
        raise ValueError("key/value block tensors must have identical shape")
    if key_blocks.ndim != 4:
        raise ValueError("block tensors must be [blocks, heads, block_size, head_dim]")
    block_size = key_blocks.shape[2]
    needed_pages = (seq_len + block_size - 1) // block_size
    if block_table.numel() < needed_pages:
        raise ValueError("block_table is shorter than seq_len requires")

    chunks_k = []
    chunks_v = []
    remaining = seq_len
    for page in range(needed_pages):
        block = int(block_table[page].item())
        take = min(block_size, remaining)
        chunks_k.append(key_blocks[block, :, :take, :].transpose(0, 1))
        chunks_v.append(value_blocks[block, :, :take, :].transpose(0, 1))
        remaining -= take
    if not chunks_k:
        shape = (0, key_blocks.shape[1], key_blocks.shape[3])
        empty = key_blocks.new_empty(shape)
        return empty, empty.clone()
    return torch.cat(chunks_k, dim=0), torch.cat(chunks_v, dim=0)


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """
    DECODE attention for one query position (single query attending to all keys).

    query: [q_heads, head_dim]
    key:   [seq_len, kv_heads, head_dim]
    value: [seq_len, kv_heads, head_dim]

    For grouped-query attention, q_heads must be a multiple of kv_heads.
    Returns [q_heads, head_dim].

    NOTE: this is the single-query decode path — there is no causal mask because
    the one (latest) query legitimately attends to every cached key. Do NOT reuse
    it for multi-query prefill; that needs a causal mask.
    """
    if query.ndim != 2 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("expected query [q_heads, dim], key/value [seq, kv_heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key/value shapes differ")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key head_dim differs")
    q_heads = query.shape[0]
    kv_heads = key.shape[1]
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a multiple of kv_heads")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    group = q_heads // kv_heads
    key_h = key.repeat_interleave(group, dim=1).transpose(0, 1)
    value_h = value.repeat_interleave(group, dim=1).transpose(0, 1)
    # Mirror the model's eager recipe: Q·K in the working dtype, softmax in fp32,
    # P·V in the working dtype. Upcasting Q·K to fp32 instead makes the result
    # *more* precise than the model and shows up as a ~0.5-logit gap vs eager.
    scores = torch.einsum("hd,hsd->hs", query, key_h) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(value_h.dtype)
    return torch.einsum("hs,hsd->hd", probs, value_h)


def sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """
    INDEPENDENT single-query decode reference via torch SDPA.

    Same signature/semantics as dense_attention but a completely different code
    path (fused scaled_dot_product_attention rather than hand-rolled einsum), so
    it can validate the paged gather-and-attend math without being tautological.
    """
    q_heads = query.shape[0]
    kv_heads = key.shape[1]
    group = q_heads // kv_heads
    # -> [1, q_heads, seq_or_1, head_dim]
    q = query.unsqueeze(0).unsqueeze(2).float()                       # [1,q_heads,1,dim]
    k = key.repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0).float()
    v = value.repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0).float()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    return out.squeeze(2).squeeze(0).to(value.dtype)                  # [q_heads, dim]


def paged_attention_reference(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    key, value = gather_paged_kv(key_blocks, value_blocks, block_table, seq_len)
    return dense_attention(query, key, value, scale=scale)


def paged_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    *,
    scale: float | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Stable paged-attention interface.

    `backend="auto"` currently resolves to the reference gather path. The Triton
    implementation will slot in here after the indexing contract is locked down.
    """
    if backend not in {"auto", "reference"}:
        raise ValueError(f"unsupported backend: {backend}")
    return paged_attention_reference(
        query, key_blocks, value_blocks, block_table, seq_len, scale=scale
    )


def _bench() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    num_blocks, block_size, kv_heads, q_heads, head_dim = 128, 16, 8, 32, 128
    seq_len = 1024
    key = torch.randn(num_blocks, kv_heads, block_size, head_dim, device=device, dtype=dtype)
    val = torch.randn_like(key)
    table = torch.arange((seq_len + block_size - 1) // block_size, device=device)
    query = torch.randn(q_heads, head_dim, device=device, dtype=dtype)

    for _ in range(5):
        paged_attention(query, key, val, table, seq_len)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        paged_attention(query, key, val, table, seq_len)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"[paged_attn] reference backend {50 / (time.perf_counter() - t0):.1f} calls/s")


def _selfcheck() -> None:
    from core.paged_cache import PagedKVCache

    torch.manual_seed(0)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=3,
        dtype=torch.float32,
    )
    cache.create_sequence(0)
    key = torch.randn(9, 2, 3)
    val = torch.randn(9, 2, 3)
    cache.append_tokens(0, [key], [val])
    query = torch.randn(4, 3)
    table = cache.block_table(0)
    out_paged = paged_attention(query, cache.key_cache[0], cache.value_cache[0], table, 9)
    # Validate against an INDEPENDENT reference (SDPA), not the internal dense path.
    out_ref = sdpa_reference(query, key, val)
    torch.testing.assert_close(out_paged, out_ref, rtol=1e-4, atol=1e-5)
    print("[paged_attn] selfcheck PASS (paged == SDPA reference)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0
    if args.bench:
        _bench()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
