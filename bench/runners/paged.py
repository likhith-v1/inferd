"""
bench.runners.paged -- cache-level phase-05 microbenchmark.

This runner exercises the page-table allocator and paged-attention reference
path without claiming full model integration. It records the useful accounting
phase 06 needs: allocated blocks, logical tokens, internal fragmentation, and a
naive contiguous preallocation comparison.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from bench.metrics import env_stamp, write_result_json
from bench.workload import MAX_TOKENS
from core.paged_attn import paged_attention
from core.paged_cache import PagedKVCache


def _microbench_hash(
    *,
    seed: int,
    max_tokens: int,
    block_size: int,
    concurrency_grid: list[int],
    model_shape: dict,
) -> str:
    """Hash the synthetic cache sweep params (not the frozen CANONICAL workload)."""
    payload = json.dumps(
        {
            "kind": "paged_cache_microbench",
            "seed": seed,
            "max_tokens": max_tokens,
            "block_size": block_size,
            "concurrency_grid": concurrency_grid,
            "model_shape": model_shape,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _sequence_lengths(concurrency: int, max_tokens: int) -> list[int]:
    # Deterministic uneven lengths so the page-table accounting shows the real
    # win over "reserve max length for every sequence".
    base = [17, 31, 32, 33, 47, 64, 79, 96, 113, 128, 191, max_tokens]
    return [min(max_tokens, base[i % len(base)]) for i in range(concurrency)]


def _bytes_for_tokens(tokens: int, *, layers: int, kv_heads: int, head_dim: int, dtype) -> int:
    bytes_per = torch.empty((), dtype=dtype).element_size()
    return tokens * layers * kv_heads * head_dim * bytes_per * 2


def _measure_alloc_mb(build, device: str) -> float:
    """Peak CUDA bytes (MiB) allocated by `build()`; 0.0 on CPU (not measurable)."""
    if not str(device).startswith("cuda"):
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    obj = build()  # noqa: F841 — keep alive through the measurement
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del obj
    torch.cuda.empty_cache()
    return round(peak / 1024**2, 2)


def _run_point(
    concurrency: int,
    *,
    max_tokens: int,
    block_size: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype,
    device: str = "cpu",
) -> dict:
    lengths = _sequence_lengths(concurrency, max_tokens)
    total_blocks = sum((length + block_size - 1) // block_size for length in lengths)

    # Measured on-GPU footprint: paged blocks (exactly what the sequences need)
    # vs the naive "reserve max_tokens for every sequence" contiguous KV.
    def _build_paged():
        return PagedKVCache(
            num_layers=layers, num_blocks=total_blocks, block_size=block_size,
            num_kv_heads=kv_heads, head_dim=head_dim, dtype=dtype, device=device,
        )

    def _build_naive():
        # [layers] x K,V of shape [concurrency, max_tokens, kv_heads, head_dim]
        return [
            (torch.empty(concurrency, max_tokens, kv_heads, head_dim, dtype=dtype, device=device),
             torch.empty(concurrency, max_tokens, kv_heads, head_dim, dtype=dtype, device=device))
            for _ in range(layers)
        ]

    paged_mb_measured = _measure_alloc_mb(_build_paged, device)
    naive_mb_measured = _measure_alloc_mb(_build_naive, device)

    cache = PagedKVCache(
        num_layers=layers,
        num_blocks=total_blocks,
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    t0 = time.perf_counter()
    for seq_id, length in enumerate(lengths):
        cache.create_sequence(seq_id)
        layer_keys = [
            torch.randn(length, kv_heads, head_dim, dtype=dtype, device=device) for _ in range(layers)
        ]
        layer_values = [torch.randn_like(k) for k in layer_keys]
        cache.append_tokens(seq_id, layer_keys, layer_values)
    cache.assert_consistent()
    append_s = time.perf_counter() - t0

    query = torch.randn(kv_heads, head_dim, dtype=dtype, device=device)
    t1 = time.perf_counter()
    for seq_id, length in enumerate(lengths):
        paged_attention(
            query,
            cache.key_cache[0],
            cache.value_cache[0],
            cache.block_table(seq_id),
            length,
        )
    attn_s = time.perf_counter() - t1

    logical_tokens = sum(lengths)
    allocated_tokens = total_blocks * block_size
    naive_tokens = concurrency * max_tokens
    paged_bytes = _bytes_for_tokens(
        allocated_tokens, layers=layers, kv_heads=kv_heads, head_dim=head_dim, dtype=dtype
    )
    naive_bytes = _bytes_for_tokens(
        naive_tokens, layers=layers, kv_heads=kv_heads, head_dim=head_dim, dtype=dtype
    )
    return {
        "concurrency": concurrency,
        "lengths": lengths,
        "logical_tokens": logical_tokens,
        "allocated_blocks": total_blocks,
        "allocated_tokens": allocated_tokens,
        "internal_fragmentation_tokens": allocated_tokens - logical_tokens,
        "naive_prealloc_tokens": naive_tokens,
        "paged_kv_mb_analytic": round(paged_bytes / 1024**2, 2),
        "naive_prealloc_kv_mb_analytic": round(naive_bytes / 1024**2, 2),
        "memory_ratio_vs_naive_analytic": round(paged_bytes / naive_bytes, 4) if naive_bytes else 0.0,
        "paged_kv_mb_measured": paged_mb_measured,
        "naive_prealloc_kv_mb_measured": naive_mb_measured,
        "memory_ratio_vs_naive_measured": (
            round(paged_mb_measured / naive_mb_measured, 4) if naive_mb_measured else None
        ),
        "device": str(device),
        "append_s": round(append_s, 6),
        "attention_reference_s": round(attn_s, 6),
    }


def run(
    *,
    concurrency_grid: list[int] | None = None,
    max_tokens: int = MAX_TOKENS,
    block_size: int = 16,
    results_dir: Path | None = None,
    seed: int = 0,
) -> dict:
    if concurrency_grid is None:
        concurrency_grid = [1, 2, 4, 8, 16]
    torch.manual_seed(seed)
    # Match the real engine: Qwen3.5-9B full-attention shape, bf16, on GPU when
    # available so the VRAM comparison is measured rather than purely analytic.
    layers, kv_heads, head_dim = 36, 8, 128
    dtype = torch.bfloat16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_shape = {
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "block_size": block_size,
    }
    points = [
        _run_point(
            c,
            max_tokens=max_tokens,
            block_size=block_size,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        for c in concurrency_grid
    ]
    result = {
        "engine": "paged",
        "role": "cache_microbench",
        "model_shape": model_shape,
        "max_tokens": max_tokens,
        "concurrency_grid": concurrency_grid,
        "points": points,
        "notes": [
            "Cache-level accounting only; full Qwen model_runner paged integration is not claimed.",
            "Qwen3.5 hybrid linear-attention states are fixed recurrent state, not paged KV.",
        ],
        "env": env_stamp(
            seed,
            _microbench_hash(
                seed=seed,
                max_tokens=max_tokens,
                block_size=block_size,
                concurrency_grid=concurrency_grid,
                model_shape=model_shape,
            ),
        ),
    }
    _write(result, results_dir)
    return result


def _write(result: dict, results_dir: Path | None) -> Path:
    out_path = write_result_json(result, "paged_cache", results_dir)
    print(f"\n[paged] result written to {out_path}")
    return out_path
