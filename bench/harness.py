"""
bench.harness — CLI entry point for the inferd benchmark harness.

Usage:
    # Self-check (no GPU needed; verifies metric math on synthetic fixtures):
    uv run python -m bench.harness --selfcheck

    # HF naive floor:
    uv run python -m bench.harness --engine hf --model Qwen3.5-9B \\
        --seed 0 --max-tokens 256 --concurrency 1,2,4,8,16 --warmup 3

    # vLLM ceiling (isolated venv, best-effort — defers cleanly if sm_120 fails):
    uv run python -m bench.harness --engine vllm --model Qwen3.5-9B \\
        --seed 0 --max-tokens 256 --concurrency 1,2,4,8,16

Results written to bench/results/<timestamp>_<engine>_<model>/result.json.
Results are append-only; existing files are never overwritten.

Metric contract (frozen — import from bench.metrics, never redefine):
  TTFT, ITL    — single-stream (concurrency=1) only.
  throughput   — total tokens / wall time; the concurrency-sweep metric.
  peak_vram_mb — nvidia-smi polling (engine-agnostic / comparable).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Self-check: verifies metric arithmetic on synthetic fixtures; no GPU needed.
# ---------------------------------------------------------------------------

def _selfcheck() -> bool:
    """
    Assert metric math on synthetic timing fixtures.
    Returns True on pass, prints failures and returns False on any assertion.
    """
    from bench.metrics import itl, throughput, ttft
    from bench.workload import CANONICAL, GREEDY, workload_hash

    failures: list[str] = []

    def _check(name: str, got, expected, tol: float = 1e-9):
        if abs(got - expected) > tol:
            failures.append(f"  FAIL {name}: got {got!r}, expected {expected!r}")

    # ttft
    _check("ttft", ttft(0.0, 0.5), 0.5)
    _check("ttft", ttft(1.0, 1.2), 0.2)

    # itl — (total - ttft) / (n - 1)
    _check("itl 10-tok", itl(2.0, 0.2, 10), (2.0 - 0.2) / 9)
    _check("itl 1-tok", itl(2.0, 2.0, 1), 0.0)

    # throughput
    _check("throughput", throughput(100, 10.0), 10.0)
    _check("throughput zero", throughput(0, 5.0), 0.0)
    _check("throughput wall=0", throughput(100, 0.0), 0.0)

    # Consistency: TTFT < total when more than 1 token.
    ttft_s = ttft(0.0, 0.3)
    total = 3.0
    n = 20
    itl_s = itl(total, ttft_s, n)
    assert ttft_s < total, "TTFT must be less than total time"
    assert itl_s > 0, "ITL must be positive when n > 1"
    reconstructed = ttft_s + itl_s * (n - 1)
    _check("time reconstruction", reconstructed, total, tol=1e-6)

    # throughput consistency: tokens / time
    tps = throughput(256, 8.0)
    _check("tps", tps, 32.0)

    # workload_hash is deterministic
    h1 = workload_hash(CANONICAL)
    h2 = workload_hash(CANONICAL)
    if h1 != h2:
        failures.append("  FAIL workload_hash: not deterministic")
    if workload_hash(CANONICAL) == workload_hash(GREEDY):
        failures.append("  FAIL workload_hash: CANONICAL == GREEDY (should differ)")

    if failures:
        print("[selfcheck] FAIL")
        for f in failures:
            print(f)
        return False
    else:
        print("[selfcheck] PASS — metric arithmetic verified on synthetic fixtures.")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m bench.harness",
        description="inferd benchmark harness — naive HF floor + vLLM ceiling.",
    )
    p.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run metric self-check (no GPU/model needed) and exit.",
    )
    p.add_argument("--engine", choices=["hf", "vllm", "spec", "paged"], default="hf")
    p.add_argument(
        "--model", default="Qwen3.5-9B",
        help="Model subdirectory under weights/ (e.g. Qwen3.5-9B).",
    )
    # --- speculative-decoding (engine=spec) ---
    p.add_argument("--target", default="merged/9b", help="Target model path (engine=spec).")
    p.add_argument("--draft", default="weights/Qwen3.5-0.8B", help="Draft model path (engine=spec).")
    p.add_argument("--draft-adapter", default=None, dest="draft_adapter",
                   help="Distilled-draft LoRA adapter dir for the alpha-lift run.")
    p.add_argument("--gamma", default="2,4,8", help="Comma-separated gamma sweep (engine=spec).")
    p.add_argument("--n-prompts", type=int, default=None, dest="n_prompts",
                   help="Limit number of workload prompts (engine=spec).")
    # --- paged-cache (engine=paged) ---
    p.add_argument("--block-size", type=int, default=16, dest="block_size",
                   help="KV block size for paged-cache accounting (engine=paged).")
    p.add_argument("--report-vram", action="store_true", dest="report_vram",
                   help="Accepted for phase-05 command compatibility; paged runner reports KV MB.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    p.add_argument(
        "--concurrency", default="1,2,4,8,16",
        help="Comma-separated concurrency levels for the sweep.",
    )
    p.add_argument("--warmup", type=int, default=3, dest="warmup_runs")
    p.add_argument(
        "--profile", choices=["canonical", "greedy"], default="canonical",
        help="Sampling profile: canonical (temp=0.7) or greedy (temp=0).",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None, dest="results_dir",
        help="Override output directory (default: bench/results/).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.selfcheck:
        ok = _selfcheck()
        return 0 if ok else 1

    concurrency_grid = [int(c) for c in args.concurrency.split(",")]

    if args.engine == "hf":
        from bench.runners.hf import run
        result = run(
            model_name=args.model,
            seed=args.seed,
            max_tokens=args.max_tokens,
            concurrency_grid=concurrency_grid,
            warmup_runs=args.warmup_runs,
            profile_name=args.profile,
            results_dir=args.results_dir,
        )
        print("\n[harness] HF floor run complete.")
        _print_summary(result)

    elif args.engine == "vllm":
        from bench.runners.vllm import run
        result = run(
            model_name=args.model,
            seed=args.seed,
            max_tokens=args.max_tokens,
            concurrency_grid=concurrency_grid,
            warmup_runs=1,
            profile_name=args.profile,
            results_dir=args.results_dir,
        )
        if result.role == "ceiling_deferred":
            print("\n[harness] vLLM ceiling deferred (see notes in result JSON).")
        else:
            print("\n[harness] vLLM ceiling run complete.")
        _print_summary(result)

    elif args.engine == "spec":
        from bench.runners.spec import run
        gammas = [int(g) for g in args.gamma.split(",")]
        run(
            target_path=args.target,
            draft_path=args.draft,
            draft_adapter=args.draft_adapter,
            gammas=gammas,
            max_tokens=args.max_tokens,
            seed=args.seed,
            n_prompts=args.n_prompts,
            results_dir=args.results_dir,
        )
        print("\n[harness] spec-decode run complete.")

    elif args.engine == "paged":
        from bench.runners.paged import run
        result = run(
            concurrency_grid=concurrency_grid,
            max_tokens=args.max_tokens,
            block_size=args.block_size,
            results_dir=args.results_dir,
            seed=args.seed,
        )
        print("\n[harness] paged-cache microbenchmark complete.")
        for pt in result["points"]:
            print(
                f"  c={pt['concurrency']:>2} blocks={pt['allocated_blocks']:>4} "
                f"paged={pt['paged_kv_mb']:>8.2f}MiB "
                f"naive={pt['naive_prealloc_kv_mb']:>8.2f}MiB "
                f"ratio={pt['memory_ratio_vs_naive']:.3f}"
            )

    return 0


def _print_summary(result) -> None:
    print(f"\n{'='*60}")
    print(f"Engine:  {result.engine}  ({result.role})")
    print(f"Model:   {result.model}")
    print(f"Profile: {result.profile}  max_tokens={result.max_tokens}")
    print(f"GPU:     {result.env.get('gpu_name', 'n/a')}")
    print(f"Commit:  {result.env.get('git_commit', 'n/a')}")

    if result.single_stream:
        ttfts = [r.ttft_s * 1000 for r in result.single_stream]
        itls  = [r.itl_s * 1000 for r in result.single_stream]
        tpss  = [r.throughput_tok_s for r in result.single_stream]
        print(f"\nSingle-stream (n={len(result.single_stream)} prompts):")
        print(f"  TTFT   mean={sum(ttfts)/len(ttfts):.1f}ms  "
              f"min={min(ttfts):.1f}ms  max={max(ttfts):.1f}ms")
        print(f"  ITL    mean={sum(itls)/len(itls):.2f}ms/tok")
        print(f"  tok/s  mean={sum(tpss)/len(tpss):.1f}")

    if result.concurrency_sweep:
        print(f"\nConcurrency sweep:")
        print(f"  {'c':>4}  {'tok/s':>8}  {'peak_vram_MB':>14}")
        for pt in result.concurrency_sweep:
            print(f"  {pt.concurrency:>4}  {pt.throughput_tok_s:>8.1f}"
                  f"  {pt.peak_vram_mb:>14.0f}")

    if result.notes:
        print(f"\nNotes:")
        for note in result.notes:
            print(f"  - {note}")
    print("="*60)


if __name__ == "__main__":
    sys.exit(main())
