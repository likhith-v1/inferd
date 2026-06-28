"""
bench.runners.batched -- phase-06 continuous batching runner.

This runner exercises the FCFS iteration-level scheduler under the frozen
workload. The current runtime KV is still ModelRunner/HF-backed; the scheduler
enforces paged block admission accounting but does not yet persist KV in
PagedKVCache blocks end-to-end.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from math import ceil
from pathlib import Path

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402

from bench.metrics import BenchResult, ConcurrencySweepPoint, VramSampler, env_stamp, throughput  # noqa: E402
from bench.workload import CANONICAL, GREEDY, MAX_TOKENS, PROMPTS, request_token_budgets, workload_hash  # noqa: E402
from core.model_runner import ModelRunner  # noqa: E402
from core.scheduler import (
    ContinuousBatchScheduler,
    ModelRunnerBackend,
    RequestStatus,
    SchedulerConfig,
)  # noqa: E402

WEIGHTS_ROOT = Path(__file__).parent.parent.parent / "weights"


def _resolve_model_path(model_name: str) -> Path:
    path = Path(model_name)
    if path.exists():
        return path
    return WEIGHTS_ROOT / model_name


def _default_max_blocks(
    concurrency_grid: list[int],
    max_tokens: int,
    block_size: int,
    tokenizer,
    prompts: list[str],
) -> int:
    """Block budget sized for prompt_len + max_tokens per admitted sequence."""
    max_prompt = max(
        len(tokenizer(p, return_tensors="pt").input_ids[0]) for p in prompts
    )
    blocks_per_seq = ceil((max_prompt + max_tokens) / block_size)
    return max(concurrency_grid) * blocks_per_seq


def _build_scheduler(
    runner: ModelRunner,
    prompts: list[str],
    *,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    block_size: int,
    max_blocks: int,
    seed: int,
    profile_name: str,
    continuous: bool = True,
    vary_lengths: bool = False,
) -> ContinuousBatchScheduler:
    profile = CANONICAL if profile_name == "canonical" else GREEDY
    backend = ModelRunnerBackend(runner)
    scheduler = ContinuousBatchScheduler(
        backend,
        SchedulerConfig(
            max_blocks=max_blocks,
            block_size=block_size,
            max_concurrent_sequences=concurrency,
            temperature=profile.temperature,
            top_p=profile.top_p,
            seed=seed,
            continuous=continuous,
        ),
    )

    pool = [prompts[i % len(prompts)] for i in range(total_requests)]
    rng = random.Random(seed)
    budgets = request_token_budgets(
        rng, total_requests, max_tokens, vary_lengths=vary_lengths
    )
    for i, prompt in enumerate(pool):
        ids = runner.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        scheduler.submit(ids, max_tokens=budgets[i], prompt_text=prompt, request_id=i + 1)
    return scheduler


def _run_scheduler_point(
    runner: ModelRunner,
    prompts: list[str],
    *,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    block_size: int,
    max_blocks: int,
    seed: int,
    profile_name: str,
    continuous: bool = True,
    vary_lengths: bool = False,
    warmup_runs: int = 0,
) -> tuple[ConcurrencySweepPoint, dict]:
    # Warmup: discard timing, but populate CUDA kernels and batched decode paths.
    for _ in range(warmup_runs):
        scheduler = _build_scheduler(
            runner,
            prompts,
            concurrency=concurrency,
            total_requests=total_requests,
            max_tokens=max_tokens,
            block_size=block_size,
            max_blocks=max_blocks,
            seed=seed,
            profile_name=profile_name,
            continuous=continuous,
            vary_lengths=vary_lengths,
        )
        scheduler.run_until_complete()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    scheduler = _build_scheduler(
        runner,
        prompts,
        concurrency=concurrency,
        total_requests=total_requests,
        max_tokens=max_tokens,
        block_size=block_size,
        max_blocks=max_blocks,
        seed=seed,
        profile_name=profile_name,
        continuous=continuous,
        vary_lengths=vary_lengths,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with VramSampler() as vs:
        t0 = time.perf_counter()
        completed = scheduler.run_until_complete()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    metrics = scheduler.metrics_snapshot().as_dict()
    total_tokens = sum(req.generated_len for req in completed if req.status == RequestStatus.COMPLETED)
    torch_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )

    point = ConcurrencySweepPoint(
        concurrency=concurrency,
        total_time_s=wall,
        total_tokens=total_tokens,
        throughput_tok_s=throughput(total_tokens, wall),
        peak_vram_mb=vs.peak_mb,
        peak_vram_torch_mb=torch_peak_mb,
        warmup_runs=warmup_runs,
    )
    return point, metrics


def run(
    model_name: str = "merged/9b",
    seed: int = 0,
    max_tokens: int = MAX_TOKENS,
    concurrency_grid: list[int] | None = None,
    warmup_runs: int = 3,
    profile_name: str = "greedy",
    results_dir: Path | None = None,
    device: str = "cuda:0",
    block_size: int = 16,
    max_blocks: int | None = None,
    total_requests: int | None = None,
    static_baseline: bool = False,
    vary_lengths: bool = False,
) -> BenchResult:
    if concurrency_grid is None:
        concurrency_grid = [1, 2, 4, 8, 16]

    model_path = _resolve_model_path(model_name)
    runner = ModelRunner.load_target(model_path, device=device)
    profile = CANONICAL if profile_name == "canonical" else GREEDY
    # Fixed total workload, processed at every batch width in the sweep, so the
    # throughput curve compares like-for-like work as concurrency grows.
    if total_requests is None:
        total_requests = max(concurrency_grid)
    if max_blocks is None:
        max_blocks = _default_max_blocks(
            concurrency_grid, max_tokens, block_size, runner.tokenizer, PROMPTS
        )

    result = BenchResult(
        engine="batched",
        role="phase06_scheduler",
        model=str(model_path),
        profile=profile_name,
        max_tokens=max_tokens,
        env=env_stamp(seed, workload_hash(profile, max_tokens)),
        notes=[
            "Continuous batching: FCFS admission, one BATCHED decode forward per iteration over "
            "the running set (per-seq caches stacked, full-attn KV left-padded, linear states cat'd).",
            "Batched decode is numerically equivalent to single-stream (bench.batched_equiv, "
            "max|Δlogit| at the bf16 floor).",
            f"Fixed pool of {total_requests} requests processed at each batch width; surplus waits "
            "so continuous backfill (vs static-cohort drain) is exercised when concurrency < total.",
            "Block budget is admission accounting over the phase-05 paged free-block model; actual KV "
            "still lives in ModelRunner/HF caches (no persistent paged runtime cache yet).",
            "Speculative decoding is measured separately; phase 06 does not implement batched accept/replay.",
        ],
    )

    scheduler_points: list[dict] = []
    print(
        f"\n[batched] Concurrency sweep {concurrency_grid} "
        f"(pool={total_requests}, warmup={warmup_runs}) ..."
    )
    for c in concurrency_grid:
        print(f"  concurrency={c} ...", end="", flush=True)
        point, metrics = _run_scheduler_point(
            runner,
            PROMPTS,
            concurrency=c,
            total_requests=total_requests,
            max_tokens=max_tokens,
            block_size=block_size,
            max_blocks=max_blocks,
            seed=seed,
            profile_name=profile_name,
            continuous=True,
            vary_lengths=vary_lengths,
            warmup_runs=warmup_runs,
        )
        result.concurrency_sweep.append(point)
        row = {"concurrency": c, "continuous_tok_s": point.throughput_tok_s, **metrics}

        if static_baseline and c < total_requests:
            static_point, _ = _run_scheduler_point(
                runner,
                PROMPTS,
                concurrency=c,
                total_requests=total_requests,
                max_tokens=max_tokens,
                block_size=block_size,
                max_blocks=max_blocks,
                seed=seed,
                profile_name=profile_name,
                continuous=False,
                vary_lengths=vary_lengths,
                warmup_runs=warmup_runs,
            )
            row["static_tok_s"] = static_point.throughput_tok_s
            row["continuous_speedup_vs_static"] = (
                point.throughput_tok_s / static_point.throughput_tok_s
                if static_point.throughput_tok_s else float("nan")
            )

        scheduler_points.append(row)
        msg = (
            f"  toks/s={point.throughput_tok_s:.1f} "
            f"peak_blocks={metrics['max_blocks_used']} "
            f"peak_vram={point.peak_vram_mb:.0f}MiB"
        )
        if "static_tok_s" in row:
            msg += (f"  static={row['static_tok_s']:.1f} "
                    f"cont/static={row['continuous_speedup_vs_static']:.2f}x")
        print(msg)

    _write_result(result, scheduler_points, results_dir)
    return result


def _write_result(
    result: BenchResult,
    scheduler_points: list[dict],
    results_dir: Path | None,
) -> Path:
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = results_dir / f"{ts}_{result.engine}_{Path(result.model).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"

    payload = dataclasses.asdict(result)
    payload["scheduler_points"] = scheduler_points
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[batched] Result written to {out_path}")
    return out_path
