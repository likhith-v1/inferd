"""Apple-only MLX scheduler benchmark rung."""

from __future__ import annotations

import platform
import resource
import subprocess
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from math import ceil
from pathlib import Path

from bench.metrics import env_stamp, itl, throughput, ttft, write_result_json
from bench.workload import CANONICAL, GREEDY, MAX_TOKENS, PROMPTS, workload_hash

APPLE_RESULTS = (Path(__file__).parent.parent / "results" / "apple").resolve()


@dataclass
class MlxSingleStreamResult:
    ttft_s: float
    itl_s: float
    total_time_s: float
    tokens_generated: int
    throughput_tok_s: float
    mlx_peak_allocated_mb: float
    process_rss_peak_mb: float
    prompt: str
    warmup_runs: int


@dataclass
class MlxSweepPoint:
    concurrency: int
    total_time_s: float
    total_tokens: int
    throughput_tok_s: float
    mlx_peak_allocated_mb: float
    process_rss_peak_mb: float
    warmup_runs: int


@dataclass
class MlxBenchResult:
    engine: str
    role: str
    model: str
    profile: str
    max_tokens: int
    env: dict
    artifact_fingerprint: str
    single_stream: list[MlxSingleStreamResult] = field(default_factory=list)
    concurrency_sweep: list[MlxSweepPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _apple_results_dir(results_dir: Path | None) -> Path:
    candidate = APPLE_RESULTS if results_dir is None else results_dir.expanduser().resolve()
    try:
        candidate.relative_to(APPLE_RESULTS)
    except ValueError as exc:
        raise ValueError(f"MLX results must stay under {APPLE_RESULTS} (bench/results/apple)") from exc
    return candidate


def _rss_peak_mb() -> float:
    if platform.system() != "Darwin":
        raise RuntimeError("the MLX benchmark rung is Apple/macOS only")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _total_ram_mb() -> float:
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1024**2


def _scheduler(runner, prompts, *, concurrency, max_tokens, profile, seed):
    from backends.mlx.backend import MlxSchedulerBackend
    from core.scheduler import ContinuousBatchScheduler, SchedulerConfig

    prompt_ids = [runner.tokenizer(prompt, add_special_tokens=True).input_ids for prompt in prompts]
    block_size = 16
    max_blocks = concurrency * ceil((max(map(len, prompt_ids)) + max_tokens) / block_size)
    scheduler = ContinuousBatchScheduler(
        MlxSchedulerBackend(runner),
        SchedulerConfig(
            max_blocks=max_blocks,
            block_size=block_size,
            max_concurrent_sequences=concurrency,
            max_model_len=max(map(len, prompt_ids)) + max_tokens,
            temperature=profile.temperature,
            top_p=profile.top_p,
            seed=seed,
        ),
    )
    for index, (prompt, ids) in enumerate(zip(prompts, prompt_ids), 1):
        scheduler.submit(ids, max_tokens=max_tokens, prompt_text=prompt, request_id=index)
    return scheduler


def _warmup(runner, prompt: str, runs: int) -> None:
    import mlx.core as mx

    for index in range(runs):
        scheduler = _scheduler(
            runner, [prompt], concurrency=1, max_tokens=2, profile=GREEDY, seed=index
        )
        scheduler.run_until_complete()
        mx.synchronize()


def _single_stream(runner, prompt, profile, max_tokens, warmup_runs, seed):
    import mlx.core as mx

    _warmup(runner, prompt, warmup_runs)
    scheduler = _scheduler(
        runner, [prompt], concurrency=1, max_tokens=max_tokens, profile=profile, seed=seed
    )
    mx.reset_peak_memory()
    start = time.perf_counter()
    scheduler.step()
    mx.synchronize()
    first = time.perf_counter()
    completed = scheduler.run_until_complete()
    mx.synchronize()
    end = time.perf_counter()
    tokens = completed[0].generated_len
    total = end - start
    first_s = ttft(start, first)
    return MlxSingleStreamResult(
        ttft_s=first_s,
        itl_s=itl(total, first_s, tokens),
        total_time_s=total,
        tokens_generated=tokens,
        throughput_tok_s=throughput(tokens, total),
        mlx_peak_allocated_mb=mx.get_peak_memory() / 1024**2,
        process_rss_peak_mb=_rss_peak_mb(),
        prompt=prompt,
        warmup_runs=warmup_runs,
    )


def _sweep(runner, concurrency, profile, max_tokens, warmup_runs, seed):
    import mlx.core as mx

    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(concurrency)]
    _warmup(runner, prompts[0], warmup_runs)
    scheduler = _scheduler(
        runner,
        prompts,
        concurrency=concurrency,
        max_tokens=max_tokens,
        profile=profile,
        seed=seed,
    )
    mx.reset_peak_memory()
    start = time.perf_counter()
    completed = scheduler.run_until_complete()
    mx.synchronize()
    wall = time.perf_counter() - start
    total_tokens = sum(request.generated_len for request in completed)
    return MlxSweepPoint(
        concurrency=concurrency,
        total_time_s=wall,
        total_tokens=total_tokens,
        throughput_tok_s=throughput(total_tokens, wall),
        mlx_peak_allocated_mb=mx.get_peak_memory() / 1024**2,
        process_rss_peak_mb=_rss_peak_mb(),
        warmup_runs=warmup_runs,
    )


def run(
    model_name: str,
    seed: int = 0,
    max_tokens: int = MAX_TOKENS,
    concurrency_grid: list[int] | None = None,
    warmup_runs: int = 3,
    profile_name: str = "greedy",
    results_dir: Path | None = None,
) -> MlxBenchResult:
    import mlx.core as mx

    from backends.mlx.runner import MlxModelRunner

    if platform.system() != "Darwin" or not mx.metal.is_available():
        raise RuntimeError("--engine mlx requires Apple Silicon with Metal")
    concurrency_grid = [1, 2, 4] if concurrency_grid is None else concurrency_grid
    if not concurrency_grid or any(value <= 0 for value in concurrency_grid):
        raise ValueError("concurrency values must be positive")
    output_root = _apple_results_dir(results_dir)
    profile = CANONICAL if profile_name == "canonical" else GREEDY
    runner = MlxModelRunner.load(model_name)
    wh = workload_hash(profile, max_tokens)
    result = MlxBenchResult(
        engine="mlx",
        role="apple_baseline",
        model="Qwen3-8B MLX 4-bit",
        profile=profile_name,
        max_tokens=max_tokens,
        artifact_fingerprint=runner.artifact.fingerprint,
        env=env_stamp(seed, wh, extra={
            "macos": platform.mac_ver()[0],
            "total_ram_mb": _total_ram_mb(),
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "artifact_fingerprint": runner.artifact.fingerprint,
        }),
        notes=[
            "Apple-only MLX 4-bit baseline; no ratio or parity claim versus CUDA bf16.",
            "mlx_peak_allocated_mb is the MLX allocator peak; process_rss_peak_mb is process-wide RSS peak.",
            "The frozen NVIDIA-comparable peak_vram_mb field is intentionally absent.",
        ],
    )
    for index, prompt in enumerate(PROMPTS):
        point = _single_stream(
            runner, prompt, profile, max_tokens, warmup_runs, seed + index
        )
        result.single_stream.append(point)
        print(f"[mlx] prompt={index + 1}/{len(PROMPTS)} TTFT={point.ttft_s * 1000:.1f}ms tok/s={point.throughput_tok_s:.1f}")
    for concurrency in concurrency_grid:
        point = _sweep(
            runner, concurrency, profile, max_tokens, warmup_runs, seed
        )
        result.concurrency_sweep.append(point)
        print(f"[mlx] c={concurrency} tok/s={point.throughput_tok_s:.1f} MLX={point.mlx_peak_allocated_mb:.0f}MiB RSS={point.process_rss_peak_mb:.0f}MiB")
    path = write_result_json(result, "mlx_Qwen3-8B-4bit", output_root)
    print(f"[mlx] Result written to {path}")
    return result
