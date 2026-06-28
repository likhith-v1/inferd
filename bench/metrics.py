"""
bench.metrics — frozen metric definitions and environment stamping.

FROZEN CONTRACT (phase 02 → all later phases):
  ttft(start, first_token_time)          → float (seconds)
  itl(total_time, ttft_s, num_tokens)    → float (seconds/token)
  throughput(total_tokens, wall_time)    → float (tokens/second)
  VramSampler                            → context manager, .peak_mb property
  env_stamp(seed, workload_hash, ...)    → dict  (stamped on every result)

Do NOT redefine these in any later phase. Import from this module instead.
If a definition needs updating, open a PR against this file and document the
change in DECISIONS.md — all previously recorded numbers become incomparable.

Design notes:
  - TTFT and ITL are single-stream (concurrency=1) metrics only.
    At concurrency>1, batched generate() emits all first tokens together after
    a shared prefill; TTFT/ITL are not per-request comparable across the sweep
    and must NOT be reported at concurrency>1.
  - peak_vram_comparable uses nvidia-smi polling (VramSampler) and is the ONLY
    VRAM number compared across engines, including the vLLM subprocess.
    torch.cuda.max_memory_allocated() is supplementary (in-process only).
  - vLLM pre-allocates KV by gpu_memory_utilization (default 0.9), so its
    "peak VRAM" ≈ 90% of the card regardless of workload — flag this rather
    than naively comparing it to HF or our engine.
  - TextIteratorStreamer first-yield can lag the true first token slightly
    (buffering to UTF-8 / word boundaries). Noted; acceptable approximation.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def ttft(start: float, first_token_time: float) -> float:
    """Time-to-first-token in seconds. Single-stream (concurrency=1) only."""
    return first_token_time - start


def itl(total_time: float, ttft_s: float, num_tokens: int) -> float:
    """
    Mean inter-token latency in seconds/token. Single-stream only.

    Defined as (decode_time) / (num_new_tokens - 1) where decode_time
    excludes the first token (which is the prefill + first decode step).
    Returns 0.0 when num_tokens <= 1 (no decode steps after the first token).
    """
    if num_tokens <= 1:
        return 0.0
    decode_time = total_time - ttft_s
    return decode_time / (num_tokens - 1)


def throughput(total_tokens: int, wall_time: float) -> float:
    """
    Throughput in tokens/second. Used for the concurrency sweep.

    total_tokens: total new tokens generated across ALL sequences in the batch.
    wall_time:    wall-clock time from batch generate() start to finish.
    """
    if wall_time <= 0:
        return 0.0
    return total_tokens / wall_time


class VramSampler:
    """
    Background thread that polls nvidia-smi every `interval_s` seconds and
    records the maximum GPU memory used (MiB) during the measurement window.

    This is the *comparable* VRAM metric — it works for in-process (HF) and
    subprocess (vLLM) engines alike.  Use as a context manager:

        with VramSampler() as vs:
            model.generate(...)
        print(vs.peak_mb)  # MiB

    For in-process engines, also call torch.cuda.max_memory_allocated() as a
    supplementary number — it is more precise but not cross-engine comparable.
    """

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.25):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_s):
            mb = self._query_mb()
            if mb is not None and mb > self._peak_mb:
                self._peak_mb = mb

    def _query_mb(self) -> Optional[float]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return float(out.strip())
        except Exception:
            return None

    def start(self) -> "VramSampler":
        # Take an initial reading before the run starts.
        mb = self._query_mb()
        if mb is not None:
            self._peak_mb = mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Final reading after run ends.
        mb = self._query_mb()
        if mb is not None and mb > self._peak_mb:
            self._peak_mb = mb
        return self._peak_mb

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    def __enter__(self) -> "VramSampler":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


@dataclass
class SingleStreamResult:
    """Metrics for a single-stream (concurrency=1) run."""
    ttft_s: float                  # time-to-first-token (seconds)
    itl_s: float                   # mean inter-token latency (seconds/token)
    total_time_s: float            # total generation wall time (seconds)
    tokens_generated: int          # new tokens produced (excluding prompt)
    throughput_tok_s: float        # tokens/second (= tokens_generated / total_time_s)
    peak_vram_mb: float            # comparable peak VRAM via nvidia-smi (MiB)
    peak_vram_torch_mb: float      # supplementary: torch allocator peak (MiB, in-process only)
    prompt: str                    # the input prompt
    warmup_runs: int               # warmup runs discarded before this measurement


@dataclass
class ConcurrencySweepPoint:
    """Metrics for one concurrency level in the sweep."""
    concurrency: int
    total_time_s: float            # batch wall time
    total_tokens: int              # total new tokens across all sequences
    throughput_tok_s: float        # total_tokens / total_time_s
    peak_vram_mb: float            # nvidia-smi peak (MiB)
    peak_vram_torch_mb: float      # torch allocator peak (MiB, in-process only)
    warmup_runs: int


@dataclass
class BenchResult:
    """Top-level result object written to bench/results/<timestamp>/result.json."""
    engine: str                    # "hf" | "vllm"
    role: str                      # "floor" | "ceiling"
    model: str                     # e.g. "Qwen3.5-9B"
    profile: str                   # "canonical" | "greedy"
    max_tokens: int
    env: dict                      # env_stamp() output
    single_stream: list[SingleStreamResult] = field(default_factory=list)
    concurrency_sweep: list[ConcurrencySweepPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def env_stamp(seed: int, workload_hash_str: str, extra: Optional[dict] = None) -> dict:
    """
    Collect a reproducibility stamp for a benchmark result.

    Includes: GPU info, CUDA/driver, Python/torch/transformers versions,
    git commit, timestamp, seed, workload hash.
    """
    import platform
    import sys

    stamp: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "workload_hash": workload_hash_str,
        "python": sys.version,
        "platform": platform.platform(),
    }

    # Torch + CUDA
    try:
        import torch
        stamp["torch"] = torch.__version__
        stamp["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            stamp["gpu_name"] = torch.cuda.get_device_name(0)
            stamp["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            stamp["cuda_version"] = torch.version.cuda
    except Exception as exc:
        stamp["torch_error"] = str(exc)

    # transformers
    try:
        import transformers
        stamp["transformers"] = transformers.__version__
    except Exception:
        pass

    # nvidia-smi driver + CUDA runtime
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        parts = [p.strip() for p in smi_out.split(",")]
        if len(parts) >= 2:
            stamp["driver_version"] = parts[0]
            stamp["vram_total_mb"] = parts[1]
    except Exception:
        pass

    # git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        stamp["git_commit"] = commit
    except Exception:
        stamp["git_commit"] = "unknown"

    if extra:
        stamp.update(extra)

    return stamp
