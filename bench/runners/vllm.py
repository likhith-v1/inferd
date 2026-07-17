"""Pinned vLLM ceiling runner executed in an isolated subprocess.

The ceiling environment is defined and locked under ``bench/vllm`` and synced
into ``bench/.venv-vllm``.  vLLM and its torch pin never enter the main project
environment or root ``uv.lock``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from bench.metrics import BenchResult, ConcurrencySweepPoint, env_stamp, write_result_json
from bench.workload import CANONICAL, GREEDY, MAX_TOKENS, PROMPTS, workload_hash

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ROOT = ROOT / "weights"
VLLM_PROJECT = ROOT / "bench" / "vllm"
VLLM_VENV = ROOT / "bench" / ".venv-vllm"
VLLM_VERSION = "0.23.0"
TORCH_VERSION = "2.11.0"
CUDA_VERSION = "13.0"
PYTHON_MINOR = (3, 13)


class VllmEnvironmentError(RuntimeError):
    """An isolated-environment operation failed with retained diagnostics."""


def _format_process_diagnostics(label: str, proc: subprocess.CompletedProcess) -> str:
    return (
        f"{label} (exit={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout or ''}\n"
        f"--- stderr ---\n{proc.stderr or ''}"
    )


def _sync_command() -> list[str]:
    return [
        "uv",
        "sync",
        "--project",
        str(VLLM_PROJECT),
        "--locked",
        "--python",
        "3.13",
        "--no-install-project",
    ]


def _sync_environment() -> subprocess.CompletedProcess:
    sync_env = os.environ.copy()
    sync_env["UV_PROJECT_ENVIRONMENT"] = str(VLLM_VENV)
    return subprocess.run(
        _sync_command(),
        cwd=ROOT,
        env=sync_env,
        capture_output=True,
        text=True,
    )


_VALIDATE_SCRIPT = f"""
import importlib.metadata as metadata
import json
import sys
import torch
import vllm

facts = {{
    "python_version": sys.version,
    "python_minor": list(sys.version_info[:2]),
    "torch_version": metadata.version("torch"),
    "vllm_version": metadata.version("vllm"),
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_capability": (
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    ),
}}
print(json.dumps(facts, sort_keys=True))
assert tuple(sys.version_info[:2]) == {PYTHON_MINOR!r}, facts
assert facts["torch_version"] == {TORCH_VERSION!r}, facts
assert facts["vllm_version"] == {VLLM_VERSION!r}, facts
assert facts["torch_cuda_version"] == {CUDA_VERSION!r}, facts
assert facts["cuda_available"], facts
assert facts["gpu_capability"] == [12, 0], facts
"""


def _validate_environment(python_bin: Path) -> dict:
    proc = subprocess.run(
        [str(python_bin), "-c", _VALIDATE_SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VllmEnvironmentError(_format_process_diagnostics("vLLM validation failed", proc))
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise VllmEnvironmentError(
            f"vLLM validation returned invalid JSON: {exc}\n{proc.stdout}\n{proc.stderr}"
        ) from exc


def _remove_isolated_environment() -> None:
    """Remove only the exact, repo-local vLLM environment before a repair sync."""
    if VLLM_VENV.exists():
        shutil.rmtree(VLLM_VENV)


def _ensure_vllm_venv() -> tuple[Path, dict]:
    """Locked-sync and validate the isolated environment, repairing it once."""
    python_bin = VLLM_VENV / "bin" / "python"
    diagnostics: list[str] = []

    for attempt in range(2):
        proc = _sync_environment()
        if proc.returncode == 0:
            try:
                facts = _validate_environment(python_bin)
                if proc.stdout.strip():
                    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
                if proc.stderr.strip():
                    print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
                return python_bin, facts
            except Exception as exc:
                diagnostics.append(str(exc))
        else:
            diagnostics.append(_format_process_diagnostics("vLLM locked sync failed", proc))

        if attempt == 0:
            print("[vllm] Isolated environment is incomplete/stale; rebuilding once ...")
            _remove_isolated_environment()

    raise VllmEnvironmentError("\n\n".join(diagnostics))


_RUNNER_SCRIPT = r'''
"""vLLM ceiling runner — executed in the pinned isolated environment.

All executable logic runs under ``if __name__ == "__main__"``. vLLM 0.23.0 on
WSL forces the ``spawn`` multiprocessing start method (CUDA is initialised and
NVML is not fork-safe under WSL), so every engine-core worker re-imports this
module. Without the guard, that re-import would reconstruct ``LLM`` before the
parent finished bootstrapping and crash with "An attempt has been made to start
a new process before the current process has finished its bootstrapping phase".
"""
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path


def main():
    model_dir = Path(sys.argv[1]).expanduser().resolve()
    result_path = sys.argv[2]
    repo_root = sys.argv[3]
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    seed = int(sys.argv[4])
    max_tokens = int(sys.argv[5])
    profile_name = sys.argv[6]
    concurrency_grid = json.loads(sys.argv[7])
    prompts = json.loads(sys.argv[8])
    warmup_runs = int(sys.argv[9])
    expected_workload_hash = sys.argv[10]

    import torch
    from vllm import LLM, SamplingParams
    from bench.metrics import VramSampler
    from bench.workload import model_fingerprint

    temperature = 0.7 if profile_name == "canonical" else 0.0
    top_p = 0.95 if profile_name == "canonical" else 1.0
    # FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel at runtime, and
    # its bundled CCCL headers are incompatible with the cu13 toolkit on sm_120
    # ("CUDA compiler and CUDA toolkit headers are incompatible"). We disable ONLY
    # the sampler backend (VLLM_USE_FLASHINFER_SAMPLER=0, set in the parent env)
    # and fall back to vLLM's native Torch sampler. Attention stays
    # FlashAttention-2, CUDA graphs stay enabled, no quantization. For the greedy
    # cohort this is argmax either way (numerically identical), and the sampler is
    # a negligible fraction of decode time — the throughput ceiling is intact.
    effective_config = {
        "dtype": "bfloat16",
        "language_model_only": True,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": False,
        "flashinfer_sampler": False,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_tokens": max_tokens,
    }

    llm = LLM(
        model=str(model_dir),
        dtype=effective_config["dtype"],
        seed=seed,
        gpu_memory_utilization=effective_config["gpu_memory_utilization"],
        language_model_only=effective_config["language_model_only"],
        enforce_eager=effective_config["enforce_eager"],
    )
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    results = {
        "concurrency_sweep": [],
        "notes": [
            "vLLM pre-allocates KV at gpu_memory_utilization=0.9; "
            "peak_vram_mb ~= 90% of card by design — do not compare naively.",
            "VLLM_USE_FLASHINFER_SAMPLER=0: native Torch sampler (the FlashInfer "
            "sampler JIT is incompatible with the cu13 headers on sm_120). Greedy "
            "is argmax regardless; attention (FlashAttention-2) and CUDA graphs "
            "are unchanged.",
        ],
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": metadata.version("torch"),
            "vllm": metadata.version("vllm"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": (
                list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            ),
            "effective_config": effective_config,
        },
        "provenance": {
            "model_fingerprint": model_fingerprint(model_dir),
            "workload_hash": expected_workload_hash,
            "profile": profile_name,
            "seed": seed,
            "max_tokens": max_tokens,
            "warmup_runs": warmup_runs,
            "concurrency_grid": concurrency_grid,
        },
    }

    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        ).strip().split(",")
        results["env"]["driver_version"] = smi[0].strip()
        results["env"]["vram_total_mb"] = float(smi[1].strip())
    except Exception as exc:
        results["env"]["nvidia_smi_error"] = repr(exc)

    for c in concurrency_grid:
        batch = [prompts[i % len(prompts)] for i in range(c)]
        for warmup_index in range(warmup_runs):
            print(
                f"[vllm-subprocess] c={c} warmup {warmup_index + 1}/{warmup_runs} ...",
                flush=True,
            )
            llm.generate(batch, sp)
        t0 = time.perf_counter()
        with VramSampler() as vs:
            outputs = llm.generate(batch, sp)
        wall = time.perf_counter() - t0
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tps = total_tokens / wall if wall > 0 else 0.0
        results["concurrency_sweep"].append({
            "concurrency": c,
            "total_time_s": wall,
            "total_tokens": total_tokens,
            "throughput_tok_s": tps,
            "peak_vram_mb": vs.peak_mb,
            "peak_vram_torch_mb": 0.0,
            "warmup_runs": warmup_runs,
        })
        print(
            f"[vllm-subprocess] c={c} toks/s={tps:.1f} peak_vram={vs.peak_mb:.0f}MiB",
            flush=True,
        )

    with open(result_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"[vllm-subprocess] Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
'''


def _subprocess_diagnostics(proc: subprocess.CompletedProcess) -> str:
    return _format_process_diagnostics("vLLM benchmark subprocess failed", proc)


def run(
    model_name: str = "Qwen3.5-9B",
    seed: int = 0,
    max_tokens: int = MAX_TOKENS,
    concurrency_grid: list[int] | None = None,
    warmup_runs: int = 1,
    profile_name: str = "canonical",
    results_dir: Path | None = None,
    cohort_id: str | None = None,
    provenance: dict | None = None,
) -> BenchResult:
    """Run the vLLM ceiling, returning an honest deferred result on failure."""
    if concurrency_grid is None:
        concurrency_grid = [1, 2, 4, 8, 16]
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if profile_name not in {"canonical", "greedy"}:
        raise ValueError(f"unknown sampling profile: {profile_name}")

    profile = CANONICAL if profile_name == "canonical" else GREEDY
    wh = workload_hash(profile, max_tokens)
    stamp = env_stamp(seed, wh)
    base_provenance = {
        **(provenance or {}),
        "workload_hash": wh,
        "profile": profile_name,
        "seed": seed,
        "max_tokens": max_tokens,
        "warmup_runs": warmup_runs,
        "concurrency_grid": list(concurrency_grid),
    }
    deferred_result = BenchResult(
        engine="vllm",
        role="ceiling_deferred",
        model=model_name,
        profile=profile_name,
        max_tokens=max_tokens,
        env=stamp,
        notes=[],
        cohort_id=cohort_id,
        provenance=base_provenance,
    )

    model_dir = (WEIGHTS_ROOT / model_name).resolve()
    if not model_dir.is_dir():
        msg = f"vLLM ceiling deferred — stock model directory is missing: {model_dir}"
        deferred_result.notes.append(msg)
        _write_deferred(deferred_result, results_dir)
        return deferred_result

    try:
        python_bin, validation_facts = _ensure_vllm_venv()
        deferred_result.env["isolated_validation"] = validation_facts
    except Exception as exc:
        msg = f"vLLM ceiling deferred — isolated environment failed:\n{exc}"
        print(f"[vllm] {msg}")
        deferred_result.notes.append(msg)
        _write_deferred(deferred_result, results_dir)
        return deferred_result

    result_file = tempfile.NamedTemporaryFile(suffix="_vllm_result.json", delete=False)
    result_path = result_file.name
    result_file.close()
    with tempfile.NamedTemporaryFile(mode="w", suffix="_vllm_runner.py", delete=False) as handle:
        handle.write(_RUNNER_SCRIPT)
        runner_path = handle.name

    # FlashInfer's sampler JIT is incompatible with the cu13 headers on sm_120,
    # so force vLLM's native Torch sampler. This is the only vLLM behavioural
    # deviation from stock defaults; it is recorded in the result's
    # effective_config/notes and documented in DECISIONS.md. Attention
    # (FlashAttention-2), CUDA graphs, dtype, and gpu_memory_utilization are all
    # left at vLLM defaults so the ceiling stays representative.
    run_env = os.environ.copy()
    run_env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    proc: subprocess.CompletedProcess | None = None
    try:
        proc = subprocess.run(
            [
                str(python_bin),
                runner_path,
                str(model_dir),
                result_path,
                str(ROOT),
                str(seed),
                str(max_tokens),
                profile_name,
                json.dumps(concurrency_grid),
                json.dumps(PROMPTS),
                str(warmup_runs),
                wh,
            ],
            cwd=ROOT,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        if proc.returncode != 0:
            raise VllmEnvironmentError(_subprocess_diagnostics(proc))
    except subprocess.TimeoutExpired as exc:
        diagnostics = (
            "vLLM benchmark subprocess timed out after 3600 seconds\n"
            f"--- stdout ---\n{exc.stdout or ''}\n--- stderr ---\n{exc.stderr or ''}"
        )
        msg = f"vLLM ceiling deferred — subprocess failed:\n{diagnostics}"
        deferred_result.notes.append(msg)
        Path(result_path).unlink(missing_ok=True)
        _write_deferred(deferred_result, results_dir)
        return deferred_result
    except Exception as exc:
        msg = f"vLLM ceiling deferred — subprocess failed:\n{exc}"
        print(f"[vllm] {msg}")
        deferred_result.notes.append(msg)
        Path(result_path).unlink(missing_ok=True)
        _write_deferred(deferred_result, results_dir)
        return deferred_result
    finally:
        Path(runner_path).unlink(missing_ok=True)

    try:
        sub = json.loads(Path(result_path).read_text())
    except Exception as exc:
        diagnostics = _subprocess_diagnostics(proc) if proc is not None else "no diagnostics"
        msg = f"vLLM ceiling deferred — invalid subprocess result: {exc}\n{diagnostics}"
        deferred_result.notes.append(msg)
        _write_deferred(deferred_result, results_dir)
        return deferred_result
    finally:
        Path(result_path).unlink(missing_ok=True)

    sweep = [ConcurrencySweepPoint(**point) for point in sub.get("concurrency_sweep", [])]
    isolated_env = sub.get("env", {})
    stamp.update(isolated_env)
    result_provenance = {**base_provenance, **sub.get("provenance", {})}
    result = BenchResult(
        engine="vllm",
        role="ceiling",
        model=str(model_dir),
        profile=profile_name,
        max_tokens=max_tokens,
        env=stamp,
        concurrency_sweep=sweep,
        notes=sub.get("notes", []),
        cohort_id=cohort_id,
        provenance=result_provenance,
    )
    final_path = write_result_json(
        result, f"vllm_{model_name.replace('/', '_')}", results_dir
    )
    print(f"[vllm] Result written to {final_path}")
    return result


def _write_deferred(result: BenchResult, results_dir: Path | None) -> None:
    out_path = write_result_json(result, "vllm_deferred", results_dir)
    print(f"[vllm] Deferred result written to {out_path}")
