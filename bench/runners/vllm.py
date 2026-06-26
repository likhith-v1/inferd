"""
bench.runners.vllm — vLLM ceiling runner (isolated subprocess).

vLLM pins its own torch version and must NOT enter the main uv.lock.
This module:
  1. Creates bench/.venv-vllm via `uv venv` (once).
  2. Installs vllm into that venv via `uv pip install vllm`.
  3. Writes a self-contained runner script to a temp file.
  4. Executes it as a subprocess with the venv's Python interpreter.
  5. Reads the JSON result written by the subprocess.

If any step fails (sm_120 not supported, install error, etc.), it logs a
clean "deferred" message and returns a result with role="ceiling_deferred"
rather than raising. The HF floor run is unaffected.

NOTE: vLLM pre-allocates KV by gpu_memory_utilization (default 0.9), so
its reported peak VRAM ≈ 90% of the card by design — flag in result notes,
do not naively compare to HF or our engine's peak.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench.metrics import BenchResult, ConcurrencySweepPoint, env_stamp
from bench.workload import CANONICAL, GREEDY, MAX_TOKENS, PROMPTS, SamplingProfile, workload_hash

WEIGHTS_ROOT = Path(__file__).parent.parent.parent / "weights"
VLLM_VENV = Path(__file__).parent.parent / ".venv-vllm"


# ---------------------------------------------------------------------------
# Isolated venv management
# ---------------------------------------------------------------------------

def _ensure_vllm_venv() -> Path:
    """Create bench/.venv-vllm and install vllm if not already done."""
    python_bin = VLLM_VENV / "bin" / "python"
    if python_bin.exists():
        print("[vllm] venv already exists at", VLLM_VENV)
        return python_bin

    print("[vllm] Creating isolated venv at", VLLM_VENV, "...")
    subprocess.run(
        ["uv", "venv", str(VLLM_VENV), "--python", "python3"],
        check=True,
    )

    print("[vllm] Installing vllm (this may take a few minutes) ...")
    result = subprocess.run(
        [
            "uv", "pip", "install", "vllm",
            "--python", str(python_bin),
            "--index-strategy", "unsafe-best-match",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"vllm install failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    print("[vllm] vllm installed successfully.")
    return python_bin


# ---------------------------------------------------------------------------
# Self-contained subprocess runner script
# ---------------------------------------------------------------------------

_RUNNER_SCRIPT = '''
"""vLLM ceiling runner — executed as a subprocess in the isolated venv."""
import json, sys, time, os
from pathlib import Path

model_dir = sys.argv[1]
result_path = sys.argv[2]
seed = int(sys.argv[3])
max_tokens = int(sys.argv[4])
profile_name = sys.argv[5]
concurrency_json = sys.argv[6]
concurrency_grid = json.loads(concurrency_json)

prompts_json = sys.argv[7]
prompts = json.loads(prompts_json)

from vllm import LLM, SamplingParams

temperature = 0.7 if profile_name == "canonical" else 0.0
top_p = 0.95 if profile_name == "canonical" else 1.0

llm = LLM(
    model=model_dir,
    dtype="bfloat16",
    seed=seed,
    gpu_memory_utilization=0.9,
)
sp = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    seed=seed,
)

results = {"concurrency_sweep": [], "notes": [
    "vLLM pre-allocates KV at gpu_memory_utilization=0.9; "
    "peak_vram_mb ~= 90% of card by design — do not compare naively."
]}

import subprocess as _sp

def _vram_mb():
    try:
        out = _sp.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            stderr=_sp.DEVNULL, timeout=2,
        )
        return float(out.strip())
    except Exception:
        return 0.0

# Warmup
print("[vllm-subprocess] Warmup ...", flush=True)
llm.generate(prompts[:1], sp)

for c in concurrency_grid:
    batch = [prompts[i % len(prompts)] for i in range(c)]
    t0 = time.perf_counter()
    outputs = llm.generate(batch, sp)
    t1 = time.perf_counter()
    wall = t1 - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / wall if wall > 0 else 0.0
    peak_mb = _vram_mb()
    results["concurrency_sweep"].append({
        "concurrency": c,
        "total_time_s": wall,
        "total_tokens": total_tokens,
        "throughput_tok_s": tps,
        "peak_vram_mb": peak_mb,
        "peak_vram_torch_mb": 0.0,
        "warmup_runs": 1,
    })
    print(f"[vllm-subprocess] c={c}  toks/s={tps:.1f}  peak_vram={peak_mb:.0f}MiB", flush=True)

with open(result_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"[vllm-subprocess] Wrote {result_path}", flush=True)
'''


# ---------------------------------------------------------------------------
# Public run() entry point
# ---------------------------------------------------------------------------

def run(
    model_name: str = "Qwen3.5-9B",
    seed: int = 0,
    max_tokens: int = MAX_TOKENS,
    concurrency_grid: list[int] | None = None,
    warmup_runs: int = 1,
    profile_name: str = "canonical",
    results_dir: Path | None = None,
) -> BenchResult:
    """
    Run the vLLM ceiling benchmark and return a BenchResult.

    On any failure (install, sm_120 unsupported, runtime error), returns a
    result with role="ceiling_deferred" and the error in notes[].
    """
    if concurrency_grid is None:
        concurrency_grid = [1, 2, 4, 8, 16]

    profile = CANONICAL if profile_name == "canonical" else GREEDY
    wh = workload_hash(profile)
    stamp = env_stamp(seed, wh)

    deferred_result = BenchResult(
        engine="vllm",
        role="ceiling_deferred",
        model=model_name,
        profile=profile_name,
        max_tokens=max_tokens,
        env=stamp,
        notes=[],
    )

    # Attempt install.
    try:
        python_bin = _ensure_vllm_venv()
    except Exception as exc:
        msg = f"vLLM ceiling deferred — venv/install failed: {exc}"
        print(f"[vllm] {msg}")
        deferred_result.notes.append(msg)
        _write_deferred(deferred_result, results_dir)
        return deferred_result

    # Write runner script to temp file.
    weights_dir = str(WEIGHTS_ROOT / model_name)

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = results_dir / f"{ts}_vllm_{model_name.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess_result_path = str(out_dir / "vllm_subprocess_result.json")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_vllm_runner.py", delete=False
    ) as tf:
        tf.write(_RUNNER_SCRIPT)
        runner_path = tf.name

    try:
        print("[vllm] Launching subprocess runner ...")
        proc = subprocess.run(
            [
                str(python_bin),
                runner_path,
                weights_dir,
                subprocess_result_path,
                str(seed),
                str(max_tokens),
                profile_name,
                json.dumps(concurrency_grid),
                json.dumps(PROMPTS),
            ],
            capture_output=False,  # stream stdout/stderr live
            timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"vLLM subprocess exited with code {proc.returncode}")
    except Exception as exc:
        msg = f"vLLM ceiling deferred — subprocess failed: {exc}"
        print(f"[vllm] {msg}")
        deferred_result.notes.append(msg)
        _write_deferred(deferred_result, results_dir)
        return deferred_result
    finally:
        os.unlink(runner_path)

    # Read subprocess JSON and build BenchResult.
    try:
        with open(subprocess_result_path) as fh:
            sub = json.load(fh)
    except Exception as exc:
        msg = f"vLLM ceiling deferred — could not read subprocess result: {exc}"
        print(f"[vllm] {msg}")
        deferred_result.notes.append(msg)
        _write_deferred(deferred_result, results_dir)
        return deferred_result

    sweep = [ConcurrencySweepPoint(**p) for p in sub.get("concurrency_sweep", [])]
    result = BenchResult(
        engine="vllm",
        role="ceiling",
        model=model_name,
        profile=profile_name,
        max_tokens=max_tokens,
        env=stamp,
        concurrency_sweep=sweep,
        notes=sub.get("notes", []),
    )

    import dataclasses
    final_path = out_dir / "result.json"
    with open(final_path, "w") as fh:
        json.dump(dataclasses.asdict(result), fh, indent=2)
    print(f"[vllm] Result written to {final_path}")
    return result


def _write_deferred(result: BenchResult, results_dir: Path | None) -> None:
    import dataclasses

    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = results_dir / f"{ts}_vllm_deferred"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    with open(out_path, "w") as fh:
        json.dump(dataclasses.asdict(result), fh, indent=2)
    print(f"[vllm] Deferred result written to {out_path}")
