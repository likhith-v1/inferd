"""
bench.runners.spec — speculative-decoding benchmark (phase 04).

Measures, on the frozen CANONICAL workload:
  - baseline target-only autoregressive throughput (same ModelRunner.forward
    code path, for a fair wall-clock comparison),
  - for each gamma: acceptance rate alpha, measured throughput + speedup, and the
    theoretical tokens-per-round (1 - alpha^(gamma+1)) / (1 - alpha),
  - alpha-lift: pass a distilled draft adapter and compare alpha/throughput to
    the stock draft (target fixed).

Reuses bench.metrics (throughput/env_stamp), bench.workload (CANONICAL/PROMPTS/
workload_hash). Results are append-only JSON under bench/results/.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import inferd.env  # noqa: F401

import torch  # noqa: E402

from bench.metrics import env_stamp, throughput, write_result_json  # noqa: E402
from bench.workload import CANONICAL, MAX_TOKENS, PROMPTS, workload_hash  # noqa: E402
from core.spec_decode import nucleus_probs, sample_from, speculative_generate  # noqa: E402


@torch.no_grad()
def _baseline_generate(target, prompt_ids, max_tokens, temperature, top_p, seed) -> int:
    """Plain target-only nucleus decode (token-by-token). Returns #tokens."""
    torch.manual_seed(seed)
    logits, kv = target.forward(prompt_ids, None)
    n = 0
    eos = target.tokenizer.eos_token_id
    for _ in range(max_tokens):
        probs = nucleus_probs(logits[:, -1, :], temperature, top_p)
        tok = sample_from(probs)
        n += 1
        if int(tok.item()) == eos:
            break
        logits, kv = target.forward(tok, kv)
    return n


def _time_baseline(target, prompts, max_tokens, profile, seed) -> tuple[float, int]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for i, p in enumerate(prompts):
        ids = target.tokenizer(p, return_tensors="pt").input_ids.to(target.device)
        total += _baseline_generate(target, ids, max_tokens, profile.temperature,
                                    profile.top_p, seed + i)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, total


def _time_spec(target, draft, prompts, gamma, max_tokens, profile, seed):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total, accepted, proposed, rounds = 0, 0, 0, 0
    for i, p in enumerate(prompts):
        ids = target.tokenizer(p, return_tensors="pt").input_ids.to(target.device)
        out = speculative_generate(
            target, draft, ids, gamma=gamma, max_tokens=max_tokens,
            temperature=profile.temperature, top_p=profile.top_p, seed=seed + i,
            eos_id=target.tokenizer.eos_token_id,
        )
        total += out.stats.generated
        accepted += out.stats.accepted
        proposed += out.stats.proposed
        rounds += out.stats.rounds
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    alpha = accepted / proposed if proposed else 0.0
    return wall, total, alpha, rounds


def run(
    target_path: str = "merged/9b",
    draft_path: str = "weights/Qwen3.5-0.8B",
    *,
    draft_adapter: Optional[str] = None,
    gammas: list[int] | None = None,
    max_tokens: int = 128,
    seed: int = 0,
    n_prompts: int | None = None,
    results_dir: Path | None = None,
    device: str = "cuda:0",
) -> dict:
    from core.model_runner import ModelRunner

    gammas = gammas or [2, 4, 8]
    profile = CANONICAL
    prompts = PROMPTS if n_prompts is None else PROMPTS[:n_prompts]

    target = ModelRunner.load_target(target_path, device=device)
    draft = ModelRunner.load_draft(draft_path, adapter=draft_adapter, device=device)
    draft_label = "distilled" if draft_adapter else "stock"

    print(f"[spec] baseline (target-only) over {len(prompts)} prompts ...")
    base_wall, base_tokens = _time_baseline(target, prompts, max_tokens, profile, seed)
    base_tps = throughput(base_tokens, base_wall)
    print(f"  baseline tok/s={base_tps:.1f}")

    sweep = []
    for g in gammas:
        wall, tokens, alpha, rounds = _time_spec(
            target, draft, prompts, g, max_tokens, profile, seed)
        tps = throughput(tokens, wall)
        theory = (1 - alpha ** (g + 1)) / (1 - alpha) if alpha < 1 else g + 1
        sweep.append({
            "gamma": g, "alpha": round(alpha, 4),
            "throughput_tok_s": round(tps, 2),
            "speedup_vs_baseline": round(tps / base_tps, 3) if base_tps else 0.0,
            "theory_tokens_per_round": round(theory, 3),
            "rounds": rounds, "tokens": tokens,
        })
        print(f"  gamma={g}: alpha={alpha:.3f} tok/s={tps:.1f} "
              f"speedup={tps/base_tps:.2f}x theory_tpr={theory:.2f}")

    result = {
        "engine": "spec",
        "target": target_path,
        "draft": draft_path,
        "draft_label": draft_label,
        "draft_adapter": draft_adapter,
        "profile": "canonical",
        "max_tokens": max_tokens,
        "baseline_tok_s": round(base_tps, 2),
        "sweep": sweep,
        "env": env_stamp(seed, workload_hash(profile, max_tokens)),
    }
    _write(result, results_dir)
    return result


def _write(result: dict, results_dir: Path | None) -> Path:
    out_path = write_result_json(result, f"spec_{result['draft_label']}", results_dir)
    print(f"\n[spec] result written to {out_path}")
    return out_path
