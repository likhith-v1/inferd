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
from statistics import median
from typing import Optional

import inferd.env  # noqa: F401

import torch  # noqa: E402

from bench.metrics import env_stamp, throughput, write_result_json  # noqa: E402
from bench.workload import (  # noqa: E402
    CANONICAL,
    PROMPTS,
    model_fingerprint,
    workload_hash,
)
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
    return wall, total, alpha, rounds, accepted, proposed


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
    warmup_runs: int = 0,
    repeats: int = 1,
    pair_config=None,
) -> dict:
    from core.model_runner import ModelRunner

    gammas = gammas or [2, 4, 8]
    if repeats <= 0 or warmup_runs < 0:
        raise ValueError("repeats must be positive and warmup_runs non-negative")
    if pair_config and draft_adapter:
        raise ValueError("named Phase 14 pairs do not support draft adapters")
    if pair_config:
        from bench.pair_configs import validate_local_revisions
        resolved_revisions = validate_local_revisions(pair_config)
    else:
        resolved_revisions = None
    profile = CANONICAL
    prompts = PROMPTS if n_prompts is None else PROMPTS[:n_prompts]

    target = ModelRunner.load_target(target_path, device=device)
    draft = ModelRunner.load_draft(draft_path, adapter=draft_adapter, device=device)
    draft_label = "distilled" if draft_adapter else "stock"
    expected_sha = pair_config.tokenizer_sha256 if pair_config else None
    expected_vocab = pair_config.vocab_size if pair_config else None
    tokenizer_sha = target.validate_speculation_pair(
        draft,
        expected_tokenizer_sha256=expected_sha,
        expected_vocab_size=expected_vocab,
    )
    if pair_config and (
        target.cache_reconciliation != pair_config.reconciliation
        or draft.cache_reconciliation != pair_config.reconciliation
    ):
        raise ValueError(
            f"pair {pair_config.name} expected {pair_config.reconciliation}, got "
            f"target={target.cache_reconciliation}, draft={draft.cache_reconciliation}"
        )

    raw_repeats = []
    for repeat in range(repeats):
        order = gammas[repeat % len(gammas):] + gammas[:repeat % len(gammas)]
        for _ in range(warmup_runs):
            _time_baseline(target, prompts, max_tokens, profile, seed)
        print(f"[spec] repeat {repeat + 1}/{repeats} baseline over {len(prompts)} prompts ...")
        base_wall, base_tokens = _time_baseline(target, prompts, max_tokens, profile, seed)
        base_tps = throughput(base_tokens, base_wall)
        measured = []
        for g in order:
            for _ in range(warmup_runs):
                _time_spec(target, draft, prompts, g, max_tokens, profile, seed)
            wall, tokens, alpha, rounds, accepted, proposed = _time_spec(
                target, draft, prompts, g, max_tokens, profile, seed
            )
            tps = throughput(tokens, wall)
            speedup = tps / base_tps if base_tps else 0.0
            theory = (1 - alpha ** (g + 1)) / (1 - alpha) if alpha < 1 else g + 1
            measured.append({
                "gamma": g,
                "wall_time_s": wall,
                "tokens": tokens,
                "throughput_tok_s": tps,
                "alpha": alpha,
                "accepted": accepted,
                "proposed": proposed,
                "rounds": rounds,
                "speedup_vs_baseline": speedup,
                "theory_tokens_per_round": theory,
            })
            print(f"  gamma={g}: alpha={alpha:.3f} tok/s={tps:.1f} "
                  f"speedup={speedup:.2f}x theory_tpr={theory:.2f}")
        raw_repeats.append({
            "repeat": repeat + 1,
            "gamma_order": order,
            "baseline": {
                "wall_time_s": base_wall,
                "tokens": base_tokens,
                "throughput_tok_s": base_tps,
            },
            "sweep": measured,
        })

    sweep = []
    for gamma in gammas:
        rows = [next(row for row in repeat["sweep"] if row["gamma"] == gamma)
                for repeat in raw_repeats]
        speedups = [row["speedup_vs_baseline"] for row in rows]
        med_speedup = median(speedups)
        sweep.append({
            "gamma": gamma,
            "alpha": round(median(row["alpha"] for row in rows), 4),
            "wall_time_s": round(median(row["wall_time_s"] for row in rows), 6),
            "throughput_tok_s": round(median(row["throughput_tok_s"] for row in rows), 2),
            "speedup_vs_baseline": round(med_speedup, 3),
            "speedup_repeat_range": [round(min(speedups), 3), round(max(speedups), 3)],
            "net_result": (
                "net-positive" if med_speedup >= 1.05 and min(speedups) > 1.0
                else "flat" if med_speedup >= 1.0 else "negative"
            ),
            "theory_tokens_per_round": round(
                median(row["theory_tokens_per_round"] for row in rows), 3
            ),
            "rounds": round(median(row["rounds"] for row in rows)),
            "accepted": round(median(row["accepted"] for row in rows)),
            "proposed": round(median(row["proposed"] for row in rows)),
            "tokens": round(median(row["tokens"] for row in rows)),
        })

    base_tps = median(repeat["baseline"]["throughput_tok_s"] for repeat in raw_repeats)

    result = {
        "engine": "spec",
        "target": target_path,
        "draft": draft_path,
        "draft_label": draft_label,
        "draft_adapter": draft_adapter,
        "phase": 14 if pair_config else 4,
        "pair_config": pair_config.name if pair_config else None,
        "profile": "canonical",
        "max_tokens": max_tokens,
        "baseline_tok_s": round(base_tps, 2),
        "sweep": sweep,
        "repeats": raw_repeats,
        "repeat_range": [1, repeats],
        "warmup_runs_per_mode": warmup_runs,
        "target_fingerprint": model_fingerprint(Path(target_path)),
        "draft_fingerprint": model_fingerprint(Path(draft_path)),
        "tokenizer_sha256": tokenizer_sha,
        "cache_reconciliation": target.cache_reconciliation,
        "pair_provenance": vars(pair_config) if pair_config else None,
        "resolved_revisions": resolved_revisions,
        "env": env_stamp(seed, workload_hash(profile, max_tokens)),
    }
    _write(result, results_dir)
    return result


def _write(result: dict, results_dir: Path | None) -> Path:
    label = result.get("pair_config") or result["draft_label"]
    out_path = write_result_json(result, f"spec_{label}", results_dir)
    print(f"\n[spec] result written to {out_path}")
    return out_path
