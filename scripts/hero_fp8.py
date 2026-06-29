"""
scripts/hero_fp8.py — the FP8 hero demo (phase 10).

Serves the model single-stream through *our own* continuous-batching engine in
bf16 and FP8 (RTX 5090 sm_120 FP8 tensor cores) — and
reports the closing-shot numbers: decode tokens/sec, TTFT, model VRAM footprint,
peak VRAM, and a coherence spot-check (same prompt, both precisions, side by
side). FP8 is the project's one quantization exception, scoped to this path.

The intended hero is the fine-tuned 27B. On a 32 GB card we cannot materialize
the merged 27B in bf16, so the 27B path serves a load-time FP8 base with the
fine-tuned LoRA adapter attached at runtime:

    uv run python scripts/hero_fp8.py \
      --model /home/likhi/inferd/weights/Qwen3.6-27B \
      --adapter /home/likhi/inferd/adapters/27b \
      --variants fp8 --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402

from bench.metrics import VramSampler, env_stamp  # noqa: E402
from bench.workload import GREEDY, PROMPTS, workload_hash  # noqa: E402
from core.model_runner import ModelRunner  # noqa: E402
from core.scheduler import (  # noqa: E402
    ContinuousBatchScheduler,
    ModelRunnerBackend,
    RequestStatus,
    SchedulerConfig,
)


def _single_stream(runner: ModelRunner, prompt: str, max_tokens: int, seed: int):
    """One request, single-stream, through the engine. Returns (text, n_tokens, decode_tps, ttft_s)."""
    backend = ModelRunnerBackend(runner)
    sched = ContinuousBatchScheduler(
        backend,
        SchedulerConfig(
            max_blocks=2048, block_size=16, max_concurrent_sequences=1,
            temperature=GREEDY.temperature, top_p=GREEDY.top_p, seed=seed,
        ),
    )
    ids = runner.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    # TTFT proxy: prefill cost (prompt -> first-token logits) through the backend.
    torch.cuda.synchronize()
    t = time.perf_counter()
    backend.prefill(ids)
    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t

    sched.submit(ids, max_tokens=max_tokens, prompt_text=prompt, request_id=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    completed = sched.run_until_complete()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    req = completed[0]
    n = req.generated_len
    text = runner.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
    decode_tps = n / wall if wall > 0 else 0.0
    return text, n, decode_tps, ttft_s


def _measure(model: str, adapter: str | None, quantize: str | None, prompts: list[str],
             max_tokens: int, seed: int, device: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    label = quantize or "bf16"
    print(f"\n[hero] loading {label} ...")
    t = time.perf_counter()
    runner = ModelRunner.load_target(model, device=device, quantize=quantize, adapter=adapter)
    load_s = time.perf_counter() - t
    footprint_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"[hero] {label} loaded in {load_s:.1f}s · weight footprint {footprint_mb:.0f} MiB")

    # warmup
    _single_stream(runner, prompts[0], max_tokens=8, seed=seed)

    rows = []
    with VramSampler() as vs:
        for i, p in enumerate(prompts):
            text, n, tps, ttft = _single_stream(runner, p, max_tokens, seed + i)
            rows.append({"prompt": p, "tokens": n, "decode_tok_s": tps,
                         "ttft_s": ttft, "sample": text})
            print(f"  [{label}] p{i}: {tps:5.1f} tok/s · TTFT {ttft*1000:6.1f}ms · {n} tok")

    peak_mb = vs.peak_mb
    peak_torch_mb = torch.cuda.max_memory_allocated() / 1024**2
    mean_tps = sum(r["decode_tok_s"] for r in rows) / len(rows)
    mean_ttft = sum(r["ttft_s"] for r in rows) / len(rows)

    del runner
    torch.cuda.empty_cache()
    return {
        "precision": label, "load_s": load_s, "weight_footprint_mb": footprint_mb,
        "peak_vram_mb": peak_mb, "peak_vram_torch_mb": peak_torch_mb,
        "mean_decode_tok_s": mean_tps, "mean_ttft_s": mean_ttft, "runs": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/home/likhi/inferd/merged/9b")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter dir.")
    ap.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    ap.add_argument("--n-prompts", type=int, default=4, dest="n_prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--variants",
        default="bf16,fp8,fp8-dynamic",
        help="Comma-separated variants: bf16,fp8,fp8-dynamic.",
    )
    ap.add_argument("--results-dir", default=None, dest="results_dir")
    a = ap.parse_args(argv)

    prompts = PROMPTS[: a.n_prompts]
    is_27b_adapter = "Qwen3.6-27B" in a.model and a.adapter is not None
    out = {
        "demo": "fp8_hero",
        "model": a.model,
        "adapter": a.adapter,
        "note": (
            "Fine-tuned 27B FP8 path: load-time FP8 base with LoRA adapter attached; "
            "full bf16 merge is avoided because it does not fit local RAM/VRAM."
            if is_27b_adapter else
            "FP8 hero path; pass --adapter adapters/27b with weights/Qwen3.6-27B "
            "for the fine-tuned 27B run."
        ),
        "env": env_stamp(a.seed, workload_hash(GREEDY, a.max_tokens)),
        "max_tokens": a.max_tokens,
        "variants": {},
    }

    variant_map = {"bf16": None, "fp8": "fp8", "fp8-dynamic": "fp8-dynamic"}
    variants = [(name, variant_map[name]) for name in a.variants.split(",") if name in variant_map]
    if not variants:
        raise ValueError("--variants must include at least one of: bf16,fp8,fp8-dynamic")
    for label, quant in variants:
        try:
            out["variants"][label] = _measure(
                a.model, a.adapter, quant, prompts, a.max_tokens, a.seed, a.device
            )
        except Exception as exc:  # FP8 kernel/path immaturity on Blackwell — surface, don't fake.
            import traceback
            out["variants"][label] = {"error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()}
            print(f"\n[hero] {label} path failed: {exc}")

    # summary — each FP8 recipe vs bf16
    bf16 = out["variants"].get("bf16")
    out["summary"] = {}
    title = "FP8 27B HERO SUMMARY" if is_27b_adapter else "FP8 HERO SUMMARY"
    print("\n" + "=" * 70 + f"\n[hero] {title} (single-stream, through the engine)\n" + "=" * 70)
    print(f"  {'variant':<14}{'tok/s':>9}{'footprint':>12}{'peak(torch)':>13}")
    for label, _ in variants:
        v = out["variants"][label]
        if "error" in v:
            print(f"  {label:<14}{'FAILED':>9}  ({v['error']})")
            continue
        print(f"  {label:<14}{v['mean_decode_tok_s']:>9.1f}"
              f"{v['weight_footprint_mb']:>10.0f}MB{v['peak_vram_torch_mb']:>11.0f}MB")
        if label != "bf16" and bf16 and "error" not in bf16:
            out["summary"][label] = {
                "decode_speedup": v["mean_decode_tok_s"] / bf16["mean_decode_tok_s"],
                "footprint_ratio": v["weight_footprint_mb"] / bf16["weight_footprint_mb"],
            }
    for label, s in out["summary"].items():
        print(f"  --> {label}: {s['decode_speedup']:.2f}x decode speed · "
              f"{s['footprint_ratio']:.2f}x weight footprint vs bf16")
    print("\n[hero] coherence spot-check (prompt 0):")
    if bf16 and "error" not in bf16:
        print(f"  bf16: {bf16['runs'][0]['sample'][:150]!r}")
    for label, _ in variants:
        v = out["variants"][label]
        if label != "bf16" and "error" not in v:
            print(f"  {label}: {v['runs'][0]['sample'][:150]!r}")
    # 27B projection: FP8 weight-only would let the 27B fit on a 32GB card.
    if "fp8" in out["summary"] and bf16:
        bf16_27b = bf16["weight_footprint_mb"] * 27 / 9 / 1024
        fp8_27b = bf16_27b * out["summary"]["fp8"]["footprint_ratio"]
        out["projection_27b_gb"] = {"bf16": bf16_27b, "fp8": fp8_27b, "card_gb": 31.8}
        print(f"\n[hero] 27B projection: bf16 ≈ {bf16_27b:.1f} GB (will NOT fit) · "
              f"FP8 ≈ {fp8_27b:.1f} GB (fits the 32 GB 5090) — the reason FP8 is scoped to the 27B.")

    results_dir = Path(a.results_dir) if a.results_dir else Path(__file__).parent.parent / "bench" / "results"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "fp8_27b_hero" if is_27b_adapter else "fp8_hero"
    d = results_dir / f"{ts}_{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(out, indent=2))
    print(f"\n[hero] result written to {d / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
