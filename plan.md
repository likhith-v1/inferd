# plan.md — Inference Optimization Server (with explicit fine-tuning lane)

A from-scratch LLM stack: **QLoRA fine-tuning** → **speculative decoding** → **paged KV-cache** → **continuous batching**, benchmarked against a naive baseline and vLLM as the ceiling. Train it, serve it, consume it — all in one repo. Runs locally on the RTX 5090. No cloud, no API billing, CUDA-native (no MLX, no GemForge).

---

## 1. Thesis & the resume bullet

One repo that does the whole loop: fine-tune a model you can showcase, then serve it through an engine you built yourself. The differentiator is depth on both ends — most portfolios stop at "I LoRA'd a model" or "I wrapped vLLM." This does the training *and* proves systems-level understanding of why fast inference is fast.

The bullets this is engineered to produce (numbers are placeholders to fill from real measurement):

> **Fine-tuning:** QLoRA fine-tuned Qwen3.6-27B (4-bit, single RTX 5090) on a domain corpus; +N% win-rate over base on a 50-prompt golden set.
>
> **Inference:** Implemented speculative decoding and paged KV-cache from scratch (PyTorch + Triton). 2.8× single-stream throughput at 74% draft acceptance, 9× concurrent throughput via continuous batching — within 1.4× of vLLM, with provably-identical output distribution to the target.
>
> **The bridge:** Fine-tuning the draft model on the target's corpus raised acceptance rate from 61% → 79%, improving speculative throughput 22% — one system, not two.

That last bullet is the whole point of adding fine-tuning: it makes the training pipeline and the inference engine a single coherent contribution.

**Hard constraint check:** fully local on owned hardware, zero external dependencies at inference time, flat-rate (no API billing). The only network step is downloading weights once, then offline.

---

## 2. Scope decisions (read before writing any code)

**In scope:**
- QLoRA fine-tuning (4-bit NF4 base + LoRA adapters) of a separate 27B-class Qwen — the showcase artifact.
- Fine-tune the showpiece model with maintained tooling only: Unsloth first, then Axolotl / Llama-Factory / ms-swift / TRL+PEFT as fallbacks. **No GemForge in v1.**
- Speculative decoding with the exact rejection-sampling acceptance rule (Leviathan et al. 2023 / Chen et al. 2023).
- Paged KV-cache: block allocator + page table + a paged-attention kernel via **Triton** (not raw CUDA).
- Continuous batching (iteration-level scheduling — Orca, OSDI 2022).
- A measurement harness built *first*, before any optimization.
- One model family: **Qwen** (clean draft/target tokenizer sharing).
- FastAPI serving layer + React metrics dashboard.

**Explicitly OUT of scope (and why):**
- **Full fine-tuning of the 27B.** Infeasible on 32GB — optimizer states alone need an order of magnitude more VRAM. QLoRA is the only realistic single-card path, and it's the correct one.
- **Raw CUDA kernels.** Writing PagedAttention in raw CUDA turns this into a CUDA project. Triton gets ~80% of the value at ~20% of the pain. CUDA credential = plan v2.
- **Beating vLLM on absolute throughput.** You won't; it has years of kernel tuning. Frame it as the *reference ceiling*: "within K× from scratch" is the win.
- **Arbitrary model support.** One family, matched draft/target. Generality is a maintenance tax with no portfolio payoff.
- **General quantization (GPTQ/AWQ).** The engine is fp16/bf16 by default. The *one* exception is FP8 to serve the 27B hero demo (§6 Phase 6) — the 5090 has FP8 hardware, so it's the right tool there and nowhere else in v1.
- **Tree-based speculation (Medusa / EAGLE / SpecInfer).** Impressive, large. Stretch goal.
- **GemForge.** Excluded for v1 because it has been stale/outdated for too long; this project should depend on actively maintained training stacks.

**The honest tension to internalize now:** speculative decoding primarily helps *single-stream latency*; continuous batching + paging primarily helps *multi-request throughput*. At large batch sizes the GPU is already compute-saturated, so speculation's benefit shrinks. The two speedups do **not** multiply. Lead with speculative decoding as the headline; treat batching as the second act. Measure them on separate axes and report where they interact.

---

## 3. Model choices (this is the load-bearing decision)

Qwen3.6 currently has a strong 27B dense checkpoint and a 35B-A3B MoE checkpoint, but no tiny same-generation draft model. That makes it excellent as a **fine-tuning showpiece**, but awkward as the main speculative-decoding pair. The Qwen3.5 generation gives the usable engine ladder: 0.8B / 2B / 4B / 9B / 27B, so the engine should still use a matched Qwen3.5 target/draft pair. Split the roles instead of forcing one model family member to do everything:

### 3.1 Fine-tuning showpiece — separate 27B-class model
Primary choice: **Qwen/Qwen3.6-27B**. Treat it as the hero fine-tune, not the everyday engine target. QLoRA only; full fine-tuning is out of scope on a single 32GB card.
- 4-bit base footprint ≈ 14–17GB; QLoRA training with gradient checkpointing + paged 8-bit AdamW ≈ 18–22GB → fits on 32GB with modest sequence length (2–4k) and batch 1–2 + grad accumulation.
- Text-first rule: Qwen3.6-27B is image-text capable, but this project should fine-tune the language path only and freeze/ignore the vision path unless you deliberately want a VLM project.
- Fallback if 27B tooling is painful on Blackwell: fine-tune **Qwen3.5-27B** or **Qwen3.5-9B** first, keep the pipeline identical, then swap back to Qwen3.6-27B after the environment is stable.
- Output artifact: LoRA adapter, merged weights for serving, 30–50 prompt golden-set eval, and a short comparison report against the base checkpoint.

### 3.2 Inference-engine target/draft — Qwen3.5 matched pair
A 27B in bf16 is ~54GB and **does not fit 32GB** — so it can't be the engine's primary served model. Use a smaller same-generation pair where draft+target tokenizers are guaranteed identical:
- **Target: Qwen3.5-9B** — bf16 ≈ 18GB, leaving ~14GB for KV-cache and batch headroom (the headroom is what makes the batching story shine).
- **Draft: Qwen3.5-0.8B** — bf16 ≈ 1.6GB.
- Both are QLoRA-fine-tunable; the draft is then *distilled against the fine-tuned target* to raise α → the α-lift experiment (§3.4).

### 3.3 The 27B hero demo — FP8
Serve the fine-tuned 27B single-stream via FP8 (~27GB, tight, no batching headroom) to prove the engine scales to a flagship model. FP8 is scoped to exactly this (Phase 6), not sprinkled through the engine.

### 3.4 Speculative decoding — and why it's exact
The acceptance rule is the crux, and it's what makes output *exactly* the target's distribution, not an approximation. For each speculative position with target prob `p(x)` and draft prob `q(x)` for the drafted token `x`:

```
accept x with probability min(1, p(x) / q(x))
on first rejection at position k:
    resample that token from the residual distribution
        p_resid(x) = max(0, p(x) - q(x)) / Σ_x max(0, p(x) - q(x))
    discard all drafted tokens after k
if all γ accepted:
    sample one bonus token from p at the final position
```

This rejection-sampling construction guarantees the emitted sequence is distributed identically to direct target sampling — provable and testable (§7). The residual-resampling branch is the most-commonly-botched part of from-scratch implementations; write it carefully.

Expected accepted tokens per target call ≈ `(1 − α^(γ+1)) / (1 − α)`, with α the acceptance rate and γ the draft length. Net wall-clock gain also depends on `c = draft_cost / target_cost` — too large a draft eats its own savings. Measuring real α against this theory curve is a great plot.

**The α-lift experiment (the bridge):** α depends on how closely the draft's distribution `q` matches *this specific target's* `p`, so the lever is making `q` mimic the fine-tuned target — not just training on similar text. Two refinements that matter:
- **Order:** fine-tune the 9B target *first*. That shifts its distribution, so a stock 0.8B draft now matches it *worse* and α can drop. Train the draft against the **fine-tuned** target, never the base one.
- **Method:** distill the draft on the target's own generations (sequence-level KD — sample from the fine-tuned 9B, train the 0.8B on those outputs). This moves `q` toward `p` far more directly than sharing a human corpus does.

The clean comparison, target held fixed: (fine-tuned target + stock draft) vs (fine-tuned target + distilled draft) → Δα, Δthroughput. This is the single result that fuses the training pipeline and the inference engine.

### 3.5 Fine-tuning tooling decision — no GemForge
GemForge stays out. Use maintained, reproducible stacks:
- **Primary:** Unsloth for single-GPU QLoRA SFT.
- **Fallbacks:** Axolotl, Llama-Factory, ms-swift, or plain TRL + PEFT + bitsandbytes.
- **Rule:** pick the first stack that cleanly supports the chosen Qwen checkpoint, Blackwell/CUDA, 4-bit NF4, gradient checkpointing, adapter export, and merge-for-serving. Do not spend project time reviving stale tooling.
- **Success condition:** `finetune/train_qlora.py` works from a pinned `uv.lock`; the training stack is an implementation detail, not the project headline.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Dashboard (React + Vite)                                    │
│  tokens/sec · TTFT · inter-token latency · draft accept rate │
│  VRAM utilization · concurrent requests · throughput curves  │
└───────────────────────────┬─────────────────────────────────┘
                            │ SSE / WebSocket
┌───────────────────────────┴─────────────────────────────────┐
│  Serving layer (FastAPI, async)                              │
│  request queue · iteration-level scheduler · token streaming │
│  /generate (SSE) · /metrics · /healthz                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ in-process
┌───────────────────────────┴─────────────────────────────────┐
│  Inference core (Python + Triton)                            │
│  spec decoding (draft runner + accept/resample) · paged      │
│  KV-cache (allocator + page table) · paged-attn kernel       │
│  model runner (target + draft, bf16; 27B via FP8)            │
└───────────────────────────┬─────────────────────────────────┘
                            │ loads adapters / merged weights
┌───────────────────────────┴─────────────────────────────────┐
│  Fine-tuning stage (Unsloth first; no GemForge, offline)      │
│  QLoRA SFT of 27B showpiece + 9B/0.8B engine pair            │
│  golden-set eval · adapter export · merge-for-serving        │
└──────────────────────────────────────────────────────────────┘
```

Keep the core importable and headless-benchmarkable (a Python script, no HTTP) so measurement never depends on the web stack. The fine-tuning stage is fully offline and feeds adapters/merged weights to the engine.

---

## 5. Environment & the Windows reality (read before Phase 0)

You just returned to Windows. The kernel-heavy parts (Triton, FlashAttention) and Unsloth are smooth on Linux, rough on native Windows.

The RTX 5090 is **Blackwell (sm_120)** — bleeding edge. Requirements: CUDA 12.8+ toolkit, a Blackwell-supported PyTorch wheel, Triton (smoother on Linux/WSL2), and bitsandbytes/Unsloth builds that support Blackwell + 4-bit.

**Recommendation: do everything in WSL2 (Ubuntu).** Near-native CUDA on the 5090, full Linux toolchain, weights on local NVMe, fully offline — consistent with the local-first philosophy. Keep the Windows host for the desktop; develop engine + fine-tuning inside WSL2. This sidesteps the Triton/bitsandbytes/FlashAttention-on-Windows tax entirely. Pin everything in a `uv` lockfile the moment a working environment exists — bleeding-edge CUDA stacks are fragile.

---

## 6. Phased milestones

Each phase is a defensible stopping point with standalone resume value.

### Phase 0 — Baseline + measurement harness *(build first)*
No speedup claim without a baseline and a way to measure it.
- Load the engine target (stock Qwen3.5-9B), get HF `generate()` (default KV-cache) working.
- Build the headless harness: fixed workload (fixed prompts/seed/output length) → tokens/sec, TTFT, inter-token latency, peak VRAM.
- Stand up vLLM on the same model as the *ceiling* reference, recorded once.
- **Exit:** reproducible numbers for naive HF and vLLM on identical workloads, plus throughput-vs-concurrency curves.

### Phase 1 — QLoRA fine-tuning *(the showpiece, runs largely independent of the engine)*
- Set up Unsloth first; if Blackwell/Qwen support is broken, fall back to Axolotl / Llama-Factory / ms-swift / TRL+PEFT. **Do not use GemForge.** 4-bit NF4 base, LoRA adapters, gradient checkpointing, paged 8-bit AdamW, bf16 compute.
- SFT the separate 27B-class showpiece model, preferably Qwen/Qwen3.6-27B, on a chosen domain corpus (text path only; freeze/ignore vision path). Also SFT the 9B engine target on the same corpus. **Defer the 0.8B draft** — it gets *distilled against the fine-tuned 9B* as part of the α-lift in Phase 2, so it can't be trained until the target is final.
- Eval discipline (your own cohort standard): 30–50 golden prompts, win-rate vs base, held-out set. Log to `DECISIONS.md`.
- **Exit:** a fine-tuned 27B with measured win-rate over base; fine-tuned 9B target ready for the engine. Adapters exported; merge-for-serving path tested.

### Phase 2 — Speculative decoding, single stream *(headline engine result)*
- Draft runner (γ-token proposal), target verification in one forward pass over γ+1 positions, exact accept/resample (§3.4 — including the residual branch).
- Sweep γ ∈ {2,4,8}; measure α and wall-clock speedup vs the theory curve.
- **Run the α-lift experiment:** distill a 0.8B draft on the fine-tuned 9B's own generations; hold the target fixed and compare stock draft vs distilled draft → Δα, Δthroughput. (Target is fine-tuned *before* the draft — a stock draft matches a fine-tuned target worse.)
- **Exit:** correctness test passes (§7); α/speedup plots; the α-lift result quantified.

### Phase 3 — Paged KV-cache, single stream
- Block allocator + page table (Python, block size 16). Paged-attention gather-and-attend kernel in Triton (FlashAttention varlen as fallback).
- Validate numerically against the contiguous-cache path (same logits within tolerance).
- **Exit:** paged path numerically equivalent to baseline; VRAM-per-sequence measured below naive pre-allocation.

### Phase 4 — Continuous batching
- Iteration-level scheduler: running batch, evict-finished, admit-waiting under a free-block budget; prefill for new admits.
- **Exit:** throughput-vs-concurrency win over naive batching; live "active sequences" view; speculative-with-batching measured and the benefit-fade reported honestly (the §2 tension).

### Phase 5 — Dashboard + writeup
- React dashboard: live tokens/sec, TTFT, α, VRAM utilization, concurrency, throughput curves (SSE/WebSocket from `/metrics`).
- `DECISIONS.md` + benchmark report with comparison plots. Short demo video under load (the live α + throughput curves are the payoff shot).
- **Exit:** every number in the resume bullets is real and reproducible from one command.

### Phase 6 — 27B FP8 hero demo *(scoped, optional capstone)*
- Serve the fine-tuned 27B single-stream via FP8 on the 5090's FP8 hardware.
- **Exit:** the flagship fine-tune served through your own engine — the closing shot of the demo.

---

## 7. Measurement strategy (where portfolio projects live or die)

**Metrics, precisely:** throughput (single-stream and aggregate — different stories), TTFT (prefill latency), inter-token latency (decode steady state), α (draft acceptance), throughput-vs-concurrency curve (1→N, naive/yours/vLLM), VRAM-vs-concurrency (sequences before OOM, naive vs paged). For fine-tuning: golden-set win-rate vs base, held-out loss.

**Baselines (three rungs):** HF `generate()` (naive floor) · your engine · vLLM (ceiling). Framing is always "within K× of the production engine, from scratch."

**The correctness test (do not skip — it's the differentiator):** with the accept/resample rule correct, speculative output is *distributionally identical* to direct target sampling. Fix a seed, generate many continuations both ways, check next-token distributions match within sampling noise (χ² or total-variation). A passing test is the evidence behind "provably-identical output distribution."

**Workload discipline:** identical prompts/seeds/max-tokens/sampling across all rungs. Warm up before timing. Report hardware + CUDA + model versions alongside numbers.

---

## 8. Tech stack (pinned intentions)

- **Fine-tuning:** Unsloth first; Axolotl / Llama-Factory / ms-swift / TRL+PEFT as maintained fallbacks. bitsandbytes 4-bit NF4. **No GemForge.**
- **Core:** Python, PyTorch (Blackwell CUDA 12.8+ wheel), Triton (paged-attn kernel), HF Transformers (loading + tokenizer).
- **Serving:** FastAPI + uvicorn, async queue, SSE streaming.
- **Dashboard:** React + Vite, Recharts/uPlot for live streams.
- **Models:** Qwen/Qwen3.6-27B as the preferred separate fine-tune showpiece (QLoRA→FP8 serve); Qwen3.5-9B target + Qwen3.5-0.8B draft for the engine, bf16. *Verify exact current model IDs on HF before pinning.*
- **Reference:** vLLM (ceiling only — not a runtime dependency).
- **Env:** WSL2 Ubuntu, `uv` lockfile, everything pinned.

---

## 9. DECISIONS.md (seed)

- **QLoRA, not full FT** — only realistic single-5090 path for a 27B; correct technique regardless.
- **Qwen3.6-27B for the separate fine-tuning showpiece, Qwen3.5 pair for the engine** — 3.6 is the impressive 27B artifact, while 3.5 gives a practical matched 9B/0.8B target/draft pair with KV headroom.
- **27B served via FP8 only** — bf16 27B doesn't fit; FP8 is the one quantization exception, scoped to the hero demo.
- **Triton over raw CUDA** — kernel concept without becoming a CUDA project.
- **vLLM as ceiling, not competitor** — success = within K× from scratch.
- **Measurement harness before optimization** — no claim without a reproducible baseline.
- **WSL2 dev environment** — sidesteps Triton/bitsandbytes/FlashAttention-on-Windows friction, stays local/offline.
- **Draft distillation as the bridge** — distilling the draft on the fine-tuned target's own outputs (not a shared corpus) raises α; target fine-tuned first; the one experiment that unifies training and inference.
- **No GemForge** — stale tooling is a distraction; use maintained QLoRA stacks only.
- **Spec decoding = latency win, batching = throughput win** — separate axes; interaction reported honestly.

---

## 10. Stretch goals (after v1 ships — do not start early)

- DPO/GRPO post-training of the showpiece (ties to your R1-style reasoning interest).
- Tree-based / multi-candidate speculation (Medusa / EAGLE / SpecInfer) to push α further.
- Serve the 35B-A3B MoE and exploit its native multi-token prediction as a speculative baseline to compare against your hand-rolled one.
- Prefix-sharing via copy-on-write KV blocks (shared system-prompt prefixes).
- Chunked prefill; quantized KV-cache to stretch concurrency.

---

## 11. Suggested repo layout

```
infer-stack/
├── finetune/
│   ├── train_qlora.py        # QLoRA SFT (27B showpiece + 9B/0.8B pair; no GemForge)
│   ├── eval_golden.py        # 50-prompt golden set, win-rate vs base
│   └── export.py             # adapter export + merge-for-serving
├── core/
│   ├── spec_decode.py        # draft runner + accept/resample
│   ├── paged_cache.py        # block allocator + page table
│   ├── paged_attn.py         # Triton kernel (+ flashattn fallback)
│   ├── scheduler.py          # iteration-level continuous batching
│   └── model_runner.py       # target/draft loading, bf16; 27B FP8 path
├── serve/
│   ├── app.py                # FastAPI: /generate, /metrics, /healthz
│   └── stream.py             # SSE token streaming
├── bench/
│   ├── harness.py            # headless workload runner (source of truth)
│   ├── correctness.py        # distribution-equivalence test
│   └── results/              # pinned numbers + plots
├── dashboard/                # React + Vite
├── DECISIONS.md
├── uv.lock
└── README.md                 # the bullets, the plots, how to reproduce
```

---

*Build order in one line: harness → QLoRA fine-tune (separate 27B showpiece, no GemForge) → speculative decoding (+ α-lift) → paged cache → scheduler → dashboard → FP8 hero. Each arrow is a place you could stop and still have something real.*
