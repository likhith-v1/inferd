# inferd

A from-scratch local LLM inference stack: **QLoRA fine-tuning → speculative decoding → paged KV-cache → continuous batching**, served via FastAPI with a React metrics dashboard. Benchmarked against a naive Hugging Face baseline and vLLM as the reference ceiling. Runs fully offline on a single RTX 5090 — no cloud APIs, no external inference dependencies.

The thesis is depth on both ends: fine-tune a showcase model *and* serve it through an engine you built yourself. Most projects stop at "I LoRA'd a model" or "I wrapped vLLM." This repo closes the loop.

---

## What it does

| Layer | Responsibility |
|-------|----------------|
| **Fine-tuning** | QLoRA SFT of a 27B showpiece + 9B engine target on a domain corpus; draft distillation for acceptance-rate lift |
| **Inference core** | Exact speculative decoding, paged KV-cache (Triton kernel), iteration-level continuous batching |
| **Serving** | FastAPI async queue, SSE token streaming, `/metrics` and `/healthz` |
| **Benchmarking** | Headless harness (source of truth), distribution-equivalence test, throughput-vs-concurrency curves |
| **Dashboard** | Live tokens/sec, TTFT, draft acceptance rate α, VRAM, concurrency |

---

## Architecture

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
│  Fine-tuning stage (Unsloth first; offline)                  │
│  QLoRA SFT of 27B showpiece + 9B/0.8B engine pair            │
│  golden-set eval · adapter export · merge-for-serving        │
└──────────────────────────────────────────────────────────────┘
```

The core is importable and headless-benchmarkable — measurement never depends on the HTTP stack. Fine-tuning is fully offline and feeds adapters/merged weights into the engine.

---

## Models

Roles are split across Qwen generations: a 27B showpiece for fine-tuning, and a matched 9B/0.8B pair for the engine.

| Role | Model | Notes |
|------|-------|-------|
| Fine-tune showpiece | `Qwen/Qwen3.6-27B` | QLoRA only; served via FP8 in the hero demo |
| Engine target | `Qwen/Qwen3.5-9B` | bf16 ≈ 18 GB; leaves KV-cache headroom for batching |
| Engine draft | `Qwen/Qwen3.5-0.8B` | bf16 ≈ 1.6 GB; distilled against the fine-tuned 9B |
| Fallback drafts | `Qwen3.5-2B` / `4B` | Same family if 0.8B acceptance rate is too low |

All three checkpoints are multimodal (`qwen3_5` arch). v1 is **text-only**: load the wrapper + processor, extract the `language_model` backbone, strip the vision tower.

vLLM is recorded once as the **ceiling reference** — never a runtime dependency.

---

## Speculative decoding (exact)

Output must be **distributionally identical** to direct target sampling. The acceptance rule (Leviathan et al. 2023 / Chen et al. 2023):

```
accept x with probability min(1, p(x) / q(x))
on first rejection at position k:
    resample from residual  p_resid(x) = max(0, p(x) - q(x)) / Σ max(0, p(x) - q(x))
    discard all drafted tokens after k
if all γ accepted:
    sample one bonus token from p at the final position
```

The residual-resampling branch is the most commonly botched part of from-scratch implementations. A correctness test (`bench/correctness.py`) proves equivalence via χ² or total-variation on next-token distributions.

**The α-lift experiment** bridges fine-tuning and inference: distill the draft on the fine-tuned 9B's own generations (sequence-level KD), hold the target fixed, and compare stock vs distilled draft → Δα, Δthroughput.

---

## Tech stack

| Area | Choices |
|------|---------|
| Fine-tuning | Unsloth first; Axolotl / Llama-Factory / ms-swift / TRL+PEFT as fallbacks. bitsandbytes 4-bit NF4. **No GemForge.** |
| Core | Python, PyTorch (Blackwell CUDA 12.8+), Triton (paged attention), Hugging Face Transformers |
| Serving | FastAPI + uvicorn, async queue, SSE streaming |
| Dashboard | React + Vite, Recharts/uPlot |
| Environment | WSL2 Ubuntu, `uv` lockfile, everything pinned |
| Reference | vLLM (ceiling only) |

FP8 is the **one quantization exception**, scoped to the fine-tuned 27B hero demo only.

---

## Target repo layout

```
inferd/
├── finetune/
│   ├── train_qlora.py        # QLoRA SFT (27B showpiece + 9B target)
│   ├── eval_golden.py        # 50-prompt golden set, win-rate vs base
│   ├── distill_draft.py      # sequence-level KD for α-lift
│   └── export.py             # adapter export + merge-for-serving
├── core/
│   ├── spec_decode.py        # draft runner + accept/resample
│   ├── paged_cache.py        # block allocator + page table
│   ├── paged_attn.py         # Triton kernel (+ FlashAttention varlen fallback)
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
└── README.md
```

`core/model_runner.py` is the shared hot file — phases extend it via new methods; callers treat `kv` as an opaque handle (contiguous early, paged later).

---

## Phased roadmap

Build order: **harness → QLoRA → spec-decode (+ α-lift) → paged cache → continuous batching → serving → dashboard → FP8 hero**. Each phase is a defensible stopping point.

| Phase | Deliverable |
|-------|-------------|
| 01 — Environment | Pinned WSL2 + CUDA 12.8+ stack; smoke test on Qwen3.5-9B text backbone |
| 02 — Harness | Reproducible baseline: naive HF floor + vLLM ceiling on identical workloads |
| 03 — QLoRA | Fine-tuned 27B showpiece + 9B target; golden-set win-rate measured |
| 04 — Spec decode | Exact rejection sampling + correctness test; α-lift experiment quantified |
| 05 — Paged KV | Block allocator + Triton paged attention; numerically equivalent to contiguous |
| 06 — Batching | Iteration-level scheduler; throughput-vs-concurrency vs naive batching |
| 07 — Serving | FastAPI + SSE; core stays headless-benchmarkable |
| 08 — Dashboard | Live metrics; demo under load |
| 09 — Bench/report | Aggregated plots; all numbers reproducible from one command |
| 10 — FP8 hero | Fine-tuned 27B served single-stream via FP8 (optional capstone) |
| 11 — Docs | `DECISIONS.md` + portfolio-ready README with real numbers |

Development uses one git worktree per phase (`phase-NN-slug` → merge into `dev` in order; `dev` → `main` at milestones).

---

## Hardware & environment

- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120), 32 GB VRAM
- **OS:** WSL2 Ubuntu (recommended over native Windows for Triton, bitsandbytes, FlashAttention)
- **CUDA:** 12.8+ with a Blackwell-supported PyTorch wheel
- **Policy:** local-first, offline after one-time weight download; no cloud/API inference

```bash
# Phase 01 target (once implemented)
uv sync
uv run python scripts/smoke_load.py        # load text backbone, one forward pass
HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py   # prove offline
```

---

## Measurement strategy

Three baseline rungs, identical prompts/seeds/max-tokens/sampling across all:

1. **HF `generate()`** — naive floor (default KV cache)
2. **inferd engine** — yours, from scratch
3. **vLLM** — production ceiling (reference only)

Metrics: throughput (single-stream and aggregate), TTFT, inter-token latency, draft acceptance rate α, throughput-vs-concurrency curves, VRAM-vs-concurrency. For fine-tuning: golden-set win-rate vs base.

**Honest framing:** speculative decoding primarily helps single-stream latency; paging + batching help multi-request throughput. At large batch sizes the GPU saturates and speculation's benefit fades. They do **not** multiply — measure and report on separate axes.

---

## Current status

**Early planning phase.** The repository defines architecture, constraints, and a phased execution plan. Implementation directories (`core/`, `bench/`, `finetune/`, `serve/`, `dashboard/`) are not yet present. Active development happens on the `dev` branch; `main` carries project documentation.

---

## Key design decisions

- **QLoRA, not full fine-tuning** — only realistic single-5090 path for a 27B
- **Qwen3.6-27B for showpiece, Qwen3.5 pair for engine** — 27B bf16 doesn't fit 32 GB; 9B leaves KV headroom
- **Triton over raw CUDA** — kernel concept without becoming a CUDA project
- **Benchmark harness before optimization** — no speedup claim without a reproducible baseline
- **Draft distillation as the bridge** — train draft on fine-tuned target's own outputs, not a shared corpus
- **No GemForge** — use maintained QLoRA stacks only (Unsloth first)

---

## License

MIT — see [LICENSE](LICENSE).
