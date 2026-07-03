# inferd

> **Status:** pre-release candidate. The full pipeline — fine-tuning, engine, serving, benchmarks, the FP8 27B hero, and the live **React dashboard** — is built and measured. Every number below traces to a `result.json` under `bench/results/` and regenerates from one command. Remaining release work is the under-load **demo capture** (`docs/demo.md`) and the final gate in [`docs/PRE_RELEASE.md`](docs/PRE_RELEASE.md). See [Current status](#current-status).

A from-scratch local LLM inference stack: **QLoRA fine-tuning → speculative decoding → paged KV-cache → continuous batching**, served via FastAPI with a React metrics dashboard. Benchmarked against a naive Hugging Face baseline and vLLM as the reference ceiling. Runs fully offline on a single RTX 5090 — no cloud APIs, no external inference dependencies.

The thesis is depth on both ends: fine-tune a showcase model *and* serve it through an engine you built yourself. Most projects stop at "I LoRA'd a model" or "I wrapped vLLM." This repo closes the loop.

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for maintainer and assistant acknowledgements.

---

## What it does

| Layer | Responsibility | Status |
|-------|----------------|--------|
| **Fine-tuning** | QLoRA SFT of a 27B showpiece + 9B engine target; draft distillation for acceptance-rate lift | Implemented (`finetune/`) |
| **Inference core** | Exact speculative decoding, paged KV-cache, iteration-level continuous batching | Implemented (`core/`) |
| **Benchmarking** | Headless harness, distribution-equivalence test, throughput-vs-concurrency curves | Implemented (`bench/`) |
| **Serving** | FastAPI async queue, SSE token streaming, `/metrics` and `/healthz` | Implemented (`serve/`) |
| **Dashboard** | Live tokens/sec, TTFT, draft acceptance rate α, VRAM, concurrency | Implemented (`dashboard/`) |

---

## Quick start

**Requirements:** WSL2 Ubuntu, RTX 5090 (Blackwell sm_120), CUDA 12.8+, [uv](https://docs.astral.sh/uv/), `gcc`/`g++` for Triton JIT, and [Bun](https://bun.sh/) for the dashboard. Full stack details in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

```bash
# 1. Install dependencies (pinned in uv.lock)
uv sync
sudo apt-get install -y gcc g++   # Triton kernel JIT

# 2. Download weights (one-time; requires HF token + accepted license)
hf auth login
hf download Qwen/Qwen3.5-9B --local-dir ./weights/Qwen3.5-9B

# 3. Smoke test — load text backbone, one forward pass
uv run python scripts/smoke_load.py

# 4. Run unit tests (no GPU required for most)
uv run python -m unittest discover -s tests -v

# 5. Benchmark self-check (no GPU)
uv run python -m bench.harness --selfcheck
```

Weights, adapters, merged checkpoints, and datasets are gitignored and live under `./weights/`, `./adapters/`, `./merged/`, and `./data/` locally. See [`DECISIONS.md`](DECISIONS.md) for training artifacts and pinned choices.

---

## Architecture

```mermaid
flowchart TB
    dash["Dashboard<br/>React + Vite · live<br/>tokens/sec · TTFT · α · VRAM"]
    serve["Serving<br/>FastAPI + SSE<br/>queue · scheduler · streaming"]
    core["Inference core · live<br/>spec decode · paged KV · batching<br/>model runner (target + draft, bf16)"]
    ft["Fine-tuning · live<br/>Unsloth QLoRA · golden eval · export · draft KD"]

    dash -->|SSE / WebSocket| serve
    serve -->|in-process| core
    ft -->|adapters / merged weights| core
```

| Layer | Package | Status |
|-------|---------|--------|
| Dashboard | `dashboard/` | Live |
| Serving | `serve/` | Live |
| Inference core | `core/` | Live |
| Fine-tuning | `finetune/` | Live |
| Benchmarks | `bench/` | Live (headless; no HTTP required) |

The core is importable and headless-benchmarkable — measurement never depends on the HTTP stack.

---

## Repository layout

```
inferd/
├── inferd/                 # Package root: env bootstrap, CUDA lib preload
├── core/                   # Inference engine
│   ├── model_runner.py     # Shared hot file: load + forward(tokens, kv)
│   ├── spec_decode.py      # Exact rejection sampling + resample
│   ├── paged_cache.py      # Block allocator + page table
│   ├── paged_attn.py       # Paged-attention reference (Triton kernel follow-up)
│   ├── batched_cache.py    # Stack/split hybrid caches for batched decode
│   └── scheduler.py        # FCFS continuous batching
├── finetune/               # QLoRA training pipeline
│   ├── train_qlora.py      # Unsloth-first SFT entrypoint
│   ├── eval_golden.py      # Golden-set regression checks
│   ├── distill_draft.py    # Sequence-level KD for draft α-lift
│   └── export.py           # Adapter export + merge-for-serving
├── bench/                  # Headless benchmark harness (source of truth)
│   ├── harness.py          # CLI: hf / spec / paged / batched / vllm
│   ├── correctness.py      # Distribution-equivalence gate
│   ├── workload.py         # Frozen prompts + sampling profiles
│   ├── runners/            # Engine-specific runners
│   └── results/            # Pinned JSON results (append-only)
├── tests/                  # Unit + equivalence tests
├── scripts/                # smoke_load.py and other entrypoints
├── docs/                   # ENVIRONMENT.md and setup notes
├── plans/                  # Phased execution pack (00–11)
├── plan.md                 # Design vision
├── DECISIONS.md            # Load-bearing decisions with dates
└── uv.lock                 # Pinned dependency lockfile
```

`dashboard/` is present (React + Vite + TS); see [`dashboard/`](dashboard/).

`core/model_runner.py` is the shared hot file — phases extend it via new methods; callers treat `kv` as an opaque handle.

---

## Models

Roles are split across Qwen generations: a 27B showpiece for fine-tuning, and a matched 9B/0.8B pair for the engine.

| Role | Model | Notes |
|------|-------|-------|
| Fine-tune showpiece | `Qwen/Qwen3.6-27B` | QLoRA only; served via FP8 in the hero demo (phase 10) |
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

`bench/correctness.py` exercises this via a multi-token statistical gate against a bootstrapped direct-vs-direct null envelope. Qwen3.5's hybrid linear-attention cache requires a custom parallel-verify patch (`core/qwen35_patch.py`) and snapshot/replay rollback — see `DECISIONS.md` for the honest throughput findings.

---

## Tech stack

| Area | Choices |
|------|---------|
| Fine-tuning | Unsloth first; Axolotl / Llama-Factory / ms-swift / TRL+PEFT as fallbacks. bitsandbytes 4-bit NF4. **No GemForge.** |
| Core | Python, PyTorch (Blackwell CUDA 12.8+), Triton (paged attention), Hugging Face Transformers |
| Serving | FastAPI + uvicorn, async queue, SSE streaming |
| Dashboard | React + Vite, Recharts |
| Environment | WSL2 Ubuntu, `uv` lockfile, everything pinned |
| Reference | vLLM (ceiling only) |

FP8 is the **one quantization exception**, scoped to the fine-tuned 27B hero demo only (phase 10).

---

## Current status

**All phases 01–11 are code-complete and merged to `main`**, including the live dashboard (08).
This is a pre-release candidate, not a tagged release: the remaining work is the under-load
**demo capture** (`docs/demo.md`) plus the final gate in [`docs/PRE_RELEASE.md`](docs/PRE_RELEASE.md).

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 01 — Environment | Pinned WSL2 + CUDA stack; smoke test | Done |
| 02 — Harness | Reproducible HF floor + vLLM ceiling | Done |
| 03 — QLoRA | Fine-tuned 9B + 27B adapters; golden-set eval | Done (9B merged; 27B adapter restored) |
| 04 — Spec decode | Exact rejection sampling + correctness gate; α-lift | Done (correctness proven; net speedup negative on hybrid model) |
| 05 — Paged KV | Block allocator + reference paged attention | Done (runtime persistent cache TBD) |
| 06 — Batching | Iteration-level scheduler + batched decode | Done |
| 07 — Serving | FastAPI + SSE | Done |
| 08 — Dashboard | Live metrics UI | Done |
| 09 — Bench/report | Aggregated plots; one-command reproduce | Done |
| 10 — FP8 hero | Fine-tuned 27B via FP8 | Done as capacity proof; latency impractical |
| 11 — Docs | Portfolio-ready README with final numbers | Done |

### Results

Every figure below regenerates with `uv run python bench/run_all.py --plots`; plots land in `bench/results/plots/` and the full table in [`bench/report.md`](bench/report.md).

**Throughput vs concurrency — matched workload, naive HF floor vs the inferd engine (9B, tok/s):**

| concurrency | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| naive HF | 29.8 | 104.9 | 116.0 | 24.4 | 23.3 |
| **inferd** | **44.3** | **141.6** | **236.8** | **356.8** | **461.8** |

- Continuous batching wins at **every** concurrency; **19.8× over the naive HF floor at c=32**, where HF collapses on KV-cache-less recompute.

| Experiment | Result | Notes |
|------------|--------|-------|
| Spec-decode correctness | **✅ PASS** distribution-equivalence gate, n=1500 | Per-position TV within bootstrapped null; proves exact accept rule + residual resample |
| Spec decode (0.8B draft) | α ≈ 0.63–0.68; **0.6–0.7× baseline throughput** | Hybrid linear-attention replay tax; correctness + α-lift are the wins, not net speed |
| Draft distillation α-lift | Δα up to **+0.056** (mean +0.048) | Replay tax dominates net throughput, not α |
| Paged KV equivalence | max\|Δlogit\| = 0 on model-level compute gate | Reference path; no Triton kernel yet |
| 27B FP8 hero | Fine-tuned 27B fits at **28.9 GiB** (peak 32.1/32.6 GiB on-card); **0.121 tok/s** | Capacity proof — FP8 halves weight bytes so the 27B fits the 5090; not a latency win |

**Honest gaps:** no persistent paged runtime cache; no batched speculative decoding; the 27B FP8 hero is a capacity/coherence proof, not a usable latency point; vLLM ceiling deferred on sm_120 (subprocess failed on Blackwell — reported, not faked).

---

## Running benchmarks

All engines share the frozen workload in `bench/workload.py` (prompts, seeds, sampling profiles).

```bash
# Regenerate plots/report from committed benchmark results (no model load)
uv run python bench/run_all.py --plots

# HF naive floor
uv run python -m bench.harness --engine hf --model Qwen3.5-9B \
    --seed 0 --max-tokens 256 --concurrency 1,2,4,8

# Speculative decoding (needs merged/9b target + draft weights)
uv run python -m bench.harness --engine spec \
    --target merged/9b --draft weights/Qwen3.5-0.8B

# Paged-cache microbenchmark
uv run python -m bench.harness --engine paged --model merged/9b

# Continuous batching
uv run python -m bench.harness --engine batched --model merged/9b \
    --concurrency 1,4,8

# Distribution-equivalence gate
uv run python -m bench.correctness --target merged/9b --draft weights/Qwen3.5-0.8B
```

Results are written to `bench/results/<timestamp>_<engine>_<model>/result.json` (append-only).

---

## Running the service and dashboard

```bash
# API + engine
INFERD_MODEL=merged/9b uv run uvicorn serve.app:app --host 0.0.0.0 --port 8000

# Dashboard
cd dashboard
bun install
bun run dev
```

The Vite dev server proxies `/metrics`, `/healthz`, and `/generate` to
`localhost:8000`. For preview/static serving from another origin, set
`VITE_INFERD_API` as shown in `dashboard/.env.example`.

---

## Phased roadmap

Build order: **harness → QLoRA → spec-decode (+ α-lift) → paged cache → continuous batching → serving → dashboard → FP8 hero**. Each phase is a defensible stopping point.

Development uses one git worktree per phase (`phase-NN-slug` → merge into `dev` in order). See `plans/00_MASTER_ORCHESTRATION.md` for cross-phase rules.

---

## Future plans

Not committed — early ideation and follow-ups beyond the v1 CUDA stack:

- **Apple Silicon port** — out of scope for this CUDA v1; a separate Metal/MLX exploration may happen later.
- **Persistent paged runtime KV cache** — wire the phase-05 block allocator into live decode instead of stacking HF caches.
- **Batched speculative decoding** — extend accept/replay through the continuous batching scheduler.
- **vLLM ceiling on Blackwell** — re-run when sm_120 wheels are available.
- **Triton paged-attention kernel** — replace the reference Python path with a fused kernel.

---

## Hardware & environment

- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120), 32 GB VRAM
- **OS:** WSL2 Ubuntu (recommended over native Windows for Triton, bitsandbytes, FlashAttention)
- **CUDA:** 12.8+ with a Blackwell-supported PyTorch wheel
- **Policy:** local-first, offline after one-time weight download; no cloud/API inference

```bash
uv sync
uv run python scripts/smoke_load.py
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

## Key design decisions

- **QLoRA, not full fine-tuning** — only realistic single-5090 path for a 27B
- **Qwen3.6-27B for showpiece, Qwen3.5 pair for engine** — 27B bf16 doesn't fit 32 GB; 9B leaves KV headroom
- **Triton over raw CUDA** — kernel concept without becoming a CUDA project
- **Benchmark harness before optimization** — no speedup claim without a reproducible baseline
- **Draft distillation as the bridge** — train draft on fine-tuned target's own outputs, not a shared corpus
- **No GemForge** — use maintained QLoRA stacks only (Unsloth first)

Full rationale and dated entries: [`DECISIONS.md`](DECISIONS.md).

---

## License

MIT — see [LICENSE](LICENSE).
