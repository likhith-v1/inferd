# AGENTS.md — `inferd`

A from-scratch local LLM inference stack: **QLoRA fine-tune → speculative decoding → paged KV-cache → continuous batching**, served via FastAPI + a React dashboard, benchmarked against naive HF and vLLM. Runs fully local on a single RTX 5090. See `plan.md` for the vision and `plans/` for the execution pack (one file per phase, `00`–`11`).

## Current state (2026-06-29)
Phases **01–07, 09, 10, 11 are complete and merged to `dev`**; only the **React dashboard (08)** remains.
- **Engine + serving live:** spec-decode, paged cache, continuous batching, FastAPI/SSE serving (`serve/`), the headless harness, plots, and `bench/report.md`.
- **Fine-tuning is real, not hypothetical:** 9B merged; the **27B QLoRA adapter exists** (`adapters/27b`) and the **fine-tuned 27B is served in FP8 as a capacity proof** (`bench/results/*_fp8_27b_hero/`).
- **Headline numbers (all trace to `bench/results/`):** continuous batching ~**19.8× over naive HF at c=32**; spec-decode **distribution-equivalence PASS @ n=1500**; 27B FP8 fits at ≈**28.9 GiB** on the 32 GB card.
- **Release status:** held on `dev` — not merged to `main`, no tags, nothing pushed beyond `origin/dev`.

## Before you implement — read first
- **Read the whole codebase before writing any code.** Reuse what exists; do not re-implement helpers, types, or patterns that already live here.
- Read `plan.md` (the design) and the `plans/NN_*.md` file for the phase you're working on (scope, owned/avoided files, commands, exit criteria, open questions).
- Read `plans/00_MASTER_ORCHESTRATION.md` for cross-phase rules (execution/merge order, the shared-file contract, quality gates).

## Hard constraints (non-negotiable)
- **Local-first, offline:** WSL2 Ubuntu, RTX 5090 (Blackwell sm_120), CUDA 12.8+. No cloud/API inference, ever. Network only for the one-time weight download, then air-gapped.
- **No MLX. No GemForge.** Fine-tuning uses maintained QLoRA stacks: **Unsloth first**, then Axolotl / Llama-Factory / ms-swift / TRL+PEFT.
- **Text-only.** All three Qwen models are multimodal (`AutoModelForMultimodalLM`, arch `qwen3_5`) — load the wrapper + processor, **extract the `language_model` backbone, strip the vision tower**. v1 is text in, text out.
- **Models:** `Qwen/Qwen3.6-27B` (fine-tune showpiece) · `Qwen/Qwen3.5-9B` target + `Qwen/Qwen3.5-0.8B` draft (engine). Fallback drafts: `Qwen3.5-2B`/`4B`.
- **vLLM is the ceiling only** — a reference number, never a runtime dependency.
- **FP8 is the one quantization exception**, scoped to the 27B hero demo (`plans/10`) and nowhere else. The hero is the *fine-tuned* 27B (load-time FP8 base + runtime LoRA, since a merged 27B bf16 won't fit ~54 GB), not the stock `Qwen3.6-27B-FP8` (reference only). **Finding (measured): FP8 here is a *capacity* play, not a latency one** — on Blackwell sm_120 / torch 2.11 torchao has no fused M=1 GEMM, so FP8 *slows* single-stream decode; the win is halved weight bytes (the 27B fits the card). Frame it that way; don't sell FP8 as a speedup.
- **Speculative decoding must be exact:** rejection-sampling accept rule + residual resampling. Output must be distributionally identical to direct target sampling, proven by the correctness test (PASS @ n=1500). **Finding:** on the 9B/0.8B hybrid-attention pair net throughput is **~0.6–0.7× baseline** (replay tax > acceptance gain) — the correctness proof and α-lift are the deliverables, not raw speed. Never claim spec×batch speedups multiply.
- **Pin everything** in a `uv` lockfile the moment a working environment exists.

## Workflow
- **Benchmark harness before optimization** (`plans/02`) — no speedup claim without a reproducible baseline.
- **One worktree per phase:** branch `phase-NN-slug`, worktree `../inferd-wt/NN-slug`, merge into `dev` in phase order. `dev` → `main` at milestones.
- **`core/model_runner.py` is the hot shared file** (phases 04/05/06/10). Extend it via new methods per the interface contract in `plans/00`; never rewrite another phase's method without updating that contract.
- **Gates per merge:** lint, the phase's tests, numerical-equivalence where applicable, benchmark reproducibility, `code-review` (+ `security-review` for serving). Run the full gate set again before any `dev` → `main` release.
- **Commit only when explicitly asked.** The maintainer (`likhith-v1`) commits and merges; do the work and leave it uncommitted unless told otherwise. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Never commit weights/adapters/merged/data symlinks or `__pycache__`. A release (merge to `main`, tag, GitHub release, any push) is outward-facing — confirm first.
- **Serving must be crash-safe:** the engine thread owns the scheduler; an error there (e.g. CUDA OOM in `step()`) must surface to clients (error the in-flight + inbox channels, flip `alive`), never silently die and hang them. `/generate` returns 503 when the engine is down.
- Model choice per phase is **deferred, not static** — each `plans/NN` header is a recommendation; the body's Model Selection section is the real decision.

## Repo layout (✅ = present, ⏳ = pending)
- ✅ `finetune/` (QLoRA + eval + export) · `core/` (spec_decode, paged_cache, paged_attn, scheduler, model_runner, qwen35_patch) · `serve/` (FastAPI + SSE: app, engine, schemas) · `bench/` (harness, correctness, runners, results, `run_all.py`, `report.md`) · `scripts/` (`hero_fp8.py`, `smoke_load.py`) · `tests/` · `plans/` · `docs/` (`ENVIRONMENT.md`, `demo.md`) · `DECISIONS.md` · `README.md`
- ⏳ `dashboard/` (React + Vite) — phase 08, not yet created; the under-load demo capture (`docs/demo.md`) is blocked on it.
- Gitignored, local-only: `weights/`, `adapters/`, `merged/`, `data/`, `runs/` (never commit these or `__pycache__`).
