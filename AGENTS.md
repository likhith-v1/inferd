# AGENTS.md — `inferd`

A from-scratch local LLM inference stack: **QLoRA fine-tune → speculative decoding → paged KV-cache → continuous batching**, served via FastAPI + a React dashboard, benchmarked against naive HF and vLLM. Runs fully local on a single RTX 5090. See `plan.md` for the vision and `plans/` for the execution pack (one file per phase, `00`–`11`).

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
- **FP8 is the one quantization exception**, scoped to the 27B hero demo (`plans/10`) and nowhere else. The hero is the *fine-tuned* 27B (merge → FP8), not the stock `Qwen3.6-27B-FP8` (reference only).
- **Speculative decoding must be exact:** rejection-sampling accept rule + residual resampling. Output must be distributionally identical to direct target sampling, proven by the correctness test.
- **Pin everything** in a `uv` lockfile the moment a working environment exists.

## Workflow
- **Benchmark harness before optimization** (`plans/02`) — no speedup claim without a reproducible baseline.
- **One worktree per phase:** branch `phase-NN-slug`, worktree `../inferd-wt/NN-slug`, merge into `dev` in phase order. `dev` → `main` at milestones.
- **`core/model_runner.py` is the hot shared file** (phases 04/05/06/10). Extend it via new methods per the interface contract in `plans/00`; never rewrite another phase's method without updating that contract.
- **Gates per merge:** lint, the phase's tests, numerical-equivalence where applicable, benchmark reproducibility, `code-review` (+ `security-review` for serving).
- Model choice per phase is **deferred, not static** — each `plans/NN` header is a recommendation; the body's Model Selection section is the real decision.

## Repo layout (target)
`finetune/` (QLoRA + eval + export) · `core/` (spec_decode, paged_cache, paged_attn, scheduler, model_runner) · `serve/` (FastAPI + SSE) · `bench/` (harness, correctness, results) · `dashboard/` (React + Vite) · `plans/` · `DECISIONS.md` · `README.md`.
