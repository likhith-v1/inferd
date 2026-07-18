Preferred model: Claude Opus 4.8 | Effort: high

# 00 — Master Orchestration

> Conducts the whole build: order, ownership, merges, gates, and per-phase model choice for `inferd` — a from-scratch local LLM stack (QLoRA → spec-decode → paged KV → continuous batching → serve → dashboard), benchmarked vs naive HF and vLLM on a single RTX 5090.

This file is the source of truth for cross-phase coordination. Phase files (01–11) stay lean and defer their shared rules here.

## Constraints (global — every phase inherits)
local-first · WSL2 Ubuntu · RTX 5090 (Blackwell sm_120) · CUDA-native (CUDA 12.8+) · no cloud/API inference · no MLX · no GemForge · QLoRA fine-tuning · Unsloth first, then Axolotl / Llama-Factory / ms-swift / TRL+PEFT · Qwen3.6-27B showpiece · Qwen3.5-9B target + Qwen3.5-0.8B draft · speculative decoding with **exact** rejection sampling · residual resampling on rejection · draft distillation for α-lift · paged KV-cache · Triton paged attention (FlashAttention varlen fallback) · continuous batching · FastAPI · React + Vite dashboard · **benchmark harness BEFORE optimization** · vLLM as ceiling only (never a runtime dep) · optional FP8 27B hero · **text-only (vision tower stripped)** · **parameterized fine-tuning corpus**.

Verified (HF, Jun 2026): all three Qwen IDs exist; **all three are multimodal** (`AutoModelForMultimodalLM`, arch `qwen3_5`) → load wrapper + processor, extract `language_model` backbone, strip vision. `Qwen/Qwen3.6-27B-FP8` exists (reference only; hero is our *fine-tuned* FP8). Fallback drafts: `Qwen3.5-2B`, `Qwen3.5-4B`.

## Execution order (each arrow is a defensible stop)
```
01 env → 02 harness → 03 QLoRA → 04 spec-decode (+α-lift) → 05 paged cache
   → 06 continuous batching → 07 serving → 08 dashboard → 09 bench/report
   → 10 FP8 hero (optional) → 11 docs
```
03 can run largely parallel to 04–06 once 02 lands (it feeds adapters, not engine internals). 08 needs 07's `/metrics`. 09 needs 02 + 04–07. 11 closes everything.

## Branch / worktree strategy
- Branch per phase: `phase-NN-slug` (e.g. `phase-04-spec-decode`).
- Worktree: `../inferd-wt/NN-slug` (sibling dir; keeps the main checkout clean).
- Integration branch: `dev`. Phases merge into `dev` **in phase order**; `dev` → `main` at milestone tags (`v0.2-baseline`, `v0.4-spec`, `v0.6-batch`, `v0.8-serve`, `v1.0`).
- Create: `git worktree add ../inferd-wt/NN-slug -b phase-NN-slug dev`
- Before merge: rebase on latest `dev`, re-run gates. Cleanup: `git worktree remove ../inferd-wt/NN-slug`.

## Subagent ownership map
| Phase | Branch | Worktree | Primary owned paths |
|------|--------|----------|----------|
| 01 | phase-01-env | ../inferd-wt/01-env | `pyproject.toml`,`uv.lock`,`.python-version`,`.gitignore` |
| 02 | phase-02-harness | ../inferd-wt/02-harness | `bench/harness.py`,`bench/results/` |
| 03 | phase-03-qlora | ../inferd-wt/03-qlora | `finetune/*` |
| 04 | phase-04-spec-decode | ../inferd-wt/04-spec | `core/spec_decode.py`,`bench/correctness.py` |
| 05 | phase-05-paged-cache | ../inferd-wt/05-paged | `core/paged_cache.py`,`core/paged_attn.py` |
| 06 | phase-06-batching | ../inferd-wt/06-batch | `core/scheduler.py` |
| 07 | phase-07-serving | ../inferd-wt/07-serve | `serve/*` |
| 08 | phase-08-dashboard | ../inferd-wt/08-dash | `dashboard/*` |
| 09 | phase-09-bench | ../inferd-wt/09-bench | `bench/results/`, report md |
| 10 | phase-10-fp8 | ../inferd-wt/10-fp8 | FP8 path + hero script |
| 11 | phase-11-docs | ../inferd-wt/11-docs | `DECISIONS.md`,`README.md` |

## Merge order & conflict rules
- Merge strictly in phase order into `dev`; never merge a later phase ahead of an unmerged dependency.
- **`core/model_runner.py` is the hot shared file** (04, 05, 06, 10). It has an **interface contract** (below) owned by this master file. Each phase *extends* via new, clearly-named methods; no phase rewrites another's method without editing the contract here first. Conflicts on this file are resolved by re-applying against the contract, not by ad-hoc merge.
- `bench/correctness.py` shared by 04 (author) and 09 (extends) — 04 lands first; 09 only adds.
- `bench/results/` is append-only per run (timestamped subdirs); never overwrite.

### `core/model_runner.py` interface contract (extend, don't break)
- `load_target() / load_draft()` → return the **text-only** `language_model` backbone (+ processor.tokenizer); vision modules dropped.
- `forward(tokens, kv) -> logits` — single source of truth for a forward pass; both spec-decode and batching call it.
- `kv` is an opaque handle: contiguous in 02/04, paged in 05+. Callers never assume layout.
- `ModelRunner.cache_reconciliation` declares the architecture strategy;
  `checkpoint_speculation()` returns an opaque rollback handle and
  `reconcile_speculation()` commits accepted tokens plus the residual/bonus token.
- `validate_speculation_pair()` rejects tokenizer, special-token, or logit-width
  mismatches before speculative generation.
- 10 adds an FP8 load variant behind the same `load_target()` signature (dtype flag).

## Quality gates (run at every phase merge)
1. Lint/format clean. 2. Phase's own tests pass. 3. **Numerical-equivalence** gate where applicable (04 distribution test, 05 logits-within-tol). 4. **Benchmark reproducibility** — harness re-run gives stable numbers (±noise). 5. `code-review` (and `security-review` for serving) before merge to `dev`.

## Stopping points (resume value at each)
After 02: "built a reproducible inference benchmark harness." After 03: "QLoRA fine-tuned a 27B + 9B, win-rate measured." After 04: "from-scratch exact speculative decoding + the α-lift bridge result." After 06: "paged KV + continuous batching, throughput-vs-concurrency vs vLLM." After 08: "live dashboard + full writeup." After 10: "flagship 27B served via FP8 through my own engine."

## Model Selection (the framework all phases reference)
Pick by **task nature, not brand**. Candidates and where each shines:
- **GPT-5.5 | very high / high** — deepest long-horizon reasoning; correctness-critical kernels & math (spec-decode rule, Triton paged-attn).
- **GPT-5.4 | high / medium** — strong but lighter/cheaper; bounded integration & glue, quantization plumbing.
- **Claude Opus 4.8 | xhigh / high** — deep code reasoning + orchestration; scheduling, multi-stack fallback reasoning, subtle stats.
- **Claude Sonnet 4.6 | high** — cost-efficient; **mandated for frontend/UX (08)**; serving glue, env setup, docs.

Recommended default for THIS file: **Claude Opus 4.8 | Effort: high** (orchestration). Alternative: GPT-5.5 | high. **Confirm or override before starting.**

## The honest tension to hold across phases (plan §2)
Spec-decode helps **single-stream latency**; paging + batching help **multi-request throughput**. They do **not** multiply — at large batch the GPU saturates and speculation's benefit fades. Lead the story with spec-decode; batching is act two; report the interaction in 06/09, don't hide it.

## Execution questions for this phase
1. Confirm `dev` as the integration branch and the `../inferd-wt/` sibling layout, or prefer in-repo `.worktrees/` (gitignored)?
2. Milestone tag names/cadence OK, or tag only at `v1.0`?
3. Should 03 (QLoRA) run on its own machine/time budget in parallel, or strictly serialized after 02?
4. Confirm the `model_runner.py` contract owner = this file (changes require a PR to 00).

## Skills & MCPs to decide at implementation
- **Recommend:** `code-review` (every merge), `security-review` (07 serving), Hugging Face MCP (model/version metadata across phases).
- **Candidates to confirm:** `huggingface-trackio` (cross-phase metric logging), firecrawl (vendor docs), `verify`/`run` (end-to-end checks).
- **Question:** which gates become blocking CI vs. advisory? Which MCPs are allowed offline (HF MCP needs network — restrict to the one online setup window)?
