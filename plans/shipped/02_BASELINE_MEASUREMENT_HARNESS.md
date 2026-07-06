Preferred model: Claude Sonnet 4.6 | Effort: high

# 02 — Baseline + Measurement Harness (build FIRST)

> No speedup claim without a baseline and a way to measure it. Build the headless harness, record the naive HF floor and the vLLM ceiling on identical workloads — before any optimization exists.

## Constraints (this phase)
local-first · WSL2 · RTX 5090 · text-only (Qwen3.5-9B language backbone) · vLLM as **ceiling only**, recorded once, never a runtime dep · identical workload across all rungs · warmup before timing · stamp hardware/CUDA/model versions on every result.

## Scope
**In:** a headless, importable harness (no HTTP) with a fixed workload (fixed prompts/seed/max-tokens/sampling) → tokens/sec, TTFT, inter-token latency, peak VRAM; the **naive floor** = HF `generate()` (default KV cache) on Qwen3.5-9B; the **ceiling** = vLLM on the same model, recorded once; throughput-vs-concurrency curves.
**Out:** any engine optimization; serving; dashboard.
**Standalone value:** "a reproducible inference benchmark with naive + vLLM reference numbers."

## Subagent breakdown
- **harness-builder** — workload spec, metric collection (timers, `torch.cuda.max_memory_allocated`), result schema in `bench/results/` (timestamped, version-stamped JSON).
- **baseline-runner** — HF `generate()` on the 9B text backbone; single-stream + concurrency sweep.
- **ceiling-runner** — vLLM on the same model/workload, recorded once; clearly labeled "ceiling, not a dependency."

## Git worktree workflow
- Branch `phase-02-harness`, worktree `../inferd-wt/02-harness`.
- Create off `dev` after 01 merges; rebase before merge; remove worktree after.

## Owned / Avoided files
- **Owns:** `bench/harness.py` (source of truth), `bench/results/` (schema + first numbers), `bench/workload.py` (fixed prompts/seed).
- **Avoids:** `core/` internals (don't pre-build the engine), `serve/`, `dashboard/`.

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run python -m bench.harness --engine hf --model Qwen3.5-9B --seed 0 \
    --max-tokens 256 --concurrency 1,2,4,8,16 --warmup 3
uv run python -m bench.harness --engine vllm --model Qwen3.5-9B --seed 0 \
    --max-tokens 256 --concurrency 1,2,4,8,16 --warmup 3   # ceiling, once
uv run python -m bench.harness --selfcheck    # asserts metric math on a tiny fixture
```
- **Validation:** re-running gives stable numbers within noise; metrics sane (TTFT < total; tokens/sec consistent with token count/time); results carry env stamp; the `--selfcheck` asserts timing/throughput arithmetic on a fixture (the one runnable check).

## Risks / Rollback / Exit / Handoff
- **Risks:** vLLM Blackwell support friction; inconsistent warmup skews numbers; metric definition drift across phases.
- **Mitigation/Rollback:** if vLLM won't run on sm_120 yet, record ceiling later and proceed with HF floor (note the gap); freeze metric definitions in `harness.py` docstring so all later phases reuse them verbatim.
- **Exit:** reproducible naive-HF + vLLM numbers on identical workloads + throughput-vs-concurrency curves committed to `bench/results/`.
- **Handoff:** every later phase imports `bench.harness` and reuses the workload + metric definitions; 09 aggregates and plots these.

## Model Selection (confirm or override)
- **Claude Sonnet 4.6 | high** *(recommended)* — measurement plumbing + careful metric definitions, cost-efficient.
- **GPT-5.4 | high** — equally fine for harness code; pick on availability/cost.
- **Claude Opus 4.8 | high** — only if metric-definition subtleties (fair vLLM parity) need deeper reasoning.
> Recommendation: Sonnet 4.6 high. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** Hugging Face MCP for exact model/version metadata to stamp results.
- **Candidates:** `huggingface-trackio` (log/visualize baseline runs), `huggingface-community-evals` (only if borrowing eval harness patterns).
- **Question:** adopt Trackio now for run logging, or plain JSON in `bench/results/` until the dashboard (08)?

## Execution questions for this phase
1. Exact workload: how many prompts, what lengths, which sampling (greedy vs temp>0)? (Greedy simplifies, but the correctness test in 04/09 needs temp>0 — pick a fixed sampling profile now.)
2. Concurrency grid: `1,2,4,8,16` or push to OOM to characterize the naive ceiling?
3. Is vLLM runnable on this Blackwell stack yet — record ceiling now or defer to 09?
4. Result format: plain JSON now, or Trackio from the start?
