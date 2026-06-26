Preferred model: Claude Opus 4.8 | Effort: high

# 09 — Benchmarks, Correctness & Reporting

> Aggregate every metric across the three rungs (naive HF / ours / vLLM ceiling), prove the speculative output is distributionally identical to the target, and produce the report + plots where every resume number is real and reproducible from one command.

## Constraints (this phase)
identical workload across all rungs (reuse 02's `bench.harness`) · warmup before timing · stamp hardware/CUDA/model versions · vLLM as **ceiling only** · framing always "within K× of the production engine, from scratch" · the correctness test is the differentiator — do not skip.

## Scope
**In:** full metric aggregation (single-stream + aggregate throughput, TTFT, inter-token latency, α, throughput-vs-concurrency 1→N, VRAM-vs-concurrency to OOM); three-rung comparison; the distribution-equivalence correctness test (extends 04's `bench/correctness.py`); comparison plots; one-command reproduction; fine-tuning metrics (golden-set win-rate, held-out loss) folded in.
**Out:** new engine features; the dashboard (08).
**Standalone value:** "a reproducible benchmark report with a provable spec-decode correctness result vs a vLLM ceiling."

## Subagent breakdown
- **correctness-stat author** — harden `bench/correctness.py`: fixed seed, large `n`, χ²/total-variation with a justified threshold; document what "passes" means.
- **benchmark-aggregator** — run all rungs on the identical workload; collect into `bench/results/` with env stamps; one-command runner.
- **plot / report author** — comparison plots (throughput-vs-concurrency, VRAM-vs-concurrency, α-vs-γ vs theory, α-lift Δ) + the benchmark report md.

## Git worktree workflow
- Branch `phase-09-bench`, worktree `../inferd-wt/09-bench`. Needs 02 + 04–07. Extends `bench/correctness.py` (append only; 04 owns the core).

## Owned / Avoided files
- **Owns:** `bench/results/` (final numbers + plots), `bench/report.md`, `bench/run_all.py`.
- **Shared:** `bench/correctness.py` (extend, don't rewrite 04's rule). **Avoids:** `core/`, `serve/`, `dashboard/`.

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run python bench/run_all.py --rungs hf,ours,vllm --concurrency 1,2,4,8,16,32 --seed 0   # one command
uv run python -m bench.correctness --target merged/9b --draft <draft> --n 5000 --test tv   # MUST pass
uv run python bench/run_all.py --plots   # regenerate all comparison figures
```
- **Validation:** correctness test passes at high `n` (the headline evidence); numbers reproduce within noise on re-run; every plot regenerates from committed results; the three resume bullets each trace to a number in `bench/results/`.

## Risks / Rollback / Exit / Handoff
- **Risks:** unfair rung comparison (different sampling/warmup) inflating/deflating claims; vLLM parity hard to match exactly; correctness test flaky at low `n`.
- **Mitigation/Rollback:** single shared workload + sampling profile enforced in `bench.harness`; document any unavoidable vLLM config differences; raise `n` until the test is stable; if vLLM still won't run on Blackwell, report "ceiling pending" honestly rather than faking it.
- **Exit:** every resume number real + reproducible from one command; correctness test green; plots committed.
- **Handoff:** numbers + plots → 11's README/DECISIONS and the demo; 08 reuses the same metric definitions.

## Model Selection (confirm or override)
- **Claude Opus 4.8 | high** *(recommended)* — the distribution-equivalence stats (χ²/TV thresholds) are subtle and reward careful reasoning.
- **GPT-5.5 | high** — co-equal; pick on availability/cost.
- **Claude Sonnet 4.6 | high** — fine for the aggregation/plotting parts if the correctness stats are already locked from 04.
> Recommendation: Opus 4.8 high for the correctness stats; Sonnet 4.6 acceptable for the reporting half. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** `huggingface-trackio` (aggregate/visualize runs), HF MCP (exact version stamps for provenance).
- **Candidates:** `code-review` on the correctness test; `huggingface-community-evals` (if reusing eval-stat patterns).
- **Question:** Trackio for the final aggregation or plain JSON + matplotlib? Re-review the correctness threshold with `code-review`?

## Execution questions for this phase
1. χ² vs total-variation as the primary test; final `n` and pass threshold?
2. Concurrency grid for the final curves; push to OOM to show the VRAM-vs-concurrency story?
3. If vLLM can't run on Blackwell yet, ship with "ceiling pending" or block the report?
4. Which exact numbers become the three resume bullets (lock them here)?
