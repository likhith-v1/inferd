Preferred model: Claude Sonnet 4.6 | Effort: high

# 17 — vLLM Ceiling on Blackwell (sm_120)

> Fill the deferred rung. The measurement thesis is "within K× of vLLM," but the
> vLLM ceiling is currently *pending* — it wouldn't build on sm_120 in v1 (reported,
> never faked). Re-run when Blackwell wheels land and slot the number in. Blocked on
> upstream, not on us — a watch-item, not active work.

## Constraints (this phase)
RTX 5090 (Blackwell sm_120) · vLLM as **ceiling reference only**, never a runtime
dependency · identical frozen workload (`bench/workload.py`: prompts, seeds,
max-tokens, sampling) across all rungs · **nothing faked** — if it still won't
build, keep it reported as "pending," per the phase-09 rollback rule.

## Scope
**In:** stand up vLLM on sm_120 once wheels exist; run the three-rung sweep on the
identical workload; slot the ceiling into `bench/report.md` and the README results
table.
**Out:** any dependence of the engine on vLLM; tuning vLLM for a better ceiling.
**Standalone value:** "the 'within K× of vLLM' framing is closed with a real
ceiling number on Blackwell."

## Approach
- Reuse the existing vLLM runner scaffold under `bench/runners/`; the harness and
  workload are unchanged — this is an environment/availability unblock plus a
  measurement, not new engine code.
- Re-run `bench/run_all.py --plots` so the ceiling flows into plots + report the
  same way every other number does.

## Owned / Avoided files
- **Owns:** `bench/runners/` vLLM path, `bench/run_all.py`, `bench/report.md`, the
  README results table. **Avoids:** `core/`, `serve/`.

## Commands / validation
```bash
uv run python -m bench.harness --engine vllm --model Qwen3.5-9B --concurrency 1,4,8,16,32
uv run python bench/run_all.py --plots
```
- **Gate:** vLLM runs the *identical* matched workload as the naive-HF and inferd
  rungs (the phase-09 apples-to-apples rule); numbers reproducible ±noise.

## Risks / Rollback / Exit
- **Risks:** sm_120 wheels still unavailable or unstable; workload drift makes the
  comparison apples-to-oranges (the phase-09 trap).
- **Rollback:** keep "ceiling pending" in the docs — the honest v1 posture.
- **Exit:** a real vLLM ceiling on Blackwell in the three-rung table, or an updated
  "still pending" note with the current blocker.

## Model Selection
- **Claude Sonnet 4.6 | high** *(recommended)* — environment plumbing + measurement,
  cost-efficient.

## Execution questions
1. Poll for sm_120 vLLM wheels on a cadence, or wait for a maintainer trigger?
2. Pin a specific vLLM version for the ceiling so the number is reproducible?
