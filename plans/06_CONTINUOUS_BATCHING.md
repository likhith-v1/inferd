Preferred model: Claude Opus 4.8 | Effort: high

# 06 — Continuous Batching

> Iteration-level scheduling (Orca-style): a running batch that evicts finished and admits waiting requests under a free-block budget, with per-iteration prefill for new admits — turning the paged cache into a concurrent-throughput win.

## Constraints (this phase)
RTX 5090 · text-only · iteration-level (not request-level) scheduling · admit/evict under the **free-block budget** from 05 · honest reporting of the spec-decode × batching interaction (plan §2: they don't multiply) · reuse `bench.harness`.

## Scope
**In:** the scheduler (running batch, evict-finished, admit-waiting under free blocks, prefill new admits); throughput-vs-concurrency win over naive static batching; spec-decode-with-batching measured with the benefit-fade reported; a live "active sequences" hook for the dashboard.
**Out:** serving/HTTP (07); chunked prefill (stretch).
**Standalone value:** "from-scratch continuous batching with a throughput-vs-concurrency curve approaching vLLM, and an honest spec×batch interaction analysis."

## Subagent breakdown
- **scheduler-implementer** — `core/scheduler.py`: waiting/running queues, per-iteration step, admit/evict against free blocks, prefill of new admits.
- **throughput-benchmark-runner** — throughput-vs-concurrency vs naive batching and vs the vLLM ceiling from 02.
- **interaction-analyst** — measure spec-decode under batching; quantify and write up the benefit-fade (the §2 tension).

## Git worktree workflow
- Branch `phase-06-batching`, worktree `../inferd-wt/06-batch`. Needs 05 (free-block budget) + 04 (`forward`, spec path). Touches `core/model_runner.py` per contract (batched `forward`).

## Owned / Avoided files
- **Owns:** `core/scheduler.py`, `tests/test_scheduler.py`.
- **Shared (per contract):** `core/model_runner.py` (batched forward). **Avoids:** `core/paged_cache.py` (consume its API), `core/spec_decode.py` (call, don't rewrite).

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run python -m pytest tests/test_scheduler.py     # admit/evict/no-starvation invariants
uv run python -m bench.harness --engine batched --model merged/9b --concurrency 1,2,4,8,16,32
uv run python -m bench.harness --engine batched+spec --gamma 4 --concurrency 1,8,32   # interaction
```
- **Validation:** scheduler invariants (no lost/starved seq, never exceeds free-block budget, finished evicted promptly); throughput-vs-concurrency beats naive static batching and trends toward the vLLM ceiling; spec×batch numbers show the expected fade; `test_scheduler.py` asserts admit/evict logic on a fixture.

## Risks / Rollback / Exit / Handoff
- **Risks:** admission deadlock/starvation under block pressure; prefill stalls decode latency; spec-decode complicates batched verification (variable accepted lengths per seq).
- **Mitigation/Rollback:** simple FCFS admission + a starvation guard first (optimize later, `# ponytail:` the heuristic with its ceiling); cap prefill chunk per iteration; if batched spec is too complex for v1, ship batching without spec and report spec as single-stream only (still a complete story).
- **Exit:** throughput-vs-concurrency win over naive batching; live active-sequences view; spec×batch fade measured and reported honestly.
- **Handoff:** scheduler is the engine 07 wraps with HTTP; the active-sequences + per-iteration metrics feed 07's `/metrics` → 08's dashboard.

## Model Selection (confirm or override)
- **Claude Opus 4.8 | high** *(recommended)* — scheduler/concurrency correctness + the interaction analysis reward strong reasoning.
- **GPT-5.5 | high** — co-equal; pick on availability/cost.
- **GPT-5.4 | high** — viable if the scheduler stays a simple FCFS + budget loop.
> Recommendation: Opus 4.8 high or GPT-5.5 high. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** `huggingface-papers` (Orca, OSDI 2022 — iteration-level scheduling design), `code-review` gate on the scheduler.
- **Candidates:** `huggingface-trackio` (live throughput/active-seq logging that 08 can reuse).
- **Question:** pull the Orca paper now? Adopt Trackio for the live metrics stream, or hand-roll the `/metrics` feed in 07?

## Execution questions for this phase
1. Admission policy: FCFS only, or priority/length-aware? Starvation guard required?
2. Prefill: chunked or whole-prompt per admit? Max prefill tokens per iteration?
3. Batched spec-decode in v1, or batching-only with spec kept single-stream?
4. Max concurrent sequences target (tied to 05's block budget + 9B KV headroom)?
