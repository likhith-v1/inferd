Preferred model: Claude Opus 4.8 | Effort: xhigh

# 16 — Batched Speculative Decoding

> Extend the exact accept/replay through the continuous-batching scheduler so
> speculation and batching run on the same step — the one axis v1 measured
> separately. Scoped as **"implemented + measured honestly,"** not a speedup claim.

## Constraints (this phase)
RTX 5090 · exact rejection sampling + residual resample per sequence · reuse
`core/batched_cache.py` stack/split surgery · **honest reporting** — on the hybrid
model the Phase-04 replay tax likely keeps this net-negative; report it, don't hide
it. Pairs naturally with phase 14's full-attention target, where it may go positive.

## Scope
**In:** per-sequence draft proposal + parallel verify + accept/replay inside
`ContinuousBatchScheduler.step()`; ragged accept lengths across the batch; the
benefit-fade-at-high-concurrency interaction measured (the §2 tension).
**Out:** tree speculation (backlog).
**Standalone value:** "speculative decoding and continuous batching on one step,
with the interaction measured — the completeness the v1 writeup flagged as open."

## Approach
- Build on `core/batched_cache.py` (`stack_caches`/`split_caches`) and
  `core/spec_decode.py`; the hard part is heterogeneous accept lengths per row and
  the hybrid replay tax multiplied across the batch.
- Strongly prefer sequencing **after phase 14** (full-attention target) so accept
  lengths don't each drag a replay pass; measure on both architectures for contrast.

## Owned / Avoided files
- **Owns:** batched accept/replay in `core/scheduler.py` + `core/spec_decode.py`
  extensions. **Shared:** `core/batched_cache.py`. **Depends on:** phase 14 for a
  non-replay path to actually win.

## Commands / validation
```bash
uv run python -m bench.batched_equiv --target <target>    # batched == serial logits
uv run python -m bench.harness --engine batched --spec --concurrency 1,4,8
```
- **Gate:** batched-vs-serial logits at the bf16 floor (as in Phase 06); correctness
  gate still exact per sequence; throughput reported per concurrency with the fade.

## Risks / Rollback / Exit
- **Risks:** replay tax × batch width dominates on hybrid; ragged accept lengths
  complicate the stack/split; sync overhead.
- **Rollback:** report the net-negative honestly on hybrid and the (hoped)
  net-positive on the phase-14 full-attention pair — the interaction curve is the
  deliverable regardless of sign.
- **Exit:** batched spec implemented, exactness preserved, interaction measured on
  ≥1 architecture.

## Model Selection
- **Claude Opus 4.8 | xhigh** *(recommended)* — subtle batched-state + accept-length
  bookkeeping. **GPT-5.5 | very high** — co-equal.

## Execution questions
1. Gate this phase behind phase 14, or measure hybrid batched-spec first for the
   honest contrast?
2. Cap γ per sequence, or adapt per row by recent acceptance?
