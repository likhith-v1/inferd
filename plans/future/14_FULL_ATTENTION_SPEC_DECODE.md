Preferred model: GPT-5.5 | Reasoning: high

# 14 — Full-Attention Target: Spec-Decode Net-Positive

> The single highest-leverage experiment. v1 proved the exact accept rule but
> measured ~0.6–0.7× wall-clock — **because Qwen3.5's hybrid linear-attention state
> can't be cropped**, forcing a replay pass every round. Swap in a *pure
> full-attention* target where rollback is a slice, and the already-proven
> implementation should finally show net-positive speedup.

## Constraints (this phase)
RTX 5090 · text-only · exact rejection sampling + residual resample (unchanged) ·
matched draft/target tokenizers · reuse the frozen `bench.workload.CANONICAL`
profile and the `bench/correctness.py` distribution gate · **honest contrast** —
keep the hybrid (Qwen3.5) result side-by-side; this adds a data point, it does not
retract the v1 finding.

## Scope
**In:** select a full-attention Qwen (or same-family) target + matched draft; run
the γ∈{2,4,8} sweep, α, and wall-clock; re-run the correctness gate on the new pair;
report net-positive (expected) vs the hybrid net-negative.
**Out:** tree speculation (backlog, sequence after this); batched spec (16).
**Standalone value:** "exact speculative decoding is net-positive on a croppable-KV
architecture and exact on both — the v1 caveat was the model, not the method."

## Approach
- `core/spec_decode.py` already branches crop-vs-replay for the hybrid cache; a
  full-attention target takes the **cheap crop path** (no snapshot→restore→replay),
  so no new algorithm — the work is the model load path and the measurement.
- Reuse the α-lift machinery (`finetune/distill_draft.py`) if a distilled draft is
  wanted for the new pair; optional.

## Owned / Avoided files
- **Owns:** model-selection config + a new `bench/results/` subtree for the pair.
- **Shared:** `core/model_runner.py` (load the full-attention target),
  `bench/harness.py`. **Avoids:** the hybrid patch `core/qwen35_patch.py` (leave the
  Qwen3.5 path intact for the contrast run).

## Commands / validation
```bash
uv run python -m bench.harness --engine spec --target <full-attn-target> --draft <matched-draft>
uv run python -m bench.correctness --target <full-attn-target> --draft <matched-draft>
```
- **Gate:** correctness (distribution-equivalence within the bootstrapped null) must
  PASS on the new pair; wall-clock reported honestly whichever way it lands.

## Risks / Rollback / Exit
- **Risks:** a good full-attention target/draft pair with shared tokenizer + a
  small-enough draft may be scarce in the current lineup; if net speedup is still
  flat, the bottleneck was draft cost, not rollback — informative either way.
- **Rollback:** even a flat result cleanly isolates rollback tax vs draft cost — a
  publishable clarification of the v1 finding.
- **Exit:** γ-sweep + correctness on a full-attention pair, with the net-positive
  (or not) result recorded against the hybrid baseline in `DECISIONS.md`.

## Model Selection
- **GPT-5.5 | high** *(recommended)* — the accept-rule branch is already written;
  this is careful measurement + model plumbing.

## Execution questions
1. Which full-attention target/draft pair (shared tokenizer, draft small enough to
   pay for itself)?
2. Distill a draft for the new pair, or measure stock-draft α first?
3. Is this the model the MLX demo (phase 12) should also adopt?
