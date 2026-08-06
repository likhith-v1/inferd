Preferred model: GPT-5.5 | Reasoning: high

# 14 — Full-Attention Target: Spec-Decode Measured

> **Completed 2026-07-18.** The maintainer approved the family-wise max bootstrap
> as the correctness decision rule; both pairs pass at n=1500. The dense pair
> measured `0.508×` baseline at best and the hybrid pair `0.435×`; both are
> negative.

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
**Out:** draft distillation, tree speculation (backlog, sequence after this); batched spec (16).
**Standalone value:** "exact speculative decoding is net-positive on a croppable-KV
architecture and exact on both — the v1 caveat was the model, not the method."

## Approach
- Pin `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`
  with `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
  Local paths are `weights/Qwen3-8B` and `weights/Qwen3-0.6B`; expected
  `tokenizer.json` SHA-256 is
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
  and vocabulary/logit width is `151936`. This pair is a Phase-12 candidate,
  not an automatic MLX choice.
- Keep the exact sampler unchanged. `ModelRunner` owns opaque speculative-cache
  checkpoints: dense Qwen crops to the accepted prefix and forwards only the
  residual/bonus token; Qwen3.5 restores linear state and replays every emitted
  token. Unknown, sliding, malformed, or mislabeled caches fail before mutation.
- Run `python -m bench.phase14`: full and hybrid sequence-correctness gates first
  with a joint max-statistic bootstrap across positions (the individual p99 TVs
  remain diagnostics),
  then the existing harness over all 12 canonical prompts, max tokens 128,
  γ `{2,4,8}`, one warmup, and three paired repeats with rotated γ order.
- Classify net-positive only when median paired speedup is at least `1.05×` and
  every repeat exceeds `1.00×`; otherwise report flat or negative.

## Owned / Avoided files
- **Owns:** model-selection config + a new `bench/results/` subtree for the pair.
- **Shared:** `core/model_runner.py` (load the full-attention target),
  `core/spec_decode.py`, `bench/harness.py`, `bench/correctness.py` (opt-in
  family-wise decision rule), and report generation. **Avoids:**
  `bench/workload.py` and `core/qwen35_patch.py`.

## Commands / validation
```bash
uv run python -m bench.phase14
uv run python -m bench.phase14_microgate --pair both
```
- **CPU gates:** dense reject-at-zero/partial/all-accepted reconciliation; dense
  forward width 1; hybrid snapshot restore + full replay; loader routing; cache and
  tokenizer mismatch rejection.
- **GPU gates:** correctness PASS at n=1500, length 6, γ=4, three prompts,
  bootstrap 200; reconciliation logits match direct prefill at accepted counts
  0/partial/γ; dense forwards width 1 and hybrid forwards `accepted+1`; both Qwen3
  models load together on the RTX 5090.
- **Regression:** all unit tests, spec/correctness self-checks, `paged_equiv --mode
  both`, `batched_equiv`, report regeneration, `git diff --check`, and code review.

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
