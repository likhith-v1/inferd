Preferred model: Claude Opus 4.8 | Effort: xhigh

# 04 — Speculative Decoding (headline engine result)

> Implement exact speculative decoding from scratch — draft proposal, single-pass target verification, and the **exact accept/resample rule** (incl. the residual branch) — then run the α-lift experiment that fuses fine-tuning and inference.

## Constraints (this phase)
RTX 5090 · text-only backbone (9B target + 0.8B draft, shared `qwen3_5` processor) · **exact rejection sampling** (Leviathan/Chen) · **residual resampling** on first rejection · draft **distilled against the fine-tuned 9B** for α-lift · output must be **distributionally identical** to direct target sampling · reuse `bench.harness` metrics.

## Scope
**In:** draft runner (γ-token proposal); target verification in one forward pass over γ+1 positions; exact accept/resample; γ-sweep {2,4,8} with α + wall-clock vs theory; the **α-lift experiment** (stock vs distilled draft, target fixed) → Δα, Δthroughput; the correctness test (in `bench/correctness.py`).
**Out:** paging (05), batching (06), tree speculation (stretch).
**Standalone value:** "from-scratch exact speculative decoding with a provable correctness test + the α-lift bridge result."

### The exact rule (must implement precisely)
For drafted token `x` with target `p(x)`, draft `q(x)`: accept w.p. `min(1, p(x)/q(x))`. On first rejection at position k: resample from residual `p_resid(x) = max(0, p(x)-q(x)) / Σ max(0, p(x)-q(x))`, discard all drafts after k. If all γ accepted: sample one bonus token from `p`. Expected accepted/target-call ≈ `(1-α^(γ+1))/(1-α)`; net gain also depends on `c = draft_cost/target_cost`.

## Subagent breakdown
- **spec-decode-implementer** — draft loop, single-pass verify, accept/resample; the residual branch is the crux — write it carefully.
- **correctness-test author** — `bench/correctness.py`: fixed seed, many continuations spec vs direct, next-token distributions match within sampling noise (χ² / total-variation).
- **alpha-lift / distillation runner** — sequence-level KD: sample from the fine-tuned 9B, train the 0.8B on those outputs; compare stock vs distilled draft, target fixed.
- **benchmark-runner** — γ-sweep α + speedup vs the theory curve; plots.

## Git worktree workflow
- Branch `phase-04-spec-decode`, worktree `../inferd-wt/04-spec`. Needs merged 9B from 03. Touches `core/model_runner.py` per the master contract (text-backbone load only).

## Owned / Avoided files
- **Owns:** `core/spec_decode.py`, `bench/correctness.py`, `finetune/distill_draft.py`.
- **Shared (per contract):** `core/model_runner.py` (add draft load + `forward` reuse). **Avoids:** `core/paged_cache.py`, `core/scheduler.py` (later phases).

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run python finetune/distill_draft.py --target merged/9b --draft Qwen/Qwen3.5-0.8B --out adapters/draft-distilled
uv run python -m core.spec_decode --target merged/9b --draft adapters/draft-distilled --gamma 4 --max-tokens 256
uv run python -m bench.harness --engine spec --gamma 2,4,8 --target merged/9b --draft <draft>   # α + speedup
uv run python -m bench.correctness --target merged/9b --draft <draft> --n 2000 --test tv   # MUST pass
```
- **Validation (the differentiator):** correctness test passes — spec output distribution matches direct target sampling within sampling noise (χ²/TV under threshold). α measured per γ; speedup tracked vs theory; α-lift quantified (stock vs distilled Δα, Δthroughput).

## Risks / Rollback / Exit / Handoff
- **Risks:** residual-resampling botched → silent distribution skew (correctness test catches it); 0.8B α too low to net a speedup once `c` is paid; distillation lifts α less than hoped.
- **Mitigation/Rollback:** gate everything on the correctness test before any speedup claim; if 0.8B α is poor, try `Qwen3.5-2B`/`4B` drafts (same family); if distillation underperforms, report the honest (possibly small) Δα — the *method* and correctness are the contribution.
- **Exit:** correctness test green; α/speedup plots; α-lift result quantified.
- **Handoff:** `core/spec_decode.py` + `model_runner` `forward` consumed by 06 (spec-with-batching); `bench/correctness.py` extended by 09.

## Model Selection (confirm or override)
- **Claude Opus 4.8 | xhigh** *(recommended)* — correctness-critical math + the residual branch reward deep reasoning.
- **GPT-5.5 | very high** — co-equal; pick on availability/cost.
- **GPT-5.5 | high / Opus 4.8 | high** — acceptable if reviewing against a known reference impl.
> Recommendation: Opus 4.8 xhigh or GPT-5.5 very-high — this is the phase to spend reasoning budget. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** `huggingface-papers` (Leviathan 2023 / Chen 2023 — get the rule exactly right), `code-review` (mandatory gate on accept/resample).
- **Candidates:** HF MCP (draft-model candidates 0.8B/2B/4B), `huggingface-trackio` (α/speedup logging).
- **Question:** pull the papers via `huggingface-papers` now? Make `code-review` on the accept/resample rule a blocking gate?

## Execution questions for this phase
1. Sampling profile for α (temperature/top-p) — must match the harness profile from 02.
2. γ values beyond {2,4,8}? Adaptive γ a stretch?
3. Distillation budget: how many samples from the 9B, how long to train the 0.8B?
4. Correctness test: χ² vs total-variation, sample count `n`, and the pass threshold?
5. If 0.8B α is poor, authorize 2B/4B fallback drafts in the same experiment?
