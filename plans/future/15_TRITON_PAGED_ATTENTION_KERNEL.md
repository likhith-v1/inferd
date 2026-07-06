Preferred model: GPT-5.5 | Reasoning: very high

# 15 — Triton Paged-Attention Kernel

> Replace the Python gather + dense/SDPA *reference* in `core/paged_attn.py` with a
> fused Triton gather-and-attend kernel — the perf half of paging and the
> "kernel credential" v1 explicitly deferred to "plan v2". Meaningful only on top
> of phase 13's persistent paged runtime.

## Constraints (this phase)
RTX 5090 (Blackwell sm_120) · Triton · **FlashAttention-varlen as a first-class
fallback**, not an afterthought · numerically equivalent to the existing
`sdpa_reference` within the bf16 noise floor · block size 16.

## Scope
**In:** a Triton kernel gathering paged K/V blocks and attending in one pass,
validated against `sdpa_reference`; wired into the phase-13 live path; kernel-vs-
fallback timing.
**Out:** non-paged attention; changes to the allocator (13 owns it).
**Standalone value:** "a from-scratch Triton paged-attention kernel, numerically
equivalent to SDPA, driving the live paged runtime."

## Approach
- Reuse `tests/test_paged_equiv.py` and the `sdpa_reference` in `core/paged_attn.py`
  as the equivalence oracle (already validates gather-and-attend across page
  boundaries and GQA/MHA head ratios).
- Keep the FlashAttention-varlen fallback behind the same interface so a JIT/perf
  failure on sm_120 still ships a working, faster-than-Python path.

## Owned / Avoided files
- **Owns:** `core/paged_attn.py` (kernel), `tests/test_paged_equiv.py` (extend).
- **Depends on:** phase 13 (live paged cache) to be exercised end-to-end.
  **Avoids:** allocator internals, scheduler.

## Commands / validation
```bash
uv run python -m core.paged_attn --selfcheck
uv run python -m unittest tests.test_paged_equiv
uv run python -m core.paged_attn --bench      # Triton vs flashattn-varlen fallback
```
- **Gate:** kernel logits == SDPA within bf16 tol across boundaries
  {1,15,16,17,31,32,33}; kernel within a sane factor of the fallback (or ship
  fallback and document the attempt, per the v1 phase-05 rollback rule).

## Risks / Rollback / Exit
- **Risks:** Triton on sm_120 immature → won't JIT or is slow; off-by-one page
  indexing → subtle logit drift.
- **Rollback:** ship the FlashAttention-varlen fallback; keep Triton as a
  documented attempt — still a real paged-runtime win from phase 13.
- **Exit:** paged attention runs through a fused kernel (or documented fallback),
  equivalence green, timing recorded.

## Model Selection
- **GPT-5.5 | very high** *(recommended)* — kernel correctness + index math.
- **Claude Opus 4.8 | xhigh** — co-equal.

## Execution questions
1. Hand-written Triton first, or lead with FlashAttention-varlen and treat Triton as
   the stretch?
2. Scrape current Triton/Blackwell docs up front, or only on a JIT failure?
