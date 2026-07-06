Preferred model: GPT-5.5 | Reasoning: very high

# 13 — Persistent Paged Runtime KV Cache

> Wire the phase-05 block allocator into live decode so KV actually persists in
> pages across steps — turning the analytic VRAM win into a measured one and the
> scheduler's block budget from accounting into real memory bounding.

## Constraints (this phase)
RTX 5090 · text-only backbone · block size 16 · must handle Qwen3.5's **hybrid**
layout (per-position pages for `full_attention`; fixed recurrent/conv state for
`linear_attention`) · numerically equivalent to the current HF-cache path (bf16
noise floor) · reuse the existing page-table contract and equivalence gates.

## Scope
**In:** a paged `Cache` subclass replacing `DynamicCache` in the live decode path,
persisting full-attention K/V in `PagedKVCache` blocks and carrying the hybrid
linear states across steps; scheduler block budget bound to *actual* allocations;
end-to-end VRAM-vs-concurrency measured.
**Out:** the Triton kernel (phase 15 — this phase can still gather+dense/SDPA);
batched spec (16); prefix-share/COW (backlog).
**Standalone value:** "the from-scratch paged KV-cache now backs live generation —
measured VRAM/seq below naive, not just analytically."

## Approach
- Reuse `core/paged_cache.py` (allocator/page-table, already bit-exact) and the
  `PagedHybridCache` adapter; the new work is making it a live `Cache` HF will
  write into and read from every step, not a one-shot round-trip.
- Route through the existing `bench/paged_equiv.py --mode compute` gate first
  (single decode step already proven `max_abs=0`), then extend to multi-step
  persistence.
- Update `core/scheduler.py` so the free-block budget reflects real page
  allocation, closing the Phase 06 "accounting, not VRAM" gap.

## Owned / Avoided files
- **Owns:** new paged `Cache` subclass (near `core/paged_cache.py` /
  `core/batched_cache.py`); extensions to the runner's cache wiring.
- **Shared (per contract):** `core/model_runner.py` (`kv` becomes the persistent
  paged handle), `core/scheduler.py` (budget). **Avoids:** `core/spec_decode.py`
  logic, dashboard.

## Commands / validation
```bash
uv run python -m core.paged_cache --selfcheck
uv run python -m bench.paged_equiv --target merged/9b        # extend to multi-step
uv run python -m bench.harness --engine batched --model merged/9b --concurrency 1,8,16 --report-vram
```
- **Gate:** live paged logits match the HF-cache path within bf16 tolerance across
  block boundaries {1,15,16,17,31,32,33}; measured VRAM/seq below naive prealloc;
  no block leaks under admit/evict churn.

## Risks / Rollback / Exit
- **Risks:** the hybrid linear state is the same reason spec-decode rollback was
  hard — persisting/advancing it inside a paged `Cache` is the crux; subtle
  off-by-one in page indexing → logit drift.
- **Rollback:** keep the compute-equivalence reference path; ship the
  full-attention-only paged runtime and document the linear-layer handling as the
  remaining work.
- **Exit:** paging backs live decode; VRAM win measured end-to-end; scheduler budget
  bounds real memory. **Handoff:** phase 15's Triton kernel plugs into this live
  path; prefix-share/KV-quant build on it.

## Model Selection
- **GPT-5.5 | very high** *(recommended)* — index math + hybrid-state correctness.
- **Claude Opus 4.8 | xhigh** — co-equal; pick on availability.

## Execution questions
1. Subclass HF's `Cache` directly, or a standalone paged cache the runner adapts?
2. Block size fixed at 16 or swept now that it's runtime, not reference?
3. Eviction policy when blocks are exhausted mid-decode — hard fail vs preempt?
