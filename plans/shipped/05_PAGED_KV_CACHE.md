Preferred model: GPT-5.5 | Reasoning: very high

# 05 — Paged KV-Cache (single stream)

> Replace the contiguous KV-cache with a block allocator + page table and a Triton paged-attention kernel, proven numerically equivalent to the contiguous path and measurably lighter on VRAM per sequence.

## Constraints (this phase)
RTX 5090 · text-only backbone · block size 16 · paged-attention via **Triton** (FlashAttention varlen as a **first-class fallback**, not an afterthought) · numerically equivalent to the contiguous path (logits within tolerance) · reuse `bench.harness`.

## Scope
**In:** block allocator + page table (Python); paged-attention gather-and-attend Triton kernel; numerical equivalence vs contiguous cache; VRAM-per-sequence below naive pre-allocation.
**Out:** batching/scheduling (06); prefix-sharing/COW (stretch); quantized KV (stretch).
**Standalone value:** "a from-scratch paged KV-cache with a Triton paged-attention kernel, numerically equivalent to baseline, with lower VRAM/seq."

## Subagent breakdown
- **allocator / page-table author** — `core/paged_cache.py`: free-block pool, per-seq page tables, alloc/free/append; invariants + asserts.
- **triton-kernel author** — `core/paged_attn.py`: gather-and-attend over paged K/V; FlashAttention-varlen fallback behind the same interface.
- **numerical-equivalence validator** — logits paged vs contiguous within tolerance across seq lengths/positions.

## Git worktree workflow
- Branch `phase-05-paged-cache`, worktree `../inferd-wt/05-paged`. Touches `core/model_runner.py` per contract (`kv` becomes the paged handle; callers stay layout-agnostic).

## Owned / Avoided files
- **Owns:** `core/paged_cache.py`, `core/paged_attn.py`, `tests/test_paged_equiv.py`.
- **Shared (per contract):** `core/model_runner.py` (swap `kv` handle to paged). **Avoids:** `core/scheduler.py` (06), `core/spec_decode.py` (don't rewrite 04's logic; it consumes `forward`).

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run python -m core.paged_cache --selfcheck        # allocator invariants (assert-based)
uv run python -m pytest tests/test_paged_equiv.py    # paged logits == contiguous within tol
uv run python -m bench.harness --engine paged --model merged/9b --concurrency 1 --report-vram
uv run python -m core.paged_attn --bench             # kernel vs flashattn-varlen fallback timing
```
- **Validation:** allocator self-check (no leaks/double-free, page-table consistency); paged logits match contiguous within tolerance (the gate); VRAM-per-sequence < naive contiguous pre-allocation; Triton kernel within a sane factor of the fallback (or use fallback and note it).

## Risks / Rollback / Exit / Handoff
- **Risks:** Triton on Blackwell sm_120 immature → kernel won't JIT or is slow; off-by-one in page indexing → subtle logit drift; tolerance set too loose hides bugs.
- **Mitigation/Rollback:** if Triton stalls, ship the **FlashAttention-varlen fallback** and keep the allocator/page-table win (still a real result); pick tolerance from the contiguous path's own fp noise floor, not an arbitrary number; fuzz page boundaries in the equivalence test.
- **Exit:** paged path numerically equivalent; VRAM/seq measured below naive.
- **Handoff:** the allocator + free-block accounting is what 06's scheduler budgets against (admit/evict under free blocks).

## Model Selection (confirm or override)
- **GPT-5.5 | very high** *(recommended)* — Triton kernel correctness + index math reward deep reasoning.
- **Claude Opus 4.8 | xhigh** — co-equal; pick on availability/cost.
- **GPT-5.5 | high** — acceptable if leaning on the FlashAttention-varlen fallback rather than a hand-written kernel.
> Recommendation: GPT-5.5 very-high or Opus 4.8 xhigh. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** firecrawl for current Triton docs (Blackwell quirks), `huggingface-papers` (PagedAttention / vLLM paper for the design).
- **Candidates:** HF MCP for FlashAttention-varlen reference usage; `code-review` gate on the kernel + indexing.
- **Question:** scrape Triton docs up front, or only on a JIT failure? Make the equivalence test a blocking gate (recommended)?

## Execution questions for this phase
1. Block size fixed at 16, or sweep {8,16,32}?
2. Equivalence tolerance — derive from fp noise floor; what's the threshold and across which positions/lengths?
3. If the Triton kernel underperforms the fallback, ship fallback for v1 and keep Triton as a documented attempt?
4. Max sequence length / max blocks per sequence to budget for?
