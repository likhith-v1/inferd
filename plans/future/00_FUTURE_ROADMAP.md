Preferred model: Claude Opus 4.8 | Effort: high

# 00 — Future Roadmap (post-v0.1.0)

> Conducts the *next* build cycle for `inferd`. The v1 execution pack (phases
> 01–11) is complete and shipped as `v0.1.0` — those files now live in
> `plans/shipped/`. This folder holds the **not-yet-committed** follow-up work:
> directions, each tied to a real gap or measured finding from `DECISIONS.md`.

This file is the source of truth for cross-phase coordination of the future work,
the same role `plans/shipped/00_MASTER_ORCHESTRATION.md` plays for v1. Future
phase files (12+) stay lean and defer their shared rules here.

## Where v1 left off (the two facts everything hangs on)
- **Spec-decode's net-negative wall-clock is architectural, not a bug.** Qwen3.5
  is a *hybrid linear-attention* model; its `GatedDeltaNet` layers carry
  fixed-size recurrent state that can't be cropped, so every spec round pays a
  snapshot→restore→**replay** tax that cancels the parallel-verify win
  (`DECISIONS.md` 2026-06-27, Phase 04). α is healthy (~0.63–0.68); wall-clock is
  ~0.6–0.7×. The correctness proof (PASS @ n=1500) and the α-lift bridge are the
  v1 wins — not raw speed.
- **The paged cache is a correctness reference, not a runtime.** Phase 05 proved
  page-table round-trip is bit-exact (`max_abs=0`) but live decode still stores KV
  in HF's contiguous `DynamicCache`; the VRAM win is analytic, the block budget is
  admission *accounting*, and there is no fused Triton kernel yet.

## Priority order (as decided with the maintainer)
MLX/Apple Silicon reach is the maintainer's flagged **top priority**; the
engine-completion items follow in leverage order. Note the honest tension: by pure
integrity, phase 13 (persistent paged runtime cache) unblocks the most and would
lead a "make v1's claims fully true" ordering — MLX is a breadth bet on a *separate
codebase*, not a fix to the existing engine. Both framings are recorded so the call
is explicit, not accidental.

```
12 MLX / Apple Silicon port        (breadth — separate track, parallelizable)
13 persistent paged runtime cache  (integrity — unblocks 14, prefix-share, KV-quant)
14 full-attention spec-decode      (flips the biggest honest caveat net-positive)
15 Triton paged-attention kernel   (the deferred kernel credential; needs 13)
16 batched speculative decoding    (completeness; may stay net-neg on hybrid)
17 vLLM ceiling on Blackwell       (DONE 2026-07-17: within ~4.6× at c=32 on sm_120)
```
12 shares no v1 files, so it runs fully parallel to 13–17. 14 depends on nothing
structural (the crop-vs-replay branch already exists in `core/spec_decode.py`).
15 depends on 13 to matter in the live path. 17 is done (vLLM 0.23.0 runs on sm_120).

## Hard-constraint delta from v1 (read before phase 12)
v1's global constraints (`plans/shipped/00_MASTER_ORCHESTRATION.md`) include
**"no MLX"** and **CUDA-native**. Those still govern the **v1 engine**. Phase 12 is
an explicit, scoped **exception**: a *separate* Metal/MLX track (its own directory,
its own baselines), not a change to the CUDA runtime. It does not relax the
constraint for `core/` / `serve/` — if MLX were ever to enter mainline, that
requires first revising the v1 hard constraint in the shipped master, deliberately.
Every other v1 constraint (local-first, offline, text-only, exact spec-decode,
benchmark-before-optimization, vLLM as ceiling only, nothing faked) carries forward
unchanged.

## Shared-file contract (unchanged from v1)
`core/model_runner.py` remains the hot shared file; extend via new methods per the
contract in `plans/shipped/00_MASTER_ORCHESTRATION.md`. Phase 12 adds an MLX
`ModelRunner` backend *behind the same `forward(tokens, kv) -> logits` seam* — the
scheduler already treats `kv` as opaque, which is exactly the plug point. Phase 13
swaps the `kv` handle to a persistent paged `Cache`; callers stay layout-agnostic.

## Quality gates (every future phase inherits v1's gates)
1. Lint/format clean. 2. Phase tests pass. 3. **Numerical/distribution equivalence**
where applicable — reuse `bench/paged_equiv.py`, `bench/batched_equiv.py`,
`bench/correctness.py`, `tests/test_paged_equiv.py`; targets stay at the bf16 noise
floor and the accept rule stays exact. 4. **Benchmark reproducibility** — every perf
claim regenerates via `bench/run_all.py --plots` into `bench/results/` before it
enters the README. 5. `code-review` (+ `security-review` for any serving change).
6. **Honest framing** — net-negative results (e.g. batched spec on hybrid) are
reported as such in `DECISIONS.md`, per the standing rule.

## Backlog (not yet scoped into phase files)
Kept here so they aren't lost; promote to a numbered file when picked up.
- **MoE (Qwen3.6 35B-A3B) native multi-token prediction** as a self-speculation
  baseline vs the hand-rolled exact accept rule — the most novel research angle.
- **Prefix-sharing via copy-on-write KV blocks** (shared system-prompt prefixes) —
  depends on phase 13.
- **DPO/GRPO post-training** of the 27B showpiece — extends `finetune/`.
- **Chunked prefill + quantized KV-cache** — stretch concurrency; depends on 13.
- **Per-request sampling** — Phase 07 open item: sampling is server-level today.
- **Streaming / sharded 27B merge** (or `llm-compressor`) — a true standalone FP8
  artifact instead of load-time-FP8 + runtime LoRA.
- **FP8 latency watch-item** — re-measure when torchao ships a fused M=1 GEMM on
  sm_120; today FP8 is capacity-only.

## Execution questions for this cycle
1. Confirm MLX (12) leads over persistent paged cache (13), accepting that 13 is the
   higher-integrity "make v1 true" item?
2. Same worktree/`dev`→`main` flow as v1, or a separate long-lived branch for the
   MLX track since it's a distinct codebase?
3. Which items graduate from the backlog into numbered phase files now vs later?
