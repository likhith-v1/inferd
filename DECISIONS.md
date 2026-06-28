# DECISIONS

## 2026-06-26 — Phase 03 Corpus

- **Decision:** Use `TokenBender/code_instructions_122k_alpaca_style` as the Phase 03 SFT corpus.
- **Revision:** `19b59da67914b5fb2e0a5dff937e9917c0cfb7e4`.
- **Local raw path:** `data/raw/TokenBender_code_instructions_122k_alpaca_style/code_instructions_120k.json`.
- **Observed schema:** `instruction`, `input`, `output`, `text`.
- **Observed rows:** `121,959`.
- **Rationale:** Apache-2.0 code-instruction dataset, large enough to subsample a stable 20k portfolio SFT set, simple Alpaca-style fields, and aligned with the project’s coding/systems evaluation story.

## 2026-06-26 — Phase 03 Training Order

- **Decision:** Train and validate `Qwen/Qwen3.5-9B` first, then attempt `Qwen/Qwen3.6-27B` with the same dataset and pipeline.
- **Rationale:** The 9B target is the Phase 04 handoff for speculative decoding. Proving the full data/train/eval/export loop on 9B reduces risk before spending GPU time on the 27B showpiece.
- **Fallback:** If 27B QLoRA stalls on Blackwell/VRAM/tooling, keep the 9B result as the completed engine-target deliverable and document 27B as a follow-up attempt.

## 2026-06-26 — Phase 03 Trainer Stack

- **Decision:** Implement Unsloth-first QLoRA with TRL `SFTTrainer`, PEFT LoRA adapters, NF4 4-bit base loading, bf16 compute, gradient checkpointing, and paged 8-bit AdamW.
- **Rationale:** Matches the project constraints: maintained QLoRA tooling, local-first, no GemForge, and single-GPU feasibility.
- **Known constraint:** Unsloth import requires a visible GPU accelerator, so no-GPU CI only runs script selfchecks.

## 2026-06-26 — Phase 03 Evaluation

- **Decision:** Seed a 30-prompt coding/systems golden set with deterministic `expected_contains` checks and JSONL generation output for human review.
- **Rationale:** This gives a local-only evaluation gate immediately. It is not a replacement for human pairwise review, but it catches gross regressions and produces review artifacts without cloud/API judging.

## 2026-06-27 — Phase 03 9B Training Result

- **Model:** `Qwen/Qwen3.5-9B`, loaded from `weights/Qwen3.5-9B`.
- **Adapter path:** `adapters/9b`.
- **Config:** `finetune/configs/qwen3_5_9b.toml`.
- **Training:** 20,000 examples, 1 epoch, 1,250 steps, batch size 1, gradient accumulation 16, effective batch size 16.
- **Trainable parameters:** 29,097,984 of 9,438,911,728 (0.31%).
- **Final train loss:** `0.5261`.
- **Final eval loss:** `0.5148`.
- **Runtime:** `4,904s`.
- **Golden eval:** `30/30` deterministic expected-contains checks passed; output written to `runs/golden_eval.jsonl`.
- **Notes:** Flash Attention 2 was unavailable; Unsloth used fallback kernels. Triton `_POSIX_C_SOURCE` and bitsandbytes future warnings were non-blocking.

## 2026-06-27 — Phase 03 27B Merge: Deferred (OOM, skip for now)

- **Decision:** Skip the 27B adapter merge for Phase 03. Phase 04–09 will load the 27B as base-in-4bit + PEFT adapter directly. Merged weights are not required until Phase 10 FP8 quantization.
- **Root cause of OOM:** Merging 27B bf16 requires ~54GB to materialize the output. Both VRAM (32GB) and system RAM were insufficient. The `export.py` `--device-map cpu` flag added here is still needed but does not solve the RAM constraint.
- **Phase 10 action required:** Design a streaming/sharded merge or use a quantize-from-PEFT path (e.g., `llm-compressor`) that avoids materializing full bf16 weights. Decide at Phase 10 implementation.
- **9B merge:** Completed successfully with `device_map="cuda:0"` (18GB fits in 32GB VRAM). No change needed for 9B.

## 2026-06-27 — Phase 03 27B Training Result

- **Model:** `Qwen/Qwen3.6-27B`, loaded from `weights/Qwen3.6-27B`.
- **Adapter path:** `adapters/27b`.
- **Config:** `finetune/configs/qwen3_6_27b.toml`.
- **Training:** 20,000 examples, 1 epoch, 625 steps, batch size 1, gradient accumulation 32, effective batch size 32.
- **Trainable parameters:** 79,691,776 of 27,436,420,336 (0.29%).
- **Final train loss:** `0.5024`.
- **Best observed eval loss:** `0.4972` at epoch `0.8`; final checkpoint saved at step 625.
- **Runtime:** `9,423s`.
- **Notes:** 27B QLoRA fit on the single RTX 5090. Flash Attention 2 was unavailable; Unsloth used fallback kernels. Triton `_POSIX_C_SOURCE`, bitsandbytes future warnings, and Unsloth `num_items_in_batch` warnings were non-blocking.

## 2026-06-27 — Distro crash recovery (Phase 03 → 04 handoff rebuilt)

- **Context:** The WSL2 distro was deleted and recreated after a system crash mid-Phase-03. All `.gitignore`'d artifacts were lost: `weights/`, `adapters/`, `merged/`, `data/`, `.venv/`, and `uv` itself. Nothing had been pushed to the HF Hub (`likhith-v1` has no repos). Only the git-tracked source + `DECISIONS.md` survived.
- **Rebuild:** Reinstalled `uv` 0.11.25 → `uv sync` from the committed lockfile. Result: torch `2.11.0+cu130`, RTX 5090 capability `(12,0)` / sm_120 confirmed.
- **Missing prereq caught:** the fresh distro had no C compiler; Triton kernel JIT failed with "Failed to find C compiler". Fixed with `sudo apt-get install -y gcc g++` (already documented in `docs/ENVIRONMENT.md`). gcc 15.2.0.
- **Re-pull:** `Qwen/Qwen3.5-9B` (target) + `Qwen/Qwen3.5-0.8B` (draft) into `weights/`; dataset re-pulled at the pinned revision. **27B intentionally skipped** — not needed until Phase 10.
- **Re-train parity (deterministic, seed=0):** 9B QLoRA reproduced the original within noise — train_loss `0.5262` (was `0.5261`), eval_loss `0.5148` (identical), runtime `4,822s` (was `4,904s`). Merge → `merged/9b` (17G, vision tower stripped: `visual`). Golden eval `30/30` (pass_rate `1.000`).
- **State:** Phase-04 prerequisites fully restored. `merged/9b` (target) and `weights/Qwen3.5-0.8B` (draft) are ready.

## 2026-06-27 — Phase 04 finding: Qwen3.5 is a hybrid linear-attention model

- **Discovery:** `Qwen3.5` (9B target and 0.8B draft) is **hybrid**: layers alternate `full_attention` (standard croppable KV) and `linear_attention` `GatedDeltaNet` (fixed-size recurrent `conv_states`/`recurrent_states` that summarize the whole prefix and **cannot be length-cropped or rolled back**). The cache class is `Qwen3_5DynamicCache(Qwen3NextDynamicCache)` — no `.crop`.
- **Impact on speculative decoding (two problems):**
  1. *Rollback:* rejected draft tokens can't be undone by cropping the linear state. **Resolution:** snapshot the fixed-size linear states (O(1) in seq len), crop attention KV by slicing, then restore + replay the committed tokens each round (`core/spec_decode.py`).
  2. *Parallel verify:* the stock `Qwen3_5GatedDeltaNet.forward` only continues the recurrent state when `seq_len == 1`; any multi-token forward from a populated cache silently restarts the recurrence (`initial_state=None`, conv rebuilt). Measured: multi-token-from-cache vs full-prefill ground truth = max|Δlogit| 18.0, argmax match 0.67 (BROKEN); token-by-token = 0.22 (bf16 noise), argmax 1.00.
- **Decision:** patch `Qwen3_5GatedDeltaNet.forward` (`core/qwen35_patch.py`, installed by `core/model_runner.py`) to add a "continuation, seq_len>1" branch. Conv state continues via `torch_causal_conv1d_update` (cheap); the recurrence uses the **CUDA `chunk_gated_delta_rule` with `initial_state=recurrent_state`** (the kernel already supports `initial_state`; HF just hardcodes `None`). Single-step decode and prefill keep the stock fast paths. **Post-patch: multi-token-from-cache matches ground truth (max|Δ| 0.22 = bf16 noise, argmax 1.00).**
- **Phases 05/06/10 must account for the hybrid cache** — paged cache + batching can't assume croppable, per-position KV for the linear layers.

## 2026-06-27 — Phase 04 result: statistical correctness gate; speedup net-negative (honest)

- **Correctness gate (the differentiator):** `bench/correctness.py` now defaults to a multi-token sequence-mode statistical gate that exercises `ps[k>0]` and replay state, with the original first-token TV/χ² test still available via `--mode first`. The recorded run so far is the first-token gate at n=2000, 4 prompts — spec-decode next-token distribution was within the bootstrapped direct-vs-direct null envelope (prompt[0] TV=0.0190 ≤ null_p99=0.0315, χ²_p=0.100; others near-deterministic). Treat this as statistical evidence for exactness, not a formal proof by itself.
- **α / speedup (stock 0.8B draft, CANONICAL, merged/9b, 6 prompts, max_tokens 128):** baseline 45 tok/s. γ=2 α=0.69 → 0.47×; γ=4 α=0.62 → 0.55×; γ=8 α=0.41 → 0.46×. **Net slower than baseline.**
- **Root cause (architectural, not a bug):** the hybrid linear-attention cache can't be cropped, so each round needs snapshot→restore→**replay** of committed tokens to undo speculation. The replay is a second target pass/round that roughly cancels the parallel-verify savings; with a fast 9B baseline (~22 ms/tok) the per-round machinery (replay + draft cost + Python nucleus/accept with CPU↔GPU syncs) exceeds the tokens saved. Even at zero Python overhead the **replay tax** alone ≈ break-even. Fallback 2B/4B drafts would raise α but also draft cost → won't rescue wall-clock.
- **Finding:** speculative decoding's latency benefit is neutralized on hybrid linear-attention models by the rollback replay tax. This is the headline honest tension for the writeup (09/11). The method plus correctness gate is the contribution; α is healthy, the bottleneck is the architecture.

## 2026-06-27 — Phase 04 α-lift (draft distillation) result

- **Method:** sequence-level KD — sampled 10k completions from `merged/9b` (CANONICAL), trained the 0.8B draft as a LoRA SFT on them (`finetune/distill_draft.py` → `adapters/draft-distilled`; eval_loss 0.452, 1 epoch, ~13 min).
- **Δα (stock → distilled, target fixed, 6 prompts):** γ=2 0.691→0.755 (+0.064); γ=4 0.621→0.603 (−0.018, within noise); γ=8 0.415→0.492 (+0.077).
- **Read:** a modest, noisy lift at this budget (clearest at γ=8). Throughput stays ~0.5× baseline — the replay tax, not α, is the binding constraint, so the α-lift doesn't flip the speedup sign. A larger lift would need more KD data/epochs; not worth it while the replay tax dominates on this architecture.
- **Note:** α numbers measured on 6 prompts (noisy); the correctness gate (n=2000) is the rigorous result. The α-lift *method* is demonstrated; the absolute gain is small.

## 2026-06-27 — Phase 04 design parameters (locked)

- **Sampling profile:** reuse the frozen `bench.workload.CANONICAL` (temp=0.7, top_p=0.95, seed=0).
- **Correctness gate:** multi-token per-position TV vs a bootstrapped direct-vs-direct null envelope by default; first-token TV + χ² remains available for faster checks. No magic threshold.
- **α-lift budget:** moderate — ~10k completions sampled from `merged/9b`, train 0.8B ~1 epoch.
- **Fallback drafts:** authorized (Qwen3.5-2B/4B) if 0.8B α nets no speedup.

## 2026-06-27 — Phase 05 start: page-table contract first

- **Decision:** Start paged KV with an isolated `PagedKVCache` allocator/page-table, `PagedHybridCache` Qwen3.5 cache adapter, and reference `paged_attention` gather path before replacing Qwen attention internals.
- **Rationale:** Qwen3.5 is hybrid: only `full_attention` layers have per-position KV pages, while `linear_attention` layers carry fixed recurrent/conv state. The page-table contract and hybrid cache round-trip must be correct before model-runner integration or phase-06 scheduler budgeting.
- **Current validation:** `core.paged_cache --selfcheck`, `core.paged_attn --selfcheck`, `bench.paged_equiv --selfcheck`, and `tests/test_paged_equiv.py` pass on CPU, including block-boundary lengths `{1,15,16,17,31,32,33}` and synthetic Qwen hybrid cache round-trip.
- **Benchmark status:** `bench.harness --engine paged` is a cache-level microbenchmark that reports formula-based KV MB and fragmentation. It does **not** yet claim full paged runtime speedup.
- **Model-level gate:** `UV_PROJECT_ENVIRONMENT=/home/likhi/inferd/.venv uv run python -m bench.paged_equiv --target /home/likhi/inferd/merged/9b` PASS on 3 prompts with block size 16: `max_abs=0`, `max_rel=0` for all prompts. This validates lossless Qwen3.5 hybrid-cache page-table round-trip against real next-token logits.

## 2026-06-27 — Phase 05 review fixes (post-Codex review)

Addressed findings from a code review of the initial Phase-05 implementation:
- **Equivalence test was tautological → independent reference.** The synthetic gate compared `paged_attention` to `dense_attention`, but the former *calls* the latter. Added `sdpa_reference` (torch SDPA) in `core/paged_attn.py` and rewrote `tests/test_paged_equiv.py` to validate paged gather-and-attend against SDPA across page boundaries `{1,15,16,17,31,32,33}` and GQA/MHA head ratios.
- **Storage-only equivalence → model-level COMPUTE equivalence.** Added `core/paged_attn_interface.py` + `bench.paged_equiv --mode compute`: routes the real 9B's full-attention **decode** step through the paged path (store K/V into `PagedKVCache`, gather, attend) and compares next-token logits to native attention. Result: **`max_abs=0`** across prompts (decode attention computed from pages is bit-identical to the model's eager attention). Routing is done by patching the module `eager_attention_forward` global (not a custom `_attn_implementation` name), because HF couples attention-mask preparation to the impl name — a custom name gets sdpa-style (mask=None) prep that breaks causal prefill. `dense_attention` was aligned to eager's precision recipe (Q·K in working dtype, softmax fp32) so the match is exact.
- **VRAM win measured, not analytic.** `bench/runners/paged.py` now allocates paged blocks vs naive contiguous prealloc on GPU and reports `torch.cuda.max_memory_allocated`. Measured ratios match analytic: paged uses ~12.5%–31% of naive prealloc (c=1→0.125, c=16→0.31) on the Qwen3.5-9B full-attention shape (36 layers, 8 kv heads, head_dim 128, bf16).
- **dtype + safety:** paged cache/runner default to **bf16** (match the model); `dense_attention` documented/guarded as single-query decode-only (no causal mask).
- **Tests:** run via `python -m unittest tests.test_paged_equiv` (pytest is not a project dependency; the plan's `uv run python -m pytest` command should be updated or pytest added).

**Still open (honest):** (1) no Triton paged-attention kernel and no FlashAttention-varlen fallback — the paged compute path is a correctness reference (gather + dense/SDPA), so there is no kernel-level perf win yet; (2) no *runtime* paged cache: generation still stores KV in HF's contiguous `DynamicCache` — the compute gate proves correctness by routing one decode step through pages, but the model does not yet persist KV in pages across steps (that needs a paged `Cache` subclass replacing `DynamicCache`, handling the hybrid linear states). These two are the remaining Phase-05 work before the VRAM win is realized end-to-end.

## 2026-06-28 — Phase 06 scheduler scope with runtime KV gap

- **Decision:** Implement Phase 06 as a scheduler-first continuous batching layer: FCFS whole-prompt admission, prompt+max-token block reservation, finished-request eviction, live metrics, and a headless `batched` benchmark runner.
- **Important limitation:** The scheduler enforces the paged-cache free-block budget as accounting, but actual generation still uses `ModelRunner.forward()` with HF-backed opaque caches. Phase 05 has not yet landed a persistent paged runtime `Cache` subclass for Qwen3.5's hybrid full-attention + linear-attention state.
- **Spec interaction:** Batched speculative decoding is not implemented in Phase 06. Speculative decoding remains measured separately from batching, with the writeup reporting the expected benefit fade and the existing Qwen3.5 replay tax from Phase 04.
- **Rationale:** This gives Phase 07 a real request scheduler and metrics surface now, without hiding the remaining paged-runtime-cache work or taking on batched accept/replay complexity prematurely.

## 2026-06-28 — Phase 06 review fixes: real batched execution (the throughput win)

A review (code-review of Codex's first pass) found the headline gap: the initial
scheduler decoded **one sequence at a time** (a per-`req` `backend.decode` loop),
so it was an admission/eviction *simulator* with **no batched compute** — the
throughput-vs-concurrency win the phase exists to show could not appear (each
sequence paid its own full forward; the curve would be flat). Fixed:

- **Real batched decode.** `core/batched_cache.py` (`stack_caches` / `split_caches`)
  stacks the running sequences' per-seq caches into one batched cache each step:
  full-attention K/V is **left-padded** to a common length and `cat` on the batch
  dim; the hybrid **linear-attention conv/recurrent states are length-independent**,
  so they `cat` directly. `ContinuousBatchScheduler.step()` now samples all running
  sequences, then runs the still-active set through **one** `decode_batch` forward
  (left-padded attention mask + per-row `position_ids` for correct RoPE), then
  splits the cache back. `ModelRunner.forward` gained optional `position_ids` /
  `cache_position` (contract-compatible extend).
- **Why this is the win:** decode at 9B is bandwidth-bound on weight loads; batching
  amortizes one weight sweep across N sequences. Measured (greedy, 9B):
  **34.9 → 123 → 199 tok/s at concurrency 1 / 4 / 8 (5.7× at 8)**.
- **Exactness proven:** `bench/batched_equiv.py` teacher-forces ragged prompts and
  compares batched-vs-serial **logits**: `max|Δlogit| ≈ 0.49`, at the bf16 floor
  (cf. eager-vs-sdpa ~0.12; a real positional/masking bug showed |Δ|=18). The lone
  greedy argmax flip was a confirmed **bf16 near-tie** (top-2 margin 0.125 < noise
  0.156) — same class as the Phase-04 finding, not a bug. `tests/test_batched_cache.py`
  covers the stack/split surgery on CPU.
- **Continuous vs naive static batching:** added a `continuous=False` static-cohort
  policy + `--static-baseline`/`--vary-lengths`. With uniform lengths the two are
  ~parity (nothing to backfill); with length variance continuous wins
  (**1.13× at c=8**), and that's a *lower* bound because the static run is measured
  second on a warm cache (back-to-back runs have a warmup-order bias — noted; it's
  why c=4 reads 0.88×). The continuous advantage is workload-dependent, as expected.

**Still open (honest):** (1) block budget is still admission *accounting* over the
Phase-05 paged free-block model — real KV lives in HF caches, so it bounds
concurrency, not VRAM precisely; (2) no persistent paged runtime cache (the
Phase-05 follow-up); the per-step stack/split is cheap vs a 9B weight sweep but is
not a paged kernel; (3) batched speculative decoding still not implemented; (4) the
continuous-vs-static numbers need a warmup pass to remove the back-to-back ordering
bias before being quoted as headline figures.

## 2026-06-28 — Phase 07 FastAPI serving layer

- **Decision:** Wrap the engine in an async FastAPI service (`serve/`) without making
  the core depend on HTTP. The scheduler is synchronous/step-driven with no streaming
  hooks, so serving runs a **single background "engine thread"** (`serve/engine.py`)
  that exclusively owns the `ContinuousBatchScheduler`: HTTP `/generate` handlers
  tokenize + post a Submit command to a thread-safe inbox and `await` token chunks off
  a per-request `StreamChannel` (asyncio.Queue); the engine thread drains the inbox,
  calls `step()`, and pushes new token text via `loop.call_soon_threadsafe`. No locks
  on scheduler state (single owner thread); one tiny lock guards the in-flight counter.
- **Locked choices:** backpressure = **bounded queue + HTTP 429** (cap = max_concurrent
  + max_queue_depth, enforced by an atomic in-flight counter); disconnect cleanup = a
  new public **`scheduler.cancel(request_id)`** (the one core touch — drops a waiting req
  or evicts a running one and frees its blocks; unit-tested) called from the SSE
  generator's `finally`; streaming = **hand-rolled `StreamingResponse`** with `data:`
  SSE framing (no extra dep); oversized requests rejected **synchronously with 400**
  (pre-checked against `max_model_len`/`max_blocks` before submit).
- **Detok:** cumulative-decode-and-diff (`decode(all_ids)[prev_len:]`) — robust to BPE
  merges / leading-space artefacts that per-token decode breaks.
- **`/metrics` contract (frozen for 08):** `SchedulerMetrics.as_dict()` + server fields
  `tokens_per_second` (rolling 5s window), `last_ttft_s`, `peak_vram_mb`, `uptime_s`,
  `model` — see `serve/schemas.py:MetricsResponse`.
- **Verified:** 22 unit tests green (no GPU; `create_app(engine=...)` injects a
  FakeEngine to cover stream/429/400/disconnect→cancel/metrics-shape); live 9B smoke
  streamed a coherent completion incrementally, `/metrics` showed blocks freed after
  completion (used_blocks→0) at ~45 tok/s / TTFT ~1.7s; headless `bench.harness
  --engine batched` still runs unchanged.
- **Still open (honest):** (1) **sampling is server-level**, not per-request — the
  scheduler samples with one shared temperature/top_p; per-request sampling needs
  per-request params in `_sample_next` (documented follow-up; `GenerateRequest` only
  takes prompt + max_tokens in v1). (2) No auth / multi-tenant / HTTPS (local
  single-user, per scope). (3) Single served model at a time. (4) `security-review` and
  `code-review` gates on the request-handling/threading path not yet run.

## 2026-06-28 — Phase 09: benchmarks, correctness & report (one command)

`bench/run_all.py` is the one-command aggregator: three-rung throughput sweep,
spec-decode γ-sweep (stock + distilled draft), the correctness gate, and the
plots + `bench/report.md` — all regenerable from committed `bench/results/`.

**The three-rung table must compare the *identical* workload.** First pass put the
naive-HF rung (exactly `c` requests × max_tokens) next to the Phase-06 batched
runner in *fixed-pool-of-32 + vary_lengths* mode (constant 2063-token drain) and
produced a nonsense "0.66× / naive is faster" headline — apples-to-oranges. Fix:
`run_all` runs **ours in matched mode** — per-c calls with `total_requests == c`,
no length variance — so both rungs do like-for-like work at every batch width. The
headline now compares **per-concurrency** (not peak-vs-peak).

**Measured (greedy, 9B, max_tokens=96, RTX 5090):**

| c | naive HF tok/s | ours tok/s | ratio |
|---|---|---|---|
| 1 | 29.8 | 44.3 | 1.49× |
| 8 | 116.0 | 236.8 | 2.04× |
| 16 | 24.4 | 356.8 | 14.6× |
| 32 | 23.3 | 461.8 | **19.8×** |

Ours wins at **every** concurrency; the gap widens with load — naive HF has no KV
cache and collapses past c=8 on quadratic recompute, while continuous batching
scales. VRAM scales sanely (19→24 GB).

**Correctness gate: PASS @ n=1500** (multi-token per-position TV test, 3 prompts ×
6 positions, all within the bootstrapped direct-vs-direct null 99th pctile). This
is the differentiator: spec-decode output is statistically indistinguishable from
direct target sampling → the rejection-sampling accept rule + residual resampling
are exact.

**Spec-decode is net-negative on throughput here (honest).** α is healthy (stock
0.31–0.63, distilled 0.36–0.68) and **draft distillation lifts α** (Δα up to
+0.056, mean +0.048), but end-to-end speedup is ~0.6–0.7× the target-only baseline:
draft+verify overhead exceeds the acceptance gain at 9B/0.8B on this hardware. The
*correctness proof* and the *α-lift bridge* are the wins, not the wall-clock.

**vLLM ceiling: deferred** — won't build on Blackwell (sm_120); reported as
"ceiling pending," never faked (per the phase's rollback rule).

**Still open:** (1) vLLM ceiling pending on sm_120; (2) the matched-mode "ours"
curve leans on naive HF's no-KV-cache collapse at high c for the 19.8× headline —
the more conservative, defensible numbers are the 1.5–2.0× at c≤8; (3) spec-decode
throughput would need a faster draft path / batched accept to go net-positive.
