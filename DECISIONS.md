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
