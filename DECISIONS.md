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
