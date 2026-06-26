Preferred model: GPT-5.5 | Reasoning: high

# 03 — QLoRA Fine-Tuning (the showpiece)

> QLoRA-fine-tune the 27B showpiece and the 9B engine target on a parameterized corpus, prove a win-rate over base, and export merge-for-serving weights. Runs largely independent of the engine; it feeds adapters/merged weights downstream.

## Constraints (this phase)
QLoRA only (4-bit NF4 base + LoRA adapters) · **Unsloth first**, fallbacks Axolotl / Llama-Factory / ms-swift / TRL+PEFT · **no GemForge** · gradient checkpointing + paged 8-bit AdamW + bf16 compute · text path only (vision frozen/ignored) · single RTX 5090 · **parameterized corpus** (requirements below, not a fixed dataset) · offline after weight pull.

## Scope
**In:** SFT `Qwen/Qwen3.6-27B` (showpiece) and `Qwen/Qwen3.5-9B` (engine target) on the chosen corpus; golden-set eval (30–50 prompts, win-rate vs base, held-out); adapter export + merge-for-serving tested. **Defer the 0.8B draft** — it gets *distilled against the fine-tuned 9B* in Phase 04.
**Out:** the engine; the draft; DPO/GRPO (stretch).
**Standalone value:** "QLoRA fine-tuned a 27B + 9B on a single 5090 with a measured win-rate over base."

### Corpus (parameterized — decide at implementation)
Requirements, not a name: instruction/SFT format (prompt→response or chat turns); license permits local fine-tune; size ~5k–50k examples for a portfolio SFT; held-out split (~10%); a **30–50 prompt golden set** disjoint from train. Placeholder: `DATASET = <hf-dataset-id>`; lock the exact id + revision in `DECISIONS.md` once chosen.

## Subagent breakdown
- **trainer-stack-selector** — try Unsloth on Blackwell+Qwen3.5/3.6+4-bit+adapter export; fall back down the list; record which stack won + why.
- **sft-runner** — 27B + 9B SFT configs (seq len 2–4k, batch 1–2 + grad-accum), checkpointing, loss logging.
- **golden-eval-runner** — win-rate vs base on the golden set + held-out loss.
- **exporter** — adapter export + merge-for-serving; verify merged weights load via the engine's text-backbone path.

## Git worktree workflow
- Branch `phase-03-qlora`, worktree `../inferd-wt/03-qlora`. May run in parallel with 04–06 once 02 lands (no engine-internal files touched).

## Owned / Avoided files
- **Owns:** `finetune/train_qlora.py`, `finetune/eval_golden.py`, `finetune/export.py`, `finetune/configs/`, `finetune/golden_set.jsonl`.
- **Avoids:** `core/`, `serve/`, `bench/harness.py` (reuse, don't edit).

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run python finetune/train_qlora.py --model Qwen/Qwen3.5-9B  --dataset <id> --out adapters/9b
uv run python finetune/train_qlora.py --model Qwen/Qwen3.6-27B --dataset <id> --out adapters/27b
uv run python finetune/eval_golden.py --base Qwen/Qwen3.5-9B --adapter adapters/9b --golden finetune/golden_set.jsonl
uv run python finetune/export.py --base Qwen/Qwen3.5-9B --adapter adapters/9b --merge --out merged/9b
uv run python -c "from core.model_runner import load_target; load_target('merged/9b')"  # serving path loads
```
- **Validation:** training stays within 32GB (watch peak VRAM); golden-set **win-rate > 50% vs base**; held-out loss drops; merged weights load through the text-backbone path; `eval_golden.py` ships a small assert-based self-check on its scoring math.

## Risks / Rollback / Exit / Handoff
- **Risks:** Unsloth/bitsandbytes Blackwell breakage on 27B; 27B OOM at chosen seq len; merge-for-serving mismatch with the multimodal wrapper (vision keys).
- **Mitigation/Rollback:** fall back through the trainer list (record in `DECISIONS.md`); drop seq len / raise grad-accum / reduce LoRA rank for the 27B; if 27B tooling stalls, fine-tune `Qwen3.5-27B` or `Qwen3.5-9B` first with an identical pipeline, then swap back; on merge, strip vision keys and keep `language_model` only.
- **Exit:** fine-tuned 27B with measured win-rate; fine-tuned 9B ready for the engine; adapters exported; merge path tested.
- **Handoff:** merged 9B → 04 (target for spec-decode + the α-lift draft distillation); merged 27B → 10 (FP8 hero).

## Model Selection (confirm or override)
- **GPT-5.5 | high** *(recommended)* — strong at multi-stack tooling integration + getting fiddly 4-bit/LoRA/optimizer configs right the first time.
- **Claude Opus 4.8 | high** — equally strong; favored if the fallback-stack reasoning (which trainer, why) gets gnarly.
- **GPT-5.4 | medium** — viable if the chosen stack "just works" and it's mostly config.
> Recommendation: GPT-5.5 high; switch to Opus 4.8 high if you hit a Blackwell tooling wall. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** HF MCP + `hf-cli` (weights/dataset pull), `hf-mem` (pre-flight 27B QLoRA VRAM), `huggingface-datasets` (vet the corpus: configs/splits/sizes).
- **Candidates:** `huggingface-llm-trainer` / `trl-training` (reference TRL/Unsloth *patterns*, but run **local**, not HF Jobs cloud — cloud violates local-first), `huggingface-community-evals` (golden-set scaffolding), `huggingface-trackio` (loss curves).
- **Question:** which trainer stack to attempt first (Unsloth assumed)? Use Trackio for loss curves or plain logs? Confirm local-only (no HF Jobs).

## Execution questions for this phase
1. **Which corpus/domain?** (Parameterized now — needs an HF dataset id + revision to become executable.)
2. Fine-tune 27B and 9B on the *same* corpus, or different domains per model?
3. LoRA rank/alpha/target-modules and seq len for each model?
4. Golden-set judging: pairwise human, LLM-judge, or task metric? If LLM-judge, which model (local only)?
5. If 27B tooling stalls, proceed with the 9B-first fallback and swap later — acceptable?
