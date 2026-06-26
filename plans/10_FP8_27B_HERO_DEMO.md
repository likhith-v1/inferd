Preferred model: GPT-5.4 | Reasoning: high

# 10 — FP8 27B Hero Demo (optional capstone)

> Serve the *fine-tuned* 27B single-stream via FP8 on the 5090's FP8 hardware, through your own engine — the closing shot proving the stack scales to a flagship model. FP8 is scoped to exactly this phase, nowhere else.

## Constraints (this phase)
RTX 5090 FP8 hardware · FP8 is the **one** quantization exception, scoped here only · serve the **fine-tuned → merged 27B** (not the stock checkpoint) · text-only backbone · single-stream (~27GB, tight, no batching headroom) · official `Qwen/Qwen3.6-27B-FP8` is a **reference only**, not the artifact.

## Scope
**In:** merge the fine-tuned 27B LoRA → FP8 quantize → load via `model_runner` FP8 path → serve single-stream through the engine; a hero/demo script + capture.
**Out:** FP8 anywhere else in the engine; batching the 27B (no headroom); general quantization (GPTQ/AWQ — out of scope).
**Standalone value:** "the flagship fine-tune, FP8-served through a from-scratch engine."

## Subagent breakdown
- **fp8-quantizer** — merge 27B adapter, FP8 quantize, verify load + a sane single-stream forward; sanity-check output vs the bf16 fine-tune on a few prompts.
- **hero-serve runner** — single-stream serve through the engine; measure tokens/sec, TTFT, peak VRAM (confirm it fits ~27GB).
- **demo-capture author** — the closing-shot capture (live generation + VRAM headroom).

## Git worktree workflow
- Branch `phase-10-fp8`, worktree `../inferd-wt/10-fp8`. Needs 03 (merged 27B) + 04–07 (engine + serving). Touches `core/model_runner.py` FP8 path **per the master contract** (same `load_target()` signature, dtype flag).

## Owned / Avoided files
- **Owns:** `scripts/hero_fp8.py`, FP8 quant config.
- **Shared (per contract):** `core/model_runner.py` (FP8 load variant). **Avoids:** spec/paged/scheduler internals (consume them).

## Commands, tests, validation
```bash
uv sync
uv run python finetune/export.py --base Qwen/Qwen3.6-27B --adapter adapters/27b --merge --out merged/27b
uv run python scripts/hero_fp8.py --in merged/27b --quantize fp8 --out merged/27b-fp8
uv run python -m core.model_runner --load merged/27b-fp8 --dtype fp8 --selfcheck   # loads + 1 forward
uv run uvicorn serve.app:app   # then single-stream /generate against the 27B-FP8
uv run python -m bench.harness --engine fp8 --model merged/27b-fp8 --concurrency 1 --report-vram
```
- **Validation:** FP8 27B fits (~27GB, peak VRAM checked); single-stream generation coherent (spot-check vs bf16 fine-tune); tokens/sec + TTFT recorded; serves through the same engine path (not a special-case bypass).

## Risks / Rollback / Exit / Handoff
- **Risks:** FP8 quant quality loss on the fine-tune; ~27GB leaves no margin → OOM with KV; FP8 kernel/path immaturity on Blackwell.
- **Mitigation/Rollback:** spot-check FP8 vs bf16 outputs; cap context/KV to fit the tight budget; if our FP8 path is unstable, validate against the official `Qwen3.6-27B-FP8` to isolate quant vs engine; if it won't fit, fall back to the FP8 9B hero and note the 27B as stretch.
- **Exit:** fine-tuned 27B served single-stream via FP8 through the engine; demo captured.
- **Handoff:** the closing-shot capture → 11's demo/README.

## Model Selection (confirm or override)
- **GPT-5.4 | high** *(recommended)* — bounded quantization + serving-glue work; cost-efficient.
- **Claude Sonnet 4.6 | high** — co-equal for the plumbing.
- **GPT-5.5 | high / Opus 4.8 | high** — only if FP8 numerics/kernel debugging on Blackwell turns deep.
> Recommendation: GPT-5.4 high; escalate to a high-reasoning model only on FP8 kernel trouble. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** HF MCP (`Qwen/Qwen3.6-27B-FP8` as the reference), `hf-mem` (FP8 footprint pre-flight).
- **Candidates:** firecrawl for FP8 quant tooling docs (whichever quantizer is chosen).
- **Question:** which FP8 quantizer/tooling? Validate against the official FP8 checkpoint as a control?

## Execution questions for this phase
1. FP8 format/recipe (e2m1/e4m3, per-tensor vs per-channel) and which tool?
2. Is this an actual v1 deliverable or strictly the optional capstone (run only if time allows)?
3. If 27B-FP8 won't fit with usable context, accept the FP8-9B hero fallback?
4. Quality bar for "coherent" — eyeball spot-check or a mini golden-set pass?
