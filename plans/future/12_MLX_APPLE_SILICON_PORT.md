Preferred model: Claude Opus 4.8 | Effort: high

# 12 — MLX / Apple Silicon Port (top priority)

> Serve the inferd engine on Apple Silicon via a Metal/MLX backend. A **separate
> track / distinct codebase**, not a change to the CUDA v1 runtime. Broadens the
> project beyond the single RTX 5090 to "runs on the laptop too."

## Constraints (this phase)
Apple Silicon (M-series, unified memory) · MLX / Metal · **isolated from `core/`
and `serve/` CUDA paths** — no edits that relax the v1 "no MLX / CUDA-native"
constraint for the existing engine (see `plans/future/00_FUTURE_ROADMAP.md` →
hard-constraint delta) · reuse the hardware-agnostic scheduler/serving surface
where it is genuinely portable · **new baselines only** — never mix Apple-silicon
numbers into the 5090 tables.

## Scope
**In:** an MLX `ModelRunner` backend behind the existing
`forward(tokens, kv) -> logits` seam; port the pure-Python, hardware-agnostic
surfaces first (`core/scheduler.py`, `serve/` request path); an Apple-silicon rung
of the harness reported on its own axis; a text-only Qwen load path in MLX.
**Out:** spec-decode / paged Triton kernels on Metal (later); fine-tuning on Apple
silicon; parity claims vs the 5090.
**Standalone value:** "the from-scratch engine's scheduler + serving layer run on
Apple Silicon through an MLX backend — same request contract, new hardware."

## Approach (staged)
1. **Prove the seam is portable.** The scheduler treats `kv` as opaque and the
   serving thread owns the scheduler (`serve/engine.py`) — stand these up against a
   trivial MLX backend stub to confirm nothing CUDA-specific leaked into them.
2. **MLX backend.** New `mlx/` (or `backends/mlx/`) module implementing the
   `ModelRunner` contract: load the text-only Qwen `language_model` backbone in MLX,
   `forward` returning logits, an MLX-native KV handle.
3. **Baseline rung.** Add an Apple-silicon engine to `bench/harness.py` behind a
   flag; record tokens/sec, TTFT, VRAM(unified)-vs-concurrency into its own
   `bench/results/` subtree. Do not touch the 5090 result files.

## Owned / Avoided files
- **Owns:** a new `backends/mlx/` (or `mlx/`) tree; new harness runner; new results
  subtree.
- **Avoids / does not modify semantics:** `core/model_runner.py` CUDA paths,
  `core/paged_*`, `core/qwen35_patch.py`, `serve/` CUDA assumptions. Extend the
  runner contract only additively (a backend selector), never rewrite the bf16/FP8
  CUDA load paths.

## Risks / Rollback / Exit
- **Risks:** Qwen3.5 hybrid linear-attention (`GatedDeltaNet`) may lack an MLX
  kernel → fall back to a full-attention Qwen for the Apple demo (ties to phase 14);
  MLX op coverage gaps; scope creep into a second engine.
- **Rollback:** ship the serving+scheduler port with a simple (non-spec, non-paged)
  MLX decode path — still a real "runs on Apple Silicon" result.
- **Exit:** a text-only Qwen served through the inferd scheduler on Apple Silicon,
  with an independent baseline rung; the CUDA engine untouched and still green.

## Model Selection (confirm or override)
- **Claude Opus 4.8 | high** *(recommended)* — porting/orchestration across a new
  hardware stack with careful isolation from v1.
- **GPT-5.5 | high** — acceptable for the MLX kernel/op-coverage spelunking.

## Execution questions
1. Which Qwen for the Apple demo — accept the hybrid-attention port risk, or lead
   with a full-attention model (shared with phase 14)?
2. Separate repo/submodule vs a `backends/mlx/` tree in this repo?
3. Minimum Apple-silicon target (unified-memory floor) to size the served model?
