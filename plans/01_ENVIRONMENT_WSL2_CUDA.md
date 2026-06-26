Preferred model: Claude Sonnet 4.6 | Effort: high

# 01 — Environment: WSL2 + CUDA (Blackwell sm_120)

> Stand up a reproducible, offline-capable CUDA stack on WSL2 Ubuntu for the RTX 5090, pin it in a `uv` lockfile, and prove it with a one-forward-pass smoke test. Everything downstream assumes this holds.

## Constraints (this phase)
local-first · WSL2 Ubuntu · RTX 5090 Blackwell **sm_120** · CUDA-native, **CUDA 12.8+** · no cloud/API inference · no MLX · no GemForge · `uv` lockfile pinned · weights downloaded **once** then air-gapped · text-only (we never load the vision tower).

## Scope
**In:** WSL2 Ubuntu + NVIDIA driver/CUDA 12.8+ for sm_120; Blackwell-supported PyTorch wheel; Triton + bitsandbytes + Unsloth Blackwell builds; `uv` project + lockfile; `hf` CLI one-time weight pull; smoke test loading the **Qwen3.5-9B text backbone** (strip vision) for a single forward pass.
**Out:** any engine/finetune/serve code; multi-GPU; Windows-native toolchain.
**Standalone value:** "a pinned, reproducible Blackwell CUDA env that loads a 9B and runs a forward pass offline."

## Subagent breakdown
- **toolchain-installer** — driver/CUDA/PyTorch/Triton/bitsandbytes/Unsloth; verifies `torch.cuda.get_device_capability() == (12, 0)`.
- **lockfile-pinner** — freezes the *working* set into `uv.lock` + `pyproject.toml` the moment it works; records CUDA/driver versions in env docs.
- **smoke-tester** — loads Qwen3.5-9B as `AutoModelForMultimodalLM`, extracts `language_model`, one forward pass, prints device/dtype/VRAM.

## Git worktree workflow
- Branch `phase-01-env`, worktree `../inferd-wt/01-env`.
- `git worktree add ../inferd-wt/01-env -b phase-01-env dev` → work → rebase on `dev` → merge → `git worktree remove`.

## Owned / Avoided files
- **Owns:** `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `docs/ENVIRONMENT.md`, `scripts/smoke_load.py`.
- **Avoids:** `core/`, `serve/`, `dashboard/`, `finetune/`, `bench/`.

## Commands, tests, validation
```bash
# inside WSL2 Ubuntu
nvidia-smi                       # driver sees the 5090
uv python install 3.13 && uv init
uv add "torch --index <blackwell-cu128-wheel-index>"  # pin the sm_120 wheel
uv add triton bitsandbytes unsloth transformers accelerate
uv run python -c "import torch;print(torch.cuda.get_device_capability())"  # (12,0)
hf download Qwen/Qwen3.5-9B --local-dir ./weights/Qwen3.5-9B   # one-time, then offline
uv run python scripts/smoke_load.py        # load text backbone, 1 forward pass
HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py           # prove offline
```
- **Validation:** capability `(12,0)`; Triton imports + a trivial kernel JITs; bitsandbytes 4-bit linear runs; smoke test offline OK; `uv.lock` regenerates the env on a clean clone.

## Risks / Rollback / Exit / Handoff
- **Risks:** bleeding-edge Blackwell wheels (PyTorch/Triton/bitsandbytes) mismatch or missing; CUDA/driver skew; Unsloth Blackwell support lagging.
- **Mitigation/Rollback:** keep last-known-good `uv.lock`; if a package breaks Blackwell, pin to the prior working commit and note it in `docs/ENVIRONMENT.md`; nightly wheels as a documented fallback.
- **Exit:** capability `(12,0)`, offline smoke pass, `uv.lock` committed, versions recorded.
- **Handoff:** every later phase runs `uv sync`; the env-version stamp is consumed by 02/09 for result provenance.

## Model Selection (confirm or override)
- **Claude Sonnet 4.6 | high** *(recommended)* — systematic, cost-efficient for procedural setup + doc capture.
- **Claude Opus 4.8 | high** — escalate here if Blackwell toolchain debugging gets deep (wheel ABI / CUDA mismatches).
- **GPT-5.4 | medium** — viable lighter alternative for scripted install.
> Recommendation: start Sonnet 4.6 high; escalate to Opus 4.8 only on a hard toolchain wall. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** Hugging Face MCP + `hf-cli` (weight download), `hf-mem` (pre-flight VRAM estimate for 9B/27B).
- **Candidates:** `huggingface-best` (sanity-check chosen checkpoints), firecrawl (Triton/Unsloth/bitsandbytes Blackwell install docs).
- **Question:** which to actually use? Recommend HF MCP + `hf-cli` + `hf-mem`; firecrawl only if install docs are unclear.

## Execution questions for this phase
1. Confirm CUDA 12.8 vs a newer 12.x; pin which exact PyTorch Blackwell wheel index?
2. Weights location: `./weights/` (gitignored) or a shared WSL2 path reused across worktrees?
3. Python 3.12 OK, or does any chosen trainer stack force a different minor?
4. Allow nightly wheels if stable releases lack sm_120, or strictly stable only?
