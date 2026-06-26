# Environment — phase 01 (validated 2026-06-26)

Reproducible local CUDA stack for **inferd** on WSL2 Ubuntu, RTX 5090 (Blackwell sm_120). Validated by `scripts/smoke_load.py` (online + `HF_HUB_OFFLINE=1`).

## Host

| Item | Value |
|------|-------|
| OS | Ubuntu 26.04 LTS (WSL2) |
| Kernel | `6.18.33.1-microsoft-standard-WSL2` |
| Package manager | [uv](https://docs.astral.sh/uv/) `0.11.24` |
| Python | **3.13.14** (`.python-version`) |

## GPU / driver

| Item | Value |
|------|-------|
| GPU | **NVIDIA GeForce RTX 5090** (32 GB VRAM) |
| Compute capability | **(12, 0)** — Blackwell sm_120 |
| Driver (KMD) | **610.62** |
| NVIDIA-SMI | `610.43.02` |
| CUDA UMD (driver report) | **13.3** |

Verify:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
# NVIDIA GeForce RTX 5090 (12, 0)
```

## Python stack (pinned in `uv.lock`)

| Package | Locked version | Notes |
|---------|----------------|-------|
| torch | **2.11.0** (`+cu130`) | Built against CUDA **13.0** |
| triton | **3.6.0** | |
| bitsandbytes | **0.49.2** | Installed; 4-bit linear not yet smoke-tested |
| transformers | **5.3.0** | |
| accelerate | **1.14.0** | |
| huggingface-hub | **1.21.0** | `hf` CLI for one-time weight pull |
| unsloth | **2026.3.11** | Import before `transformers` when used for finetune |
| cuDNN (via PyTorch) | **9.1.9** (`91900`) | |

Reproduce the env on a clean clone:

```bash
uv sync
```

## Weights (local, gitignored)

| Model | Path | Size |
|-------|------|------|
| Qwen/Qwen3.5-9B | `./weights/Qwen3.5-9B` | ~19 GB |

One-time download (requires HF read token + accepted model license):

```bash
hf auth login
hf download Qwen/Qwen3.5-9B --local-dir ./weights/Qwen3.5-9B
```

After download, all inference/load tests must work with:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py
```

## Smoke test results (2026-06-26)

Command:

```bash
uv run python scripts/smoke_load.py
HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py
```

| Check | Result |
|-------|--------|
| Capability `(12, 0)` | **PASS** |
| Load Qwen3.5-9B from `./weights/` | **PASS** |
| Single forward pass | **PASS** (`"The capital of France is"` → `" Paris"`) |
| Offline (`HF_HUB_OFFLINE=1`) | **PASS** |

Observed at load (bf16, `device_map=cuda:0`, text backbone only — vision tower stripped):

- Params: **8.95B** (`model.model.language_model` + `lm_head`)
- VRAM allocated: **~16.7 GB**
- Dtype: `torch.bfloat16`
- Logits shape: `[1, 5, 248320]`

## Known gaps (non-blocking for phase 01)

1. **Flash linear attention** — `flash-linear-attention` and `causal-conv1d` are not installed; Qwen3.5 linear-attention layers fall back to PyTorch (slower, correct).
2. **Unsloth / FlashAttention** — Unsloth reports broken Flash Attention 2; uses xformers fallback. Fine for env validation; revisit before QLoRA finetune.
3. **bitsandbytes** — Package is pinned but a dedicated 4-bit linear sanity check is still TODO per phase-01 validation list.

## Quick reference

```bash
# Sync env
uv sync

# GPU check
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"

# Smoke test
uv run python scripts/smoke_load.py
HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py
```
