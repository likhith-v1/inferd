# Environment — validated 2026-06-26

Reproducible local CUDA stack for **inferd** on WSL2 Ubuntu, RTX 5090 (Blackwell sm_120). Validated by `scripts/smoke_load.py`.

## Host

| Item | Value |
|------|-------|
| OS | Ubuntu 26.04 LTS (WSL2) |
| Kernel | `6.18.33.1-microsoft-standard-WSL2` |
| Package manager | [uv](https://docs.astral.sh/uv/) `0.11.24` |
| Python | **3.13.14** (pinned in `.python-version`) |
| C compiler | `gcc` / `g++` (required by Triton kernel JIT) |

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
| torch | **2.11.0** (`+cu130`) | Built against CUDA 13.0; bundles CUDA runtime |
| triton | **3.6.0** | Needs `gcc` on host for kernel JIT |
| bitsandbytes | **0.49.2** | 4-bit linear verified PASS |
| transformers | **5.3.0** | |
| accelerate | **1.14.0** | |
| huggingface-hub | **1.21.0** | `hf` CLI for one-time weight pull |
| flash-linear-attention | **0.5.1** | Import as `fla`; fast path needs `causal-conv1d` (pinned via GitHub wheel in `pyproject.toml`) |
| causal-conv1d | **1.6.1** (`+cu13torch2.10`) | Prebuilt wheel; enables `fla` fast path on linear-attn layers |
| unsloth | **2026.3.11** | Call `bootstrap_finetune()` from `inferd.env` before importing `transformers` in finetune scripts |
| cuDNN (via PyTorch) | **9.1.9** (`91900`) | |

**Note on bitsandbytes:** `libnvJitLink.so.13` is bundled inside the venv (`nvidia/cu13/lib/`) but is not on `LD_LIBRARY_PATH` by default. `inferd.env` pre-loads it via `ctypes` (with error handling) before GPU imports — import it at the top of any entry point:

```python
import inferd.env  # noqa: F401
```

Reproduce the env on a clean clone:

```bash
uv sync
sudo apt-get install -y gcc g++   # Triton JIT requirement
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

## Smoke test results (2026-06-26, verified)

```bash
uv run python scripts/smoke_load.py
```

| Check | Result |
|-------|--------|
| Capability `(12, 0)` | **PASS** |
| bitsandbytes 4-bit linear | **PASS** |
| Triton kernel JIT | **PASS** |
| flash-linear-attention (`fla`) import | **PASS** |
| `causal-conv1d` fast path | **PASS** (when wheel installed) |
| Load Qwen3.5-9B (text backbone, vision stripped) | **PASS** |
| Single forward pass | **PASS** (`"The capital of France is"` → `" Paris"`) |

Observed at load (bf16, `device_map=cuda:0`, vision tower stripped):

- Backbone: `model.model.language_model` + `lm_head` (vision tower `model.model.visual` deleted)
- Params: **8.95B**
- VRAM allocated: **16.68 GB**
- Dtype: `torch.bfloat16`
- Logits shape: `[1, 5, 248320]`

**First run only:** Triton compiles its C driver on first invocation (gcc warnings about `_POSIX_C_SOURCE` redefinition are harmless). Kernels are cached; subsequent runs are silent.

## Deferred (non-blocking)

| Item | Status | When to revisit |
|------|--------|-----------------|
| `flash-attn` (FA2) | No pre-built sm_120 wheel; needs `nvcc` to build | Phase-03 (QLoRA) |
| Unsloth Flash Attention 2 | Broken on Blackwell; xformers fallback in use | Phase-03 |

## Quick reference

```bash
# Sync env (also install gcc if on a fresh clone)
uv sync && sudo apt-get install -y gcc g++

# GPU check
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"

# Smoke test (all checks)
uv run python scripts/smoke_load.py
```
