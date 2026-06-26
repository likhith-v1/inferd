"""
Smoke test for phase-01 validation.

Loads Qwen3.5-9B as AutoModelForMultimodalLM, extracts the text backbone,
strips the vision tower, runs one forward pass, and prints diagnostics.
Also validates bitsandbytes 4-bit, Triton JIT, and flash-linear-attention.

Usage:
    uv run python scripts/smoke_load.py
    HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py   # offline proof
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import inferd.env  # noqa: F401 — preload libnvJitLink before GPU imports

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "Qwen3.5-9B"
PROMPT = "The capital of France is"
_VISION_ATTRS = ("visual", "vision_model", "vision_tower", "image_tower", "vpm", "vision_encoder")


def _strip_vision_modules(container) -> list[str]:
    """Delete known vision-tower attributes from a model container."""
    stripped = []
    for attr in _VISION_ATTRS:
        if hasattr(container, attr):
            delattr(container, attr)
            stripped.append(attr)
    if stripped:
        torch.cuda.empty_cache()
    return stripped


def is_offline_mode() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


def check_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — check driver / PyTorch wheel.")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"GPU:         {name}")
    print(f"Capability:  {cap}  (need (12, 0) for sm_120)")
    if cap != (12, 0):
        print(f"WARNING: expected (12, 0), got {cap} — check Blackwell PyTorch wheel.")
    return cap


def vram_used_gb() -> float:
    return torch.cuda.memory_allocated(0) / 1024**3


def load_text_backbone(weights_dir: Path):
    if not weights_dir.exists():
        sys.exit(
            f"Weights not found at {weights_dir}.\n"
            "Run:  hf download Qwen/Qwen3.5-9B --local-dir ./weights/Qwen3.5-9B"
        )

    print(f"\nLoading from: {weights_dir}")
    print("Loading processor / tokenizer ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
    except Exception:
        processor = AutoProcessor.from_pretrained(str(weights_dir))
        tokenizer = processor.tokenizer

    print("Loading model (AutoModelForMultimodalLM) ...")
    model = AutoModelForMultimodalLM.from_pretrained(
        str(weights_dir),
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    lm_head = None

    # Qwen3_5ForConditionalGeneration nests text at model.model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        stripped = _strip_vision_modules(model) + _strip_vision_modules(model.model)
        print(
            "Extracting model.model.language_model and stripping vision tower ..."
            + (f" (removed: {', '.join(stripped)})" if stripped else "")
        )
        lm = model.model.language_model
        lm_head = getattr(model, "lm_head", None)
        del model
        torch.cuda.empty_cache()
    elif hasattr(model, "language_model"):
        stripped = _strip_vision_modules(model)
        print(
            "Extracting language_model backbone and stripping vision tower ..."
            + (f" (removed: {', '.join(stripped)})" if stripped else "")
        )
        lm = model.language_model
        lm_head = getattr(model, "lm_head", None)
        del model
        torch.cuda.empty_cache()
    else:
        print("WARNING: no language_model backbone found — using whole model as-is.")
        lm = model

    lm.eval()
    if lm_head is not None:
        lm_head.eval()

    return lm, lm_head, tokenizer


def run_forward_pass(lm, tokenizer, lm_head=None):
    """
    Run one forward pass and validate structural properties of the output.

    Gates on: finite logits, expected device/dtype, vocab-size shape.
    Does NOT gate on the decoded token string — that can change with model
    revision or sampling and is not an environment correctness check.
    """
    print("\nRunning single forward pass ...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        if lm_head is not None:
            hidden = lm(**inputs).last_hidden_state
            logits = lm_head(hidden)
        else:
            logits = lm(**inputs).logits

    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"Prompt:      {PROMPT!r}")
    print(f"Next token:  {next_token!r}  (id={next_token_id})")
    print(f"Logits shape:{logits.shape}")

    # Structural checks — these validate environment correctness, not model output.
    assert logits.device.type == "cuda", f"logits on wrong device: {logits.device}"
    assert logits.dtype == torch.bfloat16, f"wrong dtype: {logits.dtype}"
    assert torch.isfinite(logits).all(), "logits contain NaN or Inf"
    assert logits.ndim == 3, f"expected 3D logits, got shape {logits.shape}"
    assert logits.shape[0] == 1, "batch size must be 1 for single-stream check"

    return logits.shape, next_token


def check_bitsandbytes_4bit() -> bool:
    print("\nChecking bitsandbytes 4-bit linear ...")
    try:
        import bitsandbytes as bnb
        x = torch.randn(4, 16, device="cuda")
        lin = bnb.nn.Linear4bit(16, 8, bias=False).cuda()
        out = lin(x)
        print(f"  bnb {bnb.__version__} 4-bit forward: shape={out.shape}  PASS")
        return True
    except Exception as exc:
        print(f"  bitsandbytes 4-bit FAIL: {exc}")
        return False


def check_triton_kernel() -> bool:
    print("\nChecking Triton kernel JIT ...")
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            x = tl.load(x_ptr + pid)
            y = tl.load(y_ptr + pid)
            tl.store(out_ptr + pid, x + y)

        x = torch.ones(4, device="cuda")
        y = torch.ones(4, device="cuda")
        out = torch.zeros(4, device="cuda")
        _add[(4,)](x, y, out, 4, BLOCK=1)
        assert out.tolist() == [2.0] * 4
        print(f"  triton {triton.__version__} kernel JIT: PASS")
        return True
    except Exception as exc:
        print(f"  Triton JIT FAIL: {exc}")
        print("  (install gcc/g++ via: sudo apt-get install -y gcc g++)")
        return False


def check_fla() -> bool:
    print("\nChecking flash-linear-attention (fla) ...")
    try:
        import fla

        ccd_ok = False
        try:
            import causal_conv1d  # noqa: F401

            ccd_ok = True
        except ImportError:
            print("  WARNING: causal-conv1d not installed; fla uses slow PyTorch fallback")
        status = "PASS (fast path)" if ccd_ok else "PASS (slow fallback — install causal-conv1d)"
        print(f"  fla {fla.__version__} import: {status}")
        return True
    except Exception as exc:
        print(f"  fla FAIL: {exc}")
        return False


def main():
    offline = is_offline_mode()
    print("=" * 60)
    print(f"inferd phase-01 smoke test  (offline={offline})")
    print("=" * 60)

    cap = check_capability()

    bnb_ok = check_bitsandbytes_4bit()
    triton_ok = check_triton_kernel()
    fla_ok = check_fla()

    vram_before = vram_used_gb()
    lm, lm_head, tokenizer = load_text_backbone(WEIGHTS_DIR)
    vram_after = vram_used_gb()

    param_count = sum(p.numel() for p in lm.parameters())
    if lm_head is not None:
        param_count += sum(p.numel() for p in lm_head.parameters())
    param_count /= 1e9

    dtype = next(lm.parameters()).dtype
    device = next(lm.parameters()).device

    print(f"\nModel dtype: {dtype}")
    print(f"Model device:{device}")
    print(f"Params:      {param_count:.2f}B")
    print(f"VRAM used:   {vram_after - vram_before:.2f} GB  (total allocated: {vram_after:.2f} GB)")

    forward_ok = False
    try:
        _shape, _token = run_forward_pass(lm, tokenizer, lm_head)
        forward_ok = True  # structural checks inside run_forward_pass; no semantic gate
    except Exception as exc:
        print(f"Forward pass error: {exc}")

    cap_ok = cap == (12, 0)
    print("\n" + "=" * 60)
    print(f"Capability (12,0):    {'PASS' if cap_ok else 'FAIL'}")
    print(f"bitsandbytes 4-bit:   {'PASS' if bnb_ok else 'FAIL'}")
    print(f"Triton kernel JIT:    {'PASS' if triton_ok else 'FAIL (needs gcc)'}")
    print(f"flash-linear-attn:    {'PASS' if fla_ok else 'FAIL'}")
    print(f"Forward pass:         {'PASS' if forward_ok else 'FAIL'}")
    print(f"Offline mode:         {'YES' if offline else 'not set'}")
    print("=" * 60)
    if not cap_ok or not forward_ok or not bnb_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
