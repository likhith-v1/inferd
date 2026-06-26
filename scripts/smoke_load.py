"""
Smoke test for phase-01 validation.

Loads Qwen3.5-9B as AutoModelForMultimodalLM, extracts the text backbone,
strips the vision tower, runs one forward pass, and prints diagnostics.

Usage:
    uv run python scripts/smoke_load.py
    HF_HUB_OFFLINE=1 uv run python scripts/smoke_load.py   # offline proof
"""

import sys
import os
import torch
from pathlib import Path
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
        tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(str(weights_dir), trust_remote_code=True)
        tokenizer = processor.tokenizer

    print("Loading model (AutoModelForMultimodalLM) ...")
    model = AutoModelForMultimodalLM.from_pretrained(
        str(weights_dir),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
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
        lm_head = model.lm_head
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
    return logits.shape, next_token


def main():
    offline = is_offline_mode()
    print("=" * 60)
    print(f"inferd phase-01 smoke test  (offline={offline})")
    print("=" * 60)

    cap = check_capability()

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
    logits_shape = None
    try:
        logits_shape, next_token = run_forward_pass(lm, tokenizer, lm_head)
        forward_ok = next_token.strip() == "Paris"
        print(f"Logits shape:{logits_shape}")
    except Exception as exc:
        print(f"Forward pass error: {exc}")

    cap_ok = cap == (12, 0)
    print("\n" + "=" * 60)
    print(f"Capability (12,0): {'PASS' if cap_ok else 'FAIL'}")
    print(f"Forward pass:      {'PASS' if forward_ok else 'FAIL'}")
    print(f"Offline mode:      {'YES' if offline else 'not set'}")
    print("=" * 60)
    if not cap_ok or not forward_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
