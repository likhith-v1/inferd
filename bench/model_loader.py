"""
bench.model_loader — load Qwen3.5 text backbone (vision stripped).

Returns (lm, lm_head, tokenizer). Used by ModelRunner and bench runners.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import inferd.env  # noqa: F401

import torch  # noqa: E402
from transformers import AutoModelForMultimodalLM, AutoTokenizer  # noqa: E402

_VISION_ATTRS = (
    "visual",
    "vision_model",
    "vision_tower",
    "image_tower",
    "vpm",
    "vision_encoder",
)


def _strip_vision(container) -> list[str]:
    stripped = []
    for attr in _VISION_ATTRS:
        if hasattr(container, attr):
            delattr(container, attr)
            stripped.append(attr)
    if stripped:
        torch.cuda.empty_cache()
    return stripped


def _quantize_fp8(lm, recipe: str = "fp8") -> None:
    """
    In-place FP8 quantization of the transformer's nn.Linear weights (phase 10).

    Two recipes (e4m3 weights), both on the RTX 5090's (sm_120) FP8 hardware:
      - "fp8"  weight-only FP8 — halves the weight bytes read from HBM each step.
        Single-token decode is memory-bandwidth-bound on weight loads, so this is
        the right recipe for the single-stream hero (memory win *and* faster
        decode).
      - "fp8-dynamic" W8A8 dynamic-activation FP8 — routes matmuls through
        `_scaled_mm` FP8 tensor cores; wins at compute-bound prefill / large
        batch, but the per-step activation-quant overhead hurts M=1 decode.

    The lm_head stays bf16 (one large vocab projection — quantizing it dents
    quality for negligible memory). FP8 is the project's one quantization
    exception, scoped to this hero path only.
    """
    from torchao.quantization import quantize_

    if recipe == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig
        quantize_(lm, Float8WeightOnlyConfig())
    elif recipe == "fp8-dynamic":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        quantize_(lm, Float8DynamicActivationFloat8WeightConfig())
    else:
        raise ValueError(f"unknown fp8 recipe {recipe!r}")
    torch.cuda.empty_cache()


def load(
    weights_dir: Path,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    quantize: str | None = None,
) -> tuple:
    """
    Load text backbone from a Qwen3.5 multimodal checkpoint.

    quantize="fp8" applies in-place W8A8 FP8 to the backbone Linears after the
    vision tower is stripped (phase-10 hero path). Default (None) is unchanged.
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
    model = AutoModelForMultimodalLM.from_pretrained(
        str(weights_dir),
        dtype=dtype,
        device_map=device,
    )
    model.eval()

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        _strip_vision(model.model)
        _strip_vision(model)
    elif hasattr(model, "language_model"):
        lm = model.language_model
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        _strip_vision(model)
    else:
        lm = model
        lm_head = None

    if quantize in ("fp8", "fp8-dynamic"):
        _quantize_fp8(lm, recipe=quantize)
    elif quantize is not None:
        raise ValueError(f"unsupported quantize={quantize!r} (use 'fp8' or 'fp8-dynamic')")

    return lm, lm_head, tokenizer
