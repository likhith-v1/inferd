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


def load(
    weights_dir: Path,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load text backbone from a Qwen3.5 multimodal checkpoint."""
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

    return lm, lm_head, tokenizer
