"""
bench.model_loader — interim text-backbone loader.

Loads Qwen3.5-* as AutoModelForMultimodalLM, extracts the text backbone
(model.model.language_model + model.lm_head), and strips the vision tower.
Returns (lm, lm_head, tokenizer).

This is an INTERIM loader used by phases 02–03. Phase 04 introduces
core/model_runner.py with the load_target()/forward() contract; at that point
bench/runners/*.py will swap the import without other changes because the
returned tuple signature stays identical.

Usage:
    from bench.model_loader import load
    lm, lm_head, tokenizer = load(Path("weights/Qwen3.5-9B"))
"""

from __future__ import annotations

import ctypes
import sysconfig
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Pre-load libnvJitLink.so.13 BEFORE any torch/transformers import.
# PyTorch cu130 bundles CUDA 13 libs inside the venv; bitsandbytes needs
# libnvJitLink at dlopen time and it is not on LD_LIBRARY_PATH by default.
# ctypes.RTLD_GLOBAL makes the symbols globally visible, equivalent to
# adding the dir to LD_LIBRARY_PATH for this process only.
# ---------------------------------------------------------------------------
_nvjitlink = (
    Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/lib/libnvJitLink.so.13"
)
if _nvjitlink.exists():
    ctypes.CDLL(str(_nvjitlink), mode=ctypes.RTLD_GLOBAL)

import torch  # noqa: E402  (must come after ctypes preload)
from transformers import AutoModelForMultimodalLM, AutoTokenizer  # noqa: E402

# Vision attribute names to probe and remove (order matters — most common first).
_VISION_ATTRS = (
    "visual",
    "vision_model",
    "vision_tower",
    "image_tower",
    "vpm",
    "vision_encoder",
)


def _strip_vision(container) -> list[str]:
    """Remove vision sub-modules from a model container; return stripped names."""
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
    """
    Load the Qwen3.5-* text backbone.

    Returns
    -------
    lm         : the language_model backbone (Qwen3_5Model)
    lm_head    : the causal-LM head (linear weight, tied or separate)
    tokenizer  : the text tokenizer (from AutoTokenizer)

    The vision tower is stripped from memory before returning.
    The returned (lm, lm_head, tokenizer) signature is stable — phase 04 will
    swap this loader for core.model_runner without changing runner code.
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

    # Extract backbone: Qwen3_5ForConditionalGeneration wraps
    # model.model (Qwen3_5Model) which contains:
    #   .language_model  — the actual transformer stack
    #   .visual          — the vision tower (stripped below)
    # lm_head lives on the top-level wrapper as model.lm_head.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        _strip_vision(model.model)   # removes model.model.visual
        _strip_vision(model)         # safety pass on the wrapper
    elif hasattr(model, "language_model"):
        # Fallback for architectures that expose it at the top level.
        lm = model.language_model
        lm_head = model.lm_head if hasattr(model, "lm_head") else None
        _strip_vision(model)
    else:
        # Last resort: treat the full model as the backbone (no vision strip).
        lm = model
        lm_head = None

    return lm, lm_head, tokenizer


def vram_used_gb(device_index: int = 0) -> float:
    """Current torch VRAM allocated on the given device, in GB."""
    return torch.cuda.memory_allocated(device_index) / 1024**3
