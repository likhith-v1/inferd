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


def _torchao_quant_config(recipe: str = "fp8"):
    """
    Load-time FP8 quantization config for the transformer's nn.Linear weights.

    Two recipes (e4m3 weights), both on the RTX 5090's (sm_120) FP8 hardware:
      - "fp8"  weight-only FP8 — halves the weight bytes read from HBM each step.
        Single-token decode is memory-bandwidth-bound on weight loads. On the
        current torchao/Blackwell stack this is a capacity win, not a latency win.
      - "fp8-dynamic" W8A8 dynamic-activation FP8 — routes matmuls through
        `_scaled_mm` FP8 tensor cores; wins at compute-bound prefill / large
        batch, but the per-step activation-quant overhead hurts M=1 decode.

    The lm_head stays bf16 (one large vocab projection — quantizing it dents
    quality for negligible memory). FP8 is the project's one quantization
    exception, scoped to this hero path only.
    """
    from transformers import TorchAoConfig

    if recipe == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig
        config = Float8WeightOnlyConfig()
    elif recipe == "fp8-dynamic":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        config = Float8DynamicActivationFloat8WeightConfig()
    else:
        raise ValueError(f"unknown fp8 recipe {recipe!r}")
    return TorchAoConfig(config, modules_to_not_convert=["lm_head"])


def load(
    weights_dir: Path,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    quantize: str | None = None,
    adapter: str | Path | None = None,
) -> tuple:
    """
    Load text backbone from a Qwen3.5 multimodal checkpoint.

    quantize="fp8" applies load-time torchao FP8 to the backbone Linears
    (phase-10 hero path). Default (None) is unchanged. adapter attaches a LoRA
    adapter after base load; this avoids materializing a full 27B bf16 merge.
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    if quantize not in (None, "fp8", "fp8-dynamic"):
        raise ValueError(f"unsupported quantize={quantize!r} (use 'fp8' or 'fp8-dynamic')")

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
    quantization_config = _torchao_quant_config(quantize) if quantize else None
    model = AutoModelForMultimodalLM.from_pretrained(
        str(weights_dir),
        dtype=dtype,
        device_map=device,
        quantization_config=quantization_config,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter),
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
    model.eval()

    if adapter is not None and hasattr(model, "base_model"):
        model = model.base_model.model

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

    torch.cuda.empty_cache()
    return lm, lm_head, tokenizer
