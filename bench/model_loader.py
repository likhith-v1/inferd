"""Load Qwen text backbones with vision stripped."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import inferd.env  # noqa: F401

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
)

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


def _peft_inner_model(model):
    """Return the wrapped checkpoint inside a PeftModel when the path exists."""
    wrapper = getattr(model, "base_model", None)
    if wrapper is None:
        return model
    inner = getattr(wrapper, "model", None)
    return inner if inner is not None else model


def _torchao_quant_config(recipe: str = "fp8"):
    """Build a load-time torchao FP8 config; keep lm_head bf16."""
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
    """Load text backbone, optionally with FP8 quantization and a LoRA adapter.

    When an adapter is supplied without quantization, weights are merged into the
    base checkpoint before backbone extraction. The FP8 hero path keeps runtime LoRA
    because a merged bf16 27B does not fit on-card.
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    if quantize not in (None, "fp8", "fp8-dynamic"):
        raise ValueError(f"unsupported quantize={quantize!r} (use 'fp8' or 'fp8-dynamic')")

    config = AutoConfig.from_pretrained(str(weights_dir))
    if config.model_type == "qwen3_5":
        model_cls = AutoModelForMultimodalLM
    elif config.model_type == "qwen3":
        model_cls = AutoModelForCausalLM
    else:
        raise ValueError(f"unsupported model_type={config.model_type!r}; expected qwen3 or qwen3_5")

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
    quantization_config = _torchao_quant_config(quantize) if quantize else None
    model = model_cls.from_pretrained(
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
        if quantize is None:
            # Merged weights fit for bf16 paths (9B, etc.). FP8 27B keeps runtime
            # LoRA because a merged bf16 27B checkpoint does not fit the card.
            model = model.merge_and_unload()
        else:
            model = _peft_inner_model(model)
    model.eval()

    if config.model_type == "qwen3":
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise TypeError("Qwen3 causal loader did not return .model and .lm_head")
        lm, lm_head = model.model, model.lm_head
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
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
