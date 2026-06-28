"""Shared engine entry point: load text backbone, run forward(tokens, kv) -> (logits, kv)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import inferd.env  # noqa: F401

import torch  # noqa: E402

from bench.model_loader import _strip_vision, load  # noqa: E402
from core.qwen35_patch import install_parallel_verify_patch  # noqa: E402

install_parallel_verify_patch()


class ModelRunner:
    """Loaded text-only backbone with uniform forward(tokens, kv)."""

    def __init__(self, lm, lm_head, tokenizer, *, device: str = "cuda:0") -> None:
        self.lm = lm
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load_target(
        cls,
        path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ModelRunner":
        lm, lm_head, tokenizer = load(Path(path), device=device, dtype=dtype)
        return cls(lm, lm_head, tokenizer, device=device)

    @classmethod
    def load_draft(
        cls,
        path: str | Path,
        *,
        adapter: Optional[str | Path] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ModelRunner":
        """Load draft backbone; optional LoRA adapter is merged before extraction."""
        if adapter is None:
            return cls.load_target(path, device=device, dtype=dtype)

        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        base = AutoModelForMultimodalLM.from_pretrained(
            str(path), dtype=dtype, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = PeftModel.from_pretrained(base, str(adapter))
        model = model.merge_and_unload()
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
            lm, lm_head = model, None

        return cls(lm, lm_head, tokenizer, device=device)

    def forward(self, tokens: torch.Tensor, kv=None, attention_mask=None,
                position_ids=None, cache_position=None):
        """
        One forward pass. Returns (logits [batch, seq, vocab], updated kv).

        With kv set, pass only new tokens. For left-padded batches supply
        attention_mask and position_ids (each row's true position for RoPE).
        """
        dev = self.device.split(":")[0]
        if tokens.device.type != dev:
            tokens = tokens.to(self.device)
        if attention_mask is not None and attention_mask.device.type != dev:
            attention_mask = attention_mask.to(self.device)
        if position_ids is not None and position_ids.device.type != dev:
            position_ids = position_ids.to(self.device)
        if cache_position is not None and cache_position.device.type != dev:
            cache_position = cache_position.to(self.device)
        with torch.no_grad():
            out = self.lm(
                input_ids=tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=kv,
                use_cache=True,
            )
            hidden = out.last_hidden_state
            logits = self.lm_head(hidden) if self.lm_head is not None else hidden
        return logits, out.past_key_values
