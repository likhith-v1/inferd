"""
core.model_runner — the hot shared engine entry point (phases 04/05/06/10).

Implements the interface contract from plans/00:
  - load_target() / load_draft() return the TEXT-ONLY language_model backbone
    (+ lm_head, tokenizer); the vision tower is stripped.
  - forward(tokens, kv) -> (logits, kv) is the single source of truth for a
    forward pass; both spec-decode (04) and continuous batching (06) call it.
  - `kv` is an OPAQUE handle: a contiguous HF Cache in 04, paged in 05+.
    Callers never inspect its layout.

Extend this file via new methods per the contract; do not rewrite existing
methods without updating the contract in plans/00 first.

The text-backbone extraction + vision strip is delegated to
bench.model_loader.load() so there is exactly one copy of that logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Pre-load bundled CUDA 13 libs before any torch import (see inferd/env.py).
import inferd.env  # noqa: F401

import torch  # noqa: E402

from bench.model_loader import _strip_vision, load  # noqa: E402
from core.qwen35_patch import install_parallel_verify_patch  # noqa: E402

# Qwen3.5's GatedDeltaNet only continues its recurrent state for seq_len==1;
# enable correct multi-token-from-cache continuation (needed for spec-decode
# parallel verify + replay). Idempotent; safe to call on every load.
install_parallel_verify_patch()


class ModelRunner:
    """A loaded text-only backbone with a uniform forward(tokens, kv) pass."""

    def __init__(self, lm, lm_head, tokenizer, *, device: str = "cuda:0") -> None:
        self.lm = lm
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.device = device

    # -- loading -----------------------------------------------------------

    @classmethod
    def load_target(
        cls,
        path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ModelRunner":
        """Load the target model's text backbone (vision stripped)."""
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
        """
        Load the draft model's text backbone (vision stripped).

        If `adapter` is given (e.g. the distilled-draft LoRA from phase 04), it is
        applied to the base and merged, then the backbone is extracted the same
        way bench.model_loader does. Without an adapter this is identical to
        load_target.
        """
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

        # Same extraction path as bench.model_loader.load().
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

    # -- the single forward pass ------------------------------------------

    def forward(self, tokens: torch.Tensor, kv=None, attention_mask=None,
                position_ids=None, cache_position=None):
        """
        Run one forward pass over `tokens` (shape [batch, seq]).

        Returns (logits, kv):
          logits — [batch, seq, vocab]
          kv     — opaque updated cache handle (pass back in on the next call)

        `tokens` should be the *new* tokens only when `kv` is supplied; the model
        infers positions from the cache length. When `kv` is None this is a
        prefill over the full prompt.

        `attention_mask` (optional, shape [batch, past+seq]) must be supplied for
        LEFT-PADDED batches so the model does not attend to pad tokens (both the
        attention KV and the linear-attention recurrent/conv state). Single-stream
        callers (spec-decode, batch=1) leave it None.

        `position_ids` (optional, shape [batch, seq]) must be supplied alongside a
        LEFT-PADDED batched cache (continuous batching, 06): rows have different
        true lengths, so the default `cache_length + arange` positions are wrong
        for the shorter rows and would corrupt RoPE. Pass each row's true position.
        `cache_position` is forwarded as-is when given.
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
