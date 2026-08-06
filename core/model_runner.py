"""Shared engine entry point: load text backbone, run forward(tokens, kv) -> (logits, kv)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
from typing import Optional

import inferd.env  # noqa: F401

import torch  # noqa: E402

from bench.model_loader import load  # noqa: E402
from core.qwen35_patch import install_parallel_verify_patch  # noqa: E402
from core.speculation import CROP_NO_REPLAY, RESTORE_AND_REPLAY  # noqa: E402

install_parallel_verify_patch()


class ModelRunner:
    """Loaded text-only backbone with uniform forward(tokens, kv)."""

    def __init__(self, lm, lm_head, tokenizer, *, device: str = "cuda:0") -> None:
        self.lm = lm
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.device = device
        model_type = getattr(getattr(lm, "config", None), "model_type", None)
        if model_type == "qwen3":
            self.cache_reconciliation = CROP_NO_REPLAY
        elif model_type == "qwen3_5_text":
            self.cache_reconciliation = RESTORE_AND_REPLAY
        else:
            self.cache_reconciliation = "unsupported"
        self._validated_drafts: set[int] = set()
        self._tokenizer_sha256: str | None = None

    @classmethod
    def load_target(
        cls,
        path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        quantize: str | None = None,
        adapter: Optional[str | Path] = None,
    ) -> "ModelRunner":
        """Load the text backbone.

        quantize="fp8" enables the phase-10 FP8 hero path. adapter attaches a
        LoRA adapter without materializing a full merged 27B bf16 checkpoint.
        """
        lm, lm_head, tokenizer = load(
            Path(path), device=device, dtype=dtype, quantize=quantize, adapter=adapter
        )
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
        return cls.load_target(path, adapter=adapter, device=device, dtype=dtype)

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

    def validate_speculation_pair(
        self,
        draft: "ModelRunner",
        *,
        expected_tokenizer_sha256: str | None = None,
        expected_vocab_size: int | None = None,
    ) -> str:
        """Fail before speculation unless token IDs and logit columns match exactly."""
        if id(draft) not in self._validated_drafts:
            target_mapping = self.tokenizer.get_vocab()
            if target_mapping != draft.tokenizer.get_vocab():
                raise ValueError("target and draft tokenizer vocabularies differ")
            if self.tokenizer.get_added_vocab() != draft.tokenizer.get_added_vocab():
                raise ValueError("target and draft added-token mappings differ")
            special = ("bos_token_id", "eos_token_id", "pad_token_id")
            if any(getattr(self.tokenizer, key) != getattr(draft.tokenizer, key) for key in special):
                raise ValueError("target and draft BOS/EOS/PAD token IDs differ")
            target_vocab = _logit_size(self.lm_head, self.lm)
            draft_vocab = _logit_size(draft.lm_head, draft.lm)
            if target_vocab != draft_vocab:
                raise ValueError(
                    f"target/draft logit sizes differ: target={target_vocab}, "
                    f"draft={draft_vocab}"
                )
            if any(token_id < 0 or token_id >= target_vocab for token_id in target_mapping.values()):
                raise ValueError("tokenizer mapping contains IDs outside the model logit range")
            self._validated_drafts.add(id(draft))

        if expected_vocab_size is not None and _logit_size(self.lm_head, self.lm) != expected_vocab_size:
            raise ValueError(f"expected vocab/logit size {expected_vocab_size}")
        if self._tokenizer_sha256 is None:
            self._tokenizer_sha256 = _tokenizer_signature(self.tokenizer)
        signature = self._tokenizer_sha256
        if expected_tokenizer_sha256 is not None and signature != expected_tokenizer_sha256:
            raise ValueError(
                f"tokenizer SHA-256 mismatch: expected {expected_tokenizer_sha256}, got {signature}"
            )
        return signature

    def checkpoint_speculation(self, kv):
        """Return an opaque, architecture-specific rollback checkpoint."""
        if self.cache_reconciliation == CROP_NO_REPLAY:
            return _DenseCheckpoint(_validate_dense_cache(kv))
        if self.cache_reconciliation == RESTORE_AND_REPLAY:
            length = _validate_hybrid_cache(kv)
            return _HybridCheckpoint(
                length,
                _clone_states(kv.conv_states),
                _clone_states(kv.recurrent_states),
            )
        raise TypeError("model does not declare a supported speculative-cache strategy")

    def reconcile_speculation(self, kv, checkpoint, accepted_count: int, emitted_tokens):
        """Rollback verification and commit accepted tokens plus residual/bonus."""
        emitted = list(emitted_tokens)
        if accepted_count < 0 or len(emitted) != accepted_count + 1:
            raise ValueError("speculation must emit accepted tokens plus one residual/bonus token")

        if isinstance(checkpoint, _DenseCheckpoint):
            if self.cache_reconciliation != CROP_NO_REPLAY:
                raise TypeError("dense checkpoint used with a non-dense runner")
            _validate_dense_cache(kv, minimum_length=checkpoint.length + accepted_count)
            kv.crop(checkpoint.length + accepted_count)
            tokens = torch.tensor([[emitted[-1]]], dtype=torch.long, device=self.device)
        elif isinstance(checkpoint, _HybridCheckpoint):
            if self.cache_reconciliation != RESTORE_AND_REPLAY:
                raise TypeError("hybrid checkpoint used with a non-hybrid runner")
            _validate_hybrid_cache(kv, minimum_length=checkpoint.length + accepted_count)
            kv.conv_states = _clone_states(checkpoint.conv_states)
            kv.recurrent_states = _clone_states(checkpoint.recurrent_states)
            for layer in kv.transformer_layers:
                kv.key_cache[layer] = kv.key_cache[layer][:, :, :checkpoint.length, :].contiguous()
                kv.value_cache[layer] = kv.value_cache[layer][:, :, :checkpoint.length, :].contiguous()
            tokens = torch.tensor([emitted], dtype=torch.long, device=self.device)
        else:
            raise TypeError("unknown speculative-cache checkpoint")
        return self.forward(tokens, kv)


@dataclass(frozen=True)
class _DenseCheckpoint:
    length: int


@dataclass(frozen=True)
class _HybridCheckpoint:
    length: int
    conv_states: list
    recurrent_states: list


def _clone_states(states):
    return [value.clone() if value is not None else None for value in states]


def _validate_dense_cache(kv, minimum_length: int = 0) -> int:
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer

    if type(kv) is not DynamicCache or not kv.layers:
        raise TypeError("dense speculation requires a populated DynamicCache")
    if any(type(layer) is not DynamicLayer for layer in kv.layers):
        raise TypeError("sliding-window or unknown dense cache layers are unsupported")
    lengths = set()
    for layer in kv.layers:
        if not isinstance(layer.keys, torch.Tensor) or not isinstance(layer.values, torch.Tensor):
            raise TypeError("dense cache contains uninitialized layers")
        if layer.keys.shape != layer.values.shape:
            raise ValueError("dense cache key/value shapes differ")
        lengths.add(int(layer.keys.shape[-2]))
    if len(lengths) != 1:
        raise ValueError("dense cache layers have inconsistent lengths")
    length = lengths.pop()
    if length < minimum_length:
        raise ValueError("dense cache is shorter than the speculative checkpoint")
    return length


def _validate_hybrid_cache(kv, minimum_length: int = 0) -> int:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    if type(kv) is not Qwen3_5DynamicCache:
        raise TypeError("hybrid speculation requires Qwen3_5DynamicCache")
    size = len(kv.layer_types)
    fields = (kv.key_cache, kv.value_cache, kv.conv_states, kv.recurrent_states)
    if any(len(values) != size for values in fields) or not kv.transformer_layers:
        raise ValueError("malformed Qwen3.5 hybrid cache")
    lengths = set()
    for layer in range(size):
        if layer in kv.transformer_layers:
            key, value = kv.key_cache[layer], kv.value_cache[layer]
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError("hybrid attention cache contains uninitialized layers")
            if key.shape != value.shape:
                raise ValueError("hybrid cache key/value shapes differ")
            lengths.add(int(key.shape[-2]))
        elif not isinstance(kv.conv_states[layer], torch.Tensor) or not isinstance(
            kv.recurrent_states[layer], torch.Tensor
        ):
            raise TypeError("hybrid linear state contains uninitialized layers")
    if len(lengths) != 1:
        raise ValueError("hybrid attention layers have inconsistent lengths")
    length = lengths.pop()
    if length < minimum_length:
        raise ValueError("hybrid cache is shorter than the speculative checkpoint")
    return length


def _logit_size(lm_head, lm) -> int:
    if lm_head is not None and hasattr(lm_head, "out_features"):
        return int(lm_head.out_features)
    return int(getattr(getattr(lm, "config", None), "vocab_size", -1))


def _tokenizer_signature(tokenizer) -> str:
    path = Path(getattr(tokenizer, "name_or_path", "")) / "tokenizer.json"
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(
        {
            "vocab": tokenizer.get_vocab(),
            "added": tokenizer.get_added_vocab(),
            "special": [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
