"""Deterministic CPU backend used to prove the scheduler seam."""

from __future__ import annotations

import torch


class StubBackend:
    device = "cpu"
    eos_token_id = None

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((1, self.vocab_size), -1_000.0)
        logits[0, token_id % self.vocab_size] = 1_000.0
        return logits

    def prefill(self, prompt_ids: list[int]):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        kv = tuple(prompt_ids)
        return self._logits(len(kv)), kv

    def decode_batch(self, token_ids: list[int], kvs: list[object]):
        if not token_ids or len(token_ids) != len(kvs):
            raise ValueError("token_ids and kvs must be non-empty and aligned")
        updated = [tuple(kv) + (int(token),) for token, kv in zip(token_ids, kvs)]
        return [self._logits(len(kv)) for kv in updated], updated
