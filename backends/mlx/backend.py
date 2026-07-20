"""SchedulerBackend implementation for MLX."""

from __future__ import annotations

import torch

from backends.mlx.cache import extract_caches, merge_caches


class MlxSchedulerBackend:
    device = "cpu"

    def __init__(self, runner) -> None:
        self.runner = runner
        self.eos_token_id = runner.tokenizer.eos_token_id

    def prefill(self, prompt_ids: list[int]):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        logits, kv = self.runner.forward(torch.tensor([prompt_ids], dtype=torch.long), None)
        return logits[:, -1, :], kv

    def decode_batch(self, token_ids: list[int], kvs: list[object]):
        if not token_ids or len(token_ids) != len(kvs):
            raise ValueError("token_ids and kvs must be non-empty and aligned")
        tokens = torch.tensor([[token] for token in token_ids], dtype=torch.long)
        if len(kvs) == 1:
            logits, kv = self.runner.forward(tokens, kvs[0])
            return [logits[:, -1, :]], [kv]
        batched = merge_caches(kvs)
        logits, batched = self.runner.forward(tokens, batched)
        rows = extract_caches(batched, len(kvs))
        last = logits[:, -1, :]
        return [last[index:index + 1] for index in range(len(kvs))], rows
