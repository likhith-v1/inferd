"""MLX model runner with CPU torch logits for inferd sampling."""

from __future__ import annotations

from pathlib import Path


class MlxModelRunner:
    device = "cpu"

    def __init__(self, model, tokenizer, artifact) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.artifact = artifact

    @classmethod
    def load(cls, artifact_path: str | Path):
        from backends.mlx.loader import load_artifact

        model, tokenizer, artifact = load_artifact(artifact_path)
        return cls(model, tokenizer, artifact)

    def forward(self, tokens, kv=None):
        import mlx.core as mx
        import numpy as np
        import torch
        from mlx_lm.models.cache import make_prompt_cache

        if isinstance(tokens, torch.Tensor):
            token_ids = tokens.detach().cpu().tolist()
        else:
            token_ids = tokens.tolist() if hasattr(tokens, "tolist") else tokens
        if not token_ids or not token_ids[0]:
            raise ValueError("tokens must have shape [batch, sequence] with non-empty rows")
        cache = make_prompt_cache(self.model) if kv is None else kv
        logits = self.model(mx.array(token_ids), cache=cache)
        # ponytail: every caller uses only the last position, so slice in MLX before
        # copying to CPU — a long prompt otherwise copies the whole [batch, seq, vocab].
        last = logits[:, -1:, :]
        mx.eval(last)
        return torch.from_numpy(np.asarray(last, dtype=np.float32).copy()), cache
