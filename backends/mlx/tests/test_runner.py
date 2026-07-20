"""Real-artifact MLX smoke; 4-bit output is not a bf16 parity claim."""

import os

import pytest

pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from backends.mlx.runner import MlxModelRunner


def test_forward_shape_and_tiny_greedy_decode():
    artifact = os.environ.get("INFERD_MLX_ARTIFACT")
    if not artifact:
        pytest.skip("set INFERD_MLX_ARTIFACT to run the real MLX runner smoke")
    runner = MlxModelRunner.load(artifact)
    ids = runner.tokenizer("Hello", add_special_tokens=True).input_ids
    logits, kv = runner.forward(torch.tensor([ids]), None)
    assert logits.shape[:2] == (1, len(ids))
    assert logits.shape[-1] > max(ids)

    generated = []
    for _ in range(3):
        token = int(logits[:, -1, :].argmax())
        generated.append(token)
        logits, kv = runner.forward(torch.tensor([[token]]), kv)
    assert len(generated) == 3
    assert isinstance(runner.tokenizer.decode(generated), str)
