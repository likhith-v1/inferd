"""Mac stop gate: ragged MLX batch cache matches independent decoding."""

import os

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from backends.mlx.backend import MlxSchedulerBackend
from backends.mlx.runner import MlxModelRunner


def _artifact():
    path = os.environ.get("INFERD_MLX_ARTIFACT")
    if not path:
        pytest.skip("set INFERD_MLX_ARTIFACT to run the real MLX cache gate")
    return path


def _prefill_pair(backend, ids):
    serial_logits, serial_kv = backend.prefill(ids)
    batch_logits, batch_kv = backend.prefill(ids)
    torch.testing.assert_close(batch_logits, serial_logits, atol=1e-3, rtol=1e-3)
    return [serial_logits, serial_kv], [batch_logits, batch_kv]


def _advance(backend, serial, batched, names, steps):
    for _ in range(steps):
        tokens = [int(serial[name][0].argmax()) for name in names]
        expected = {}
        for name, token in zip(names, tokens):
            logits, kvs = backend.decode_batch([token], [serial[name][1]])
            serial[name] = [logits[0], kvs[0]]
            expected[name] = logits[0]
        logits, kvs = backend.decode_batch(tokens, [batched[name][1] for name in names])
        for name, logits_row, kv in zip(names, logits, kvs):
            torch.testing.assert_close(logits_row, expected[name], atol=1e-3, rtol=1e-3)
            assert int(logits_row.argmax()) == int(expected[name].argmax())
            batched[name] = [logits_row, kv]


def test_ragged_repeated_decode_with_admission_and_backfill():
    runner = MlxModelRunner.load(_artifact())
    backend = MlxSchedulerBackend(runner)
    prompts = [
        runner.tokenizer("Short prompt", add_special_tokens=True).input_ids,
        runner.tokenizer("A deliberately much longer prompt for ragged cache coverage", add_special_tokens=True).input_ids,
        runner.tokenizer("Backfilled request", add_special_tokens=True).input_ids,
    ]
    assert len(prompts[0]) != len(prompts[1])

    serial, batched = {}, {}
    serial["a"], batched["a"] = _prefill_pair(backend, prompts[0])
    serial["b"], batched["b"] = _prefill_pair(backend, prompts[1])
    _advance(backend, serial, batched, ["a", "b"], 3)

    del serial["a"], batched["a"]
    serial["c"], batched["c"] = _prefill_pair(backend, prompts[2])
    _advance(backend, serial, batched, ["b", "c"], 3)
