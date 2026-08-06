"""
PagedRuntimeCache equivalence tests: multi-step, single-sequence decode
through the live paged cache vs. a manually-accumulated ground truth
(the same `torch.cat`-per-step accumulation `Qwen3_5DynamicCache.update()`
performs), across incremental steps straddling page-block boundaries.

These are pure-tensor tests -- no model weights required, since
`PagedRuntimeCache.update()` is exercised directly with synthetic K/V
tensors, exactly as a real attention layer's forward would call it.
"""

import unittest

import torch

from core.paged_cache import PagedKVCache
from core.paged_runtime_cache import PagedRuntimeCache


class _FakeConfig:
    def __init__(self, layer_types):
        self.layer_types = layer_types
        self.num_hidden_layers = len(layer_types)


def _make_cache(num_kv_heads=2, head_dim=4, block_size=16, num_blocks=8, seq_id=0):
    layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    config = _FakeConfig(layer_types)
    pool = PagedKVCache(
        num_layers=2,  # only full_attention layers are stored in the pool
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
    )
    cache = PagedRuntimeCache(config, pool=pool, seq_id=seq_id)
    return cache, pool


class PagedRuntimeCacheTest(unittest.TestCase):
    def test_incremental_steps_match_manual_concat_across_block_boundaries(self):
        # Step sizes chosen so cumulative length hits every value in the
        # required sweep {1,15,16,17,31,32,33}: 1, +14=15, +1=16, +1=17,
        # +14=31, +1=32, +1=33.
        torch.manual_seed(0)
        cache, pool = _make_cache(num_kv_heads=3, head_dim=5, block_size=16, num_blocks=8)
        full_layers = cache.transformer_layers  # [1, 3]
        expected = {layer: {"k": [], "v": []} for layer in full_layers}

        total = 0
        for step_tokens in [1, 14, 1, 1, 14, 1, 1]:
            total += step_tokens
            for layer in full_layers:
                k = torch.randn(1, 3, step_tokens, 5)
                v = torch.randn(1, 3, step_tokens, 5)
                out_k, out_v = cache.update(k, v, layer)
                expected[layer]["k"].append(k)
                expected[layer]["v"].append(v)

                want_k = torch.cat(expected[layer]["k"], dim=2)
                want_v = torch.cat(expected[layer]["v"], dim=2)
                torch.testing.assert_close(out_k, want_k)
                torch.testing.assert_close(out_v, want_v)
                torch.testing.assert_close(cache.key_cache[layer], want_k)
                torch.testing.assert_close(cache.value_cache[layer], want_v)

            with self.subTest(total_len=total):
                self.assertEqual(cache.get_seq_length(), total)
                self.assertEqual(cache.get_seq_length(full_layers[0]), total)
                self.assertEqual(pool.sequence_length(0), total)

        pool.assert_consistent()
        # Linear-attention layers were never touched via update(); this class
        # intentionally leaves that state exactly as the base class would.
        for layer in (0, 2):
            self.assertIsNone(cache.conv_states[layer])
            self.assertIsNone(cache.recurrent_states[layer])

    def test_rejects_batch_greater_than_one(self):
        cache, _ = _make_cache()
        layer = cache.transformer_layers[0]
        k = torch.randn(2, 2, 1, 4)
        with self.assertRaises(ValueError):
            cache.update(k, k, layer)

    def test_rejects_layer_not_in_transformer_layers(self):
        cache, _ = _make_cache()
        k = torch.randn(1, 2, 1, 4)
        with self.assertRaises(ValueError):
            cache.update(k, k, 0)  # layer 0 is linear_attention

    def test_rejects_out_of_order_layer_before_first_reserve(self):
        cache, _ = _make_cache()
        k = torch.randn(1, 2, 1, 4)
        second_full_layer = cache.transformer_layers[1]
        with self.assertRaises(RuntimeError):
            cache.update(k, k, second_full_layer)

    def test_free_sequence_releases_blocks_without_leaks(self):
        cache, pool = _make_cache(block_size=4, num_blocks=6)
        for layer in cache.transformer_layers:
            k = torch.randn(1, 2, 5, 4)
            cache.update(k, k, layer)
        pool.assert_consistent()
        self.assertGreater(pool.allocator.allocated_count, 0)

        cache.free_sequence()
        pool.assert_consistent()
        self.assertEqual(pool.allocator.free_count, pool.num_blocks)

    def test_two_sequences_share_pool_without_cross_contamination(self):
        config = _FakeConfig(["full_attention", "full_attention", "linear_attention"])
        pool = PagedKVCache(
            num_layers=2, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=3, dtype=torch.float32,
        )
        a = PagedRuntimeCache(config, pool=pool, seq_id="a")
        b = PagedRuntimeCache(config, pool=pool, seq_id="b")

        ka = torch.randn(1, 2, 5, 3)
        kb = torch.randn(1, 2, 3, 3)
        for layer in a.transformer_layers:
            a.update(ka, ka, layer)
        for layer in b.transformer_layers:
            b.update(kb, kb, layer)

        for layer in a.transformer_layers:
            torch.testing.assert_close(a.key_cache[layer], ka)
        for layer in b.transformer_layers:
            torch.testing.assert_close(b.key_cache[layer], kb)
        pool.assert_consistent()

        a.free_sequence()
        b.free_sequence()
        pool.assert_consistent()
        self.assertEqual(pool.allocator.free_count, pool.num_blocks)


if __name__ == "__main__":
    unittest.main()
