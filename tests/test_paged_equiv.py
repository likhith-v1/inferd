"""Paged-attention equivalence tests (paged gather vs SDPA reference)."""

import unittest

import torch

from bench.paged_equiv import logits_equivalent
from bench.runners.paged import _microbench_hash
from core.paged_attn import paged_attention, sdpa_reference
from core.paged_cache import PagedHybridCache, PagedKVCache


class LogitsEquivalentTest(unittest.TestCase):
    def test_requires_both_absolute_and_relative_bounds(self):
        # OR would pass: large abs error on a big logit, tiny relative error.
        ref = torch.tensor([10_000.0])
        bad = torch.tensor([10_000.1])
        ok, max_abs, max_rel = logits_equivalent(ref, bad, atol=5e-3, rtol=5e-3)
        self.assertFalse(ok)
        self.assertGreater(max_abs, 5e-3)
        self.assertLess(max_rel, 5e-3)

    def test_passes_when_both_bounds_hold(self):
        ref = torch.tensor([1.0, -2.0, 0.001])
        close = torch.tensor([1.0 + 1e-6, -2.0 - 1e-6, 0.001 + 1e-6])
        ok, _, _ = logits_equivalent(ref, close, atol=5e-3, rtol=5e-3)
        self.assertTrue(ok)


class MicrobenchHashTest(unittest.TestCase):
    def test_changes_with_max_tokens(self):
        shape = {"layers": 36, "kv_heads": 8, "head_dim": 128, "dtype": "bfloat16", "block_size": 16}
        grid = [1, 2, 4]
        h256 = _microbench_hash(
            seed=0, max_tokens=256, block_size=16, concurrency_grid=grid, model_shape=shape,
        )
        h128 = _microbench_hash(
            seed=0, max_tokens=128, block_size=16, concurrency_grid=grid, model_shape=shape,
        )
        self.assertNotEqual(h256, h128)


class PagedCacheEquivalenceTest(unittest.TestCase):
    def test_append_gather_across_page_boundaries(self):
        cache = PagedKVCache(
            num_layers=3,
            num_blocks=16,
            block_size=16,
            num_kv_heads=2,
            head_dim=4,
            dtype=torch.float32,
        )
        cache.create_sequence(7)
        layer_keys = []
        layer_values = []
        for layer in range(3):
            base = torch.arange(33 * 2 * 4, dtype=torch.float32).reshape(33, 2, 4)
            layer_keys.append(base + layer * 1000)
            layer_values.append(base + layer * 1000 + 0.5)

        cache.append_tokens(7, layer_keys, layer_values)
        cache.assert_consistent()
        self.assertEqual(cache.sequence_length(7), 33)
        self.assertEqual(cache.allocator.allocated_count, 3)

        for layer in range(3):
            got_k, got_v = cache.gather_layer(7, layer)
            torch.testing.assert_close(got_k, layer_keys[layer])
            torch.testing.assert_close(got_v, layer_values[layer])

    def test_paged_attention_matches_independent_sdpa_reference(self):
        # Compare the paged gather-and-attend path against torch SDPA — an
        # INDEPENDENT implementation — so the test can actually catch an attention
        # / indexing bug (paged_attention internally uses dense_attention, so
        # comparing to dense_attention would be tautological). Sweep page
        # boundaries and grouped-query ratios.
        torch.manual_seed(0)
        lengths = [1, 15, 16, 17, 31, 32, 33]
        for kv_heads, q_heads in [(2, 4), (4, 4), (2, 8)]:  # GQA + MHA
            for seq_len in lengths:
                with self.subTest(seq_len=seq_len, kv_heads=kv_heads, q_heads=q_heads):
                    cache = PagedKVCache(
                        num_layers=1,
                        num_blocks=16,
                        block_size=16,
                        num_kv_heads=kv_heads,
                        head_dim=8,
                        dtype=torch.float32,
                    )
                    cache.create_sequence(0)
                    key = torch.randn(seq_len, kv_heads, 8)
                    value = torch.randn(seq_len, kv_heads, 8)
                    query = torch.randn(q_heads, 8)
                    cache.append_tokens(0, [key], [value])

                    out_paged = paged_attention(
                        query,
                        cache.key_cache[0],
                        cache.value_cache[0],
                        cache.block_table(0),
                        seq_len,
                    )
                    out_ref = sdpa_reference(query, key, value)
                    torch.testing.assert_close(out_paged, out_ref, rtol=1e-4, atol=1e-5)

    def test_allocator_reuses_freed_blocks_without_leaks(self):
        cache = PagedKVCache(
            num_layers=1,
            num_blocks=4,
            block_size=2,
            num_kv_heads=1,
            head_dim=1,
            dtype=torch.float32,
        )
        cache.create_sequence(1)
        cache.create_sequence(2)
        one = [torch.ones(3, 1, 1)]
        cache.append_tokens(1, one, one)
        first_blocks = list(cache.sequences[1].blocks)
        cache.free_sequence(1)
        cache.append_tokens(2, one, one)
        self.assertEqual(cache.sequences[2].blocks, first_blocks[::-1])
        cache.free_sequence(2)
        cache.assert_consistent()
        self.assertEqual(cache.allocator.free_count, cache.num_blocks)

    def test_qwen_hybrid_cache_roundtrip_preserves_attention_and_linear_state(self):
        class FakeQwenCache:
            pass

        fake = FakeQwenCache()
        fake.layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
        fake.transformer_layers = [1, 3]
        fake.last_linear_layer = 2
        fake.key_cache = [None, torch.randn(1, 2, 17, 8), None, torch.randn(1, 2, 17, 8)]
        fake.value_cache = [None, torch.randn(1, 2, 17, 8), None, torch.randn(1, 2, 17, 8)]
        fake.conv_states = [torch.randn(1, 4, 3), None, torch.randn(1, 4, 3), None]
        fake.recurrent_states = [torch.randn(1, 4, 6), None, torch.randn(1, 4, 6), None]

        paged = PagedHybridCache.from_qwen_cache(fake, block_size=16)
        roundtrip = paged.to_qwen_cache_like(fake)

        for layer in fake.transformer_layers:
            torch.testing.assert_close(roundtrip.key_cache[layer], fake.key_cache[layer])
            torch.testing.assert_close(roundtrip.value_cache[layer], fake.value_cache[layer])
        for layer in [0, 2]:
            torch.testing.assert_close(roundtrip.conv_states[layer], fake.conv_states[layer])
            torch.testing.assert_close(roundtrip.recurrent_states[layer], fake.recurrent_states[layer])


if __name__ == "__main__":
    unittest.main()
