"""
BatchedPagedCache equivalence tests: N rows sharing one pool, driven through
several batched decode steps with synthetic tensors -- no model weights
required. This is the discriminating test for the bug the design exists to
prevent: a row's write ever landing in another row's page table, or a
row's pool contents ever failing to match exactly what that row
contributed (never derived from padded/concatenated output).
"""

import unittest

import torch

from core.paged_cache import PagedKVCache
from core.paged_runtime_cache import BatchedPagedCache, PagedRuntimeCache


class _FakeConfig:
    def __init__(self, layer_types):
        self.layer_types = layer_types
        self.num_hidden_layers = len(layer_types)


def _make_rows(n_rows, *, num_kv_heads=2, head_dim=3, block_size=4, num_blocks=32):
    layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    config = _FakeConfig(layer_types)
    pool = PagedKVCache(
        num_layers=2,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
    )
    rows = [PagedRuntimeCache(config, pool=pool, seq_id=i) for i in range(n_rows)]
    for row in rows:
        for layer in (0, 2):  # linear_attention: seed non-None state, as prefill would
            row.conv_states[layer] = torch.zeros(1, 4, 3)
            row.recurrent_states[layer] = torch.zeros(1, 4, 6)
    return rows, pool


class BatchedPagedCacheTest(unittest.TestCase):
    def test_multi_step_writes_are_bit_exact_and_isolated_per_row(self):
        torch.manual_seed(0)
        n_rows = 3
        rows, pool = _make_rows(n_rows)
        full_layers = rows[0].transformer_layers  # [1, 3]

        # ground truth: what each row *should* accumulate, built independently
        # of BatchedPagedCache, from the exact tensors fed to update().
        expected = {r._seq_id: {layer: [] for layer in full_layers} for r in rows}

        for _step in range(5):
            batched = BatchedPagedCache(rows)
            for layer in full_layers:
                k = torch.randn(n_rows, 2, 1, 3)
                v = torch.randn(n_rows, 2, 1, 3)
                batched.update(k, v, layer)
                for i, row in enumerate(rows):
                    # [kv_heads, head_dim] -- one token's K/V for this row/layer/step.
                    expected[row._seq_id][layer].append((k[i, :, 0, :], v[i, :, 0, :]))
            pool.assert_consistent()

            for row in rows:
                for layer in full_layers:
                    cache_layer = row._cache_layer_for[layer]
                    got_k, got_v = pool.gather_layer(row._seq_id, cache_layer)
                    want_k = torch.stack([k for k, _ in expected[row._seq_id][layer]], dim=0)
                    want_v = torch.stack([v for _, v in expected[row._seq_id][layer]], dim=0)
                    torch.testing.assert_close(got_k, want_k)
                    torch.testing.assert_close(got_v, want_v)

        pool.assert_consistent()

    def test_scatter_linear_state_back_round_trips_per_row(self):
        rows, pool = _make_rows(2)
        batched = BatchedPagedCache(rows)
        for layer in rows[0].transformer_layers:
            k = torch.randn(2, 2, 1, 3)
            batched.update(k, k, layer)

        # Mutate the batched linear state as the model forward would, then
        # scatter back and confirm each row got its own slice, not row 0's.
        for layer in (0, 2):
            batched.conv_states[layer] = torch.stack(
                [torch.full((4, 3), float(i)) for i in range(2)], dim=0
            )
            batched.recurrent_states[layer] = torch.stack(
                [torch.full((4, 6), float(10 + i)) for i in range(2)], dim=0
            )
        batched.scatter_linear_state_back()

        for i, row in enumerate(rows):
            for layer in (0, 2):
                torch.testing.assert_close(row.conv_states[layer], torch.full((1, 4, 3), float(i)))
                torch.testing.assert_close(
                    row.recurrent_states[layer], torch.full((1, 4, 6), float(10 + i))
                )

    def test_rejects_multi_token_step(self):
        rows, _ = _make_rows(2)
        batched = BatchedPagedCache(rows)
        layer = rows[0].transformer_layers[0]
        k = torch.randn(2, 2, 3, 3)  # 3 new tokens -- decode should be exactly 1
        with self.assertRaises(ValueError):
            batched.update(k, k, layer)

    def test_rejects_batch_not_matching_row_count(self):
        rows, _ = _make_rows(3)
        batched = BatchedPagedCache(rows)
        layer = rows[0].transformer_layers[0]
        k = torch.randn(2, 2, 1, 3)  # only 2 rows of data for 3 rows of cache
        with self.assertRaises(ValueError):
            batched.update(k, k, layer)

    def test_rejects_rows_from_different_pools(self):
        rows_a, _ = _make_rows(1)
        rows_b, _ = _make_rows(1)
        with self.assertRaises(ValueError):
            BatchedPagedCache([rows_a[0], rows_b[0]])

    def test_free_all_rows_leaves_no_leaks(self):
        rows, pool = _make_rows(4)
        batched = BatchedPagedCache(rows)
        for layer in rows[0].transformer_layers:
            k = torch.randn(4, 2, 1, 3)
            batched.update(k, k, layer)
        pool.assert_consistent()
        for row in rows:
            row.free_sequence()
        pool.assert_consistent()
        self.assertEqual(pool.allocator.free_count, pool.num_blocks)


if __name__ == "__main__":
    unittest.main()
