"""
CPU unit tests for core.batched_cache stack/split surgery.

These validate the ragged left-pad / trim math deterministically without a GPU or
a model, using a fake cache that mimics Qwen3_5DynamicCache's structure (the four
state lists + get_seq_length + transformer_layers).
"""

import copy
import unittest

import torch

from core.batched_cache import split_caches, stack_caches

H, D = 2, 3  # heads, head_dim for the fake full-attention K/V


class FakeCache:
    """Minimal stand-in for Qwen3_5DynamicCache (hybrid full + linear layers)."""

    def __init__(self, layer_types, length, fill):
        n = len(layer_types)
        self.layer_types = layer_types
        self.transformer_layers = [i for i, t in enumerate(layer_types) if t == "full"]
        self.key_cache = [None] * n
        self.value_cache = [None] * n
        self.conv_states = [None] * n
        self.recurrent_states = [None] * n
        for i, t in enumerate(layer_types):
            if t == "full":
                self.key_cache[i] = torch.full((1, H, length, D), float(fill))
                self.value_cache[i] = torch.full((1, H, length, D), float(fill) + 0.5)
            else:
                self.conv_states[i] = torch.full((1, 4, 3), float(fill) + 1.0)
                self.recurrent_states[i] = torch.full((1, 2, 3, 3), float(fill) + 2.0)

    def get_seq_length(self, layer_idx=0):
        return self.key_cache[self.transformer_layers[0]].shape[-2]


LAYER_TYPES = ["full", "linear", "full", "linear"]


class BatchedCacheTest(unittest.TestCase):
    def test_stack_then_split_is_identity(self):
        lengths = [3, 5, 2]
        caches = [FakeCache(LAYER_TYPES, length=L, fill=i + 1) for i, L in enumerate(lengths)]
        originals = [copy.deepcopy(c) for c in caches]

        batched, got_lengths = stack_caches(caches)
        self.assertEqual(got_lengths, lengths)

        # Batched full-attn K/V is left-padded to max length.
        self.assertEqual(batched.key_cache[0].shape, (3, H, 5, D))
        # Linear states stack on the batch dim, untouched.
        self.assertEqual(batched.conv_states[1].shape, (3, 4, 3))

        recovered = split_caches(batched, lengths)
        self.assertEqual(len(recovered), 3)
        for orig, rec in zip(originals, recovered):
            for layer, t in enumerate(LAYER_TYPES):
                if t == "full":
                    self.assertTrue(torch.equal(orig.key_cache[layer], rec.key_cache[layer]))
                    self.assertTrue(torch.equal(orig.value_cache[layer], rec.value_cache[layer]))
                else:
                    self.assertTrue(torch.equal(orig.conv_states[layer], rec.conv_states[layer]))
                    self.assertTrue(
                        torch.equal(orig.recurrent_states[layer], rec.recurrent_states[layer])
                    )

    def test_left_padding_places_real_tokens_on_the_right(self):
        caches = [FakeCache(LAYER_TYPES, length=2, fill=7), FakeCache(LAYER_TYPES, length=4, fill=9)]
        batched, _ = stack_caches(caches)
        k = batched.key_cache[0]  # [2, H, 4, D]
        # Row 0 had length 2 -> first 2 columns are pad (zeros), last 2 are real.
        self.assertTrue(torch.equal(k[0, :, :2, :], torch.zeros(H, 2, D)))
        self.assertTrue(torch.equal(k[0, :, 2:, :], torch.full((H, 2, D), 7.0)))
        # Row 1 had length 4 -> no padding.
        self.assertTrue(torch.equal(k[1], torch.full((H, 4, D), 9.0)))

    def test_split_after_growth_trims_correct_left_pad(self):
        # Simulate the post-decode state: each row grew by one token.
        lengths = [3, 5, 2]
        caches = [FakeCache(LAYER_TYPES, length=L, fill=i + 1) for i, L in enumerate(lengths)]
        batched, _ = stack_caches(caches)
        # Append one shared "new token" column on the right of the padded tensor.
        for layer, t in enumerate(LAYER_TYPES):
            if t == "full":
                b = batched.key_cache[layer].shape[0]
                col = torch.full((b, H, 1, D), -1.0)
                batched.key_cache[layer] = torch.cat([batched.key_cache[layer], col], dim=-2)
                batched.value_cache[layer] = torch.cat([batched.value_cache[layer], col], dim=-2)

        recovered = split_caches(batched, [L + 1 for L in lengths])
        for L, rec in zip(lengths, recovered):
            # Recovered length is old+1, and the trailing token is the appended -1.
            self.assertEqual(rec.key_cache[0].shape, (1, H, L + 1, D))
            self.assertTrue(torch.equal(rec.key_cache[0][0, :, -1, :], torch.full((H, D), -1.0)))


if __name__ == "__main__":
    unittest.main()
