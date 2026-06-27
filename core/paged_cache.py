"""
core.paged_cache -- fixed-size block allocator and page table for KV tensors.

Phase 05 starts with the cache data structure in isolation: blocks are allocated
from a free pool, each sequence owns a logical page table, and K/V tensors can be
gathered back into contiguous form for numerical-equivalence tests. Model-layer
integration stays behind the opaque `kv` contract in core.model_runner.

Tensor layout is intentionally explicit:
    key_cache[layer] / value_cache[layer]:
        [num_blocks, num_kv_heads, block_size, head_dim]

Appending accepts per-layer tensors in logical token order:
    layer_keys[layer]:
        [num_new_tokens, num_kv_heads, head_dim]
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field

import torch


class BlockAllocator:
    """LIFO allocator for fixed-size KV blocks."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._allocated: set[int] = set()

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def allocated_count(self) -> int:
        return len(self._allocated)

    def allocate(self, n: int = 1) -> list[int]:
        if n < 0:
            raise ValueError("cannot allocate a negative number of blocks")
        if n > len(self._free):
            raise MemoryError(f"requested {n} blocks, only {len(self._free)} free")
        blocks = [self._free.pop() for _ in range(n)]
        self._allocated.update(blocks)
        return blocks

    def free(self, blocks: list[int]) -> None:
        seen: set[int] = set()
        for block in blocks:
            if block in seen:
                raise ValueError(f"duplicate block in free request: {block}")
            if block not in self._allocated:
                raise ValueError(f"block is not allocated: {block}")
            seen.add(block)
        for block in blocks:
            self._allocated.remove(block)
            self._free.append(block)


@dataclass
class SequencePageTable:
    """Logical sequence metadata consumed by phase 06 admission/eviction."""

    seq_id: int
    block_size: int
    blocks: list[int] = field(default_factory=list)
    length: int = 0

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def capacity_tokens(self) -> int:
        return len(self.blocks) * self.block_size

    @property
    def free_slots(self) -> int:
        return self.capacity_tokens - self.length

    def block_for_pos(self, pos: int) -> tuple[int, int]:
        if pos < 0 or pos >= self.length:
            raise IndexError(f"position {pos} outside sequence length {self.length}")
        page_idx, offset = divmod(pos, self.block_size)
        return self.blocks[page_idx], offset


class PagedKVCache:
    """
    Per-layer K/V block store plus per-sequence page tables.

    This class stores only full-attention KV tensors. Qwen3.5's linear-attention
    recurrent states are separate fixed-size state, not per-position KV pages.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int = 16,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("num_kv_heads and head_dim must be positive")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.allocator = BlockAllocator(num_blocks)
        self.sequences: dict[int, SequencePageTable] = {}

        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self.key_cache = [
            torch.empty(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.empty(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]

    def create_sequence(self, seq_id: int) -> SequencePageTable:
        if seq_id in self.sequences:
            raise ValueError(f"sequence already exists: {seq_id}")
        table = SequencePageTable(seq_id=seq_id, block_size=self.block_size)
        self.sequences[seq_id] = table
        return table

    def free_sequence(self, seq_id: int) -> None:
        table = self._table(seq_id)
        self.allocator.free(table.blocks)
        del self.sequences[seq_id]

    def append_tokens(
        self,
        seq_id: int,
        layer_keys: list[torch.Tensor],
        layer_values: list[torch.Tensor],
    ) -> None:
        """
        Append one or more logical tokens for every layer.

        `layer_keys` and `layer_values` must each contain `num_layers` tensors
        shaped [num_tokens, num_kv_heads, head_dim]. The page table is extended
        before writes, so allocation failure leaves the cache unchanged.
        """
        table = self._table(seq_id)
        self._validate_layers(layer_keys, layer_values)
        n_tokens = int(layer_keys[0].shape[0])
        if n_tokens == 0:
            return

        needed_capacity = table.length + n_tokens
        needed_blocks = (needed_capacity + self.block_size - 1) // self.block_size
        extra_blocks = needed_blocks - len(table.blocks)
        new_blocks = self.allocator.allocate(extra_blocks) if extra_blocks > 0 else []

        try:
            table.blocks.extend(new_blocks)
            for rel_pos in range(n_tokens):
                abs_pos = table.length + rel_pos
                page_idx, offset = divmod(abs_pos, self.block_size)
                block = table.blocks[page_idx]
                for layer in range(self.num_layers):
                    self.key_cache[layer][block, :, offset, :] = layer_keys[layer][rel_pos]
                    self.value_cache[layer][block, :, offset, :] = layer_values[layer][rel_pos]
            table.length += n_tokens
        except Exception:
            if new_blocks:
                del table.blocks[-len(new_blocks):]
                self.allocator.free(new_blocks)
            raise

    def gather_layer(self, seq_id: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous K/V tensors shaped [seq_len, num_kv_heads, head_dim]."""
        table = self._table(seq_id)
        self._check_layer(layer)
        keys = torch.empty(
            (table.length, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        values = torch.empty_like(keys)
        for pos in range(table.length):
            block, offset = table.block_for_pos(pos)
            keys[pos] = self.key_cache[layer][block, :, offset, :]
            values[pos] = self.value_cache[layer][block, :, offset, :]
        return keys, values

    def block_table(self, seq_id: int, *, device: str | torch.device | None = None) -> torch.Tensor:
        table = self._table(seq_id)
        out_device = self.device if device is None else device
        return torch.tensor(table.blocks, dtype=torch.int64, device=out_device)

    def sequence_length(self, seq_id: int) -> int:
        return self._table(seq_id).length

    def _table(self, seq_id: int) -> SequencePageTable:
        try:
            return self.sequences[seq_id]
        except KeyError:
            raise KeyError(f"unknown sequence: {seq_id}") from None

    def _check_layer(self, layer: int) -> None:
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(f"layer {layer} outside [0, {self.num_layers})")

    def _validate_layers(
        self, layer_keys: list[torch.Tensor], layer_values: list[torch.Tensor]
    ) -> None:
        if len(layer_keys) != self.num_layers or len(layer_values) != self.num_layers:
            raise ValueError("layer key/value lists must match num_layers")
        expected_shape = layer_keys[0].shape
        if len(expected_shape) != 3:
            raise ValueError("layer tensors must be [tokens, kv_heads, head_dim]")
        if expected_shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"expected [tokens, {self.num_kv_heads}, {self.head_dim}], "
                f"got {tuple(expected_shape)}"
            )
        for tensor in [*layer_keys, *layer_values]:
            if tensor.shape != expected_shape:
                raise ValueError("all layer tensors must have the same shape")
            if tensor.device != self.device:
                raise ValueError(f"tensor device {tensor.device} != cache device {self.device}")

    def assert_consistent(self) -> None:
        owned = [block for table in self.sequences.values() for block in table.blocks]
        if len(owned) != len(set(owned)):
            raise AssertionError("a physical block is owned by multiple sequences")
        if len(owned) != self.allocator.allocated_count:
            raise AssertionError("allocator/table allocation counts disagree")
        for block in owned:
            if block < 0 or block >= self.num_blocks:
                raise AssertionError(f"invalid block id: {block}")


@dataclass
class PagedHybridCache:
    """
    Paged representation of Qwen3.5's hybrid cache.

    Full-attention K/V tensors are stored in `attention`; linear-attention
    conv/recurrent states are preserved as fixed-size tensors because they are
    not per-position KV and cannot be paged.
    """

    attention: PagedKVCache
    layer_to_cache: dict[int, int]
    conv_states: list[torch.Tensor | None]
    recurrent_states: list[torch.Tensor | None]
    layer_types: list[str] | None
    transformer_layers: list[int]
    seq_id: int = 0

    @classmethod
    def from_qwen_cache(
        cls,
        kv,
        *,
        block_size: int = 16,
        seq_id: int = 0,
    ) -> "PagedHybridCache":
        """
        Convert a batch=1 Qwen3.5 dynamic cache into paged attention blocks.

        The returned object can be converted back with `to_qwen_cache_like` for a
        lossless cache round-trip equivalence check.
        """
        transformer_layers = list(getattr(kv, "transformer_layers", []))
        if not transformer_layers:
            transformer_layers = [
                i for i, k in enumerate(kv.key_cache)
                if k is not None and k.ndim == 4 and k.shape[-2] > 0
            ]
        if not transformer_layers:
            raise ValueError("cache contains no full-attention layers")

        first = kv.key_cache[transformer_layers[0]]
        if first.shape[0] != 1:
            raise ValueError("PagedHybridCache currently supports batch=1 caches only")
        _, num_kv_heads, seq_len, head_dim = first.shape
        num_blocks = (seq_len + block_size - 1) // block_size
        attention = PagedKVCache(
            num_layers=len(transformer_layers),
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=first.dtype,
            device=first.device,
        )
        attention.create_sequence(seq_id)

        layer_keys = []
        layer_values = []
        layer_to_cache = {}
        for cache_layer, original_layer in enumerate(transformer_layers):
            key = kv.key_cache[original_layer]
            value = kv.value_cache[original_layer]
            if key is None or value is None:
                raise ValueError(f"missing key/value cache for layer {original_layer}")
            if key.shape != value.shape:
                raise ValueError(f"key/value shape mismatch at layer {original_layer}")
            if key.shape[0] != 1:
                raise ValueError("PagedHybridCache currently supports batch=1 caches only")
            if key.shape[1:] != (num_kv_heads, seq_len, head_dim):
                raise ValueError("all attention layers must share KV shape")
            layer_to_cache[original_layer] = cache_layer
            layer_keys.append(key[0].transpose(0, 1).contiguous())
            layer_values.append(value[0].transpose(0, 1).contiguous())
        attention.append_tokens(seq_id, layer_keys, layer_values)

        return cls(
            attention=attention,
            layer_to_cache=layer_to_cache,
            conv_states=_clone_optional_tensors(kv.conv_states),
            recurrent_states=_clone_optional_tensors(kv.recurrent_states),
            layer_types=list(getattr(kv, "layer_types", [])) or None,
            transformer_layers=transformer_layers,
            seq_id=seq_id,
        )

    def to_qwen_cache_like(self, template):
        """
        Reconstruct an HF-style cache object with the same class/attributes as
        `template`. This is intended for equivalence validation and handoff, not
        as the final paged runtime cache.
        """
        out = copy.copy(template)
        out.conv_states = _clone_optional_tensors(self.conv_states)
        out.recurrent_states = _clone_optional_tensors(self.recurrent_states)
        out.key_cache = [None for _ in template.key_cache]
        out.value_cache = [None for _ in template.value_cache]
        if hasattr(template, "layer_types"):
            out.layer_types = list(template.layer_types)
        if hasattr(template, "transformer_layers"):
            out.transformer_layers = list(template.transformer_layers)
        if hasattr(template, "last_linear_layer"):
            out.last_linear_layer = template.last_linear_layer

        for original_layer, cache_layer in self.layer_to_cache.items():
            key, value = self.attention.gather_layer(self.seq_id, cache_layer)
            out.key_cache[original_layer] = key.transpose(0, 1).unsqueeze(0).contiguous()
            out.value_cache[original_layer] = value.transpose(0, 1).unsqueeze(0).contiguous()
        return out


def _clone_optional_tensors(items):
    return [item.clone() if item is not None else None for item in items]


def _selfcheck() -> None:
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=8,
        block_size=4,
        num_kv_heads=3,
        head_dim=5,
        dtype=torch.float32,
    )
    cache.create_sequence(10)
    keys = [
        torch.arange(7 * 3 * 5, dtype=torch.float32).reshape(7, 3, 5) + layer * 10_000
        for layer in range(2)
    ]
    values = [k + 0.25 for k in keys]
    cache.append_tokens(10, keys, values)
    cache.assert_consistent()
    assert cache.sequence_length(10) == 7
    assert cache.allocator.allocated_count == 2
    for layer in range(2):
        got_k, got_v = cache.gather_layer(10, layer)
        torch.testing.assert_close(got_k, keys[layer])
        torch.testing.assert_close(got_v, values[layer])

    cache.create_sequence(11)
    one = [torch.ones(1, 3, 5) for _ in range(2)]
    cache.append_tokens(11, one, one)
    cache.free_sequence(10)
    cache.free_sequence(11)
    cache.assert_consistent()
    assert cache.allocator.free_count == cache.num_blocks

    class _FakeQwenCache:
        pass

    fake = _FakeQwenCache()
    fake.layer_types = ["full_attention", "linear_attention", "full_attention"]
    fake.transformer_layers = [0, 2]
    fake.last_linear_layer = 1
    fake.key_cache = [torch.randn(1, 2, 9, 4), None, torch.randn(1, 2, 9, 4)]
    fake.value_cache = [torch.randn(1, 2, 9, 4), None, torch.randn(1, 2, 9, 4)]
    fake.conv_states = [None, torch.randn(1, 8, 3), None]
    fake.recurrent_states = [None, torch.randn(1, 8, 16), None]
    paged = PagedHybridCache.from_qwen_cache(fake, block_size=4)
    roundtrip = paged.to_qwen_cache_like(fake)
    for layer in fake.transformer_layers:
        torch.testing.assert_close(roundtrip.key_cache[layer], fake.key_cache[layer])
        torch.testing.assert_close(roundtrip.value_cache[layer], fake.value_cache[layer])
    torch.testing.assert_close(roundtrip.conv_states[1], fake.conv_states[1])
    torch.testing.assert_close(roundtrip.recurrent_states[1], fake.recurrent_states[1])
    print("[paged_cache] selfcheck PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
