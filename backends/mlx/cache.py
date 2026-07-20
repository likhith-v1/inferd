"""mlx-lm architecture-aware cache merge/extract adapter."""

from __future__ import annotations


def merge_caches(caches: list[list[object]]) -> list[object]:
    if not caches:
        raise ValueError("merge_caches requires at least one cache")
    if any(len(cache) != len(caches[0]) for cache in caches):
        raise ValueError("MLX caches have different layer counts")
    merged = []
    for layers in zip(*caches):
        if any(type(layer) is not type(layers[0]) for layer in layers):
            raise TypeError("MLX cache layer types differ")
        merge = getattr(type(layers[0]), "merge", None)
        if merge is None:
            raise RuntimeError(
                f"{type(layers[0]).__name__} has no mlx-lm batch merge; use concurrency=1"
            )
        merged.append(merge(list(layers)))
    return merged


def extract_caches(cache: list[object], batch_size: int) -> list[list[object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = [[] for _ in range(batch_size)]
    for layer in cache:
        extract = getattr(layer, "extract", None)
        if extract is None:
            raise RuntimeError(
                f"{type(layer).__name__} has no mlx-lm batch extract; use concurrency=1"
            )
        for index, row in enumerate(rows):
            row.append(extract(index))
    return rows
