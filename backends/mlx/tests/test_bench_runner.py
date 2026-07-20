from pathlib import Path

import pytest

from bench.runners.mlx import _apple_results_dir


def test_results_path_cannot_escape_apple_subtree(tmp_path: Path):
    with pytest.raises(ValueError, match="bench/results/apple"):
        _apple_results_dir(tmp_path)
