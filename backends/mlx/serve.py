"""Construct the existing inferd FastAPI app with an MLX engine."""

from __future__ import annotations

import os
from math import ceil

from serve.engine import Engine


class MlxEngine(Engine):
    """Engine whose metrics report MLX unified-memory instead of CUDA VRAM."""

    def metrics(self) -> dict:
        import mlx.core as mx

        snap = super().metrics()
        # ponytail: torch.cuda peak is 0 on Mac; report the MLX allocator peak in the
        # same field — an Apple compatibility value, not an NVIDIA-comparable VRAM figure.
        snap["peak_vram_mb"] = round(mx.get_peak_memory() / 1024**2, 1)
        return snap


def build_mlx_engine_from_env() -> MlxEngine:
    from backends.mlx.backend import MlxSchedulerBackend
    from backends.mlx.runner import MlxModelRunner
    from core.scheduler import ContinuousBatchScheduler, SchedulerConfig

    artifact = os.environ.get("INFERD_MLX_ARTIFACT")
    if not artifact:
        raise RuntimeError("INFERD_MLX_ARTIFACT must name a converted local artifact")
    block_size = int(os.environ.get("INFERD_BLOCK_SIZE", "16"))
    max_model_len = int(os.environ.get("INFERD_MAX_MODEL_LEN", "4096"))
    max_concurrent = int(os.environ.get("INFERD_MAX_CONCURRENT", "4"))
    max_queue_depth = int(os.environ.get("INFERD_MAX_QUEUE_DEPTH", "32"))
    default_blocks = max_concurrent * ceil(max_model_len / block_size)
    max_blocks = int(os.environ.get("INFERD_MAX_BLOCKS", str(default_blocks)))

    runner = MlxModelRunner.load(artifact)
    scheduler = ContinuousBatchScheduler(
        MlxSchedulerBackend(runner),
        SchedulerConfig(
            max_blocks=max_blocks,
            block_size=block_size,
            max_concurrent_sequences=max_concurrent,
            max_model_len=max_model_len,
            temperature=float(os.environ.get("INFERD_TEMPERATURE", "0.0")),
            top_p=float(os.environ.get("INFERD_TOP_P", "1.0")),
            seed=int(os.environ.get("INFERD_SEED", "0")),
        ),
    )
    return MlxEngine(
        scheduler,
        runner.tokenizer,
        model_name="Qwen3-8B MLX 4-bit",
        device="Apple Metal/MLX",
        max_concurrent=max_concurrent,
        max_queue_depth=max_queue_depth,
    )


def create_mlx_app():
    from serve.app import create_app

    return create_app(engine=build_mlx_engine_from_env())
