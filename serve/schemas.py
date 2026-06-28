"""
serve.schemas — request/response models for the serving layer (phase 07).

`MetricsResponse` is the JSON contract the phase-08 dashboard binds to: it is the
superset of `core.scheduler.SchedulerMetrics.as_dict()` plus server-level
aggregates the engine tracks. Keep it in sync with `Engine.metrics()`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(256, gt=0)
    # NOTE: v1 sampling (temperature/top_p) is server-level config, not per-request
    # — the scheduler samples with one shared profile. Per-request sampling is a
    # documented follow-up (needs per-request params in the scheduler's _sample_next).


class MetricsResponse(BaseModel):
    # --- from SchedulerMetrics ---
    waiting_sequences: int
    active_sequences: int
    completed_sequences: int
    failed_sequences: int
    admitted_sequences: int
    evicted_sequences: int
    iterations: int
    total_generated_tokens: int
    used_blocks: int
    free_blocks: int
    max_blocks_used: int
    # --- server-level aggregates ---
    tokens_per_second: float
    last_ttft_s: float | None
    peak_vram_mb: float
    uptime_s: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    engine_alive: bool
    device: str
