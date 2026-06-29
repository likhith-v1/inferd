"""Serving request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(256, gt=0)


class MetricsResponse(BaseModel):
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
