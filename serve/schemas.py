"""Serving request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ponytail: DoS ceiling on untrusted network input, not the engine's token budget.
# The prompt is tokenized synchronously on the event loop before limit_violation()
# can see its length, so cap the raw bytes here (Pydantic rejects with 422 before
# the handler buffers/tokenizes). 1M chars sits well above any realistic context
# (~131K tokens worst-case at 1 char/token); over-budget prompts still get a clean
# 400 from limit_violation. Raise if a genuine >1M-char context is ever configured.
MAX_PROMPT_CHARS = 1_000_000


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    max_tokens: int = Field(256, gt=0)
    temperature: float | None = Field(None, ge=0.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)


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
