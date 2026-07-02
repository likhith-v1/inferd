"""
serve.app — FastAPI serving layer over the continuous-batching engine (phase 07).

Routes:
  POST /generate  -> SSE token stream (text/event-stream)
  GET  /metrics   -> MetricsResponse (the contract phase-08 binds to)
  GET  /healthz   -> HealthResponse

The core never depends on HTTP: this module imports core/ and serve.engine only;
the headless bench.harness path is untouched. `create_app(engine=...)` is an app
factory so tests can inject a fake engine and exercise the whole HTTP/SSE/metrics
surface without a GPU.

    uv run uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from math import ceil

import inferd.env  # noqa: F401  (CUDA preload before torch)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from serve.engine import Done, Engine, Error, TokenChunk  # noqa: E402
from serve.schemas import GenerateRequest, HealthResponse, MetricsResponse  # noqa: E402


def build_engine_from_env() -> Engine:
    """Construct the real Engine (loads the model). Called from app startup."""
    from core.model_runner import ModelRunner
    from core.scheduler import ContinuousBatchScheduler, ModelRunnerBackend, SchedulerConfig

    model = os.environ.get("INFERD_MODEL", "merged/9b")
    device = os.environ.get("INFERD_DEVICE", "cuda:0")
    block_size = int(os.environ.get("INFERD_BLOCK_SIZE", "16"))
    max_model_len = int(os.environ.get("INFERD_MAX_MODEL_LEN", "4096"))
    max_concurrent = int(os.environ.get("INFERD_MAX_CONCURRENT", "8"))
    max_queue_depth = int(os.environ.get("INFERD_MAX_QUEUE_DEPTH", "32"))
    temperature = float(os.environ.get("INFERD_TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("INFERD_TOP_P", "1.0"))
    seed = int(os.environ.get("INFERD_SEED", "0"))
    default_blocks = max_concurrent * ceil(max_model_len / block_size)
    max_blocks = int(os.environ.get("INFERD_MAX_BLOCKS", str(default_blocks)))

    runner = ModelRunner.load_target(model, device=device)
    backend = ModelRunnerBackend(runner)
    scheduler = ContinuousBatchScheduler(
        backend,
        SchedulerConfig(
            max_blocks=max_blocks,
            block_size=block_size,
            max_concurrent_sequences=max_concurrent,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ),
    )
    return Engine(
        scheduler,
        runner.tokenizer,
        model_name=str(model),
        device=device,
        max_concurrent=max_concurrent,
        max_queue_depth=max_queue_depth,
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine if engine is not None else build_engine_from_env()
        eng.start()
        app.state.engine = eng
        try:
            yield
        finally:
            eng.stop()

    app = FastAPI(title="inferd", version="0.7.0", lifespan=lifespan)

    @app.post("/generate")
    async def generate(body: GenerateRequest, request: Request):
        eng: Engine = request.app.state.engine
        if not eng.alive:
            # Engine thread died (e.g. a fatal error in step()); without this gate
            # the request would submit into an undrained inbox and hang forever.
            raise HTTPException(status_code=503, detail="engine unavailable")
        prompt_ids = eng.encode(body.prompt)
        violation = eng.limit_violation(len(prompt_ids), body.max_tokens)
        if violation:
            raise HTTPException(status_code=400, detail=violation)
        try:
            channel = eng.submit(prompt_ids, body.max_tokens)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if channel is None:
            raise HTTPException(status_code=429, detail="engine saturated; retry later")

        async def event_stream():
            try:
                while True:
                    item = await channel.queue.get()
                    if isinstance(item, TokenChunk):
                        yield _sse({"type": "token", "text": item.text})
                    elif isinstance(item, Done):
                        yield _sse({
                            "type": "done",
                            "finish_reason": item.finish_reason,
                            "tokens": item.generated_tokens,
                        })
                        return
                    elif isinstance(item, Error):
                        yield _sse({"type": "error", "message": item.message})
                        return
            finally:
                # client disconnect or normal end: free the sequence's blocks
                # (idempotent: a no-op once the request is already terminal).
                eng.cancel(channel.request_id)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics(request: Request):
        return request.app.state.engine.metrics()

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request):
        eng: Engine = request.app.state.engine
        return HealthResponse(
            status="ok" if eng.alive else "degraded",
            model=eng.model_name,
            engine_alive=eng.alive,
            device=eng.device,
        )

    return app


# Module-level ASGI app for `uvicorn serve.app:app`.
app = create_app()
