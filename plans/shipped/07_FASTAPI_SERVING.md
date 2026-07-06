Preferred model: Claude Sonnet 4.6 | Effort: high

# 07 — FastAPI Serving Layer

> Wrap the engine in an async FastAPI service — request queue into the iteration scheduler, SSE token streaming, and `/metrics` / `/healthz` — without making the core depend on HTTP (the harness must keep working headless).

## Constraints (this phase)
local-first · async FastAPI + uvicorn · SSE token streaming · core stays **importable & headless-benchmarkable** (HTTP never required for measurement) · text-only · `/metrics` exposes the live engine stats 08 consumes.

## Scope
**In:** FastAPI app; async request queue feeding the 06 scheduler; `/generate` (SSE), `/metrics`, `/healthz`; token streaming; graceful backpressure when the engine is saturated.
**Out:** the dashboard (08); auth/multi-tenant (local single-user); HTTPS.
**Standalone value:** "an async serving layer streaming tokens from a from-scratch engine, with live metrics."

## Subagent breakdown
- **api-endpoint author** — `serve/app.py`: routes, request/response models, queue submission, lifecycle (startup loads engine, shutdown drains).
- **sse-streaming author** — `serve/stream.py`: per-request token stream, client-disconnect cleanup (cancel + free blocks).
- **integration-tester** — end-to-end: HTTP request → streamed tokens; `/metrics` shape; confirm the headless harness path is untouched.

## Git worktree workflow
- Branch `phase-07-serving`, worktree `../inferd-wt/07-serve`. Needs 06 (scheduler). Imports `core/` only — does **not** edit engine internals.

## Owned / Avoided files
- **Owns:** `serve/app.py`, `serve/stream.py`, `serve/schemas.py`, `tests/test_serve.py`.
- **Avoids:** `core/` internals (import only), `dashboard/`, `bench/harness.py`.

## Commands, tests, benchmarks, validation
```bash
uv sync
uv run uvicorn serve.app:app --host 0.0.0.0 --port 8000
curl -N -X POST localhost:8000/generate -d '{"prompt":"hi","max_tokens":64}'   # streams tokens
curl localhost:8000/metrics    # tokens/sec, TTFT, α, VRAM, active seqs, throughput
curl localhost:8000/healthz
uv run python -m pytest tests/test_serve.py        # e2e stream + metrics shape
uv run python -m bench.harness --engine batched    # proves core still runs headless (no HTTP)
```
- **Validation:** tokens stream incrementally (not buffered); client disconnect frees the sequence's blocks; `/metrics` matches the schema 08 expects; the headless harness still runs unchanged; `security-review` on the request-handling path.

## Risks / Rollback / Exit / Handoff
- **Risks:** async/event-loop blocking on GPU calls; disconnects leaking KV blocks; coupling the core to HTTP and breaking headless benchmarking.
- **Mitigation/Rollback:** run engine steps off the event loop (executor/thread or a dedicated loop); explicit on-disconnect cancellation → free blocks; keep all engine logic in `core/`, `serve/` is a thin adapter; if SSE is flaky, fall back to chunked HTTP and note it.
- **Exit:** `/generate` streams from the live engine; `/metrics` + `/healthz` up; headless harness unaffected.
- **Handoff:** `/metrics` schema is the contract 08 binds to; the running server is the demo target for 09/11.

## Model Selection (confirm or override)
- **Claude Sonnet 4.6 | high** *(recommended)* — async serving + streaming glue, cost-efficient and reliable here.
- **GPT-5.4 | high** — co-equal for API plumbing.
- **Claude Opus 4.8 | high** — only if the async/GPU-offload concurrency model needs deeper reasoning.
> Recommendation: Sonnet 4.6 high. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** `security-review` (request-handling path), `verify`/`run` (drive the server end-to-end).
- **Candidates:** firecrawl for FastAPI/SSE/uvicorn streaming docs.
- **Question:** make `security-review` a blocking gate for the serving merge? Use `run` to capture the live demo for 11?

## Execution questions for this phase
1. SSE vs WebSocket for streaming (plan mentions both)? SSE is simpler for one-way tokens — confirm.
2. `/metrics` shape: lock the JSON schema now so 08 can build against it.
3. Backpressure when saturated: 429, queue-with-wait, or reject? Max queue depth?
4. Single served model at a time, or hot-swap target/draft/FP8 via config?
