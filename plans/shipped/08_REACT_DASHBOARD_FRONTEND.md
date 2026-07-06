Preferred model: Claude Sonnet 4.6 | Effort: high

# 08 — React + Vite Dashboard (frontend)

> A live metrics dashboard — tokens/sec, TTFT, α, VRAM, concurrency, throughput curves streaming from `/metrics` — that is the payoff shot of the demo. **Model is hard-pinned to Claude Sonnet 4.6** per project rule for UX, layout, responsive design, metrics visualization, accessibility, and polish.

## Constraints (this phase)
local-first · React + Vite · Recharts/uPlot for live streams · binds to 07's `/metrics` (SSE/WebSocket) · responsive + accessible baseline · the live α + throughput curves are the headline visual.

## Scope
**In:** dashboard UI — live tokens/sec, TTFT, α (draft acceptance), VRAM utilization, concurrent requests, throughput-vs-concurrency curves; live-updating charts; responsive layout; a11y baseline; polish.
**Out:** backend (`serve/`, `core/`); auth; build/deploy beyond local `vite dev`/`build`.
**Standalone value:** "a live dashboard visualizing a from-scratch inference engine under load."

## Subagent breakdown
- **layout / UX author** — information hierarchy, the metric grid, the hero throughput/α panel; intentional visual direction (not templated defaults).
- **live-charts author** — Recharts/uPlot streaming bindings to `/metrics`; smooth high-frequency updates without jank.
- **a11y / responsive / polish author** — keyboard/contrast/ARIA baseline, breakpoints, loading/empty/error states.

## Git worktree workflow
- Branch `phase-08-dashboard`, worktree `../inferd-wt/08-dash`. Needs 07's `/metrics` schema. Pure frontend — touches no backend files.

## Owned / Avoided files
- **Owns:** `dashboard/` (Vite app, components, hooks).
- **Avoids:** `serve/`, `core/`, `bench/` (consumes `/metrics` over HTTP only).

## Commands, tests, validation
```bash
cd dashboard && npm install
npm run dev          # against a running serve.app on :8000
npm run build && npm run preview
npm run lint
# manual: load under bench load, watch α + throughput update live
```
- **Validation:** charts update live under load without jank; layout holds at mobile/desktop widths; a11y baseline (keyboard nav, contrast, ARIA on charts); graceful empty/error states when `/metrics` is down; the α + throughput hero panel reads clearly in a screen-capture.

## Risks / Rollback / Exit / Handoff
- **Risks:** high-frequency metric updates causing re-render jank; chart lib churn; `/metrics` schema drift from 07.
- **Mitigation/Rollback:** throttle/batch updates, use uPlot if Recharts struggles at frequency; pin the chart lib; treat 07's `/metrics` schema as a contract and fail visibly on mismatch; if WebSocket is flaky, poll `/metrics` on an interval.
- **Exit:** live dashboard rendering every headline metric from `/metrics`, responsive + a11y baseline, demo-ready.
- **Handoff:** the dashboard-under-load screen capture is the payoff shot for 11's writeup/demo video.

## Model Selection (HARD-PINNED)
- **Claude Sonnet 4.6** — **required** for this phase (UX, layout, responsive, metrics viz, a11y, polish), per project rule. Only the **effort level** is open:
  - `Effort: high` *(recommended)* — full polish + a11y + responsive pass.
  - `Effort: medium` — if scope is trimmed to a single-screen dashboard.
> Other models are **not** options here. **Confirm the effort level.**

## Skills & MCPs to decide at implementation
- **Recommend:** the `frontend-design` skill (mandated lens for distinctive, intentional visual design).
- **Candidates:** firecrawl for Recharts/uPlot docs; `verify`/`run` to screenshot the dashboard under load.
- **Question:** Recharts (ergonomic) vs uPlot (fast at high frequency) — decide based on update rate. Use `frontend-design` from the first commit?

## Execution questions for this phase
1. Recharts vs uPlot — driven by the `/metrics` update frequency from 07.
2. SSE vs WebSocket consumption (match 07's choice).
3. Single-page dashboard, or tabs (single-stream vs concurrency views)?
4. Visual direction: dark "ops console" vs light analytical — any brand/color preference?
