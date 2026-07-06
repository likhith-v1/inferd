# Phase 08 — Dashboard Implementation Plan (Codex handoff)

> Companion to `plans/08_REACT_DASHBOARD_FRONTEND.md` (the phase spec). This is the concrete
> build plan for the **executor (Codex)**. Visual spec: `plans/assets/inferd-dashboard-design.html`
> (open in a browser — it is the Overview mockup). After Codex finishes, the maintainer brings
> Claude back to verify and do final polish.

## Context
Phase 08 is the last remaining piece of inferd: a live metrics dashboard — "the payoff shot of the
demo." The engine, serving layer, and `/metrics` / `/healthz` / `/generate` SSE feed all exist
(`serve/`); the UI that visualizes them does not, and `docs/demo.md` is explicitly blocked on it.
We reimplement the supplied design as a real React + Vite app under `dashboard/`, wiring every panel
to its true data source.

The design (`plans/assets/inferd-dashboard-design.html`) is the **visual spec only**. The numbers in
it are mockup values. **Hard project rule (AGENTS.md): every number must trace to a real source —
nothing faked.** The organizing principle of this build is therefore *data provenance*: each panel
is tagged **live** / **benchmark** / **rederived**, and invented mockup numbers are removed.

## Locked decisions
- **Stack:** Vite + React + **TypeScript**, `react-router-dom`, **Recharts** for data-driven charts
  (`/metrics` polls at ~1 Hz — no high-frequency jank risk), **custom SVG** for the two radial
  gauges, self-hosted fonts via `@fontsource/*` (offline constraint — the mockup's
  `fonts.googleapis.com` link would break the air-gapped demo).
- **Metrics transport:** **poll** `GET /metrics` every ~1.5 s (it is a plain JSON GET, not SSE).
  Only `/generate` is SSE.
- **No backend edits.** `serve/`, `core/`, `bench/` are untouched. CORS is absent in `serve/app.py`,
  so dev uses a **Vite dev proxy**; benchmark data is a **committed snapshot**, not a runtime call
  into `bench/`.
- **Active-sequences table + demo load:** driven by a built-in `/generate` **playground** — real
  prompts stream into the table via SSE; the dashboard generates its own under-load traffic for the
  capture.
- **Scope: build all five nav pages** (Overview, Streams, Spec decode, Memory, Benchmarks). Each maps
  to a real engine subsystem; none are invented filler.
- **Invented metrics rederived transparently:** replace mockup "health score 9.3" with a real
  **VRAM-headroom % gauge**; **drop** the fabricated delta badges (`+23% vs last run`, `−12% TTFT`,
  `+0.05`, `+19.8×` as a "delta") unless a genuine in-session prior value exists. The `+19.8×` is a
  real benchmark fact and stays, but as a benchmark-sourced figure, not a live delta.

## The data contract (provenance map)
Backend types live in `serve/schemas.py` and `serve/app.py` — mirror them exactly; fail visibly on
mismatch (a phase exit criterion).

**`GET /metrics` → `MetricsResponse`** (`serve/schemas.py:13-29`):
`waiting_sequences, active_sequences, completed_sequences, failed_sequences, admitted_sequences,
evicted_sequences, iterations, total_generated_tokens, used_blocks, free_blocks, max_blocks_used,
tokens_per_second, last_ttft_s (float|None), peak_vram_mb, uptime_s, model`.

**`GET /healthz` → `HealthResponse`** (`serve/schemas.py:32-36`):
`status ("ok"|"degraded"), model, engine_alive (bool), device`.

**`POST /generate`** (`serve/app.py:91-127`), body `GenerateRequest {prompt: str, max_tokens: int}`,
returns SSE `text/event-stream`, one JSON object per `data:` line (no `event:` field — discriminate
on `type`):
- `{"type":"token","text":"<delta>"}` — concatenate `text` for full output.
- `{"type":"done","finish_reason":"<reason>","tokens":<int>}` — terminal.
- `{"type":"error","message":"<str>"}` — terminal.
- Error statuses to surface: **503** engine unavailable (`not alive`), **400** limit violation,
  **429** saturated (retry later).

Provenance per panel:
- **LIVE (poll /metrics + /healthz):** Tokens/sec, TTFT, Peak VRAM (`/32 GiB`), Headroom
  (`vram_total − peak`), Free/Used/Max blocks, the whole **Engine-activity** block (Iterations,
  Tokens, Sequences, Blocks-used; `streaming = active_sequences`, `queued = waiting_sequences`),
  HEALTHY pill + model + device. **Do not hardcode the engine-activity numbers** — they are live.
- **BENCHMARK (committed snapshot, labeled "benchmark · report.md @ <commit>"):** Draft α 0.66 +
  acceptance gauge; throughput-vs-concurrency curve (the 19.8× hero — needs the **naive-HF baseline
  that only exists in `bench/results/`**, not recoverable from live polling); VRAM-vs-concurrency;
  spec-decode timeline (correctness PASS n=1500, Δα +0.05, honest 0.6–0.7× net caveat).
- **REDERIVED / REMOVED:** health-score → VRAM-headroom % gauge; fabricated delta badges dropped.
- **Two distinct throughput visuals — never conflate:** (a) a genuinely-live tokens/sec **sparkline**
  over wall-clock, sampled from polls; (b) the static benchmark **concurrency curve**.

### Benchmark snapshot
`dashboard/scripts/snapshot-bench.mjs` reads `bench/results/*/result.json` + parses
`bench/report.md`, emitting `dashboard/src/data/benchmarks.json` with a `sourceCommit` header. Run it
once and **commit the JSON**; the app imports the JSON only (keeps the "consumes results, not
runtime-coupled" boundary, and every benchmark number traces to a file). Numbers to capture (from the
current `bench/report.md`):
- throughput vs concurrency rungs `{1,2,4,8,16,32}` → naive HF `{29.8,58.5,104.9,116.0,24.4,23.3}`,
  ours `{44.3,78.3,141.6,236.8,356.8,461.8}`, vLLM `pending`; headline **19.80× at c=32**.
- spec-decode: stock draft α by gamma `{2:0.628, 4:0.494, 8:0.313}`, distilled
  `{2:0.684, 4:0.532, 8:0.363}`, target-only baselines ~10 tok/s, net speedup 0.6–0.7×, Δα mean
  +0.048 (up to +0.056); correctness **PASS** (multi-token per-position TV, n=1500, len=6, gamma=4).
- env: RTX 5090, 32607 MiB total VRAM (use for headroom math).

## Project setup
- `dashboard/`: `npm create vite@latest` (react-ts), then add `react-router-dom`, `recharts`,
  `@fontsource/plus-jakarta-sans`, `@fontsource/jetbrains-mono`, `@fontsource/ibm-plex-mono`, ESLint
  (`npm run lint` must pass).
- **`vite.config.ts` dev proxy** for `/metrics`, `/healthz`, `/generate` → `http://localhost:8000`.
  API base overridable via `VITE_INFERD_API` (default same-origin) for `build`/`preview`; document in
  `.env.example`.
- **`src/theme.css`** — CSS variables extracted from the mockup: bg `#05081e`; card surface
  `linear-gradient(160deg,rgba(255,255,255,.07),rgba(255,255,255,.02))`; border
  `rgba(255,255,255,.08)`; accents cyan `#21d4fd`, blue `#2152ff`/`#0075ff`, purple `#4318ff`, green
  `#2ee6a6`/`#01b574`, amber `#ffb86a`, danger `#ff7a7a`; muted text `#8b96c4`/`#6f7bab`; radii
  14–20 px; background radial-gradient stack from the mockup's root `<div>`.

## Directory layout (new — `dashboard/`)
```
dashboard/
  index.html  package.json  vite.config.ts  tsconfig.json  .eslintrc.cjs  .env.example
  scripts/snapshot-bench.mjs           # reads bench/results + report.md -> benchmarks.json
  src/
    main.tsx  App.tsx  theme.css
    data/benchmarks.json               # committed benchmark snapshot
    lib/api.ts                         # fetch helpers + MetricsResponse/HealthResponse/GenerateRequest types
    hooks/useMetrics.ts useHealth.ts useGenerate.ts   # poll + SSE, loading/error/backoff
    components/  AppShell Sidebar TopBar KpiCard HeroPanel
                 RadialGauge(SVG) ThroughputChart LiveSparkline EngineActivity
                 ActiveSequencesTable SpecDecodeTimeline ConnectionState Playground
    pages/  Overview Streams SpecDecode Memory Benchmarks
```

## Page mapping (all five, all data-traced)
- **Overview** — the supplied design faithfully: live KPI row, live engine-activity, benchmark
  hero/acceptance/throughput panels, VRAM-headroom gauge, playground-driven active-sequences table,
  spec-decode timeline. The mockup's three "connection states" become the **real** loading / idle /
  disconnected UI states.
- **Streams** — the playground in depth: submit prompts, watch multiple concurrent SSE streams,
  per-stream incremental tokens + live TTFT/tokens (this is what drives Overview's table).
- **Spec decode** — benchmark JSON: α-vs-gamma (stock vs distilled), correctness PASS, Δα +0.05,
  honest net-throughput caveat.
- **Memory** — paged KV-cache: live used/free/max blocks + a block-occupancy grid, live VRAM
  headroom, benchmark VRAM-vs-concurrency curve.
- **Benchmarks** — full throughput-vs-concurrency table + curve (naive HF / ours / vLLM "pending"),
  VRAM-vs-concurrency, headline 19.8×, all from the snapshot.

## Accessibility & responsive (phase exit criteria)
- Keyboard nav for sidebar/playground; visible focus rings; ARIA on charts/gauges (`role="img"` +
  `aria-label` carrying the value); contrast pass on muted text; `prefers-reduced-motion` disables
  the blink/shimmer animations.
- Grid collapses at tablet/mobile; sidebar becomes a drawer; charts stay readable.
- Graceful loading / idle / disconnected states when `/metrics` is down, with reconnect backoff — so
  the dashboard is demoable even with no live engine, falling back to benchmark panels + a clear
  "engine offline" banner.

## Verification (run before handing back)
1. `cd dashboard && npm install && npm run lint && npm run build` — clean lint + build.
2. `npm run dev`; with engine up (`uv run uvicorn serve.app:app --port 8000`) confirm KPIs,
   engine-activity, health pill update live; submit a playground prompt and watch tokens stream into
   Streams + the active-sequences table.
3. Generate under-load traffic (playground or `bench/` harness) and confirm the live sparkline +
   sequence counts move without jank.
4. With the engine **down**, confirm disconnected/idle states render and benchmark panels still show
   (provenance label visible).
5. The Overview "α + throughput" hero is the screen-capture `docs/demo.md` is waiting on.
6. Note: a full live run needs the RTX 5090 / WSL2 box; a non-GPU box can still validate build, lint,
   routing, and the offline/disconnected states.

## Out of scope / house rules
- No backend edits (`serve/`, `core/`, `bench/` untouched — proxy + committed snapshot instead).
- No auth; no deploy beyond `vite dev` / `build` / `preview`.
- Per AGENTS.md: **leave uncommitted** — the maintainer (`likhith-v1`) commits and merges. Suggested
  worktree `phase-08-dashboard` / `../inferd-wt/08-dash`. Never commit `node_modules/` (add
  `dashboard/node_modules/` to `.gitignore`).
