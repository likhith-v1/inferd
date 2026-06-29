# Demo capture checklist

The payoff shot for the repo: the **dashboard under load**, with the draft-acceptance
rate (α) and throughput-vs-concurrency curves moving in real time. This file is the
capture script so the recording is reproducible and the numbers on screen match
`bench/results/`.

> **Blocked on phase 08 (dashboard).** The engine, serving layer, and `/metrics` SSE
> feed all exist; the live UI that visualizes them does not yet. Record once the
> dashboard is up. Until then, the static plots in `bench/results/plots/` are the
> portfolio artifact.

## What to show (in order)

1. **Cold start → healthy.** `GET /healthz` green; model name + device visible.
2. **Single stream.** One `POST /generate`; tokens stream incrementally (not one
   buffered blob); TTFT and tokens/sec settle on the dashboard.
3. **Ramp concurrency.** Fire 1 → 4 → 8 → 16 → 32 concurrent requests; the
   throughput curve climbs while naive HF (overlaid reference) collapses past c=8.
   This is the **19.8× at c=32** headline, live.
4. **Acceptance rate.** Toggle speculative decoding on; α settles ≈0.63–0.68; show
   the honest caveat that net throughput is 0.6–0.7× (correctness + α-lift are the
   wins, not raw speed).
5. **VRAM headroom.** Peak VRAM vs concurrency stays within the 32 GB card.

## Setup

```bash
# 1. Serve the 9B target through the engine
INFERD_MODEL=/home/likhi/inferd/merged/9b \
  uv run uvicorn serve.app:app --port 8000

# 2. (separate shell) start the dashboard dev server  [phase 08]
#    cd dashboard && npm run dev   # points at http://localhost:8000

# 3. Drive load — reuse the frozen benchmark workload so on-screen numbers
#    match bench/results/ exactly
uv run python bench/run_all.py --rungs hf,ours --concurrency 1,4,8,16,32 --plots
```

## Recording

- **Length:** 60–90s. Audience: recruiters/engineers skimming the repo.
- **Resolution:** 1080p; capture the browser window only, not the whole desktop.
- **Hosting:** commit a compressed `.mp4`/`.gif` under `docs/assets/` (keep < 10 MB)
  or link an external upload from the README. Decide at capture time.
- **Cross-check:** pause on the final frame and confirm the peak throughput and α on
  screen match the latest `bench/results/` snapshot before publishing.

## Asset manifest

| Asset | Path | Status |
|-------|------|--------|
| Throughput vs concurrency | `bench/results/plots/throughput_vs_concurrency.png` | ✅ generated |
| VRAM vs concurrency | `bench/results/plots/vram_vs_concurrency.png` | ✅ generated |
| α vs gamma | `bench/results/plots/alpha_vs_gamma.png` | ✅ generated |
| Spec speedup vs gamma | `bench/results/plots/spec_speedup_vs_gamma.png` | ✅ generated |
| Under-load demo video | `docs/assets/demo.mp4` | ⏳ pending dashboard (phase 08) |
