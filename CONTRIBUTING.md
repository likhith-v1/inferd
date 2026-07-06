# Contributing

`inferd` is a from-scratch local LLM inference stack — QLoRA fine-tune →
speculative decoding → paged KV-cache → continuous batching, served via FastAPI
with a React dashboard. It's primarily a personal / portfolio project maintained
by [@likhith-v1](https://github.com/likhith-v1). Issues, forks, and experiments
are welcome; this guide covers how it's built and the bar a change has to clear.

For the full picture see [`AGENTS.md`](AGENTS.md) (design + hard constraints),
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) (validated environment), and
[`docs/PRE_RELEASE.md`](docs/PRE_RELEASE.md) (release gate).

## Environment

Local-first and offline by design: WSL2 Ubuntu, RTX 5090 (Blackwell sm_120),
CUDA 12.8+. `uv.lock` and `dashboard/bun.lock` are the source of truth for
dependencies — install with `uv sync` / `bun install`, never a plain `pip install`
(the pinned cu130 / sm_120 stack is not reproducible from PyPI defaults).

```bash
uv sync
sudo apt-get install -y gcc g++     # Triton kernel JIT
cd dashboard && bun install         # dashboard
```

Validate the stack with `uv run python scripts/smoke_load.py`.

## Workflow

- **Read first, then write.** Read the relevant `plans/shipped/NN_*.md` (v1) or
  `plans/future/NN_*.md` (follow-up work) and reuse what already exists — don't
  re-implement helpers, types, or patterns that live here.
- **One worktree per phase:** branch `phase-NN-slug`, merge into `dev` in phase
  order; `dev` → `main` at milestones.
- **`core/model_runner.py` is the shared hot file** — extend it via new methods
  per the interface contract in `plans/shipped/00`; don't rewrite another phase's method.

## Quality gates (before opening a PR)

Run the no-GPU and dashboard gates from [`docs/PRE_RELEASE.md`](docs/PRE_RELEASE.md):

```bash
uv run python -m unittest discover -s tests
# plus the module `--selfcheck` commands listed in docs/PRE_RELEASE.md
cd dashboard && bun run lint && bun run build
```

- Lint clean; tests pass; numerical-equivalence preserved where it applies
  (speculative decoding must stay distributionally exact).
- No speedup claim without a reproducible benchmark under `bench/`.
- Never commit weights, adapters, merged checkpoints, datasets, or `__pycache__`
  (all gitignored and local-only).

## Hard constraints

- Local / offline only — no cloud or API inference, ever.
- Pin everything in `uv.lock`.
- QLoRA via Unsloth first (no MLX, no GemForge). Text-only v1.

The complete, authoritative list is in [`AGENTS.md`](AGENTS.md).

## Commits

The maintainer commits and merges. Keep commit messages simple and imperative.
AI coding-assistant help is acknowledged in [`CONTRIBUTORS.md`](CONTRIBUTORS.md),
not via commit co-author trailers.
