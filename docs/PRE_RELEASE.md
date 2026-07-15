# Pre-release Checklist

`inferd` is at a pre-release-candidate stage: the local engine, serving layer,
dashboard, benchmark report, and FP8 27B capacity proof are implemented and
measured. The items below are the remaining checks before cutting a tagged
release.

## Status — `v0.1.5` staged, not yet tagged (2026-07-15)

Patch over `v0.1.1` — adds per-request sampling (`temperature`/`top_p` overridable
per `/generate` call, resolved against the server default when omitted; see
`DECISIONS.md` 2026-07-15 entry). Gates for this cut:

- **No-GPU gate** — **not run from the authoring session**: that session ran on
  macOS/arm64, and `uv run` cannot resolve the project's environment there at all
  (`causal-conv1d` is pinned to a `linux_x86_64` CUDA wheel with no macOS build).
  The new/updated tests (`tests/test_scheduler.py`:
  `test_submit_resolves_per_request_sampling_against_config_default`,
  `test_sample_next_reads_per_request_not_shared_config`; `tests/test_serve.py`:
  `test_generate_passes_per_request_sampling_to_engine`,
  `test_generate_omits_sampling_defaults_to_none`) were written and manually
  traced against the code, but never executed. **Must run
  `uv run python -m unittest discover -s tests -v` (plus the usual selfchecks) on
  the WSL2/RTX 5090 box before tagging.**
- **Dashboard gate** — unaffected: this change does not touch `dashboard/`.
  `bun run lint && bun run build` still recommended as a sanity check.
- **GPU spot-checks** — the diff touches `core/scheduler.py` (sampling only, not
  attention/cache logic), not `core/model_runner.py`; `paged_equiv`/`batched_equiv`
  are not expected to be affected, but were not re-run for this cut.
- **Under-load demo video** — remains intentionally waived.

## Status — `v0.1.1` released (2026-07-06)

Patch over `v0.1.0` — benchmark JSON output, dashboard charting polish, and internal
cleanup (`core/model_runner.py`, `inferd/__init__.py`, `inferd/env.py`). Gates for this cut:

- **Dashboard gate** — `bun run lint` + `bun run build`: **PASS**.
- **Headline agreement** — README / `bench/report.md` / `benchmarks.json` consistent
  (19.8×, 461.8 tok/s).
- **GPU spot-checks** — **not re-run for this cut**; pending on the RTX 5090 box. The diff
  touches `core/model_runner.py`, so rerun `paged_equiv --mode both` and `batched_equiv`
  (see *Required Before Tagging* below) to confirm numerical equivalence held.
- **Under-load demo video** — remains intentionally waived.

### `v0.1.0` released (2026-07-03)

First stable release cut off `main`, superseding pre-release `v0.1.0-rc.1`. Sign-off:

- **No-GPU gate** — 33 unit tests + all module `--selfcheck`s: **PASS**.
- **Dashboard gate** — `bun run lint` + `bun run build`: **PASS**.
- **GPU spot-checks** — `paged_equiv --mode both` bit-exact (`max_abs=0`); `batched_equiv`
  within tol (`max|Δlogit|=0.73 < 1.0`, `bad_flips=0`). Run against the base `Qwen3.5-9B`
  — equivalence is weights-independent, and `merged/9b` was not on the build machine.
- **Headline agreement** — README / `bench/report.md` / `benchmarks.json` consistent
  (19.8×, 461.8 tok/s).
- **Tracked artifacts** — none (clean).
- **Under-load demo video** — intentionally waived for this cut (not a blocker).

The checklist below is retained as the reusable template for future releases.

## Required Before Tagging

- Capture the under-load dashboard demo and place/link it from `docs/demo.md`.
- Run the no-GPU gate:
  ```bash
  uv run python -m unittest discover -s tests -v
  uv run python -m bench.harness --selfcheck
  uv run python -m bench.correctness --selfcheck
  uv run python -m core.paged_cache --selfcheck
  uv run python -m core.paged_attn --selfcheck
  uv run python -m bench.paged_equiv --selfcheck
  uv run python finetune/prepare_dataset.py --selfcheck
  uv run python finetune/train_qlora.py --selfcheck
  uv run python finetune/eval_golden.py --selfcheck
  uv run python finetune/export.py --selfcheck
  uv run python finetune/distill_draft.py --selfcheck
  ```
- Run the dashboard gate:
  ```bash
  cd dashboard
  bun install
  bun run lint
  bun run build
  ```
- On the RTX 5090 machine, rerun or spot-check the GPU gates:
  ```bash
  uv run python scripts/smoke_load.py
  uv run python bench/run_all.py --plots
  uv run python -m bench.paged_equiv --target merged/9b --mode both
  uv run python -m bench.batched_equiv --target merged/9b
  ```
- Confirm `bench/report.md`, `dashboard/src/data/benchmarks.json`, and
  `README.md` still agree on the headline numbers.
- Confirm no local-only artifacts are tracked:
  ```bash
  git status --short
  git ls-files | rg '(^|/)(weights|adapters|merged|runs|node_modules|dist|__pycache__)(/|$)|(^|/)data/(raw|processed)(/|$)|(^|/)\.DS_Store$|^dashboard/package-lock\.json$'
  ```

## Known Non-blocking Gaps

- vLLM remains a deferred ceiling on this Blackwell setup.
- The paged KV implementation is a correctness/reference path; serving still uses
  HF-backed runtime caches with scheduler block accounting.
- Batched speculative decoding is not implemented.
- The FP8 27B path is a capacity proof, not a latency result.

## Release Notes Seed

- Local QLoRA + inference stack for Qwen3.6/Qwen3.5 on a single RTX 5090.
- Exact speculative decoding with residual resampling and a PASS @ n=1500
  distribution-equivalence gate.
- Continuous batching reaches 461.8 tok/s at c=32, 19.8x over the naive HF
  floor on the matched benchmark workload.
- Fine-tuned 27B adapter is served through the engine with load-time FP8 at
  about 28.9 GiB model footprint.
