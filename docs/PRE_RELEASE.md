# Pre-release Checklist

`inferd` is at a pre-release-candidate stage: the local engine, serving layer,
dashboard, benchmark report, and FP8 27B capacity proof are implemented and
measured. The items below are the remaining checks before cutting a tagged
release.

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
