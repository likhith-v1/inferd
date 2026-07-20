# inferd MLX backend

This is a standalone Apple Silicon environment. It reuses inferd's scheduler,
sampling, engine thread, and FastAPI app without changing the CUDA runtime.
MLX 0.31.2 and mlx-lm 0.31.3 are pinned because cache merge/extract is part of
the backend contract.

On an Apple Silicon Mac (macOS 14+, native arm Python), from the repo root:

```sh
uv lock --project backends/mlx
uv sync --project backends/mlx --locked
uv run --project backends/mlx python -m backends.mlx.loader convert \
  --output weights/mlx/Qwen3-8B-4bit
```

Conversion is the only online step. It downloads the pinned Qwen3-8B revision,
converts once to bfloat16/4-bit (group size 64), and writes a provenance
manifest. Runtime loading accepts only that local artifact; it never loads
bf16 and quantizes at startup.

After conversion, run offline:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  INFERD_MLX_ARTIFACT=weights/mlx/Qwen3-8B-4bit \
  uv run --project backends/mlx uvicorn --factory backends.mlx.serve:create_mlx_app

uv run --project backends/mlx pytest backends/mlx/tests

uv run --project backends/mlx python -m bench.harness --engine mlx \
  --model weights/mlx/Qwen3-8B-4bit --profile greedy --concurrency 1,2,4
```

Apple results are forced under `bench/results/apple/`. They report MLX allocator
peak and process RSS peak, never NVIDIA's `peak_vram_mb`, and are not compared
to CUDA bf16. The 4-bit artifact can differ from source bf16; compare a fixed
prompt set or perplexity before making a quality claim. If
`test_batched_equiv.py` fails for a future mlx-lm/model combination, serve and
benchmark with concurrency `1` until its architecture-aware batch cache works.
