"""Engine-specific benchmark runners invoked by ``bench.harness``.

  hf      — naive Hugging Face generate() floor
  vllm    — vLLM ceiling (isolated venv; best-effort on sm_120)
  spec    — exact speculative decoding
  paged   — paged KV-cache microbenchmark
  batched — continuous batching scheduler
"""
