"""inferd inference core.

Modules:
  model_runner   — load text backbone; forward(tokens, kv) -> (logits, kv)
  spec_decode    — exact speculative decoding (accept + residual resample)
  paged_cache    — block allocator and page table for KV tensors
  paged_attn     — paged-attention reference path (Triton kernel TBD)
  batched_cache  — stack/split hybrid caches for batched decode
  scheduler      — FCFS continuous batching with block budgeting
  qwen35_patch   — Qwen3.5 parallel-verify patch for hybrid linear attention
"""
