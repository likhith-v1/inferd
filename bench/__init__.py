"""
bench — headless, importable inference benchmark harness for inferd.

Public contract (frozen; every later phase imports without modification):
  from bench.workload import CANONICAL, GREEDY, PROMPTS, MAX_TOKENS, request_token_budgets, workload_hash
  from bench.metrics  import ttft, itl, throughput, env_stamp, VramSampler
"""
