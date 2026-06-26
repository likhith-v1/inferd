"""
bench.workload — frozen workload definition.

FROZEN: Do not change PROMPTS, CANONICAL, GREEDY, or MAX_TOKENS after phase 02
lands. Every later phase (04, 06, 09) reuses these verbatim so numbers stay
comparable. Any modification invalidates all previously recorded baselines.

If you need a different workload for a specific experiment, create a new
SamplingProfile and prompt list in your own module — do not modify this file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingProfile:
    temperature: float
    top_p: float
    seed: int

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "seed": self.seed}


# Canonical profile: temperature > 0 so phase-04 speculative-decoding
# correctness test can reuse it directly (rejection sampling needs temp > 0).
CANONICAL = SamplingProfile(temperature=0.7, top_p=0.95, seed=0)

# Greedy profile: deterministic, lowest-variance throughput numbers.
GREEDY = SamplingProfile(temperature=0.0, top_p=1.0, seed=0)

# Default max new tokens — fixed across all phases.
MAX_TOKENS: int = 256

# ~12 prompts spanning short / medium / long input token counts.
# Hardcoded; no external dataset dependency.
PROMPTS: list[str] = [
    # short (< 20 tokens)
    "The capital of France is",
    "Explain gravity in one sentence.",
    "What is 17 multiplied by 13?",
    "Translate 'hello world' to Spanish.",
    # medium (40–80 tokens)
    (
        "Write a brief introduction to the concept of neural networks, "
        "suitable for a high-school student with no prior knowledge of AI."
    ),
    (
        "Summarize the key differences between supervised and unsupervised "
        "machine learning in two to three sentences."
    ),
    (
        "A user asks: 'My Python script is running very slowly when processing "
        "a large CSV file. What are three techniques I can try to speed it up?' "
        "Provide concise, actionable advice."
    ),
    (
        "Describe the role of the transformer architecture in modern natural "
        "language processing and why it replaced recurrent neural networks "
        "for most tasks."
    ),
    # long (100–160 tokens)
    (
        "You are an expert systems programmer. A colleague has written a "
        "memory allocator in C that uses a free list. They are seeing "
        "fragmentation issues and occasional segfaults under high load. "
        "Walk through the most common causes of fragmentation in a free-list "
        "allocator, explain how to diagnose each, and suggest two concrete "
        "implementation strategies to reduce fragmentation without sacrificing "
        "throughput."
    ),
    (
        "Consider the following scenario: a startup is building a real-time "
        "inference service that must handle 500 concurrent requests per second "
        "with a p99 latency under 200 ms. Their model is a 7-billion-parameter "
        "transformer running on two A100 GPUs. They are currently using naive "
        "single-request batching. Propose an architecture that would allow them "
        "to meet these requirements, covering batching strategy, KV-cache "
        "management, and load balancing."
    ),
    (
        "Speculative decoding is a technique for accelerating autoregressive "
        "LLM inference. Explain the rejection-sampling acceptance rule in "
        "detail: under what condition is a draft token accepted, what happens "
        "when it is rejected, and why does this procedure guarantee that the "
        "final output distribution is identical to sampling from the target "
        "model directly?"
    ),
    (
        "You are reviewing a pull request that adds paged KV-cache management "
        "to an LLM inference engine. The author claims their implementation "
        "eliminates memory fragmentation by allocating fixed-size 'pages' of "
        "key-value tensors and mapping logical sequence positions to physical "
        "page slots via a block table. List five concrete things you would "
        "check or test to convince yourself the implementation is correct and "
        "production-ready."
    ),
]


def workload_hash(profile: SamplingProfile = CANONICAL) -> str:
    """SHA-256 of (prompts + profile + MAX_TOKENS) for provenance stamping."""
    payload = json.dumps(
        {"prompts": PROMPTS, "profile": profile.as_dict(), "max_tokens": MAX_TOKENS},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
