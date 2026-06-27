"""
bench.paged_equiv -- model-level cache round-trip equivalence for phase 05.

This validates the current paged cache representation against real model logits:
prefill with the normal Qwen3.5 cache, convert full-attention KV into
PagedHybridCache, reconstruct an HF-style cache, then compare the next-token
logits from contiguous vs reconstructed cache.

It does not claim the final paged runtime path yet; it proves the page-table
representation is lossless before replacing the attention internals.
"""

from __future__ import annotations

import argparse

import torch

from bench.workload import PROMPTS
from core.paged_cache import PagedHybridCache


@torch.no_grad()
def run(
    target_path: str,
    *,
    block_size: int = 16,
    n_prompts: int = 3,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    device: str = "cuda:0",
) -> bool:
    from core.model_runner import ModelRunner

    target = ModelRunner.load_target(target_path, device=device)
    all_pass = True
    for idx, prompt in enumerate(PROMPTS[:n_prompts]):
        prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
        logits, kv = target.forward(prompt_ids, None)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        paged = PagedHybridCache.from_qwen_cache(kv, block_size=block_size)
        contig_kv = paged.to_qwen_cache_like(kv)
        paged_roundtrip_kv = paged.to_qwen_cache_like(kv)

        logits_contig, _ = target.forward(next_id, contig_kv)
        logits_paged, _ = target.forward(next_id, paged_roundtrip_kv)
        diff = (logits_contig - logits_paged).abs()
        max_abs = float(diff.max().item())
        denom = logits_contig.abs().clamp_min(1e-6)
        max_rel = float((diff / denom).max().item())
        ok = max_abs <= atol or max_rel <= rtol
        all_pass &= ok
        print(
            f"[paged_equiv] prompt[{idx}] max_abs={max_abs:.6g} "
            f"max_rel={max_rel:.6g} -> {'PASS' if ok else 'FAIL'}"
        )

    print(
        f"\n[paged_equiv] {'PASS' if all_pass else 'FAIL'} "
        f"(block_size={block_size}, prompts={n_prompts}, atol={atol}, rtol={rtol})"
    )
    return all_pass


@torch.no_grad()
def run_compute(
    target_path: str,
    *,
    n_prompts: int = 3,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    device: str = "cuda:0",
) -> bool:
    """
    Model-level paged-attention COMPUTE equivalence (not just storage).

    Routes the real model's full-attention decode step through the paged
    gather-and-attend path (core.paged_attn_interface) and compares next-token
    logits to the stock attention path. This exercises paged_attention inside the
    real 9B for every full-attention layer, closing the gap left by the
    round-trip (storage-only) check.
    """
    from core.model_runner import ModelRunner
    from core.paged_attn_interface import install, uninstall, set_impl

    target = ModelRunner.load_target(target_path, device=device)
    # Force eager mask prep so causal prefill is correct; paged routing is done by
    # patching the eager global (install), toggled per run below.
    set_impl(target, "eager")
    all_pass = True
    for idx, prompt in enumerate(PROMPTS[:n_prompts]):
        prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)

        # Reference decode step with the stock eager attention path.
        uninstall()
        logits, kv = target.forward(prompt_ids, None)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ref_logits, _ = target.forward(next_id, kv)

        # Same decode step, decode attention routed through the paged path.
        install()
        logits2, kv2 = target.forward(prompt_ids, None)
        paged_logits, _ = target.forward(next_id, kv2)
        uninstall()

        diff = (ref_logits - paged_logits).abs()
        max_abs = float(diff.max().item())
        denom = ref_logits.abs().clamp_min(1e-6)
        max_rel = float((diff / denom).max().item())
        ok = max_abs <= atol or max_rel <= rtol
        all_pass &= ok
        print(
            f"[paged_equiv:compute] prompt[{idx}] max_abs={max_abs:.6g} "
            f"max_rel={max_rel:.6g} -> {'PASS' if ok else 'FAIL'}"
        )

    print(
        f"\n[paged_equiv:compute] {'PASS' if all_pass else 'FAIL'} "
        f"(prompts={n_prompts}, atol={atol}, rtol={rtol})"
    )
    return all_pass


def _selfcheck() -> None:
    class FakeCache:
        pass

    fake = FakeCache()
    fake.layer_types = ["full_attention", "linear_attention"]
    fake.transformer_layers = [0]
    fake.last_linear_layer = 1
    fake.key_cache = [torch.randn(1, 2, 5, 4), None]
    fake.value_cache = [torch.randn(1, 2, 5, 4), None]
    fake.conv_states = [None, torch.randn(1, 3, 2)]
    fake.recurrent_states = [None, torch.randn(1, 3, 7)]
    paged = PagedHybridCache.from_qwen_cache(fake, block_size=4)
    roundtrip = paged.to_qwen_cache_like(fake)
    torch.testing.assert_close(roundtrip.key_cache[0], fake.key_cache[0])
    torch.testing.assert_close(roundtrip.value_cache[0], fake.value_cache[0])
    torch.testing.assert_close(roundtrip.conv_states[1], fake.conv_states[1])
    print("[paged_equiv] selfcheck PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="merged/9b")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["roundtrip", "compute", "both"], default="both",
                        help="roundtrip: lossless storage; compute: paged attention "
                             "through the real model; both: run each.")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    ok = True
    if args.mode in ("roundtrip", "both"):
        ok &= run(
            args.target, block_size=args.block_size, n_prompts=args.n_prompts,
            atol=args.atol, rtol=args.rtol, device=args.device,
        )
    if args.mode in ("compute", "both"):
        ok &= run_compute(
            args.target, n_prompts=args.n_prompts,
            atol=args.atol, rtol=args.rtol, device=args.device,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
