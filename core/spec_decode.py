"""
core.spec_decode — exact speculative decoding (phase 04, the headline result).

Implements draft proposal + single-pass target verification + the EXACT
accept/resample rule (Leviathan 2023 / Chen 2023), including the residual
branch. The rejection-sampling construction is distributionally identical to
direct target sampling when the implementation is correct; bench/correctness.py
provides statistical evidence for that contract. q (the draft proposal) affects
only the acceptance rate, never the output distribution.

The accept rule, per drafted token x_k with target prob p_k(x) and draft prob
q_k(x):
    accept x_k with probability min(1, p_k(x_k) / q_k(x_k))
On the first rejection at position k, resample from the residual
    p_resid(x) = normalize(max(0, p_k(x) - q_k(x)))
and discard all drafts after k. If all gamma drafts are accepted, sample one
bonus token from p_{gamma+1}.

Single-stream (batch=1) here; phase 06 layers continuous batching on top via the
same core.model_runner.forward contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Sampling distribution (must be identical for target p and draft q)
# ---------------------------------------------------------------------------

def nucleus_probs(
    logits_row: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """
    Full-vocab probability vector after temperature + top-p (nucleus) truncation.

    Returns shape [1, vocab]: tokens outside the top-p nucleus are zeroed and the
    remainder is renormalized. This is THE sampling distribution — target (p) and
    draft (q) must both use it so the accept/residual math is exact.

    temperature <= 0 yields a one-hot argmax distribution (greedy); exact
    speculative decoding then reduces to "accept iff draft argmax == target
    argmax, else emit target argmax", i.e. token-identical to greedy decoding.
    """
    if temperature <= 0:
        probs = torch.zeros_like(logits_row, dtype=torch.float)
        probs.scatter_(-1, logits_row.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    logits_row = logits_row.float() / temperature
    probs = torch.softmax(logits_row, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest prefix whose cumulative mass first reaches top_p.
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def sample_from(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token id from a [1, vocab] probability vector → [1, 1]."""
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Hybrid-cache rollback: linear (recurrent) state can't be cropped, so snapshot
# the fixed-size conv/recurrent states (cheap, O(1) in seq len) and crop the
# growing attention KV by slicing. Restore-then-replay rebuilds the exact state.
# ---------------------------------------------------------------------------

def _clone_list(lst):
    return [t.clone() if t is not None else None for t in lst]


def _snapshot_linear(kv):
    """Snapshot only the fixed-size GatedDeltaNet states (conv + recurrent)."""
    if kv is None:
        return None
    return (_clone_list(kv.conv_states), _clone_list(kv.recurrent_states))


def _restore_to(kv, lin_snap, attn_len: int) -> None:
    """Restore linear states from snapshot and crop attention KV to attn_len."""
    if kv is None:
        return
    conv, rec = lin_snap
    kv.conv_states = _clone_list(conv)
    kv.recurrent_states = _clone_list(rec)
    for i in kv.transformer_layers:
        if kv.key_cache[i] is not None and kv.key_cache[i].shape[-2] > attn_len:
            kv.key_cache[i] = kv.key_cache[i][:, :, :attn_len, :].contiguous()
            kv.value_cache[i] = kv.value_cache[i][:, :, :attn_len, :].contiguous()


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class SpecStats:
    """Acceptance statistics for one generation."""
    proposed: int = 0          # total draft tokens proposed
    accepted: int = 0          # total draft tokens accepted
    rounds: int = 0            # spec-decode rounds (target verify passes)
    bonus_tokens: int = 0      # rounds where all gamma accepted (bonus sampled)
    generated: int = 0         # total committed new tokens

    @property
    def alpha(self) -> float:
        """Empirical acceptance rate α = accepted / proposed."""
        return self.accepted / self.proposed if self.proposed else 0.0


@dataclass
class SpecOutput:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stats: SpecStats = field(default_factory=SpecStats)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

@torch.no_grad()
def speculative_generate(
    target,
    draft,
    prompt_ids: torch.Tensor,
    *,
    gamma: int = 4,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 0,
    eos_id: Optional[int] = None,
) -> SpecOutput:
    """
    Generate up to `max_tokens` tokens from `target`, accelerated by `draft`.

    target, draft : core.model_runner.ModelRunner
    prompt_ids    : LongTensor [1, prompt_len] on the target device.

    Invariant: between rounds both caches reflect EXACTLY `committed`, and
    `p_anchor`/`q_anchor` are the target/draft logits predicting the next token.
    Qwen3.5's linear-attention recurrent state can't be cropped, so each round we
    snapshot it, run the parallel verify (which advances the cache), then RESTORE
    the snapshot and REPLAY the committed tokens to rebuild the exact state. The
    snapshot is O(1) in sequence length (fixed-size states); attention KV is
    cropped by slicing.
    """
    device = target.device
    torch.manual_seed(seed)
    prompt_ids = prompt_ids.to(device)

    committed: list[int] = prompt_ids[0].tolist()
    prompt_len = len(committed)

    # --- prefill: caches reflect the whole prompt -------------------------
    t_logits, target_kv = target.forward(prompt_ids, None)
    d_logits, draft_kv = draft.forward(prompt_ids, None)
    p_anchor = t_logits[:, -1, :]
    q_anchor = d_logits[:, -1, :]

    stats = SpecStats()

    def _finish() -> SpecOutput:
        gen_ids = committed[prompt_len:]
        return SpecOutput(token_ids=gen_ids, stats=stats,
                          text=target.tokenizer.decode(gen_ids))

    while stats.generated < max_tokens:
        base = len(committed)                     # committed length (caches reflect this)
        t_snap = _snapshot_linear(target_kv)
        d_snap = _snapshot_linear(draft_kv)

        # --- draft proposes gamma tokens (single-step, fast path) -----------
        q_last = q_anchor
        proposals: list[int] = []
        qs: list[torch.Tensor] = []
        for _ in range(gamma):
            q = nucleus_probs(q_last, temperature, top_p)
            x = sample_from(q)                    # [1,1]
            proposals.append(int(x.item()))
            qs.append(q)
            d_logits, draft_kv = draft.forward(x, draft_kv)
            q_last = d_logits[:, -1, :]

        # --- target verifies all gamma in ONE parallel pass ----------------
        # p_1 = anchor; p_{k+1} = logits after x_k (k=1..gamma).
        prop_t = torch.tensor([proposals], device=device)
        t_logits, target_kv = target.forward(prop_t, target_kv)
        ps = [nucleus_probs(p_anchor, temperature, top_p)]
        ps += [nucleus_probs(t_logits[:, k, :], temperature, top_p) for k in range(gamma)]

        stats.rounds += 1
        stats.proposed += gamma

        # --- exact accept / resample ----------------------------------------
        n_accepted = 0
        emitted: list[int] = []
        rejected = False
        for k in range(gamma):
            x = proposals[k]
            p_x = ps[k][0, x]
            q_x = qs[k][0, x]
            r = torch.rand(1, device=device)
            if r <= (p_x / q_x):
                n_accepted += 1
                emitted.append(x)
            else:
                rejected = True
                residual = torch.clamp(ps[k] - qs[k], min=0.0)
                total = residual.sum()
                resampled_dist = residual / total if total > 0 else ps[k]
                emitted.append(int(sample_from(resampled_dist).item()))
                break
        if not rejected:
            emitted.append(int(sample_from(ps[gamma]).item()))   # bonus from p_{gamma+1}
            stats.bonus_tokens += 1
        stats.accepted += n_accepted

        # --- restore to `committed`, then replay `emitted` ------------------
        _restore_to(target_kv, t_snap, base)
        _restore_to(draft_kv, d_snap, base)
        emit_t = torch.tensor([emitted], device=device)
        t_logits, target_kv = target.forward(emit_t, target_kv)
        d_logits, draft_kv = draft.forward(emit_t, draft_kv)
        p_anchor = t_logits[:, -1, :]
        q_anchor = d_logits[:, -1, :]

        # --- commit, honoring max_tokens and EOS ----------------------------
        for tok in emitted:
            committed.append(tok)
            stats.generated += 1
            if stats.generated >= max_tokens or (eos_id is not None and tok == eos_id):
                return _finish()

    return _finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    """No-GPU asserts on the sampling/residual math."""
    logits = torch.tensor([[10.0, 9.0, 1.0, -5.0]])
    p = nucleus_probs(logits, temperature=1.0, top_p=1.0)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    # top_p truncation keeps at least the top token and renormalizes.
    p2 = nucleus_probs(logits, temperature=1.0, top_p=0.5)
    assert float(p2[0, 0]) > 0 and abs(float(p2.sum()) - 1.0) < 1e-5
    assert float(p2[0, 3]) == 0.0
    # residual of identical dists is zero → fall back to p.
    resid = torch.clamp(p - p, min=0.0)
    assert float(resid.sum()) == 0.0
    print("[spec_decode] selfcheck PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="merged/9b")
    parser.add_argument("--draft", default="weights/Qwen3.5-0.8B")
    parser.add_argument("--draft-adapter", default=None,
                        help="Optional LoRA adapter dir for the distilled draft.")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="Write a Python function to reverse a string.")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    from core.model_runner import ModelRunner

    target = ModelRunner.load_target(args.target)
    draft = ModelRunner.load_draft(args.draft, adapter=args.draft_adapter)

    prompt_ids = target.tokenizer(args.prompt, return_tensors="pt").input_ids
    out = speculative_generate(
        target, draft, prompt_ids,
        gamma=args.gamma, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        eos_id=target.tokenizer.eos_token_id,
    )
    print(out.text)
    s = out.stats
    print(f"\n[spec] gamma={args.gamma} generated={s.generated} "
          f"alpha={s.alpha:.3f} accepted={s.accepted}/{s.proposed} "
          f"rounds={s.rounds} bonus={s.bonus_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
