"""Exact speculative decoding: draft proposal + target verify + accept/resample."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import inferd.env  # noqa: F401

import torch  # noqa: E402


def nucleus_probs(
    logits_row: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Full-vocab probs after temperature + top-p. Used for both target (p) and draft (q)."""
    if temperature <= 0:
        probs = torch.zeros_like(logits_row, dtype=torch.float)
        probs.scatter_(-1, logits_row.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    logits_row = logits_row.float() / temperature
    probs = torch.softmax(logits_row, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def sample_from(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1)


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0
    bonus_tokens: int = 0
    generated: int = 0

    @property
    def alpha(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


@dataclass
class SpecOutput:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stats: SpecStats = field(default_factory=SpecStats)


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
    Generate up to `max_tokens` tokens via exact speculative decoding.

    Cache rollback is delegated to each runner: dense models crop and forward one
    token; Qwen3.5 restores linear state and replays all committed tokens.
    """
    device = target.device
    target.validate_speculation_pair(draft)
    torch.manual_seed(seed)
    prompt_ids = prompt_ids.to(device)

    committed: list[int] = prompt_ids[0].tolist()
    prompt_len = len(committed)

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
        t_snap = target.checkpoint_speculation(target_kv)
        d_snap = draft.checkpoint_speculation(draft_kv)

        q_last = q_anchor
        proposals: list[int] = []
        qs: list[torch.Tensor] = []
        for _ in range(gamma):
            q = nucleus_probs(q_last, temperature, top_p)
            x = sample_from(q)
            proposals.append(int(x.item()))
            qs.append(q)
            d_logits, draft_kv = draft.forward(x, draft_kv)
            q_last = d_logits[:, -1, :]

        prop_t = torch.tensor([proposals], device=device)
        t_logits, target_kv = target.forward(prop_t, target_kv)
        ps = [nucleus_probs(p_anchor, temperature, top_p)]
        ps += [nucleus_probs(t_logits[:, k, :], temperature, top_p) for k in range(gamma)]

        stats.rounds += 1
        stats.proposed += gamma

        n_accepted = 0
        emitted: list[int] = []
        rejected = False
        for k in range(gamma):
            x = proposals[k]
            p_x = ps[k][0, x]
            q_x = qs[k][0, x]
            if torch.rand(1, device=device) <= (p_x / q_x):
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
            emitted.append(int(sample_from(ps[gamma]).item()))
            stats.bonus_tokens += 1
        stats.accepted += n_accepted

        t_logits, target_kv = target.reconcile_speculation(
            target_kv, t_snap, n_accepted, emitted
        )
        d_logits, draft_kv = draft.reconcile_speculation(
            draft_kv, d_snap, n_accepted, emitted
        )
        p_anchor = t_logits[:, -1, :]
        q_anchor = d_logits[:, -1, :]

        for tok in emitted:
            committed.append(tok)
            stats.generated += 1
            if stats.generated >= max_tokens or (eos_id is not None and tok == eos_id):
                return _finish()

    return _finish()


def _selfcheck() -> None:
    logits = torch.tensor([[10.0, 9.0, 1.0, -5.0]])
    p = nucleus_probs(logits, temperature=1.0, top_p=1.0)
    assert abs(float(p.sum()) - 1.0) < 1e-5
    p2 = nucleus_probs(logits, temperature=1.0, top_p=0.5)
    assert float(p2[0, 0]) > 0 and abs(float(p2.sum()) - 1.0) < 1e-5
    assert float(p2[0, 3]) == 0.0
    assert float(torch.clamp(p - p, min=0.0).sum()) == 0.0
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
