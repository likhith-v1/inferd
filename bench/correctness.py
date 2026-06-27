"""
bench.correctness — the speculative-decoding correctness gate (phase 04).

Statistically checks that speculative decoding produces the same token
distribution as direct sampling from the target. This is the differentiator: no
speedup claim is valid until this passes.

Method (self-calibrating, no magic threshold):
  For each fixed prefix, draw `n` next-token samples two ways —
    (a) spec-decode: one speculative round, take the FIRST committed token;
    (b) direct: nucleus-sample the target's next-token distribution.
  Build empirical histograms over the vocab and compute the Total-Variation
  distance TV(spec, direct). Compare it to a BOOTSTRAPPED NULL: many TV values
  between two independent direct-vs-direct samples of the same size n. The spec
  distribution passes if its TV lies within the null's 99th percentile (i.e. it
  is statistically indistinguishable from resampling the target itself), and a
  χ² goodness-of-fit gives p > 0.05.

Phase 09 extends this file (append-only); do not rewrite existing functions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import inferd.env  # noqa: F401

import torch  # noqa: E402

from bench.workload import CANONICAL, PROMPTS  # noqa: E402
from core.spec_decode import nucleus_probs, sample_from, speculative_generate  # noqa: E402


# ---------------------------------------------------------------------------
# Statistics (pure; covered by selfcheck)
# ---------------------------------------------------------------------------

def total_variation(p: dict[int, float], q: dict[int, float]) -> float:
    """TV distance between two discrete distributions given as {token: prob}."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _hist(samples: list[int]) -> dict[int, float]:
    n = len(samples)
    c = Counter(samples)
    return {k: v / n for k, v in c.items()}


def chi_square_pvalue(obs_a: list[int], obs_b: list[int]) -> float:
    """
    Two-sample χ² homogeneity test p-value over the pooled token support.

    Returns 1.0 if degenerate (single category). Uses scipy if available, else a
    survival-function approximation via torch's chi2 is not available, so we fall
    back to a normal approximation of the χ² statistic.
    """
    cats = sorted(set(obs_a) | set(obs_b))
    if len(cats) <= 1:
        return 1.0
    ca, cb = Counter(obs_a), Counter(obs_b)
    na, nb = len(obs_a), len(obs_b)
    n = na + nb
    stat = 0.0
    for cat in cats:
        a, b = ca.get(cat, 0), cb.get(cat, 0)
        row = a + b
        ea = row * na / n
        eb = row * nb / n
        if ea > 0:
            stat += (a - ea) ** 2 / ea
        if eb > 0:
            stat += (b - eb) ** 2 / eb
    dof = len(cats) - 1
    try:
        from scipy.stats import chi2
        return float(chi2.sf(stat, dof))
    except Exception:
        # Wilson–Hilferty normal approximation to the χ² survival function.
        import math
        x = stat / dof
        z = (x ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------------------------------------------------------------------
# Sampling collectors
# ---------------------------------------------------------------------------

@torch.no_grad()
def _direct_next_tokens(target, prompt_ids, n, temperature, top_p) -> list[int]:
    """Sample n next-tokens directly from the target's nucleus distribution."""
    logits, _ = target.forward(prompt_ids, None)
    probs = nucleus_probs(logits[:, -1, :], temperature, top_p)
    idx = torch.multinomial(probs.expand(n, -1), num_samples=1)
    return idx.squeeze(-1).tolist()


@torch.no_grad()
def _spec_first_tokens(target, draft, prompt_ids, n, gamma, temperature, top_p) -> list[int]:
    """First committed token from n independent spec-decode rounds."""
    out = []
    for i in range(n):
        r = speculative_generate(
            target, draft, prompt_ids,
            gamma=gamma, max_tokens=1,
            temperature=temperature, top_p=top_p, seed=1_000 + i,
        )
        out.append(r.token_ids[0])
    return out


# --- multi-token sequence test (exercises verify ps[k>0] AND replay) ---------

@torch.no_grad()
def _direct_continuations(target, prompt_ids, n, length, temperature, top_p, chunk=64) -> list[list[int]]:
    """n independent direct (target-only) nucleus continuations of `length` tokens."""
    eos = target.tokenizer.eos_token_id
    seqs: list[list[int]] = []
    for c0 in range(0, n, chunk):
        b = min(chunk, n - c0)
        ids = prompt_ids.expand(b, -1).contiguous()
        logits, kv = target.forward(ids, None)
        nl = logits[:, -1, :]
        cols: list[list[int]] = [[] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool, device=target.device)
        for _ in range(length):
            probs = torch.stack([nucleus_probs(nl[i:i+1], temperature, top_p)[0] for i in range(b)])
            nxt = torch.multinomial(probs, num_samples=1)
            for i in range(b):
                if not done[i]:
                    t = int(nxt[i, 0])
                    if t == eos:
                        done[i] = True
                    else:
                        cols[i].append(t)
            if done.all():
                break
            logits, kv = target.forward(nxt, kv)
            nl = logits[:, -1, :]
        seqs.extend(cols)
    return seqs


@torch.no_grad()
def _spec_continuations(target, draft, prompt_ids, n, length, gamma, temperature, top_p) -> list[list[int]]:
    """n independent speculative continuations of up to `length` tokens."""
    seqs = []
    for i in range(n):
        r = speculative_generate(
            target, draft, prompt_ids, gamma=gamma, max_tokens=length,
            temperature=temperature, top_p=top_p, seed=2_000 + i,
            eos_id=target.tokenizer.eos_token_id,
        )
        seqs.append(r.token_ids[:length])
    return seqs


def _pos_tokens(seqs: list[list[int]], t: int) -> list[int]:
    return [s[t] for s in seqs if len(s) > t]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def run(
    target_path: str,
    draft_path: str,
    *,
    draft_adapter: Optional[str] = None,
    n: int = 2000,
    gamma: int = 4,
    n_prompts: int = 4,
    bootstrap: int = 200,
    seed: int = 0,
) -> bool:
    from core.model_runner import ModelRunner

    profile = CANONICAL
    torch.manual_seed(seed)
    target = ModelRunner.load_target(target_path)
    draft = ModelRunner.load_draft(draft_path, adapter=draft_adapter)

    prompts = PROMPTS[:n_prompts]
    all_pass = True
    for pi, prompt in enumerate(prompts):
        prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)

        direct = _direct_next_tokens(target, prompt_ids, n, profile.temperature, profile.top_p)
        spec = _spec_first_tokens(target, draft, prompt_ids, n, gamma,
                                  profile.temperature, profile.top_p)

        tv_obs = total_variation(_hist(spec), _hist(direct))

        # Bootstrap the null: TV between two independent direct halves of size n.
        null_tvs = []
        gen = torch.Generator().manual_seed(seed + pi)
        for _ in range(bootstrap):
            a = _direct_next_tokens(target, prompt_ids, n, profile.temperature, profile.top_p)
            b = _direct_next_tokens(target, prompt_ids, n, profile.temperature, profile.top_p)
            null_tvs.append(total_variation(_hist(a), _hist(b)))
        null_tvs.sort()
        null_p99 = null_tvs[min(len(null_tvs) - 1, int(0.99 * len(null_tvs)))]

        pval = chi_square_pvalue(spec, direct)
        ok = (tv_obs <= null_p99) and (pval > 0.05)
        all_pass &= ok
        print(f"[correctness] prompt[{pi}] TV={tv_obs:.4f} null_p99={null_p99:.4f} "
              f"chi2_p={pval:.3f} -> {'PASS' if ok else 'FAIL'}")

    print(f"\n[correctness] {'PASS' if all_pass else 'FAIL'} "
          f"(n={n}, gamma={gamma}, prompts={len(prompts)})")
    return all_pass


def run_seq(
    target_path: str,
    draft_path: str,
    *,
    draft_adapter: Optional[str] = None,
    n: int = 600,
    length: int = 6,
    gamma: int = 4,
    n_prompts: int = 3,
    bootstrap: int = 100,
    seed: int = 0,
) -> bool:
    """
    Multi-token correctness gate: compare the per-position token distribution of
    speculative vs direct continuations. Unlike the first-token test, this
    exercises the parallel-verify logits ps[k>0] (the Qwen3.5 patch) and the
    snapshot/restore/replay cache state across rounds. A direct pool of 2n
    sequences supplies a bootstrapped direct-vs-direct null per position.
    """
    import random as _random
    from core.model_runner import ModelRunner

    profile = CANONICAL
    torch.manual_seed(seed)
    target = ModelRunner.load_target(target_path)
    draft = ModelRunner.load_draft(draft_path, adapter=draft_adapter)
    rng = _random.Random(seed)

    prompts = PROMPTS[:n_prompts]
    all_pass = True
    for pi, prompt in enumerate(prompts):
        prompt_ids = target.tokenizer(prompt, return_tensors="pt").input_ids.to(target.device)
        spec = _spec_continuations(target, draft, prompt_ids, n, length, gamma,
                                   profile.temperature, profile.top_p)
        pool = _direct_continuations(target, prompt_ids, 2 * n, length,
                                     profile.temperature, profile.top_p)
        direct = pool[:n]

        prompt_ok = True
        for t in range(length):
            spec_h = _hist(_pos_tokens(spec, t))
            dir_h = _hist(_pos_tokens(direct, t))
            if not spec_h or not dir_h:
                continue
            tv_obs = total_variation(spec_h, dir_h)
            # Null: TV between two disjoint random halves of the direct pool.
            null = []
            idx = list(range(len(pool)))
            for _ in range(bootstrap):
                rng.shuffle(idx)
                a = [pool[j] for j in idx[:n]]
                b = [pool[j] for j in idx[n:2 * n]]
                null.append(total_variation(_hist(_pos_tokens(a, t)), _hist(_pos_tokens(b, t))))
            null.sort()
            p99 = null[min(len(null) - 1, int(0.99 * len(null)))]
            ok = tv_obs <= p99
            prompt_ok &= ok
            print(f"[correctness] prompt[{pi}] pos[{t}] TV={tv_obs:.4f} "
                  f"null_p99={p99:.4f} -> {'PASS' if ok else 'FAIL'}")
        all_pass &= prompt_ok

    print(f"\n[correctness:seq] {'PASS' if all_pass else 'FAIL'} "
          f"(n={n}, length={length}, gamma={gamma}, prompts={len(prompts)})")
    return all_pass


def _selfcheck() -> None:
    # TV of identical dists is 0; of disjoint dists is 1.
    assert total_variation({0: 1.0}, {0: 1.0}) == 0.0
    assert abs(total_variation({0: 1.0}, {1: 1.0}) - 1.0) < 1e-9
    assert abs(total_variation({0: 0.5, 1: 0.5}, {0: 1.0}) - 0.5) < 1e-9
    # Identical samples → high χ² p-value; disjoint → low.
    same = chi_square_pvalue([0, 1] * 50, [0, 1] * 50)
    diff = chi_square_pvalue([0] * 100, [1] * 100)
    assert same > 0.05 and diff < 0.05, (same, diff)
    print("[correctness] selfcheck PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="merged/9b")
    parser.add_argument("--draft", default="weights/Qwen3.5-0.8B")
    parser.add_argument("--draft-adapter", default=None)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--test", choices=["tv"], default="tv")
    parser.add_argument("--mode", choices=["first", "seq"], default="seq",
                        help="'first': first-token only; 'seq': multi-token per-position "
                             "(exercises verify ps[k>0] + replay).")
    parser.add_argument("--length", type=int, default=6, help="Continuation length (mode=seq).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    if args.mode == "seq":
        ok = run_seq(args.target, args.draft, draft_adapter=args.draft_adapter,
                     n=args.n, length=args.length, gamma=args.gamma,
                     n_prompts=args.n_prompts, bootstrap=args.bootstrap, seed=args.seed)
    else:
        ok = run(args.target, args.draft, draft_adapter=args.draft_adapter,
                 n=args.n, gamma=args.gamma, n_prompts=args.n_prompts,
                 bootstrap=args.bootstrap, seed=args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
