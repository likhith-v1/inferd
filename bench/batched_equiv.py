"""
bench.batched_equiv -- continuous-batching decode equivalence for phase 06.

Proves the batched scheduler step is numerically equivalent to single-stream
decoding: with the SAME tokens fed to each sequence, decoding a set of
ragged-length prompts TOGETHER (one batched forward per step, with left-padded KV
+ per-row position_ids) produces the same next-token logits as decoding each
prompt on its own, to within the bf16 floor. A RoPE / attention-mask / position
bug from the batched cache surgery would blow `max|Δlogit|` far past that floor
(cf. the GDN-patch bug which showed |Δ|=18); batching only reorders reductions,
so the real difference sits at ~0.12-0.16 like the eager-vs-sdpa floor.

We compare LOGITS (not free-running argmax) on purpose: under greedy, a single
bf16 near-tie (top-2 margin < noise) flips one token and then cascades, which is
expected bf16 behaviour, not an equivalence failure — so the gate is a logit-floor
check, and any argmax disagreement is reported with its top-2 margin to confirm it
was a near-tie.

    uv run python -m bench.batched_equiv --target merged/9b --n 4 --max-tokens 24
"""

from __future__ import annotations

import argparse

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402

from bench.workload import PROMPTS  # noqa: E402
from core.scheduler import ModelRunnerBackend  # noqa: E402

# Observed bf16 floor for batched-vs-serial on the 9B is ~0.15-0.49 (same order as
# eager-vs-sdpa); a real positional/masking bug is orders of magnitude larger
# (the GDN-patch bug showed |Δ|=18). 1.0 leaves headroom over the floor while
# staying ~18x below the bug signal, so the gate is decisive, not flaky.
LOGIT_TOL = 1.0


@torch.no_grad()
def _serial_logits(backend: ModelRunnerBackend, prompt_ids: list[int], steps: int):
    """Single-stream greedy: return (tokens, per-step logits) over `steps` steps."""
    logits, kv = backend.prefill(prompt_ids)
    tokens, per_step = [], []
    for _ in range(steps):
        per_step.append(logits.float().clone())
        tid = int(logits.argmax(dim=-1).item())
        tokens.append(tid)
        logits, kv = backend.decode(tid, kv)
    return tokens, per_step


@torch.no_grad()
def run(
    target_path: str,
    *,
    n_prompts: int = 4,
    max_tokens: int = 24,
    tol: float = LOGIT_TOL,
    device: str = "cuda:0",
) -> bool:
    from core.model_runner import ModelRunner

    target = ModelRunner.load_target(target_path, device=device)
    backend = ModelRunnerBackend(target)
    prompts = PROMPTS[:n_prompts]
    prompt_ids = [target.tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    # Reference: each prompt decoded alone; keep its tokens to teacher-force the batch.
    serial_tokens, serial_logits = [], []
    for ids in prompt_ids:
        toks, logs = _serial_logits(backend, ids, max_tokens)
        serial_tokens.append(toks)
        serial_logits.append(logs)

    # Batched: all rows advance together, each teacher-forced with its own serial
    # tokens so the inputs are identical to the serial reference. Rows stay in the
    # batch for the full run -> maximally ragged (prompt lengths differ).
    kvs, last = [], []
    for ids in prompt_ids:
        logits, kv = backend.prefill(ids)
        last.append(logits)
        kvs.append(kv)
    batched_logits = [[last[r].float().clone()] for r in range(n_prompts)]
    for step in range(max_tokens - 1):
        feed = [serial_tokens[r][step] for r in range(n_prompts)]
        last, kvs = backend.decode_batch(feed, kvs)
        for r in range(n_prompts):
            batched_logits[r].append(last[r].float().clone())

    overall_max = 0.0
    all_pass = True
    for r in range(n_prompts):
        row_max, flips, bad_flips = 0.0, 0, 0
        for t in range(max_tokens):
            sl, bl = serial_logits[r][t][0], batched_logits[r][t][0]
            delta = (sl - bl).abs().max().item()
            row_max = max(row_max, delta)
            if int(sl.argmax()) != int(bl.argmax()):
                flips += 1
                top2 = torch.topk(sl, 2).values
                margin = (top2[0] - top2[1]).item()
                if margin > delta + 1e-3:  # a flip NOT explained by a near-tie -> real
                    bad_flips += 1
        overall_max = max(overall_max, row_max)
        row_ok = row_max <= tol and bad_flips == 0
        all_pass &= row_ok
        print(f"[batched_equiv] prompt {r}: max|Δlogit|={row_max:.4f} "
              f"argmax_flips={flips} (near-tie) bad_flips={bad_flips} -> "
              f"{'OK' if row_ok else 'FAIL'}")

    print(f"[batched_equiv] {'PASS' if all_pass else 'FAIL'}  "
          f"overall max|Δlogit|={overall_max:.4f} (tol={tol}) "
          f"over {n_prompts} ragged prompts x {max_tokens} steps")
    return all_pass


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase-06 batched-decode equivalence.")
    p.add_argument("--target", default="merged/9b")
    p.add_argument("--n", type=int, default=4, dest="n_prompts")
    p.add_argument("--max-tokens", type=int, default=24, dest="max_tokens")
    p.add_argument("--tol", type=float, default=LOGIT_TOL)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    ok = run(
        args.target,
        n_prompts=args.n_prompts,
        max_tokens=args.max_tokens,
        tol=args.tol,
        device=args.device,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
