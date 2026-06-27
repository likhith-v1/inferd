"""
finetune.distill_draft — sequence-level KD to lift the draft's acceptance rate.

The alpha-lift experiment (phase 04): sample completions from the FINE-TUNED 9B
target, then SFT the 0.8B draft on those samples so it better matches the target
→ higher acceptance rate alpha → more speculative throughput, with the target
fixed and output still exact (correctness gate unchanged).

Two stages:
  1. generate — batch-sample N completions from `merged/9b` on dataset prompts
     (CANONICAL profile), write data/processed/draft_distill/{train,validation}.jsonl
     in the same alpaca `text` SFT format used by phase 03.
  2. train    — reuse finetune/train_qlora.py (via subprocess) with the
     qwen3_5_0_8b_distill.toml config → adapters/draft-distilled.

Generation uses a batched left-padded decode over the stripped target backbone
(same pattern as bench/runners/hf), so 10k short completions finish in minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import inferd.env  # noqa: F401

import torch  # noqa: E402

from finetune.data import format_alpaca, read_jsonl, write_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / "data" / "processed" / "code_alpaca_20k" / "train.jsonl"
OUT_DIR = ROOT / "data" / "processed" / "draft_distill"


def _prompt_prefix(row: dict) -> str:
    """Alpaca prompt up to (and including) '### Response:\\n', no target text."""
    return format_alpaca(row["instruction"], row.get("input", ""), "")


@torch.no_grad()
def _batched_generate(target, prefixes, max_new_tokens, temperature, top_p, seed):
    """Left-padded batched nucleus decode over the target backbone."""
    from core.spec_decode import nucleus_probs, sample_from

    tok = target.tokenizer
    device = target.device
    torch.manual_seed(seed)
    orig_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    enc = tok(prefixes, return_tensors="pt", padding=True)
    ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    tok.padding_side = orig_side

    bsz = ids.shape[0]
    out_tokens = [[] for _ in range(bsz)]
    done = torch.zeros(bsz, dtype=torch.bool, device=device)

    # Prefill once (masked so pad tokens aren't attended), then cache-based decode.
    # The mask must cover the full sequence (pad prefix + every generated token).
    logits, kv = target.forward(ids, None, attention_mask=attn)
    next_logits = logits[:, -1, :]
    for step in range(max_new_tokens):
        probs = torch.stack([nucleus_probs(next_logits[i:i+1], temperature, top_p)[0]
                             for i in range(bsz)])
        nxt = torch.multinomial(probs, num_samples=1)  # [bsz,1]
        for i in range(bsz):
            if not done[i]:
                t = int(nxt[i, 0])
                if t == tok.eos_token_id:
                    done[i] = True
                else:
                    out_tokens[i].append(t)
        if done.all():
            break
        attn = torch.cat([attn, torch.ones(bsz, 1, dtype=attn.dtype, device=device)], dim=1)
        logits, kv = target.forward(nxt, kv, attention_mask=attn)
        next_logits = logits[:, -1, :]
    return [tok.decode(o) for o in out_tokens]


def generate(target_path, n_samples, max_new_tokens, batch_size, seed, val_frac=0.1):
    from core.model_runner import ModelRunner
    from bench.workload import CANONICAL

    rows = read_jsonl(DEFAULT_PROMPTS)[:n_samples]
    target = ModelRunner.load_target(target_path)

    samples = []
    for b in range(0, len(rows), batch_size):
        batch = rows[b:b + batch_size]
        prefixes = [_prompt_prefix(r) for r in batch]
        responses = _batched_generate(
            target, prefixes, max_new_tokens, CANONICAL.temperature,
            CANONICAL.top_p, seed + b)
        for r, resp in zip(batch, responses):
            text = format_alpaca(r["instruction"], r.get("input", ""), resp.strip())
            samples.append({"text": text})
        print(f"  generated {min(b+batch_size, len(rows))}/{len(rows)}", flush=True)

    n_val = max(1, int(len(samples) * val_frac))
    write_jsonl(OUT_DIR / "validation.jsonl", samples[:n_val])
    write_jsonl(OUT_DIR / "train.jsonl", samples[n_val:])
    print(f"[distill] wrote {len(samples)-n_val} train / {n_val} val to {OUT_DIR}")


def train(offline: bool, draft: str, out: str) -> int:
    cfg = "finetune/configs/qwen3_5_0_8b_distill.toml"
    cmd = [sys.executable, "-m", "finetune.train_qlora", "--config", cfg,
           "--model", draft, "--out", out]
    if offline:
        cmd.append("--offline")
    print(f"[distill] training: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def _selfcheck() -> None:
    row = {"instruction": "Add two numbers", "input": "", "output": "x"}
    pre = _prompt_prefix(row)
    assert pre.endswith("### Response:\n"), repr(pre[-40:])
    assert "### Instruction:" in pre
    print("[distill] selfcheck PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="merged/9b")
    ap.add_argument("--draft", default="weights/Qwen3.5-0.8B",
                    help="Draft base model path/ID to fine-tune (passed to train_qlora --model).")
    ap.add_argument("--out", default="adapters/draft-distilled")
    ap.add_argument("--n-samples", type=int, default=10_000, dest="n_samples")
    ap.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-only", action="store_true", dest="gen_only")
    ap.add_argument("--train-only", action="store_true", dest="train_only")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.train_only:
        generate(args.target, args.n_samples, args.max_new_tokens, args.batch_size, args.seed)
    if not args.gen_only:
        return train(args.offline, args.draft, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
