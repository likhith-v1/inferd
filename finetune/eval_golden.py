"""Golden-set evaluator for Phase 03 adapters.

The default scoring is deterministic and local: each prompt has one or more
`expected_contains` strings, and a generation passes if any marker appears in
the model output. This is intentionally modest; human pairwise review can be
added on top by writing generation JSONL and reviewing it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_golden(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No golden prompts found in {path}")
    return rows


def score_output(output: str, expected_contains: list[str]) -> bool:
    lowered = output.lower()
    return any(marker.lower() in lowered for marker in expected_contains)


def selfcheck() -> None:
    assert score_output("def add(a, b): return a + b", ["def add", "lambda"])
    assert not score_output("print('hello')", ["def add"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="weights/Qwen3.5-9B", help="Base model path or ID.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--golden", type=Path, default=Path("finetune/golden_set.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/golden_eval.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    return parser.parse_args()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def load_model(base: str, adapter: str | None):
    from inferd.env import bootstrap

    bootstrap()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    base_path = ROOT / base if not Path(base).is_absolute() else Path(base)
    model_ref = str(base_path) if base_path.exists() else base
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        print("[eval_golden] selfcheck PASS")
        return 0
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    rows = load_golden(args.golden)
    model, tokenizer = load_model(args.base, args.adapter)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            output = generate(model, tokenizer, row["prompt"], args.max_new_tokens)
            ok = score_output(output, row["expected_contains"])
            passed += int(ok)
            record = {
                "id": row["id"],
                "passed": ok,
                "expected_contains": row["expected_contains"],
                "prompt": row["prompt"],
                "output": output,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[eval_golden] {row['id']}: {'PASS' if ok else 'FAIL'}")

    rate = passed / len(rows)
    print(f"[eval_golden] pass_rate={rate:.3f} ({passed}/{len(rows)})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

