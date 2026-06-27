"""Materialize deterministic Phase 03 SFT train/validation splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from finetune.data import DEFAULT_RAW_JSON, SplitSpec, prepare_splits, selfcheck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, default=DEFAULT_RAW_JSON)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/code_alpaca_20k"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--selfcheck", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        print("[prepare_dataset] selfcheck PASS")
        return 0
    spec = SplitSpec(
        seed=args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
    )
    train_path, val_path = prepare_splits(args.raw_json, args.out_dir, spec)
    print(f"[prepare_dataset] wrote {train_path}")
    print(f"[prepare_dataset] wrote {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

