"""QLoRA SFT entrypoint for Phase 03.

Unsloth is the primary trainer path. This script is intentionally local-first:
it resolves checked-out model weights under ./weights before falling back to a
Hub ID, and it supports HF_HUB_OFFLINE=1 once weights are downloaded.
"""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path
from typing import Any

from finetune.data import read_jsonl, selfcheck as data_selfcheck


ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def resolve_model_name(config: dict[str, Any], override: str | None) -> str:
    if override:
        candidate = Path(override)
        return str(candidate) if candidate.exists() else override
    local_path = ROOT / config["model"].get("local_path", "")
    if local_path.exists():
        return str(local_path)
    return config["model"]["id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("finetune/configs/qwen3_5_9b.toml"))
    parser.add_argument("--model", default=None, help="Override model path or Hub ID.")
    parser.add_argument("--out", type=Path, default=None, help="Override adapter output dir.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--validation-jsonl", type=Path, default=None)
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 for this run.")
    parser.add_argument("--selfcheck", action="store_true", help="Run no-GPU config/data checks and exit.")
    return parser.parse_args()


def build_datasets(config: dict[str, Any], args: argparse.Namespace):
    from datasets import Dataset

    dataset_cfg = config["dataset"]
    train_path = args.train_jsonl or Path(dataset_cfg["train_jsonl"])
    validation_path = args.validation_jsonl or Path(dataset_cfg["validation_jsonl"])
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    return Dataset.from_list(train_rows), Dataset.from_list(validation_rows)


def run_selfcheck(config: dict[str, Any]) -> None:
    data_selfcheck()
    for section in ("model", "dataset", "lora", "training"):
        assert section in config, f"missing config section: {section}"
    assert config["lora"]["r"] > 0
    assert config["model"]["max_seq_length"] > 0
    assert config["dataset"]["text_field"] == "text"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.selfcheck:
        run_selfcheck(config)
        print("[train_qlora] selfcheck PASS")
        return 0

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Import-order contract: Unsloth must patch before transformers/trl imports.
    from inferd.env import bootstrap_finetune

    bootstrap_finetune()
    import torch
    from trl import SFTConfig
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model_name = resolve_model_name(config, args.model)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_dir = args.out or Path(train_cfg["output_dir"])
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", -1)

    train_dataset, eval_dataset = build_datasets(config, args)

    print(f"[train_qlora] model={model_name}")
    print(f"[train_qlora] output_dir={output_dir}")
    print(f"[train_qlora] train={len(train_dataset)} validation={len(eval_dataset)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=model_cfg["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=args.offline,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=train_cfg["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=train_cfg["run_name"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        max_steps=max_steps,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
        seed=train_cfg["seed"],
        dataset_text_field=config["dataset"]["text_field"],
        max_length=model_cfg["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_qlora] adapter saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
