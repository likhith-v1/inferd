"""Dataset preparation helpers for Phase 03 QLoRA fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RAW_JSON = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "TokenBender_code_instructions_122k_alpaca_style"
    / "code_instructions_120k.json"
)


@dataclass(frozen=True)
class SplitSpec:
    """Deterministic train/validation split configuration."""

    seed: int = 0
    train_size: int = 20_000
    validation_size: int = 2_000


def format_alpaca(instruction: str, input_text: str, output: str) -> str:
    """Render one instruction example in Alpaca SFT format."""
    instruction = instruction.strip()
    input_text = input_text.strip()
    output = output.strip()
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def load_raw_examples(path: Path = DEFAULT_RAW_JSON) -> list[dict]:
    """Load raw TokenBender Code Alpaca JSON examples."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset JSON not found at {path}. Download it with:\n"
            "  hf download TokenBender/code_instructions_122k_alpaca_style "
            "--repo-type dataset "
            "--local-dir ./data/raw/TokenBender_code_instructions_122k_alpaca_style"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def normalize_examples(raw: list[dict]) -> list[dict]:
    """Convert raw rows to the minimal SFT schema: instruction/input/output/text."""
    normalized: list[dict] = []
    for idx, row in enumerate(raw):
        try:
            instruction = str(row["instruction"])
            input_text = str(row.get("input", ""))
            output = str(row["output"])
        except KeyError as exc:
            raise ValueError(f"Raw row {idx} missing required key: {exc}") from exc
        if not instruction.strip() or not output.strip():
            continue
        normalized.append(
            {
                "id": idx,
                "instruction": instruction.strip(),
                "input": input_text.strip(),
                "output": output.strip(),
                "text": format_alpaca(instruction, input_text, output),
            }
        )
    if not normalized:
        raise ValueError("No usable instruction/output examples found")
    return normalized


def split_examples(examples: list[dict], spec: SplitSpec) -> tuple[list[dict], list[dict]]:
    """Deterministically sample train and validation rows without replacement."""
    needed = spec.train_size + spec.validation_size
    if len(examples) < needed:
        raise ValueError(f"Need {needed} examples, found {len(examples)}")
    rng = random.Random(spec.seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    train_idx = indices[: spec.train_size]
    val_idx = indices[spec.train_size : needed]
    train = [examples[i] for i in train_idx]
    validation = [examples[i] for i in val_idx]
    return train, validation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read UTF-8 JSONL rows."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_splits(
    raw_json: Path = DEFAULT_RAW_JSON,
    out_dir: Path = Path("data/processed/code_alpaca_20k"),
    spec: SplitSpec = SplitSpec(),
) -> tuple[Path, Path]:
    """Materialize deterministic train/validation JSONL files."""
    raw = load_raw_examples(raw_json)
    examples = normalize_examples(raw)
    train, validation = split_examples(examples, spec)
    out_dir = Path(out_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, validation)
    metadata = {
        "source": "TokenBender/code_instructions_122k_alpaca_style",
        "source_revision": "19b59da67914b5fb2e0a5dff937e9917c0cfb7e4",
        "seed": spec.seed,
        "train_size": len(train),
        "validation_size": len(validation),
        "raw_rows": len(raw),
        "usable_rows": len(examples),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train_path, val_path


def selfcheck() -> None:
    """Assert deterministic formatting and splitting on a tiny fixture."""
    rows = [
        {"instruction": f"Task {i}", "input": "", "output": f"Answer {i}"}
        for i in range(10)
    ]
    examples = normalize_examples(rows)
    assert "### Instruction:" in examples[0]["text"]
    assert "### Response:" in examples[0]["text"]
    a = split_examples(examples, SplitSpec(seed=3, train_size=6, validation_size=2))
    b = split_examples(examples, SplitSpec(seed=3, train_size=6, validation_size=2))
    assert [x["id"] for x in a[0]] == [x["id"] for x in b[0]]
    assert len(a[0]) == 6
    assert len(a[1]) == 2

