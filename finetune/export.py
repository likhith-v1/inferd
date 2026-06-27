"""Export or merge a Phase 03 LoRA adapter for serving."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VISION_ATTRS = ("visual", "vision_model", "vision_tower", "image_tower", "vpm", "vision_encoder")


def strip_vision_modules(container) -> list[str]:
    stripped: list[str] = []
    for attr in VISION_ATTRS:
        if hasattr(container, attr):
            delattr(container, attr)
            stripped.append(attr)
    return stripped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=False, default="weights/Qwen3.5-9B")
    parser.add_argument("--adapter", required=False, default="adapters/9b")
    parser.add_argument("--out", type=Path, default=Path("merged/9b"))
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base weights.")
    parser.add_argument(
        "--no-strip-vision",
        action="store_false",
        dest="strip_vision",
        default=True,
        help="Keep vision tower modules in the saved checkpoint (default: strip).",
    )
    parser.add_argument("--device-map", default="cuda:0", help="Use 'cpu' for 27B merge to avoid VRAM OOM.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    return parser.parse_args()


def selfcheck() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.visual = object()

    dummy = Dummy()
    stripped = strip_vision_modules(dummy)
    assert stripped == ["visual"]
    assert not hasattr(dummy, "visual")


def resolve_ref(ref: str) -> str:
    path = ROOT / ref if not Path(ref).is_absolute() else Path(ref)
    return str(path) if path.exists() else ref


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        print("[export] selfcheck PASS")
        return 0
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from inferd.env import bootstrap

    bootstrap()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    base_ref = resolve_ref(args.base)
    print(f"[export] loading base={base_ref}")
    base = AutoModelForMultimodalLM.from_pretrained(
        base_ref,
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_ref)
    print(f"[export] loading adapter={args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    if args.merge:
        print("[export] merging adapter into base")
        model = model.merge_and_unload()

    stripped: list[str] = []
    if args.strip_vision:
        stripped += strip_vision_modules(model)
        if hasattr(model, "model"):
            stripped += strip_vision_modules(model.model)
        print(f"[export] stripped vision attrs={stripped}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))
    print(f"[export] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
