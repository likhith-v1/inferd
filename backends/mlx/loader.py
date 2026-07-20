"""Convert once and load a pinned, local Qwen3-8B MLX artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from bench.workload import model_fingerprint

SOURCE = "Qwen/Qwen3-8B"
SOURCE_REVISION = "21073ac5a57f8ac6b159dae129728af51ac707e8"
BITS = 4
GROUP_SIZE = 64
DTYPE = "bfloat16"
MLX_VERSION = "0.31.2"
MLX_LM_VERSION = "0.31.3"
MANIFEST = "inferd_mlx_artifact.json"


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path
    fingerprint: str
    source: str
    source_revision: str
    bits: int
    group_size: int
    dtype: str
    mlx_version: str
    mlx_lm_version: str


def _manifest(path: Path) -> dict:
    manifest_path = path / MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"not an inferd MLX artifact (missing {manifest_path})")
    data = json.loads(manifest_path.read_text())
    expected = {
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "dtype": DTYPE,
        "mlx": MLX_VERSION,
        "mlx_lm": MLX_LM_VERSION,
    }
    mismatches = {key: (data.get(key), value) for key, value in expected.items() if data.get(key) != value}
    if mismatches:
        raise ValueError(f"MLX artifact provenance mismatch: {mismatches}")
    config = json.loads((path / "config.json").read_text())
    quant = config.get("quantization", config.get("quantization_config", {}))
    if quant.get("bits") != BITS or quant.get("group_size") != GROUP_SIZE:
        raise ValueError("MLX artifact config is not the pinned 4-bit/group-size-64 conversion")
    return data


def inspect_artifact(path: str | Path) -> ArtifactInfo:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"MLX artifact directory does not exist: {root}. Convert once with "
            "`python -m backends.mlx.loader convert --output <dir>`, then pass that "
            "directory (the bench runner takes it as --model <dir>)."
        )
    data = _manifest(root)
    return ArtifactInfo(
        path=root,
        fingerprint=model_fingerprint(root),
        source=data["source"],
        source_revision=data["source_revision"],
        bits=data["bits"],
        group_size=data["group_size"],
        dtype=data["dtype"],
        mlx_version=data["mlx"],
        mlx_lm_version=data["mlx_lm"],
    )


def convert_artifact(output: str | Path) -> ArtifactInfo:
    from mlx_lm import convert

    installed = (version("mlx"), version("mlx-lm"))
    if installed != (MLX_VERSION, MLX_LM_VERSION):
        raise RuntimeError(
            f"convert requires mlx=={MLX_VERSION}, mlx-lm=={MLX_LM_VERSION}; found "
            f"mlx=={installed[0]}, mlx-lm=={installed[1]}. Pin the backends/mlx env first "
            "so the artifact provenance matches at load time."
        )
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"conversion output already exists: {destination}")
    convert(
        SOURCE,
        mlx_path=str(destination),
        quantize=True,
        q_group_size=GROUP_SIZE,
        q_bits=BITS,
        dtype=DTYPE,
        revision=SOURCE_REVISION,
    )
    (destination / MANIFEST).write_text(json.dumps({
        "source": SOURCE,
        "source_revision": SOURCE_REVISION,
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "dtype": DTYPE,
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
    }, indent=2))
    return inspect_artifact(destination)


def load_artifact(path: str | Path):
    from mlx_lm import load

    info = inspect_artifact(path)
    model, tokenizer = load(str(info.path), lazy=False)
    for container in (model, getattr(model, "model", None)):
        if container is not None:
            for name in ("vision_tower", "vision_model", "visual"):
                if hasattr(container, name):
                    delattr(container, name)
    return model, tokenizer, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert_parser = sub.add_parser("convert", help="create the pinned local artifact")
    convert_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    info = convert_artifact(args.output)
    print(f"converted {info.path} fingerprint={info.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
