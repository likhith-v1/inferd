"""Named speculative-decoding pairs with pinned Phase 14 provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.speculation import CROP_NO_REPLAY, RESTORE_AND_REPLAY


@dataclass(frozen=True)
class PairConfig:
    name: str
    target: str
    draft: str
    target_repo: str
    draft_repo: str
    target_revision: str | None
    draft_revision: str | None
    tokenizer_sha256: str | None
    vocab_size: int | None
    reconciliation: str
    phase12_candidate: bool = False


PAIRS = {
    "phase14-full": PairConfig(
        name="phase14-full",
        target="weights/Qwen3-8B",
        draft="weights/Qwen3-0.6B",
        target_repo="Qwen/Qwen3-8B",
        draft_repo="Qwen/Qwen3-0.6B",
        target_revision="b968826d9c46dd6066d109eabc6255188de91218",
        draft_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        tokenizer_sha256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        vocab_size=151936,
        reconciliation=CROP_NO_REPLAY,
        phase12_candidate=True,
    ),
    "phase14-hybrid": PairConfig(
        name="phase14-hybrid",
        target="weights/Qwen3.5-9B",
        draft="weights/Qwen3.5-0.8B",
        target_repo="Qwen/Qwen3.5-9B",
        draft_repo="Qwen/Qwen3.5-0.8B",
        target_revision=None,
        draft_revision=None,
        tokenizer_sha256="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        vocab_size=248320,
        reconciliation=RESTORE_AND_REPLAY,
    ),
}
PAIR_NAMES = tuple(PAIRS)


def get_pair(name: str) -> PairConfig:
    try:
        return PAIRS[name]
    except KeyError as exc:
        raise ValueError(f"unknown pair config {name!r}; choose from {', '.join(PAIRS)}") from exc


def validate_local_revisions(pair: PairConfig) -> dict[str, str | None]:
    """Verify Hugging Face local-dir metadata against every configured revision."""
    resolved = {}
    for role, path, expected in (
        ("target", pair.target, pair.target_revision),
        ("draft", pair.draft, pair.draft_revision),
    ):
        model_dir = Path(path)
        repo = pair.target_repo if role == "target" else pair.draft_repo
        if not model_dir.is_dir():
            revision = f" --revision {expected}" if expected else ""
            raise FileNotFoundError(
                f"missing {role} weights at {model_dir}; run: "
                f"hf download {repo}{revision} --local-dir {model_dir}"
            )
        metadata = model_dir / ".cache" / "huggingface" / "download"
        revisions = set()
        for file in metadata.glob("*.metadata") if metadata.is_dir() else ():
            lines = file.read_text().splitlines()
            if lines:
                revisions.add(lines[0])
        if len(revisions) > 1:
            raise ValueError(f"{role} directory contains mixed Hugging Face revisions")
        actual = next(iter(revisions), None)
        if expected is not None and actual != expected:
            raise ValueError(f"{role} revision mismatch: expected {expected}, got {actual}")
        resolved[role] = actual
    return resolved
