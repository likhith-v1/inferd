"""Phase 14 gate-and-measure coordinator; all result writes are append-only."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from bench.metrics import write_result_json
from bench.pair_configs import PAIR_NAMES, get_pair, validate_local_revisions


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"[phase14] {' '.join(command[1:])}")
    result = subprocess.run(command, text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        revisions = {name: validate_local_revisions(get_pair(name)) for name in PAIR_NAMES}
    except (OSError, ValueError) as exc:
        write_result_json(
            {"phase": 14, "status": "preflight-failed", "error": str(exc)},
            "phase14_coordinator",
            args.results_dir,
        )
        print(f"[phase14] preflight failed: {exc}", file=sys.stderr)
        return 2

    gates = []
    for name in PAIR_NAMES:
        pair = get_pair(name)
        command = [
            sys.executable, "-m", "bench.correctness",
            "--target", pair.target, "--draft", pair.draft,
            "--mode", "seq", "--n", "1500", "--length", "6",
            "--gamma", "4", "--n-prompts", "3", "--bootstrap", "200",
            "--seed", "0", "--familywise",
        ]
        result = _run(command)
        gates.append({"pair_config": name, "passed": result.returncode == 0,
                      "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            write_result_json(
                {"phase": 14, "status": "correctness-failed", "revisions": revisions,
                 "correctness": gates},
                "phase14_coordinator",
                args.results_dir,
            )
            print("[phase14] correctness failed; timing aborted", file=sys.stderr)
            return result.returncode

    for name in PAIR_NAMES:
        command = [
            sys.executable, "-m", "bench.harness", "--engine", "spec",
            "--pair-config", name, "--gamma", "2,4,8", "--max-tokens", "128",
            "--n-prompts", "12", "--warmup", "1", "--repeats", "3", "--seed", "0",
        ]
        if args.results_dir:
            command.extend(("--results-dir", str(args.results_dir)))
        result = _run(command)
        if result.returncode:
            write_result_json(
                {"phase": 14, "status": "benchmark-failed", "failed_pair": name,
                 "revisions": revisions, "correctness": gates,
                 "stdout": result.stdout, "stderr": result.stderr},
                "phase14_coordinator",
                args.results_dir,
            )
            return result.returncode

    write_result_json(
        {"phase": 14, "status": "measured", "revisions": revisions,
         "correctness": gates},
        "phase14_coordinator",
        args.results_dir,
    )
    if args.results_dir is None:
        return _run([sys.executable, "bench/run_all.py", "--plots"]).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
