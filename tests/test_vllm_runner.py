"""CPU-only tests for the isolated Phase-17 vLLM ceiling runner."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.metrics import BenchResult
from bench.runners import vllm
from bench.workload import model_fingerprint


class VllmEnvironmentTest(unittest.TestCase):
    def test_project_and_lock_pin_ceiling_stack(self):
        project = (vllm.VLLM_PROJECT / "pyproject.toml").read_text()
        lock = (vllm.VLLM_PROJECT / "uv.lock").read_text()
        self.assertIn('requires-python = ">=3.13,<3.14"', project)
        self.assertIn('"torch==2.11.0"', project)
        self.assertIn('"vllm==0.23.0"', project)
        self.assertIn('name = "torch"\nversion = "2.11.0"', lock)
        self.assertIn('name = "vllm"\nversion = "0.23.0"', lock)
        self.assertIn('name = "nvidia-cudnn-cu13"', lock)

    def test_sync_is_locked_and_targets_isolated_project(self):
        command = vllm._sync_command()
        self.assertEqual(command[:2], ["uv", "sync"])
        self.assertIn("--locked", command)
        self.assertEqual(command[command.index("--project") + 1], str(vllm.VLLM_PROJECT))
        self.assertEqual(command[command.index("--python") + 1], "3.13")

    def test_stale_environment_is_removed_and_resynced(self):
        ok = subprocess.CompletedProcess([], 0, stdout="synced\n", stderr="")
        with tempfile.TemporaryDirectory() as td:
            isolated = Path(td) / ".venv-vllm"
            python_bin = isolated / "bin" / "python"
            python_bin.parent.mkdir(parents=True)
            python_bin.write_text("stale")
            facts = {"vllm_version": vllm.VLLM_VERSION}
            with (
                patch.object(vllm, "VLLM_VENV", isolated),
                patch.object(vllm, "_sync_environment", side_effect=[ok, ok]) as sync,
                patch.object(
                    vllm,
                    "_validate_environment",
                    side_effect=[vllm.VllmEnvironmentError("wrong torch"), facts],
                ) as validate,
            ):
                got_python, got_facts = vllm._ensure_vllm_venv()

            self.assertEqual(got_python, python_bin)
            self.assertEqual(got_facts, facts)
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(validate.call_count, 2)
            self.assertFalse(isolated.exists())

    def test_runner_uses_text_only_and_keeps_cuda_graphs_enabled(self):
        self.assertIn('"language_model_only": True', vllm._RUNNER_SCRIPT)
        self.assertIn('"enforce_eager": False', vllm._RUNNER_SCRIPT)
        self.assertIn("language_model_only=effective_config", vllm._RUNNER_SCRIPT)
        self.assertNotIn("enforce_eager=True", vllm._RUNNER_SCRIPT)

    def test_runner_body_is_guarded_for_spawn_reimport(self):
        # vLLM on WSL forces the spawn start method, which re-imports this script
        # in every engine-core worker; the guard prevents reconstructing LLM.
        self.assertIn('if __name__ == "__main__":', vllm._RUNNER_SCRIPT)
        self.assertIn("def main():", vllm._RUNNER_SCRIPT)

    def test_runner_disables_flashinfer_sampler_and_records_it(self):
        # FlashInfer's sampler JIT is incompatible with cu13 headers on sm_120;
        # the native Torch sampler is used and the deviation is recorded.
        self.assertIn('"flashinfer_sampler": False', vllm._RUNNER_SCRIPT)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER", vllm._RUNNER_SCRIPT)


class VllmRunTest(unittest.TestCase):
    def _model_root(self, td: str) -> Path:
        weights = Path(td) / "weights"
        model = weights / "Qwen3.5-9B"
        model.mkdir(parents=True)
        (model / "config.json").write_text('{"model_type":"qwen3_5"}')
        return weights

    def test_warmups_and_provenance_flow_through_subprocess(self):
        with tempfile.TemporaryDirectory() as td:
            weights = self._model_root(td)
            results_dir = Path(td) / "results"

            def fake_subprocess(command, **_kwargs):
                self.assertEqual(command[10], "3")
                Path(command[3]).write_text(
                    json.dumps(
                        {
                            "concurrency_sweep": [
                                {
                                    "concurrency": 1,
                                    "total_time_s": 2.0,
                                    "total_tokens": 10,
                                    "throughput_tok_s": 5.0,
                                    "peak_vram_mb": 100.0,
                                    "peak_vram_torch_mb": 0.0,
                                    "warmup_runs": 3,
                                }
                            ],
                            "notes": [],
                            "env": {
                                "vllm": "0.23.0",
                                "torch": "2.11.0",
                                "cuda_version": "13.0",
                            },
                            "provenance": {"model_fingerprint": "artifact123"},
                        }
                    )
                )
                return subprocess.CompletedProcess(command, 0, stdout="measured\n", stderr="")

            with (
                patch.object(vllm, "WEIGHTS_ROOT", weights),
                patch.object(vllm, "_ensure_vllm_venv", return_value=(Path("/fake/python"), {})),
                patch.object(vllm, "env_stamp", return_value={"workload_hash": "parent"}),
                patch.object(vllm.subprocess, "run", side_effect=fake_subprocess),
            ):
                result = vllm.run(
                    warmup_runs=3,
                    concurrency_grid=[1],
                    profile_name="greedy",
                    cohort_id="cohort-17",
                    results_dir=results_dir,
                )

            self.assertEqual(result.role, "ceiling")
            self.assertEqual(result.cohort_id, "cohort-17")
            self.assertEqual(result.concurrency_sweep[0].warmup_runs, 3)
            self.assertEqual(result.provenance["warmup_runs"], 3)
            self.assertEqual(result.provenance["concurrency_grid"], [1])
            self.assertEqual(result.provenance["model_fingerprint"], "artifact123")
            self.assertEqual(result.env["vllm"], "0.23.0")

    def test_failure_result_retains_complete_subprocess_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            weights = self._model_root(td)
            results_dir = Path(td) / "results"
            failed = subprocess.CompletedProcess(
                ["fake"], 17, stdout="complete stdout\nline two", stderr="complete stderr"
            )
            with (
                patch.object(vllm, "WEIGHTS_ROOT", weights),
                patch.object(vllm, "_ensure_vllm_venv", return_value=(Path("/fake/python"), {})),
                patch.object(vllm, "env_stamp", return_value={}),
                patch.object(vllm.subprocess, "run", return_value=failed),
            ):
                result = vllm.run(results_dir=results_dir)

            self.assertEqual(result.role, "ceiling_deferred")
            note = "\n".join(result.notes)
            self.assertIn("exit=17", note)
            self.assertIn("complete stdout\nline two", note)
            self.assertIn("complete stderr", note)
            persisted = json.loads(next(results_dir.glob("*/result.json")).read_text())
            self.assertIn("complete stdout", "\n".join(persisted["notes"]))

    def test_old_benchresult_constructor_remains_source_compatible(self):
        result = BenchResult(
            engine="hf", role="floor", model="m", profile="greedy", max_tokens=1, env={}
        )
        self.assertIsNone(result.cohort_id)
        self.assertEqual(result.provenance, {})


class ModelFingerprintTest(unittest.TestCase):
    def test_hashes_metadata_content_but_only_tensor_name_and_size(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            config = model / "config.json"
            shard = model / "model-00001-of-00001.safetensors"
            config.write_text('{"a":1}')
            shard.write_bytes(b"abcd")
            first = model_fingerprint(model)

            shard.write_bytes(b"wxyz")
            self.assertEqual(first, model_fingerprint(model))

            config.write_text('{"a":2}')
            self.assertNotEqual(first, model_fingerprint(model))


if __name__ == "__main__":
    unittest.main()
