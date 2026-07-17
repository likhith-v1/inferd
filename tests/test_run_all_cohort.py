"""CPU-only tests for Phase-17 three-rung cohort selection in bench.run_all.

Exercises the Python-side authority that gates whether a vLLM ceiling ratio is
publishable: provenance-matched cohort selection, fail-closed rejection of mixed
or incomplete cohorts, precedence over newer deferred attempts, and the
"within K× of vLLM" calculation at the highest *shared* concurrency.
"""

from __future__ import annotations

import inspect
import unittest

from bench import run_all

GRID = run_all.PHASE17_CONCURRENCY


def _sweep(grid=GRID, tps=None):
    tps = tps or {c: float(c) for c in grid}
    return [
        {
            "concurrency": c,
            "total_time_s": 1.0,
            "total_tokens": int(tps[c]),
            "throughput_tok_s": tps[c],
            "peak_vram_mb": 1000.0,
            "peak_vram_torch_mb": 0.0,
            "warmup_runs": run_all.PHASE17_WARMUP_RUNS,
        }
        for c in grid
    ]


def _rung(engine, role, cohort_id, *, fp="fp-A", wh="wh-A", grid=GRID, tps=None):
    provenance = {
        "model_fingerprint": fp,
        "workload_hash": wh,
        "profile": run_all.PHASE17_PROFILE,
        "seed": run_all.PHASE17_SEED,
        "max_tokens": run_all.PHASE17_MAX_TOKENS,
        "warmup_runs": run_all.PHASE17_WARMUP_RUNS,
        "concurrency_grid": list(GRID),
    }
    return {
        "engine": engine,
        "role": role,
        "profile": run_all.PHASE17_PROFILE,
        "max_tokens": run_all.PHASE17_MAX_TOKENS,
        "cohort_id": cohort_id,
        "provenance": provenance,
        "env": {"seed": run_all.PHASE17_SEED, "workload_hash": wh},
        "concurrency_sweep": _sweep(grid, tps),
    }


def _cohort(cohort_id, *, fp="fp-A", wh="wh-A"):
    return [
        _rung("hf", "floor", cohort_id, fp=fp, wh=wh),
        _rung("batched", "phase06_scheduler_matched", cohort_id, fp=fp, wh=wh),
        _rung("vllm", "ceiling", cohort_id, fp=fp, wh=wh),
    ]


def _as_results(*result_dicts):
    """Wrap dicts as (dir_name, data) newest-first, like run_all._all_results()."""
    return [(f"dir{i}", r) for i, r in enumerate(result_dicts)]


class SelectCohortTest(unittest.TestCase):
    def test_selects_complete_cohort_and_ignores_newer_deferred(self):
        deferred = {
            "engine": "vllm",
            "role": "ceiling_deferred",
            "cohort_id": None,
            "notes": ["blocked"],
        }
        # newest-first: deferred is newest, valid cohort is older
        results = _as_results(deferred, *_cohort("cohort-good"))
        selected = run_all.select_three_rung_cohort(results)
        self.assertEqual(selected.cohort_id, "cohort-good")
        self.assertIsNotNone(selected.hf)
        self.assertIsNotNone(selected.ours)
        self.assertIsNotNone(selected.vllm)
        self.assertIs(selected.deferred_vllm, deferred)
        self.assertEqual(selected.errors, ())

    def test_rejects_incomplete_cohort_missing_vllm(self):
        rungs = _cohort("cohort-partial")[:2]  # hf + ours only
        selected = run_all.select_three_rung_cohort(_as_results(*rungs))
        self.assertIsNone(selected.cohort_id)
        self.assertTrue(any("missing rungs" in e for e in selected.errors))

    def test_rejects_mismatched_workload_hash_across_rungs(self):
        rungs = _cohort("cohort-mixed")
        rungs[2] = _rung("vllm", "ceiling", "cohort-mixed", wh="wh-DIFFERENT")
        selected = run_all.select_three_rung_cohort(_as_results(*rungs))
        self.assertIsNone(selected.cohort_id)
        self.assertTrue(any("mismatched provenance field" in e for e in selected.errors))

    def test_rejects_mismatched_model_fingerprint(self):
        rungs = _cohort("cohort-fp")
        rungs[2] = _rung("vllm", "ceiling", "cohort-fp", fp="fp-DIFFERENT")
        selected = run_all.select_three_rung_cohort(_as_results(*rungs))
        self.assertIsNone(selected.cohort_id)
        self.assertTrue(any("mismatched provenance field" in e for e in selected.errors))

    def test_rejects_incomplete_measured_grid(self):
        rungs = _cohort("cohort-grid")
        rungs[2] = _rung("vllm", "ceiling", "cohort-grid", grid=[1, 2, 4, 8, 16])
        selected = run_all.select_three_rung_cohort(_as_results(*rungs))
        self.assertIsNone(selected.cohort_id)
        self.assertTrue(
            any("concurrency grid is incomplete" in e for e in selected.errors)
        )

    def test_valid_cohort_wins_over_newer_invalid_one(self):
        bad = _cohort("cohort-bad")
        bad[2] = _rung("vllm", "ceiling", "cohort-bad", wh="wh-DIFFERENT")
        good = _cohort("cohort-good")
        # newest-first: bad cohort first, good cohort after
        selected = run_all.select_three_rung_cohort(_as_results(*bad, *good))
        self.assertEqual(selected.cohort_id, "cohort-good")


class RunnerCohortThreadingTest(unittest.TestCase):
    """run_all calls every rung with cohort_id=; the runners must accept it and
    emit the provenance the cohort validator requires (regression guard for the
    hf/batched threading that makes the three-rung cohort assemblable)."""

    def test_hf_and_batched_runners_accept_cohort_id(self):
        from bench.runners import batched, hf
        for fn in (hf.run, batched.run):
            params = inspect.signature(fn).parameters
            self.assertIn("cohort_id", params, f"{fn.__module__}.run missing cohort_id")


class CeilingRatioTest(unittest.TestCase):
    def test_ratio_computed_at_highest_shared_concurrency(self):
        ours = _rung("batched", "phase06_scheduler_matched", "c",
                     tps={c: 100.0 for c in GRID})
        vllm = _rung("vllm", "ceiling", "c", tps={c: 150.0 for c in GRID})
        c, ours_tps, vllm_tps, ratio = run_all._ceiling_ratio_at_highest_shared(ours, vllm)
        self.assertEqual(c, max(GRID))
        self.assertEqual(ours_tps, 100.0)
        self.assertEqual(vllm_tps, 150.0)
        self.assertAlmostEqual(ratio, 1.5)

    def test_ratio_uses_shared_grid_not_unrelated_peaks(self):
        # ours measured to c=32, vllm only to c=8 → compare at c=8, not peak-vs-peak
        ours = _rung("batched", "phase06_scheduler_matched", "c",
                     tps={c: 10.0 * c for c in GRID})
        vllm = _rung("vllm", "ceiling", "c", grid=[1, 2, 4, 8],
                     tps={1: 15.0, 2: 30.0, 4: 60.0, 8: 120.0})
        c, ours_tps, vllm_tps, ratio = run_all._ceiling_ratio_at_highest_shared(ours, vllm)
        self.assertEqual(c, 8)
        self.assertEqual(ours_tps, 80.0)
        self.assertEqual(vllm_tps, 120.0)
        self.assertAlmostEqual(ratio, 1.5)

    def test_returns_none_when_no_vllm(self):
        ours = _rung("batched", "phase06_scheduler_matched", "c")
        self.assertIsNone(run_all._ceiling_ratio_at_highest_shared(ours, None))


if __name__ == "__main__":
    unittest.main()
