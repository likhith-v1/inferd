import assert from "node:assert/strict";
import test from "node:test";
import {
  PHASE17_CONCURRENCY_GRID,
  latestDeferredVllm,
  selectLegacyTwoRung,
  selectPhase17Cohort
} from "./snapshot-selection.mjs";

function result(dir, engine, cohortId, overrides = {}) {
  const fingerprint = overrides.fingerprint ?? "sha256:model";
  const workloadHash = overrides.workloadHash ?? "workload";
  const grid = overrides.grid ?? PHASE17_CONCURRENCY_GRID;
  // The "ours" rung is the merged matched batched result; default batched
  // fixtures to that role (mirrors bench.run_all._rung_name).
  const role =
    overrides.role ?? (engine === "batched" ? "phase06_scheduler_matched" : undefined);
  return {
    dir,
    data: {
      engine,
      cohort_id: cohortId,
      role,
      provenance: {
        model_fingerprint: fingerprint,
        workload_hash: workloadHash,
        profile: "greedy",
        seed: 0,
        max_tokens: 96,
        warmup_runs: 3,
        concurrency_grid: PHASE17_CONCURRENCY_GRID
      },
      concurrency_sweep: grid.map((concurrency) => ({ concurrency }))
    }
  };
}

test("selects the newest complete cohort and ignores a newer deferred result", () => {
  const results = [
    result("20260717_vllm_deferred", "vllm", "newer-partial", { role: "ceiling_deferred" }),
    result("20260716_vllm", "vllm", "complete"),
    result("20260716_ours", "batched", "complete"),
    result("20260716_hf", "hf", "complete")
  ];
  const selected = selectPhase17Cohort(results);
  assert.equal(selected.cohortId, "complete");
  assert.equal(selected.vllm.dir, "20260716_vllm");
  assert.equal(latestDeferredVllm(results).dir, "20260717_vllm_deferred");
});

test("picks the merged matched batched result, not per-concurrency byproducts", () => {
  // The ours rung writes one merged matched result plus several per-concurrency
  // batched byproducts that share the cohort_id but carry an incomplete grid.
  // Selection must pick the merged one, or the cohort is wrongly rejected.
  const results = [
    result("20260716_ours_c32", "batched", "complete", {
      role: "phase06_scheduler",
      grid: [32]
    }),
    result("20260716_ours_c1", "batched", "complete", {
      role: "phase06_scheduler",
      grid: [1]
    }),
    result("20260716_ours_matched", "batched", "complete"),
    result("20260716_vllm", "vllm", "complete"),
    result("20260716_hf", "hf", "complete")
  ];
  const selected = selectPhase17Cohort(results);
  assert.notEqual(selected, null);
  assert.equal(selected.cohortId, "complete");
  assert.equal(selected.ours.dir, "20260716_ours_matched");
});

test("rejects a cohort whose model fingerprint differs between rungs", () => {
  const results = [
    result("vllm", "vllm", "mixed", { fingerprint: "sha256:other" }),
    result("ours", "batched", "mixed"),
    result("hf", "hf", "mixed")
  ];
  assert.equal(selectPhase17Cohort(results), null);
});

test("rejects a cohort with an incomplete measured concurrency grid", () => {
  const results = [
    result("vllm", "vllm", "partial", { grid: [1, 2, 4, 8, 16] }),
    result("ours", "batched", "partial"),
    result("hf", "hf", "partial")
  ];
  assert.equal(selectPhase17Cohort(results), null);
});

test("legacy fallback never mixes cohort-tagged results", () => {
  const legacyHf = { dir: "legacy-hf", data: { engine: "hf", profile: "greedy" } };
  const legacyOurs = {
    dir: "legacy-ours",
    data: { engine: "batched", role: "phase06_scheduler_matched", concurrency_sweep: [{ concurrency: 1 }] }
  };
  const selected = selectLegacyTwoRung([
    result("partial-hf", "hf", "partial"),
    legacyOurs,
    legacyHf
  ]);
  assert.equal(selected.hf.dir, "legacy-hf");
  assert.equal(selected.ours.dir, "legacy-ours");
});
