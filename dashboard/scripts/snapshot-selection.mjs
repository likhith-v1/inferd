export const PHASE17_CONCURRENCY_GRID = [1, 2, 4, 8, 16, 32];

function sameArray(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function pointGrid(item) {
  return (item?.data?.concurrency_sweep ?? [])
    .map((point) => point.concurrency)
    .sort((a, b) => a - b);
}

function hasCanonicalProvenance(item) {
  const provenance = item?.data?.provenance;
  return Boolean(
    item?.data?.cohort_id
      && provenance
      && typeof provenance.model_fingerprint === "string"
      && provenance.model_fingerprint.length > 0
      && typeof provenance.workload_hash === "string"
      && provenance.workload_hash.length > 0
      && provenance.profile === "greedy"
      && provenance.seed === 0
      && provenance.max_tokens === 96
      && provenance.warmup_runs === 3
      && Array.isArray(provenance.concurrency_grid)
      && sameArray(provenance.concurrency_grid, PHASE17_CONCURRENCY_GRID)
      && sameArray(pointGrid(item), PHASE17_CONCURRENCY_GRID)
  );
}

function newest(items, predicate) {
  return items.find((item) => predicate(item.data));
}

/**
 * Select a complete Phase 17 three-rung cohort as one indivisible unit.
 * `results` must be newest-first, matching allResults() in snapshot-bench.mjs.
 */
export function selectPhase17Cohort(results) {
  const cohortIds = [...new Set(
    results.map((item) => item.data?.cohort_id).filter((value) => typeof value === "string" && value)
  )];

  for (const cohortId of cohortIds) {
    const cohort = results.filter((item) => item.data?.cohort_id === cohortId);
    const hf = newest(cohort, (result) => result.engine === "hf");
    // Only the merged matched result is the "ours" rung; the per-concurrency
    // batched byproducts share the cohort_id but carry an incomplete grid.
    // (Mirrors bench.run_all._rung_name, the Python selection authority.)
    const ours = newest(
      cohort,
      (result) => result.engine === "batched" && result.role === "phase06_scheduler_matched"
    );
    const vllm = newest(
      cohort,
      (result) => result.engine === "vllm" && result.role !== "ceiling_deferred"
    );
    if (!hf || !ours || !vllm || ![hf, ours, vllm].every(hasCanonicalProvenance)) {
      continue;
    }

    const provenance = [hf, ours, vllm].map((item) => item.data.provenance);
    if (new Set(provenance.map((item) => item.model_fingerprint)).size !== 1
      || new Set(provenance.map((item) => item.workload_hash)).size !== 1) {
      continue;
    }
    return { cohortId, hf, ours, vllm, provenance: provenance[0] };
  }
  return null;
}

/** Preserve the shipped two-rung dashboard until a complete Phase 17 cohort exists. */
export function selectLegacyTwoRung(results) {
  const legacy = results.filter((item) => !item.data?.cohort_id);
  const hf = newest(legacy, (result) => result.engine === "hf" && result.profile === "greedy")
    ?? newest(legacy, (result) => result.engine === "hf");
  const ours = newest(
    legacy,
    (result) => result.engine === "batched" && result.role === "phase06_scheduler_matched"
  ) ?? newest(
    legacy,
    (result) => result.engine === "batched" && (result.concurrency_sweep?.length ?? 0) > 1
  );
  return hf && ours ? { hf, ours } : null;
}

export function latestDeferredVllm(results) {
  return newest(
    results,
    (result) => result.engine === "vllm" && result.role === "ceiling_deferred"
  ) ?? null;
}
