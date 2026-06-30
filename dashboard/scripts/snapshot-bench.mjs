import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dashboardRoot = path.resolve(here, "..");
const repoRoot = path.resolve(dashboardRoot, "..");
const resultsRoot = path.join(repoRoot, "bench", "results");
const reportPath = path.join(repoRoot, "bench", "report.md");
const outPath = path.join(dashboardRoot, "src", "data", "benchmarks.json");

function readJson(file) {
  return JSON.parse(readFileSync(file, "utf8"));
}

function sourceCommit() {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8"
    }).trim();
  } catch {
    return "unknown";
  }
}

function allResults() {
  return readdirSync(resultsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const file = path.join(resultsRoot, entry.name, "result.json");
      try {
        return { dir: entry.name, file, data: readJson(file) };
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((a, b) => b.dir.localeCompare(a.dir));
}

function latest(results, predicate) {
  return results.find((item) => predicate(item.data));
}

function round(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }
  const scale = 10 ** digits;
  return Math.round(Number(value) * scale) / scale;
}

function sweepMap(result, key) {
  const out = new Map();
  for (const point of result?.data?.concurrency_sweep ?? []) {
    out.set(point.concurrency, round(point[key], 3));
  }
  return out;
}

function parseVramTotal(env) {
  const raw = env?.vram_total_mb;
  if (typeof raw === "number") {
    return raw;
  }
  if (typeof raw === "string") {
    const match = raw.match(/[\d.]+/);
    if (match) {
      return Number(match[0]);
    }
  }
  return 32607;
}

function chartRows(hf, ours, vllm, key) {
  const hfMap = sweepMap(hf, key);
  const oursMap = sweepMap(ours, key);
  const vllmMap = sweepMap(vllm, key);
  const concurrency = [...new Set([...hfMap.keys(), ...oursMap.keys(), ...vllmMap.keys()])]
    .sort((a, b) => a - b);
  return concurrency.map((c) => ({
    concurrency: c,
    naiveHf: hfMap.get(c) ?? null,
    ours: oursMap.get(c) ?? null,
    vllm: vllmMap.get(c) ?? null
  }));
}

function specRows(stock, distilled) {
  const byGamma = new Map();
  for (const point of stock?.data?.sweep ?? []) {
    byGamma.set(point.gamma, {
      gamma: point.gamma,
      stockAlpha: round(point.alpha, 4),
      stockTokS: round(point.throughput_tok_s, 2),
      stockSpeedup: round(point.speedup_vs_baseline, 3)
    });
  }
  for (const point of distilled?.data?.sweep ?? []) {
    const row = byGamma.get(point.gamma) ?? { gamma: point.gamma };
    row.distilledAlpha = round(point.alpha, 4);
    row.distilledTokS = round(point.throughput_tok_s, 2);
    row.distilledSpeedup = round(point.speedup_vs_baseline, 3);
    byGamma.set(point.gamma, row);
  }
  return [...byGamma.values()].sort((a, b) => a.gamma - b.gamma);
}

const results = allResults();
const hf = latest(results, (r) => r.engine === "hf" && r.profile === "greedy")
  ?? latest(results, (r) => r.engine === "hf");
const ours = latest(results, (r) => r.engine === "batched" && r.role === "phase06_scheduler_matched")
  ?? latest(results, (r) => r.engine === "batched" && (r.concurrency_sweep?.length ?? 0) > 1);
const vllm = latest(results, (r) => r.engine === "vllm");
const stock = latest(results, (r) => r.engine === "spec" && r.draft_label === "stock");
const distilled = latest(results, (r) => r.engine === "spec" && r.draft_label === "distilled");
const paged = latest(results, (r) => r.engine === "paged");
const fp8 = latest(results, (r) => r.demo === "fp8_hero");

if (!hf || !ours || !stock || !distilled) {
  throw new Error("Missing required benchmark result JSON for dashboard snapshot");
}

const throughput = chartRows(hf, ours, vllm, "throughput_tok_s");
const vram = chartRows(hf, ours, vllm, "peak_vram_mb");
const highestShared = [...throughput]
  .filter((row) => row.naiveHf && row.ours)
  .sort((a, b) => b.concurrency - a.concurrency)[0];
const ratio = highestShared ? highestShared.ours / highestShared.naiveHf : null;
const rows = specRows(stock, distilled);
const alphaLifts = rows
  .filter((row) => row.stockAlpha !== undefined && row.distilledAlpha !== undefined)
  .map((row) => row.distilledAlpha - row.stockAlpha);
const gammaTwo = rows.find((row) => row.gamma === 2);
const acceptanceAlpha = gammaTwo
  ? (gammaTwo.stockAlpha + gammaTwo.distilledAlpha) / 2
  : rows.reduce((sum, row) => sum + (row.distilledAlpha ?? row.stockAlpha ?? 0), 0) / rows.length;
// Band reflects the same operating point as acceptanceAlpha (gamma=2 stock..distilled),
// not the full alpha range across all gammas — otherwise the "settled 0.66" band reads wrong.
const bandAlphas = (gammaTwo
  ? [gammaTwo.stockAlpha, gammaTwo.distilledAlpha]
  : rows.map((row) => row.distilledAlpha ?? row.stockAlpha)
).filter((value) => typeof value === "number");

let correctness = null;
try {
  correctness = readJson(path.join(resultsRoot, "correctness_summary.json"));
} catch {
  correctness = null;
}

let reportHeadline = "";
try {
  const report = readFileSync(reportPath, "utf8");
  reportHeadline = report.match(/\*\*19\.80[x×][^*]*\*\*/)?.[0] ?? "";
} catch {
  reportHeadline = "";
}

const env = ours.data.env ?? hf.data.env ?? {};
const vramTotalMb = parseVramTotal(env);
const snapshot = {
  generatedAt: new Date().toISOString(),
  sourceCommit: sourceCommit(),
  benchmarkCommit: env.git_commit ?? "unknown",
  reportPath: "bench/report.md",
  reportHeadline,
  sources: {
    hf: hf.dir,
    ours: ours.dir,
    vllm: vllm?.dir ?? null,
    stockSpec: stock.dir,
    distilledSpec: distilled.dir,
    paged: paged?.dir ?? null,
    fp8Hero: fp8?.dir ?? null
  },
  environment: {
    gpuName: env.gpu_name ?? "NVIDIA GeForce RTX 5090",
    cudaVersion: env.cuda_version ?? null,
    driverVersion: env.driver_version ?? null,
    torch: env.torch ?? null,
    transformers: env.transformers ?? null,
    python: env.python ?? null,
    vramTotalMb
  },
  throughput: {
    rows: throughput,
    headline: {
      concurrency: highestShared?.concurrency ?? 32,
      naiveHfTokS: round(highestShared?.naiveHf, 1),
      oursTokS: round(highestShared?.ours, 1),
      speedup: round(ratio, 2)
    },
    vllmStatus: vllm?.data?.role === "ceiling_deferred"
      ? (vllm.data.notes?.[0] ?? "vLLM ceiling deferred")
      : "available"
  },
  vram: {
    rows: vram,
    totalMb: vramTotalMb,
    pagedMicrobench: paged?.data?.points ?? []
  },
  specDecode: {
    rows,
    stockBaselineTokS: round(stock.data.baseline_tok_s, 2),
    distilledBaselineTokS: round(distilled.data.baseline_tok_s, 2),
    alphaLiftMean: round(alphaLifts.reduce((sum, value) => sum + value, 0) / alphaLifts.length, 3),
    alphaLiftMax: round(Math.max(...alphaLifts), 3),
    correctness: correctness
      ? {
          passed: Boolean(correctness.passed),
          n: correctness.n,
          length: correctness.length,
          gamma: correctness.gamma,
          lines: correctness.lines ?? []
        }
      : null,
    caveat: "Spec decode measured 0.6-0.7x target-only throughput on this hybrid-attention pair; correctness and alpha lift are the deliverables."
  },
  acceptance: {
    alpha: round(acceptanceAlpha, 2),
    label: "benchmark mean of stock/distilled gamma=2",
    minBand: round(Math.min(...bandAlphas), 2),
    maxBand: round(Math.max(...bandAlphas), 2)
  },
  fp8Hero: fp8?.data?.variants?.fp8
    ? {
        model: fp8.data.model,
        weightFootprintMb: round(fp8.data.variants.fp8.weight_footprint_mb, 1),
        peakVramMb: round(fp8.data.variants.fp8.peak_vram_mb, 1),
        meanDecodeTokS: round(fp8.data.variants.fp8.mean_decode_tok_s, 3),
        meanTtftS: round(fp8.data.variants.fp8.mean_ttft_s, 2)
      }
    : null
};

mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(snapshot, null, 2)}\n`);
console.log(`wrote ${path.relative(repoRoot, outPath)}`);
