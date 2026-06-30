export interface BenchmarkRow {
  concurrency: number;
  naiveHf: number | null;
  ours: number | null;
  vllm: number | null;
}

export interface SpecRow {
  gamma: number;
  stockAlpha?: number;
  stockTokS?: number;
  stockSpeedup?: number;
  distilledAlpha?: number;
  distilledTokS?: number;
  distilledSpeedup?: number;
}

export interface BenchmarkSnapshot {
  generatedAt: string;
  sourceCommit: string;
  benchmarkCommit: string;
  reportPath: string;
  reportHeadline: string;
  sources: Record<string, string | null>;
  environment: {
    gpuName: string;
    cudaVersion: string | null;
    driverVersion: string | null;
    torch: string | null;
    transformers: string | null;
    python: string | null;
    vramTotalMb: number;
  };
  throughput: {
    rows: BenchmarkRow[];
    headline: {
      concurrency: number;
      naiveHfTokS: number | null;
      oursTokS: number | null;
      speedup: number | null;
    };
    vllmStatus: string;
  };
  vram: {
    rows: BenchmarkRow[];
    totalMb: number;
    pagedMicrobench: Array<Record<string, number | string | number[]>>;
  };
  specDecode: {
    rows: SpecRow[];
    stockBaselineTokS: number | null;
    distilledBaselineTokS: number | null;
    alphaLiftMean: number | null;
    alphaLiftMax: number | null;
    correctness: {
      passed: boolean;
      n: number;
      length: number;
      gamma: number;
      lines: string[];
    } | null;
    caveat: string;
  };
  acceptance: {
    alpha: number | null;
    label: string;
    minBand: number | null;
    maxBand: number | null;
  };
  fp8Hero: {
    model: string;
    weightFootprintMb: number;
    peakVramMb: number;
    meanDecodeTokS: number;
    meanTtftS: number;
  } | null;
}

export function sourceLabel(snapshot: BenchmarkSnapshot) {
  return `benchmark · report.md @ ${snapshot.benchmarkCommit}`;
}

