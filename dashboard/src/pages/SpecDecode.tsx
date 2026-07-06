import RadialGauge from "../components/RadialGauge";
import SourceBadge from "../components/SourceBadge";
import SpecDecodeTimeline from "../components/SpecDecodeTimeline";
import { sourceLabel } from "../lib/benchmarks";
import { useDashboard } from "../lib/dashboard";
import { fixed, rate } from "../lib/format";

type SpecKey = "stockAlpha" | "stockSpeedup" | "distilledAlpha" | "distilledSpeedup";

function specMax(rows: { [K in SpecKey]?: number }[], keys: SpecKey[]) {
  return Math.max(
    ...rows.flatMap((row) =>
      keys.map((key) => row[key]).filter((value): value is number => value !== undefined)
    ),
    1
  );
}

function specPath(
  rows: { [K in SpecKey]?: number }[],
  key: SpecKey,
  width: number,
  height: number,
  max: number
) {
  return rows
    .map((row, index) => {
      const x = rows.length <= 1 ? width / 2 : (index / (rows.length - 1)) * width;
      const y = height - ((row[key] ?? 0) / max) * (height - 16) - 8;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function SpecChart({ mode }: { mode: "alpha" | "speedup" }) {
  const { benchmarks } = useDashboard();
  const rows = benchmarks.specDecode.rows;
  const stockKey: SpecKey = mode === "alpha" ? "stockAlpha" : "stockSpeedup";
  const distilledKey: SpecKey = mode === "alpha" ? "distilledAlpha" : "distilledSpeedup";
  const width = 640;
  const height = 220;
  const max = specMax(rows, [stockKey, distilledKey]);

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <h2>{mode === "alpha" ? "Alpha vs gamma" : "Speedup vs gamma"}</h2>
          <p>{sourceLabel(benchmarks)}</p>
        </div>
        <SourceBadge kind="benchmark" />
      </div>
      <div className="chart-frame" role="img" aria-label={`Spec decode ${mode} by gamma`}>
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          {[0.25, 0.5, 0.75].map((y) => (
            <line key={y} x1="0" x2={width} y1={height * y} y2={height * y} className="chart-grid" />
          ))}
          <polyline points={specPath(rows, stockKey, width, height, max)} className="chart-line vllm-line" />
          <polyline points={specPath(rows, distilledKey, width, height, max)} className="chart-line ours-line" />
          {rows.map((row, index) => {
            const x = rows.length <= 1 ? width / 2 : (index / (rows.length - 1)) * width;
            return (
              <text key={row.gamma} x={x} y={height - 2} className="chart-tick">
                {row.gamma}
              </text>
            );
          })}
        </svg>
      </div>
      <div className="legend-row" aria-hidden="true">
        <span><i className="legend vllm" />stock draft</span>
        <span><i className="legend ours" />distilled draft</span>
      </div>
    </section>
  );
}

export default function SpecDecode() {
  const { benchmarks } = useDashboard();
  const correctness = benchmarks.specDecode.correctness;

  return (
    <div className="page-stack">
      <div>
        <h1 className="page-title">Speculative decoding</h1>
        <p className="page-subtitle">Exact rejection sampling with residual resampling — the correctness proof and α-lift are the deliverables, not raw speed.</p>
      </div>
      <div className="spec-grid">
        <RadialGauge
          title="Draft acceptance"
          subtitle={benchmarks.acceptance.label}
          value={benchmarks.acceptance.alpha}
          display={fixed(benchmarks.acceptance.alpha, 2)}
          detail={`${fixed(benchmarks.acceptance.minBand, 2)}-${fixed(benchmarks.acceptance.maxBand, 2)} band`}
          source="benchmark"
          variant="arc"
          ariaLabel={`Draft acceptance ${fixed(benchmarks.acceptance.alpha, 2)}`}
        />
        <section className="panel fact-panel">
          <div className="panel-heading">
            <div>
              <h2>Correctness gate</h2>
              <p>multi-token per-position TV test</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <strong className={correctness?.passed ? "pass-text" : "warn-text"}>
            {correctness ? (correctness.passed ? "PASS" : "FAIL") : "not available"}
          </strong>
          <dl className="fact-grid">
            <div><dt>n</dt><dd>{correctness?.n ?? "—"}</dd></div>
            <div><dt>length</dt><dd>{correctness?.length ?? "—"}</dd></div>
            <div><dt>gamma</dt><dd>{correctness?.gamma ?? "—"}</dd></div>
          </dl>
        </section>
        <section className="panel fact-panel">
          <div className="panel-heading">
            <div>
              <h2>Throughput caveat</h2>
              <p>target-only baseline comparison</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <p className="caveat-copy">{benchmarks.specDecode.caveat}</p>
          <dl className="fact-grid">
            <div><dt>stock baseline</dt><dd>{rate(benchmarks.specDecode.stockBaselineTokS, 1)}</dd></div>
            <div><dt>distilled baseline</dt><dd>{rate(benchmarks.specDecode.distilledBaselineTokS, 1)}</dd></div>
          </dl>
        </section>
      </div>
      <div className="two-col">
        <SpecChart mode="alpha" />
        <SpecChart mode="speedup" />
      </div>
      <div className="two-col">
        <SpecDecodeTimeline snapshot={benchmarks} />
        <section className="panel table-panel">
          <div className="panel-heading">
            <div>
              <h2>Gamma sweep</h2>
              <p>{sourceLabel(benchmarks)}</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>gamma</th>
                <th>stock alpha</th>
                <th>distilled alpha</th>
                <th>stock speedup</th>
                <th>distilled speedup</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.specDecode.rows.map((row) => (
                <tr key={row.gamma}>
                  <td>{row.gamma}</td>
                  <td>{fixed(row.stockAlpha, 3)}</td>
                  <td>{fixed(row.distilledAlpha, 3)}</td>
                  <td>{fixed(row.stockSpeedup, 3)}x</td>
                  <td>{fixed(row.distilledSpeedup, 3)}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
