import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import RadialGauge from "../components/RadialGauge";
import SourceBadge from "../components/SourceBadge";
import SpecDecodeTimeline from "../components/SpecDecodeTimeline";
import { sourceLabel } from "../lib/benchmarks";
import { useDashboard } from "../lib/dashboard";
import { fixed, rate } from "../lib/format";

function SpecChart({ mode }: { mode: "alpha" | "speedup" }) {
  const { benchmarks } = useDashboard();
  const rows = benchmarks.specDecode.rows;
  const stockKey = mode === "alpha" ? "stockAlpha" : "stockSpeedup";
  const distilledKey = mode === "alpha" ? "distilledAlpha" : "distilledSpeedup";

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
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 10, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,.07)" vertical={false} />
            <XAxis dataKey="gamma" tick={{ fill: "#85868b", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#85868b", fontSize: 11 }} axisLine={false} tickLine={false} width={42} />
            <Tooltip
              contentStyle={{
                background: "#1c1c20",
                border: "1px solid rgba(255,255,255,.12)",
                borderRadius: 14,
                color: "#f2f3f4"
              }}
            />
            <Legend wrapperStyle={{ color: "#85868b", fontSize: 12 }} />
            <Line type="monotone" dataKey={stockKey} name="stock draft" stroke="#75767b" strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
            <Line type="monotone" dataKey={distilledKey} name="distilled draft" stroke="#f2f3f5" strokeWidth={3} dot={{ r: 4 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
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
