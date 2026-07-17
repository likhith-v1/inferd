import { Activity, Cpu, Gauge, Trophy } from "lucide-react";
import KpiCard from "../components/KpiCard";
import SourceBadge from "../components/SourceBadge";
import ThroughputChart from "../components/ThroughputChart";
import { sourceLabel } from "../lib/benchmarks";
import { useDashboard } from "../lib/dashboard";
import { fixed, mibToGib, rate } from "../lib/format";

export default function Benchmarks() {
  const { benchmarks } = useDashboard();
  const headline = benchmarks.throughput.headline;
  const ceiling = benchmarks.throughput.ceiling;
  const range = benchmarks.throughput.oursVsHfRange;

  // Lead with the stable, reproducible ceiling ratio. The ours-vs-HF ratio is
  // shown as a range because the naive HF floor at high concurrency is not
  // reproducible (thrashes at the VRAM edge) — see the report's Reproducibility.
  const rangeLabel = range
    ? `~${fixed(range.low, 0)}–${fixed(range.high, 0)}x`
    : headline.speedup
      ? `${fixed(headline.speedup, 1)}x`
      : "—";

  return (
    <div className="page-stack">
      <div>
        <h1 className="page-title">Benchmarks</h1>
        <p className="page-subtitle">inferd against the naive HF floor and the vLLM ceiling — snapshot from bench/report.md.</p>
      </div>
      <div className="kpi-grid">
        {ceiling ? (
          <KpiCard title="Within Nx of vLLM" value={`${fixed(ceiling.ratio, 2)}x`} detail={`c=${ceiling.concurrency} ceiling · reproducible`} source="benchmark" icon={Trophy} />
        ) : (
          <KpiCard title="Headline speedup" value={`${fixed(headline.speedup, 2)}x`} detail={`c=${headline.concurrency} vs naive HF`} source="benchmark" icon={Trophy} />
        )}
        <KpiCard title="inferd c=32" value={rate(ceiling?.oursTokS ?? headline.oursTokS, 1)} detail="continuous batching" source="benchmark" icon={Activity} />
        <KpiCard title="vLLM ceiling c=32" value={rate(ceiling?.vllmTokS ?? null, 1)} detail="reference engine" source="benchmark" icon={Gauge} />
        <KpiCard title="VRAM total" value={mibToGib(benchmarks.environment.vramTotalMb, 1)} detail={benchmarks.environment.gpuName} source="benchmark" icon={Cpu} />
      </div>

      <div className="two-col wide-left">
        <ThroughputChart
          rows={benchmarks.throughput.rows}
          subtitle={sourceLabel(benchmarks)}
          ariaLabel="Benchmark throughput by concurrency"
        />
        <section className="panel fact-panel">
          <div className="panel-heading">
            <div>
              <h2>Snapshot provenance</h2>
              <p>{benchmarks.reportPath}</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <dl className="fact-list">
            <div><dt>snapshot commit</dt><dd>{benchmarks.sourceCommit}</dd></div>
            <div><dt>benchmark commit</dt><dd>{benchmarks.benchmarkCommit}</dd></div>
            <div><dt>cohort</dt><dd>{benchmarks.cohort.id ?? benchmarks.cohort.status}</dd></div>
            <div><dt>ours vs naive HF (c=32)</dt><dd>{rangeLabel} · floor VRAM-thrash noisy</dd></div>
            <div><dt>generated</dt><dd>{new Date(benchmarks.generatedAt).toLocaleString()}</dd></div>
            <div><dt>vLLM</dt><dd>{benchmarks.throughput.vllmStatus}</dd></div>
          </dl>
        </section>
      </div>

      <div className="two-col">
        <section className="panel table-panel">
          <div className="panel-heading">
            <div>
              <h2>Throughput table</h2>
              <p>tokens/sec, matched workload</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>c</th>
                <th>naive HF</th>
                <th>inferd</th>
                <th>ratio</th>
                <th>vLLM</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.throughput.rows.map((row) => (
                <tr key={row.concurrency}>
                  <td>{row.concurrency}</td>
                  <td>{fixed(row.naiveHf, 1)}</td>
                  <td>{fixed(row.ours, 1)}</td>
                  <td>{row.naiveHf && row.ours ? `${fixed(row.ours / row.naiveHf, 2)}x` : "—"}</td>
                  <td>{fixed(row.vllm, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="panel table-panel">
          <div className="panel-heading">
            <div>
              <h2>VRAM table</h2>
              <p>MiB peak from nvidia-smi</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>c</th>
                <th>naive HF</th>
                <th>inferd</th>
                <th>vLLM</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.vram.rows.map((row) => (
                <tr key={row.concurrency}>
                  <td>{row.concurrency}</td>
                  <td>{fixed(row.naiveHf, 0)}</td>
                  <td>{fixed(row.ours, 0)}</td>
                  <td>{fixed(row.vllm, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
