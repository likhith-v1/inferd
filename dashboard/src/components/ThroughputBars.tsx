import { BarChart3 } from "lucide-react";
import { BenchmarkRow } from "../lib/benchmarks";
import { fixed } from "../lib/format";
import SourceBadge from "./SourceBadge";

interface Props {
  rows: BenchmarkRow[];
  peakConcurrency: number;
  delta?: string;
  ariaLabel: string;
}

function niceMax(peak: number) {
  if (peak <= 0) {
    return 100;
  }
  return Math.ceil(peak / 100) * 100;
}

export default function ThroughputBars({ rows, peakConcurrency, delta, ariaLabel }: Props) {
  const values = rows.map((row) => row.ours ?? 0);
  const peak = Math.max(...values, 0);
  const max = niceMax(peak);
  const ticks = [1, 0.8, 0.6, 0.4, 0.2, 0].map((frac) => Math.round(max * frac));
  const refTop = `${(1 - peak / max) * 100}%`;

  return (
    <section className="panel">
      <div className="panel-heading">
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <BarChart3 size={17} color="var(--accent)" aria-hidden="true" />
          <h2>Throughput vs concurrency</h2>
        </div>
        <SourceBadge kind="benchmark" label="batched run" />
      </div>

      <div className="tp-headline">
        <span className="val">{fixed(peak, 1)}</span>
        <span className="unit">tok/s peak</span>
        {delta ? (
          <span className="delta">
            <b>{delta}</b> vs naive HF
          </span>
        ) : null}
      </div>

      <div className="throughput-bars" role="img" aria-label={ariaLabel}>
        <div className="tp-ref" style={{ top: refTop }} />
        <div className="tp-yaxis">
          {ticks.map((tick, index) => (
            <span key={index}>{tick}</span>
          ))}
        </div>
        <div className="tp-bars">
          {rows.map((row) => {
            const value = row.ours ?? 0;
            const isPeak = row.concurrency === peakConcurrency;
            const height = `${Math.max(2, (value / max) * 100)}%`;
            return (
              <div className="tp-col" key={row.concurrency}>
                {isPeak ? (
                  <div className="tp-label">
                    c={row.concurrency} : {fixed(value, 1)}
                  </div>
                ) : null}
                <div className={`tp-bar ${isPeak ? "peak" : ""}`} style={{ height }} />
              </div>
            );
          })}
        </div>
        <div className="tp-xaxis">
          {rows.map((row) => (
            <span key={row.concurrency}>c={row.concurrency}</span>
          ))}
        </div>
      </div>
    </section>
  );
}
