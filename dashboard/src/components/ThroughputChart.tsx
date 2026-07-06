import { BenchmarkRow } from "../lib/benchmarks";
import SourceBadge from "./SourceBadge";

interface Props {
  rows: BenchmarkRow[];
  title?: string;
  subtitle?: string;
  metric?: "tok/s" | "MiB";
  ariaLabel: string;
}

function linePath(rows: BenchmarkRow[], key: keyof BenchmarkRow, width: number, height: number) {
  const values = rows
    .map((row) => (typeof row[key] === "number" ? Number(row[key]) : null))
    .filter((value): value is number => value !== null);
  const max = Math.max(...values, 1);
  return rows
    .map((row, index) => {
      const value = typeof row[key] === "number" ? Number(row[key]) : null;
      if (value === null) {
        return "";
      }
      const x = rows.length <= 1 ? width / 2 : (index / (rows.length - 1)) * width;
      const y = height - (value / max) * (height - 16) - 8;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .filter(Boolean)
    .join(" ");
}

export default function ThroughputChart({
  rows,
  title = "Throughput vs concurrency",
  subtitle,
  ariaLabel
}: Props) {
  const width = 640;
  const height = 220;
  const xTicks = rows.map((row, index) => {
    const x = rows.length <= 1 ? width / 2 : (index / (rows.length - 1)) * width;
    return { x, label: row.concurrency };
  });

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        <SourceBadge kind="benchmark" />
      </div>
      <div className="chart-frame" role="img" aria-label={ariaLabel}>
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          {[0.25, 0.5, 0.75].map((y) => (
            <line key={y} x1="0" x2={width} y1={height * y} y2={height * y} className="chart-grid" />
          ))}
          <polyline points={linePath(rows, "naiveHf", width, height)} className="chart-line hf-line" />
          <polyline points={linePath(rows, "ours", width, height)} className="chart-line ours-line" />
          <polyline points={linePath(rows, "vllm", width, height)} className="chart-line vllm-line" />
          {xTicks.map((tick) => (
            <text key={tick.label} x={tick.x} y={height - 2} className="chart-tick">
              {tick.label}
            </text>
          ))}
        </svg>
      </div>
      <div className="legend-row" aria-hidden="true">
        <span><i className="legend ours" />inferd</span>
        <span><i className="legend hf" />naive HF</span>
        <span><i className="legend vllm" />vLLM pending</span>
      </div>
    </section>
  );
}
