import { MetricsPoint } from "../hooks/useMetrics";
import SourceBadge from "./SourceBadge";

function pathFor(data: MetricsPoint[], width: number, height: number) {
  if (data.length < 2) {
    return "";
  }
  const max = Math.max(...data.map((point) => point.tokensPerSecond), 1);
  return data
    .map((point, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - (point.tokensPerSecond / max) * (height - 12) - 6;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function LiveSparkline({ data }: { data: MetricsPoint[] }) {
  const width = 640;
  const height = 180;
  const path = pathFor(data, width, height);

  return (
    <section className="panel spark-panel">
      <div className="panel-heading">
        <div>
          <h2>Live tokens/sec</h2>
          <p>rolling wall-clock samples from /metrics</p>
        </div>
        <SourceBadge kind="live" />
      </div>
      <div
        className="spark-frame"
        role="img"
        aria-label="Live tokens per second sparkline sampled from metrics polling"
      >
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          {[0.33, 0.66].map((y) => (
            <line key={y} x1="0" x2={width} y1={height * y} y2={height * y} className="chart-grid" />
          ))}
          {path ? <polyline points={path} className="chart-line live-line" /> : null}
        </svg>
      </div>
    </section>
  );
}
