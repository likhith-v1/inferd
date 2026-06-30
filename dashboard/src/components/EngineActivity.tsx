import { MetricsResponse } from "../lib/api";
import { compact } from "../lib/format";
import SourceBadge from "./SourceBadge";

function bar(value: number, max: number) {
  if (max <= 0) {
    return 0;
  }
  return Math.min(100, Math.max(4, (value / max) * 100));
}

export default function EngineActivity({ metrics }: { metrics: MetricsResponse | null }) {
  const maxBlocks = metrics ? metrics.used_blocks + metrics.free_blocks : 0;
  const rows = [
    ["Iterations", compact(metrics?.iterations), bar(metrics?.iterations ?? 0, Math.max(metrics?.iterations ?? 0, 1))],
    ["Tokens", compact(metrics?.total_generated_tokens), bar(metrics?.total_generated_tokens ?? 0, Math.max(metrics?.total_generated_tokens ?? 0, 1))],
    ["Sequences", compact(metrics?.admitted_sequences), bar(metrics?.admitted_sequences ?? 0, Math.max(metrics?.admitted_sequences ?? 0, 1))],
    ["Blocks used", compact(metrics?.used_blocks), bar(metrics?.used_blocks ?? 0, maxBlocks)]
  ] as const;

  return (
    <section className="panel activity-panel">
      <div className="panel-heading">
        <div>
          <h2>Engine activity</h2>
          <p>{metrics ? `${metrics.active_sequences} streaming · ${metrics.waiting_sequences} queued` : "awaiting /metrics"}</p>
        </div>
        <SourceBadge kind="live" />
      </div>
      <div className="activity-grid">
        {rows.map(([label, value, width]) => (
          <div className="activity-item" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <div className="activity-bar" aria-hidden="true">
              <i style={{ width: `${width}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

