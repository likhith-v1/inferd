import { Square } from "lucide-react";
import { MetricsResponse } from "../lib/api";
import { StreamRun } from "../hooks/useGenerate";
import { fixed } from "../lib/format";
import SourceBadge from "./SourceBadge";

function promptLabel(prompt: string) {
  return prompt.length > 42 ? `${prompt.slice(0, 42)}...` : prompt;
}

function runtime(run: StreamRun) {
  const start = run.startedAt ?? run.createdAt;
  const end = run.completedAt ?? performance.now();
  return `${fixed((end - start) / 1000, 1)}s`;
}

export default function ActiveSequencesTable({
  metrics,
  streams,
  onStop
}: {
  metrics: MetricsResponse | null;
  streams: StreamRun[];
  onStop?: (id: string) => void;
}) {
  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <h2>Active sequences</h2>
          <p>{metrics ? `${metrics.active_sequences} engine active · ${metrics.waiting_sequences} queued` : "playground streams"}</p>
        </div>
        <SourceBadge kind="live" label="live/playground" />
      </div>
      <div className="sequence-table" role="table" aria-label="Dashboard-started generate streams">
        <div className="sequence-head" role="row">
          <span role="columnheader">Request</span>
          <span role="columnheader">Tokens</span>
          <span role="columnheader">Status</span>
          <span role="columnheader">Completion</span>
          <span role="columnheader">Time</span>
        </div>
        {streams.length === 0 ? (
          <div className="empty-row">No dashboard-started streams.</div>
        ) : (
          streams.slice(0, 8).map((run) => {
            const tokens = run.finalTokens ?? run.emittedChunks;
            const pct = Math.min(100, Math.round((tokens / run.maxTokens) * 100));
            return (
              <div className="sequence-row" role="row" key={run.id}>
                <span role="cell" className="request-cell" title={run.prompt}>{promptLabel(run.prompt)}</span>
                <span role="cell" className="mono">{tokens}</span>
                <span role="cell" className={`stream-status ${run.status}`}>{run.status}</span>
                <span role="cell" className="progress-cell">
                  <i aria-hidden="true"><b style={{ width: `${pct}%` }} /></i>
                  <em>{pct}%</em>
                </span>
                <span role="cell" className="time-cell">
                  {runtime(run)}
                  {onStop && run.status === "streaming" ? (
                    <button className="icon-button small" type="button" onClick={() => onStop(run.id)} aria-label="Stop stream">
                      <Square size={12} />
                    </button>
                  ) : null}
                </span>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}

