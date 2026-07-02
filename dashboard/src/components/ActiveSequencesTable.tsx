import { Check, Square } from "lucide-react";
import { MetricsResponse } from "../lib/api";
import { StreamRun } from "../hooks/useGenerate";
import SourceBadge from "./SourceBadge";

function promptLabel(prompt: string) {
  return prompt.length > 52 ? `${prompt.slice(0, 52)}…` : prompt;
}

function startedClock(run: StreamRun) {
  // run.createdAt is a performance.now() timestamp; recover the wall-clock start.
  const wall = Date.now() - (performance.now() - run.createdAt);
  return new Date(wall).toLocaleTimeString([], { hour12: false });
}

function targetTag(model: string | undefined) {
  const match = model?.match(/(\d+(?:\.\d+)?)\s*B/i);
  return match ? `${match[1]}B` : "LLM";
}

const STATUS_LABEL: Record<StreamRun["status"], string> = {
  queued: "Queued",
  streaming: "Streaming",
  done: "Done",
  error: "Error",
  cancelled: "Cancelled"
};

export default function ActiveSequencesTable({
  metrics,
  streams,
  onStop
}: {
  metrics: MetricsResponse | null;
  streams: StreamRun[];
  onStop?: (id: string) => void;
}) {
  const target = targetTag(metrics?.model);

  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <h2>Active Sequences</h2>
          <p>{metrics ? `${metrics.active_sequences} engine active · ${metrics.waiting_sequences} queued` : "playground streams"}</p>
        </div>
        <SourceBadge kind="live" label="live/playground" />
      </div>
      <div className="sequence-table" role="table" aria-label="Dashboard-started generate streams">
        <div className="sequence-head" role="row">
          <span role="columnheader">Stream ID</span>
          <span role="columnheader">Prompt</span>
          <span role="columnheader">Tokens</span>
          <span role="columnheader">Target</span>
          <span role="columnheader">Status</span>
          <span role="columnheader">Started</span>
        </div>
        {streams.length === 0 ? (
          <div className="empty-row">No dashboard-started streams.</div>
        ) : (
          streams.slice(0, 8).map((run) => {
            const tokens = run.finalTokens ?? run.emittedChunks;
            const pct = Math.min(100, Math.round((tokens / run.maxTokens) * 100));
            const tier = pct >= 66 ? 3 : pct >= 33 ? 2 : 1;
            const done = run.status === "done";
            return (
              <div className="sequence-row" role="row" key={run.id}>
                <span role="cell" className="seq-id">
                  <span className={`seq-check ${done ? "on" : ""}`} aria-hidden="true">
                    {done ? <Check size={11} /> : null}
                  </span>
                  <span className="mono">#{run.id.slice(0, 6)}</span>
                </span>
                <span role="cell" className="request-cell" title={run.prompt}>
                  {promptLabel(run.prompt)}
                </span>
                <span role="cell" className="priority-bars">
                  <i aria-hidden="true">
                    <b className={tier >= 1 ? "on" : ""} />
                    <b className={tier >= 2 ? "on" : ""} />
                    <b className={tier >= 3 ? "on" : ""} />
                  </i>
                  <span className="mono">{tokens}</span>
                </span>
                <span role="cell" className="model-avatar">
                  <span>{target}</span>
                  <em>target</em>
                </span>
                <span role="cell" className={`stream-status ${run.status}`}>{STATUS_LABEL[run.status]}</span>
                <span role="cell" className="time-cell">
                  {startedClock(run)}
                  {onStop && run.status === "streaming" ? (
                    <button className="icon-button small" type="button" onClick={() => onStop(run.id)} aria-label="Stop stream">
                      <Square size={11} />
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
