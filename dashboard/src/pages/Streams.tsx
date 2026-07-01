import ActiveSequencesTable from "../components/ActiveSequencesTable";
import Playground from "../components/Playground";
import { useDashboard } from "../lib/dashboard";
import { fixed } from "../lib/format";

function ttft(run: { firstTokenAt: number | null; createdAt: number }) {
  if (!run.firstTokenAt) {
    return "—";
  }
  return `${fixed((run.firstTokenAt - run.createdAt) / 1000, 2)}s`;
}

export default function Streams() {
  const { generation, metrics } = useDashboard();

  return (
    <div className="page-stack">
      <div>
        <h1 className="page-title">Streams</h1>
        <p className="page-subtitle">Drive the engine from the playground and watch live SSE generate streams.</p>
      </div>
      <div className="streams-layout">
        <Playground />
        <ActiveSequencesTable
          metrics={metrics.data}
          streams={generation.streams}
          onStop={generation.stopGeneration}
        />
      </div>
      <section className="panel stream-output-panel">
        <div className="panel-heading">
          <div>
            <h2>Stream output</h2>
            <p>{generation.active.length} active dashboard streams</p>
          </div>
        </div>
        <div className="stream-list">
          {generation.streams.length === 0 ? (
            <div className="empty-row">No stream output yet.</div>
          ) : (
            generation.streams.map((run) => (
              <article className="stream-row" key={run.id}>
                <header>
                  <strong>{run.prompt}</strong>
                  <span className={`stream-status ${run.status}`}>{run.status}</span>
                </header>
                <div className="stream-meta">
                  <span>TTFT {ttft(run)}</span>
                  <span>{run.finalTokens ?? run.emittedChunks} tokens</span>
                  {run.statusCode ? <span>HTTP {run.statusCode}</span> : null}
                </div>
                <pre>{run.error ?? (run.text || "...")}</pre>
              </article>
            ))
          )}
        </div>
      </section>
    </div>
  );
}
