import { FormEvent, useState } from "react";
import { Play, Send, Trash2, Zap } from "lucide-react";
import { useDashboard } from "../lib/dashboard";

export default function Playground({ compact = false }: { compact?: boolean }) {
  const { generation } = useDashboard();
  const [prompt, setPrompt] = useState(generation.samplePrompts[0]);
  const [maxTokens, setMaxTokens] = useState(96);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    generation.startGeneration(prompt, maxTokens);
  };

  return (
    <section className={`panel playground ${compact ? "compact" : ""}`}>
      <div className="panel-heading">
        <div>
          <h2>Generate playground</h2>
          <p>POST /generate · SSE stream</p>
        </div>
        <button className="icon-button" type="button" onClick={generation.clearTerminal} aria-label="Clear completed streams">
          <Trash2 size={17} />
        </button>
      </div>
      <form onSubmit={submit} className="play-form">
        <label>
          <span>Prompt</span>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={compact ? 3 : 5} />
        </label>
        <div className="play-controls">
          <label>
            <span>Max tokens</span>
            <input
              type="number"
              min={1}
              max={512}
              value={maxTokens}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
            />
          </label>
          <button className="primary-button" type="submit">
            <Send size={16} aria-hidden="true" />
            Send
          </button>
        </div>
      </form>
      <div className="burst-row" aria-label="Demo load controls">
        {[1, 4, 8].map((count) => (
          <button key={count} type="button" onClick={() => generation.startDemoLoad(count, maxTokens)}>
            {count === 1 ? <Play size={15} aria-hidden="true" /> : <Zap size={15} aria-hidden="true" />}
            {count} stream{count > 1 ? "s" : ""}
          </button>
        ))}
      </div>
    </section>
  );
}

