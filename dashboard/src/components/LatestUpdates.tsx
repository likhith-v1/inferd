import {
  AlertTriangle,
  BarChart3,
  Check,
  CircleX,
  Plus,
  ShieldCheck,
  Zap,
  type LucideIcon
} from "lucide-react";
import { MetricsResponse } from "../lib/api";
import { StreamRun } from "../hooks/useGenerate";
import { BenchmarkSnapshot } from "../lib/benchmarks";
import { fixed, mibToGib, seconds } from "../lib/format";

interface Update {
  key: string;
  icon: LucideIcon;
  tone: "accent" | "neutral" | "warn";
  title: string;
  detail: string;
}

function shortId(id: string) {
  return `#${id.slice(0, 6)}`;
}

function streamUpdate(run: StreamRun): Update {
  const tokens = run.finalTokens ?? run.emittedChunks;
  const ttft = run.firstTokenAt ? `${seconds((run.firstTokenAt - run.createdAt) / 1000, 2)} TTFT` : "no TTFT yet";
  switch (run.status) {
    case "done":
      return { key: run.id, icon: Check, tone: "accent", title: "Sequence completed", detail: `${shortId(run.id)} · ${tokens} tokens · ${ttft}` };
    case "streaming":
      return { key: run.id, icon: Zap, tone: "accent", title: "Sequence streaming", detail: `${shortId(run.id)} · ${tokens} tokens · ${ttft}` };
    case "queued":
      return { key: run.id, icon: Plus, tone: "neutral", title: "Sequence admitted", detail: `${shortId(run.id)} entered the queue` };
    case "error":
      return { key: run.id, icon: AlertTriangle, tone: "warn", title: "Sequence errored", detail: run.error ?? `${shortId(run.id)} failed` };
    default:
      return { key: run.id, icon: CircleX, tone: "neutral", title: "Sequence cancelled", detail: shortId(run.id) };
  }
}

export default function LatestUpdates({
  streams,
  metrics,
  benchmarks
}: {
  streams: StreamRun[];
  metrics: MetricsResponse | null;
  benchmarks: BenchmarkSnapshot;
}) {
  const updates: Update[] = [];

  for (const run of streams.slice(0, 3)) {
    updates.push(streamUpdate(run));
  }

  if (metrics && metrics.peak_vram_mb > 0) {
    updates.push({
      key: "vram",
      icon: AlertTriangle,
      tone: "warn",
      title: "VRAM peak",
      detail: `${mibToGib(metrics.peak_vram_mb, 1)} / ${mibToGib(benchmarks.environment.vramTotalMb, 0)} card`
    });
  }

  const correctness = benchmarks.specDecode.correctness;
  if (correctness) {
    updates.push({
      key: "correctness",
      icon: ShieldCheck,
      tone: "accent",
      title: "Correctness gate",
      detail: `${correctness.passed ? "PASS" : "review"} · n=${correctness.n} · per-position TV`
    });
  }

  const headline = benchmarks.throughput.headline;
  if (headline.speedup) {
    updates.push({
      key: "benchmark",
      icon: BarChart3,
      tone: "neutral",
      title: "Benchmark refreshed",
      detail: `${fixed(headline.speedup, 1)}× over naive HF at c=${headline.concurrency}`
    });
  }

  return (
    <aside className="updates-rail">
      <div className="updates-head">
        <strong>Latest Updates</strong>
      </div>
      <div className="updates-count">
        <b>{updates.length}</b> recent events
      </div>
      <div className="updates-list">
        {updates.map(({ key, icon: Icon, tone, title, detail }) => (
          <div className="update-item" key={key}>
            <div className={`update-icon ${tone === "neutral" ? "" : tone}`}>
              <Icon size={16} aria-hidden="true" />
            </div>
            <div className="update-body">
              <strong>{title}</strong>
              <span className="mono">{detail}</span>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
