import { Activity, Cpu, Gauge } from "lucide-react";
import ActiveSequencesTable from "../components/ActiveSequencesTable";
import ConnectionState from "../components/ConnectionState";
import KpiCard, { KpiDelta } from "../components/KpiCard";
import LatestUpdates from "../components/LatestUpdates";
import ThroughputBars from "../components/ThroughputBars";
import { sourceLabel } from "../lib/benchmarks";
import { useDashboard } from "../lib/dashboard";
import { fixed, mibToGib, percent, rate } from "../lib/format";

export default function Overview() {
  const { benchmarks, metrics, generation } = useDashboard();
  const live = metrics.data;
  const history = metrics.history;
  const vramTotal = benchmarks.environment.vramTotalMb;
  const peakVram = live?.peak_vram_mb ?? null;
  const headroomMb = peakVram === null ? null : Math.max(0, vramTotal - peakVram);
  const headroomPct = headroomMb === null || vramTotal <= 0 ? null : (headroomMb / vramTotal) * 100;
  const headline = benchmarks.throughput.headline;

  const spark = history.slice(-14).map((point) => point.tokensPerSecond);
  let tpsDelta: KpiDelta | undefined;
  if (history.length >= 4) {
    const latest = history[history.length - 1].tokensPerSecond;
    const prev = history[Math.max(0, history.length - 6)].tokensPerSecond;
    if (prev > 0 && latest !== prev) {
      const pct = ((latest - prev) / prev) * 100;
      tpsDelta = {
        text: `${Math.abs(pct).toFixed(1)}%`,
        direction: pct >= 0 ? "up" : "down",
        label: "vs last window"
      };
    }
  }

  return (
    <div className="page-stack">
      <div>
        <h1 className="page-title">Live inference metrics</h1>
        <p className="page-subtitle">Latest insights from the engine serving under load — every number traces to a real source.</p>
      </div>

      <ConnectionState status={metrics.status} idle={metrics.idle} error={metrics.error} />

      <div className="ov-grid">
        <div className="ov-main">
          <div className="ov-kpis">
            <KpiCard
              title="Tokens / sec"
              value={rate(live?.tokens_per_second, 1)}
              detail="rolling 5s server window"
              source="live"
              icon={Activity}
              delta={tpsDelta}
              spark={spark}
            />
            <KpiCard
              title="Draft acceptance α"
              value={fixed(benchmarks.acceptance.alpha, 2)}
              detail={benchmarks.acceptance.label}
              source="benchmark"
              icon={Gauge}
            />
            <KpiCard
              title="VRAM headroom"
              value={percent(headroomPct, 0)}
              detail={headroomMb === null ? "awaiting /metrics" : `${mibToGib(headroomMb, 1)} free`}
              source="rederived"
              icon={Cpu}
            />
          </div>

          <ThroughputBars
            rows={benchmarks.throughput.rows}
            peakConcurrency={headline.concurrency}
            delta={headline.speedup ? `${fixed(headline.speedup, 2)}×` : undefined}
            ariaLabel={`Benchmark throughput bars by concurrency, peak ${sourceLabel(benchmarks)}`}
          />
        </div>

        <LatestUpdates streams={generation.streams} metrics={live} benchmarks={benchmarks} />
      </div>

      <ActiveSequencesTable
        metrics={live}
        streams={generation.streams}
        onStop={generation.stopGeneration}
      />
    </div>
  );
}
