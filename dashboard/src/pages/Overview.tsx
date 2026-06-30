import { Activity, Cpu, Gauge, Timer } from "lucide-react";
import { Link } from "react-router-dom";
import ActiveSequencesTable from "../components/ActiveSequencesTable";
import ConnectionState from "../components/ConnectionState";
import EngineActivity from "../components/EngineActivity";
import KpiCard from "../components/KpiCard";
import LiveSparkline from "../components/LiveSparkline";
import Playground from "../components/Playground";
import RadialGauge from "../components/RadialGauge";
import SpecDecodeTimeline from "../components/SpecDecodeTimeline";
import ThroughputChart from "../components/ThroughputChart";
import { sourceLabel } from "../lib/benchmarks";
import { useDashboard } from "../lib/dashboard";
import { fixed, mibToGib, percent, rate, seconds } from "../lib/format";

export default function Overview() {
  const { benchmarks, metrics, health, generation } = useDashboard();
  const live = metrics.data;
  const vramTotal = benchmarks.environment.vramTotalMb;
  const peakVram = live?.peak_vram_mb ?? null;
  const headroomMb = peakVram === null ? null : Math.max(0, vramTotal - peakVram);
  const headroomPct =
    headroomMb === null || vramTotal <= 0 ? null : (headroomMb / vramTotal) * 100;
  const headline = benchmarks.throughput.headline;

  return (
    <div className="page-stack">
      <ConnectionState status={metrics.status} idle={metrics.idle} error={metrics.error} />

      <div className="kpi-grid">
        <KpiCard
          title="Tokens / sec"
          value={rate(live?.tokens_per_second, 1)}
          detail="rolling 5s server window"
          source="live"
          icon={Activity}
        />
        <KpiCard
          title="TTFT"
          value={seconds(live?.last_ttft_s, 2)}
          detail="last streamed request"
          source="live"
          icon={Timer}
        />
        <KpiCard
          title="Draft alpha"
          value={fixed(benchmarks.acceptance.alpha, 2)}
          detail={sourceLabel(benchmarks)}
          source="benchmark"
          icon={Gauge}
        />
        <KpiCard
          title="Peak VRAM"
          value={mibToGib(live?.peak_vram_mb, 1)}
          detail={`${mibToGib(vramTotal, 1)} card`}
          source="live"
          icon={Cpu}
        />
      </div>

      <div className="overview-middle">
        <section className="hero-panel">
          <span>inferd engine · benchmark</span>
          <h2>{fixed(headline.speedup, 2)}x over the naive HF floor</h2>
          <p>
            {fixed(headline.oursTokS, 1)} tok/s vs {fixed(headline.naiveHfTokS, 1)} tok/s at {headline.concurrency} concurrent requests.
          </p>
          <Link to="/benchmarks">View benchmark table</Link>
        </section>
        <RadialGauge
          title="Acceptance rate"
          subtitle="draft vs target"
          value={benchmarks.acceptance.alpha}
          display={fixed(benchmarks.acceptance.alpha, 2)}
          detail="exact rejection rule"
          source="benchmark"
          variant="arc"
          ariaLabel={`Benchmark draft acceptance rate ${fixed(benchmarks.acceptance.alpha, 2)}`}
        />
        <RadialGauge
          title="VRAM headroom"
          subtitle={health.data?.device ?? "device pending"}
          value={headroomPct}
          max={100}
          display={percent(headroomPct, 0)}
          detail={headroomMb === null ? "awaiting /metrics" : `${mibToGib(headroomMb, 1)} free`}
          source="rederived"
          tone={headroomPct !== null && headroomPct < 15 ? "amber" : "green"}
          ariaLabel={`VRAM headroom ${percent(headroomPct, 0)}`}
        />
      </div>

      <div className="overview-charts">
        <ThroughputChart
          rows={benchmarks.throughput.rows}
          subtitle={`${fixed(headline.speedup, 2)}x at c=${headline.concurrency} vs naive HF`}
          ariaLabel="Benchmark throughput curve comparing inferd, naive HF, and vLLM"
        />
        <div className="side-stack">
          <LiveSparkline data={metrics.history} />
          <EngineActivity metrics={live} />
        </div>
      </div>

      <div className="overview-bottom">
        <ActiveSequencesTable
          metrics={live}
          streams={generation.streams}
          onStop={generation.stopGeneration}
        />
        <SpecDecodeTimeline snapshot={benchmarks} />
      </div>

      <div className="overview-play">
        <Playground compact />
      </div>
    </div>
  );
}
