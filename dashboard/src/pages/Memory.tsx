import { Cpu, Database, Layers3 } from "lucide-react";
import KpiCard from "../components/KpiCard";
import RadialGauge from "../components/RadialGauge";
import SourceBadge from "../components/SourceBadge";
import ThroughputChart from "../components/ThroughputChart";
import { useDashboard } from "../lib/dashboard";
import { compact, fixed, mibToGib, percent } from "../lib/format";

function BlockGrid({ used, free }: { used: number; free: number }) {
  const total = Math.max(used + free, 1);
  const cells = 96;
  const usedCells = Math.round((used / total) * cells);
  return (
    <div className="block-grid" role="img" aria-label={`${used} used KV blocks and ${free} free KV blocks`}>
      {Array.from({ length: cells }, (_, index) => (
        <i key={index} className={index < usedCells ? "used" : ""} />
      ))}
    </div>
  );
}

export default function Memory() {
  const { benchmarks, metrics } = useDashboard();
  const live = metrics.data;
  const vramTotal = benchmarks.environment.vramTotalMb;
  const peak = live?.peak_vram_mb ?? 0;
  const headroomMb = Math.max(0, vramTotal - peak);
  const headroomPct = vramTotal ? (headroomMb / vramTotal) * 100 : 0;

  return (
    <div className="page-stack">
      <div className="kpi-grid three">
        <KpiCard title="Used blocks" value={compact(live?.used_blocks)} detail={`${compact(live?.free_blocks)} free`} source="live" icon={Database} />
        <KpiCard title="Max blocks used" value={compact(live?.max_blocks_used)} detail="scheduler high-water mark" source="live" icon={Layers3} />
        <KpiCard title="Peak VRAM" value={mibToGib(live?.peak_vram_mb, 1)} detail={`${mibToGib(vramTotal, 1)} card`} source="live" icon={Cpu} />
      </div>

      <div className="memory-grid">
        <section className="panel memory-panel">
          <div className="panel-heading">
            <div>
              <h2>Paged KV block occupancy</h2>
              <p>{live ? `${live.used_blocks} used · ${live.free_blocks} free` : "awaiting /metrics"}</p>
            </div>
            <SourceBadge kind="live" />
          </div>
          <BlockGrid used={live?.used_blocks ?? 0} free={live?.free_blocks ?? 1} />
        </section>
        <RadialGauge
          title="VRAM headroom"
          subtitle="rederived from live peak and 32 GiB card"
          value={headroomPct}
          max={100}
          display={percent(headroomPct, 0)}
          detail={`${mibToGib(headroomMb, 1)} free`}
          source="rederived"
          tone={headroomPct < 15 ? "amber" : "green"}
          ariaLabel={`VRAM headroom ${percent(headroomPct, 0)}`}
        />
      </div>

      <div className="two-col">
        <ThroughputChart
          rows={benchmarks.vram.rows}
          title="Peak VRAM vs concurrency"
          subtitle="nvidia-smi comparable peak"
          metric="MiB"
          ariaLabel="Benchmark VRAM by concurrency for inferd, naive HF, and vLLM"
        />
        <section className="panel table-panel">
          <div className="panel-heading">
            <div>
              <h2>Paged cache microbench</h2>
              <p>cache-level accounting, not runtime cache claim</p>
            </div>
            <SourceBadge kind="benchmark" />
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>c</th>
                <th>paged MiB</th>
                <th>naive MiB</th>
                <th>ratio</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.vram.pagedMicrobench.map((point) => (
                <tr key={String(point.concurrency)}>
                  <td>{point.concurrency}</td>
                  <td>{fixed(Number(point.paged_kv_mb_measured), 2)}</td>
                  <td>{fixed(Number(point.naive_prealloc_kv_mb_measured), 2)}</td>
                  <td>{fixed(Number(point.memory_ratio_vs_naive_measured), 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}

