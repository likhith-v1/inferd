import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MetricsPoint } from "../hooks/useMetrics";
import SourceBadge from "./SourceBadge";

export default function LiveSparkline({ data }: { data: MetricsPoint[] }) {
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
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="live-spark" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a1a2a7" stopOpacity={0.32} />
                <stop offset="100%" stopColor="#a1a2a7" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis hide dataKey="label" />
            <YAxis hide domain={[0, "dataMax + 4"]} />
            <Tooltip
              cursor={{ stroke: "rgba(255,255,255,.14)" }}
              contentStyle={{
                background: "#1c1c20",
                border: "1px solid rgba(255,255,255,.12)",
                borderRadius: 14,
                color: "#f2f3f4"
              }}
              formatter={(value: number) => [`${value.toFixed(1)} tok/s`, "live"]}
              labelFormatter={(value) => value}
            />
            <Area
              type="monotone"
              dataKey="tokensPerSecond"
              stroke="#a1a2a7"
              strokeWidth={2.5}
              fill="url(#live-spark)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

