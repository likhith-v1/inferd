import {
  Area,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { BenchmarkRow } from "../lib/benchmarks";
import SourceBadge from "./SourceBadge";

interface Props {
  rows: BenchmarkRow[];
  title?: string;
  subtitle?: string;
  metric?: "tok/s" | "MiB";
  ariaLabel: string;
}

export default function ThroughputChart({
  rows,
  title = "Throughput vs concurrency",
  subtitle,
  metric = "tok/s",
  ariaLabel
}: Props) {
  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        <SourceBadge kind="benchmark" />
      </div>
      <div className="chart-frame" role="img" aria-label={ariaLabel}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 10, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="ours-area" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#d7d8db" stopOpacity={0.22} />
                <stop offset="100%" stopColor="#d7d8db" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,.07)" vertical={false} />
            <XAxis
              dataKey="concurrency"
              tick={{ fill: "#85868b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#85868b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={42}
            />
            <Tooltip
              cursor={{ stroke: "rgba(255,255,255,.14)" }}
              contentStyle={{
                background: "#1c1c20",
                border: "1px solid rgba(255,255,255,.12)",
                borderRadius: 14,
                color: "#f2f3f4"
              }}
              formatter={(value: number | string, name: string) => [
                value === null ? "pending" : `${Number(value).toFixed(1)} ${metric}`,
                name === "naiveHf" ? "naive HF" : name
              ]}
              labelFormatter={(value) => `concurrency ${value}`}
            />
            <Area
              type="monotone"
              dataKey="ours"
              stroke="none"
              fill="url(#ours-area)"
              isAnimationActive={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="naiveHf"
              stroke="rgba(255,255,255,.48)"
              strokeDasharray="6 5"
              strokeWidth={2}
              dot={{ r: 3 }}
              isAnimationActive={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="ours"
              stroke="#f2f3f5"
              strokeWidth={3}
              dot={{ r: 4, strokeWidth: 2, fill: "#0a0a0b" }}
              isAnimationActive={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="vllm"
              stroke="#8a8b90"
              strokeDasharray="3 5"
              strokeWidth={2}
              dot={{ r: 3 }}
              isAnimationActive={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend-row" aria-hidden="true">
        <span><i className="legend ours" />inferd</span>
        <span><i className="legend hf" />naive HF</span>
        <span><i className="legend vllm" />vLLM pending</span>
      </div>
    </section>
  );
}

