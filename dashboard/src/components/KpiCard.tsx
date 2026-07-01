import { LucideIcon } from "lucide-react";
import SourceBadge, { SourceKind } from "./SourceBadge";

export interface KpiDelta {
  text: string;
  direction: "up" | "down";
  label?: string;
}

interface Props {
  title: string;
  value: string;
  detail?: string;
  source: SourceKind;
  icon: LucideIcon;
  delta?: KpiDelta;
  spark?: number[];
}

function Sparkline({ points }: { points: number[] }) {
  if (points.length < 2) {
    return null;
  }
  const width = 66;
  const height = 26;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const path = points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - ((value - min) / span) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="kpi-spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} fill="none" aria-hidden="true">
      <polyline points={path} stroke="var(--accent)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function KpiCard({ title, value, detail, source, icon: Icon, delta, spark }: Props) {
  return (
    <section className="kpi-card" aria-label={title}>
      <div className="kpi-header">
        <span>{title}</span>
        <Icon size={17} aria-hidden="true" />
      </div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-foot">
        {delta ? (
          <span className={`kpi-delta ${delta.direction}`}>
            <b>{delta.direction === "up" ? "▲" : "▼"} {delta.text}</b> {delta.label ?? ""}
          </span>
        ) : (
          <span className="kpi-detail">{detail}</span>
        )}
        <span className="kpi-foot-right">
          {spark && spark.length > 1 ? <Sparkline points={spark} /> : null}
          <SourceBadge kind={source} />
        </span>
      </div>
    </section>
  );
}
