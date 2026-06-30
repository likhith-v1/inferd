import { LucideIcon } from "lucide-react";
import SourceBadge, { SourceKind } from "./SourceBadge";

interface Props {
  title: string;
  value: string;
  detail?: string;
  source: SourceKind;
  icon: LucideIcon;
}

export default function KpiCard({ title, value, detail, source, icon: Icon }: Props) {
  return (
    <section className="kpi-card" aria-label={title}>
      <div className="kpi-copy">
        <div className="kpi-header">
          <span>{title}</span>
          <SourceBadge kind={source} />
        </div>
        <div className="kpi-value">{value}</div>
        {detail ? <div className="kpi-detail">{detail}</div> : null}
      </div>
      <div className="kpi-icon" aria-hidden="true">
        <Icon size={22} />
      </div>
    </section>
  );
}

