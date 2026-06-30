import { clamp } from "../lib/format";
import SourceBadge, { SourceKind } from "./SourceBadge";

interface Props {
  title: string;
  subtitle?: string;
  value: number | null;
  max?: number;
  display: string;
  detail?: string;
  source: SourceKind;
  variant?: "arc" | "ring";
  tone?: "cyan" | "green" | "amber";
  ariaLabel: string;
}

export default function RadialGauge({
  title,
  subtitle,
  value,
  max = 1,
  display,
  detail,
  source,
  variant = "ring",
  tone = "cyan",
  ariaLabel
}: Props) {
  const ratio = value === null ? 0 : clamp(value / max);
  const ringCircumference = 2 * Math.PI * 46;
  const arcLength = Math.PI * 70;
  const stroke = variant === "ring" ? ringCircumference : arcLength;
  const dash = `${stroke * ratio} ${stroke}`;
  const gradientId = `${title.replace(/\W+/g, "-").toLowerCase()}-${variant}`;

  return (
    <section className="panel gauge-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        <SourceBadge kind={source} />
      </div>
      <div className={`gauge ${variant}`} role="img" aria-label={ariaLabel}>
        {variant === "arc" ? (
          <svg viewBox="0 0 180 116" aria-hidden="true">
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
                <stop offset="0" stopColor={tone === "amber" ? "#ffb86a" : "#2152ff"} />
                <stop offset="1" stopColor={tone === "green" ? "#2ee6a6" : "#21d4fd"} />
              </linearGradient>
            </defs>
            <path
              d="M20,96 A70,70 0 0 1 160,96"
              fill="none"
              stroke="rgba(255,255,255,.12)"
              strokeWidth="11"
              strokeLinecap="round"
            />
            <path
              d="M20,96 A70,70 0 0 1 160,96"
              fill="none"
              stroke={`url(#${gradientId})`}
              strokeWidth="11"
              strokeLinecap="round"
              strokeDasharray={dash}
              className="gauge-stroke"
            />
            <circle cx="90" cy="96" r="20" className="gauge-hub" />
          </svg>
        ) : (
          <svg viewBox="0 0 120 120" aria-hidden="true">
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="#21d4fd" />
                <stop offset="1" stopColor={tone === "amber" ? "#ffb86a" : "#01b574"} />
              </linearGradient>
            </defs>
            <circle cx="60" cy="60" r="46" fill="none" stroke="rgba(255,255,255,.12)" strokeWidth="10" />
            <circle
              cx="60"
              cy="60"
              r="46"
              fill="none"
              stroke={`url(#${gradientId})`}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={dash}
              transform="rotate(-90 60 60)"
              className="gauge-stroke"
            />
          </svg>
        )}
        <div className="gauge-readout">
          <strong>{display}</strong>
          {detail ? <span>{detail}</span> : null}
        </div>
      </div>
    </section>
  );
}

