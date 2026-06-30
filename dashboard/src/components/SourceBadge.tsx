export type SourceKind = "live" | "benchmark" | "rederived";

export default function SourceBadge({ kind, label }: { kind: SourceKind; label?: string }) {
  return <span className={`source-badge ${kind}`}>{label ?? kind}</span>;
}

