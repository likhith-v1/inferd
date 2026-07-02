import { CheckCircle2, CircleAlert, Loader2 } from "lucide-react";
import { ConnectionStatus } from "../hooks/useMetrics";

interface Props {
  status: ConnectionStatus;
  healthy?: boolean;
  label?: string;
}

export default function StatusPill({ status, healthy = true, label }: Props) {
  const isLoading = status === "loading";
  const isDown = status === "disconnected" || !healthy;
  const text = label ?? (isLoading ? "connecting" : isDown ? "degraded" : "healthy");
  const Icon = isLoading ? Loader2 : isDown ? CircleAlert : CheckCircle2;

  return (
    <span className={`status-pill ${isDown ? "danger" : isLoading ? "loading" : "ok"}`}>
      <Icon size={14} aria-hidden="true" className={isLoading ? "spin" : ""} />
      {text}
    </span>
  );
}

