import { CircleAlert, Loader2, Radio } from "lucide-react";
import { ConnectionStatus } from "../hooks/useMetrics";

export default function ConnectionState({
  status,
  idle,
  error
}: {
  status: ConnectionStatus;
  idle: boolean;
  error: string | null;
}) {
  const state = status === "loading" ? "loading" : status === "disconnected" ? "disconnected" : idle ? "idle" : "streaming";
  const Icon = state === "loading" ? Loader2 : state === "disconnected" ? CircleAlert : Radio;

  return (
    <section className={`connection-state ${state}`} aria-live="polite">
      <Icon size={17} className={state === "loading" ? "spin" : ""} aria-hidden="true" />
      <div>
        <strong>{state === "disconnected" ? "/metrics unreachable" : state === "idle" ? "Engine idle" : state === "loading" ? "Connecting" : "Engine active"}</strong>
        <span>{state === "disconnected" ? error ?? "retrying" : state === "idle" ? "waiting for traffic" : state === "loading" ? "polling local service" : "metrics are updating"}</span>
      </div>
    </section>
  );
}

