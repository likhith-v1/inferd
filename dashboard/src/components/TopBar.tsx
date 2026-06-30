import { Menu, Server, Search } from "lucide-react";
import { useLocation } from "react-router-dom";
import { useDashboard } from "../lib/dashboard";
import StatusPill from "./StatusPill";

const titles: Record<string, string> = {
  "/": "Overview",
  "/streams": "Streams",
  "/spec-decode": "Spec decode",
  "/memory": "Memory",
  "/benchmarks": "Benchmarks"
};

export default function TopBar({ onMenu }: { onMenu: () => void }) {
  const location = useLocation();
  const { health, metrics } = useDashboard();
  const title = titles[location.pathname] ?? "Overview";
  const model = health.data?.model || metrics.data?.model || "model pending";
  const healthy = health.data?.engine_alive ?? health.status !== "disconnected";

  return (
    <header className="topbar">
      <button className="icon-button menu-button" type="button" onClick={onMenu} aria-label="Open navigation">
        <Menu size={20} />
      </button>
      <div className="crumbs">
        <span>Pages / <b>{title}</b></span>
        <h1>{title}</h1>
      </div>
      <div className="topbar-actions">
        <div className="search-chip" aria-label="Metrics endpoint">
          <Search size={14} aria-hidden="true" />
          /metrics
        </div>
        <StatusPill status={health.status} healthy={healthy} />
        <div className="model-chip" title={model}>
          <Server size={15} aria-hidden="true" />
          <span>{model}</span>
        </div>
      </div>
    </header>
  );
}

