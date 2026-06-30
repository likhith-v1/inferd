/* eslint-disable react-refresh/only-export-components */
import { createContext, ReactNode, useContext } from "react";
import { BenchmarkSnapshot } from "./benchmarks";
import { useGenerate } from "../hooks/useGenerate";
import { useHealth } from "../hooks/useHealth";
import { useMetrics } from "../hooks/useMetrics";

export interface DashboardState {
  benchmarks: BenchmarkSnapshot;
  metrics: ReturnType<typeof useMetrics>;
  health: ReturnType<typeof useHealth>;
  generation: ReturnType<typeof useGenerate>;
}

const DashboardContext = createContext<DashboardState | null>(null);

export function DashboardProvider({
  value,
  children
}: {
  value: DashboardState;
  children: ReactNode;
}) {
  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const value = useContext(DashboardContext);
  if (!value) {
    throw new Error("useDashboard must be used within DashboardProvider");
  }
  return value;
}
