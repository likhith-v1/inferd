import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import benchmarkSnapshot from "./data/benchmarks.json";
import { BenchmarkSnapshot } from "./lib/benchmarks";
import { DashboardProvider } from "./lib/dashboard";
import { useGenerate } from "./hooks/useGenerate";
import { useHealth } from "./hooks/useHealth";
import { useMetrics } from "./hooks/useMetrics";
import AppShell from "./components/AppShell";
import Benchmarks from "./pages/Benchmarks";
import Memory from "./pages/Memory";
import Overview from "./pages/Overview";
import SpecDecode from "./pages/SpecDecode";
import Streams from "./pages/Streams";

export default function App() {
  const metrics = useMetrics();
  const health = useHealth();
  const generation = useGenerate();

  return (
    <DashboardProvider
      value={{
        benchmarks: benchmarkSnapshot as BenchmarkSnapshot,
        metrics,
        health,
        generation
      }}
    >
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Overview />} />
            <Route path="/streams" element={<Streams />} />
            <Route path="/spec-decode" element={<SpecDecode />} />
            <Route path="/memory" element={<Memory />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </DashboardProvider>
  );
}
