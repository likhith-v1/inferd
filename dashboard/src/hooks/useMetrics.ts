import { useEffect, useMemo, useState } from "react";
import { fetchMetrics, MetricsResponse } from "../lib/api";

export interface MetricsPoint {
  at: number;
  label: string;
  tokensPerSecond: number;
  activeSequences: number;
  waitingSequences: number;
  usedBlocks: number;
}

export type ConnectionStatus = "loading" | "connected" | "disconnected";

export function useMetrics() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [history, setHistory] = useState<MetricsPoint[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    let failures = 0;

    const poll = async () => {
      const controller = new AbortController();
      try {
        const metrics = await fetchMetrics(controller.signal);
        if (!alive) {
          return;
        }
        failures = 0;
        setData(metrics);
        setStatus("connected");
        setError(null);
        setHistory((prev) => {
          const now = Date.now();
          const next = [
            ...prev,
            {
              at: now,
              label: new Date(now).toLocaleTimeString([], {
                hour12: false,
                minute: "2-digit",
                second: "2-digit"
              }),
              tokensPerSecond: metrics.tokens_per_second,
              activeSequences: metrics.active_sequences,
              waitingSequences: metrics.waiting_sequences,
              usedBlocks: metrics.used_blocks
            }
          ];
          return next.slice(-80);
        });
        timer = window.setTimeout(poll, 1500);
      } catch (exc) {
        if (!alive) {
          return;
        }
        failures += 1;
        setStatus("disconnected");
        setError(exc instanceof Error ? exc.message : String(exc));
        timer = window.setTimeout(poll, Math.min(8000, 1500 * failures));
      }
    };

    poll();
    return () => {
      alive = false;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  const idle = useMemo(
    () => Boolean(data && data.active_sequences === 0 && data.waiting_sequences === 0),
    [data]
  );

  return { data, history, status, error, idle };
}

