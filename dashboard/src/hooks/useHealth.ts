import { useEffect, useState } from "react";
import { fetchHealth, HealthResponse } from "../lib/api";
import { ConnectionStatus } from "./useMetrics";

export function useHealth() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    let failures = 0;

    const poll = async () => {
      const controller = new AbortController();
      try {
        const health = await fetchHealth(controller.signal);
        if (!alive) {
          return;
        }
        failures = 0;
        setData(health);
        setStatus("connected");
        setError(null);
        timer = window.setTimeout(poll, 3000);
      } catch (exc) {
        if (!alive) {
          return;
        }
        failures += 1;
        setStatus("disconnected");
        setError(exc instanceof Error ? exc.message : String(exc));
        timer = window.setTimeout(poll, Math.min(10000, 2500 * failures));
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

  return { data, status, error };
}

