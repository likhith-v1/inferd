import { useCallback, useMemo, useRef, useState } from "react";
import { ApiError, generate, GenerateEvent } from "../lib/api";

const PROMPTS = [
  "Explain speculative decoding in one paragraph.",
  "Summarize the dashboard state as an ops note.",
  "List three checks for paged KV-cache correctness.",
  "Rewrite this sentence in a concise technical tone.",
  "Extract the key metrics as JSON keys.",
  "Describe continuous batching for a new engineer.",
  "Draft a release note for the 27B FP8 capacity proof.",
  "Compare naive HF serving with inferd batching."
];

export type StreamStatus = "queued" | "streaming" | "done" | "error" | "cancelled";

export interface StreamRun {
  id: string;
  prompt: string;
  maxTokens: number;
  text: string;
  status: StreamStatus;
  emittedChunks: number;
  finalTokens: number | null;
  finishReason: string | null;
  error: string | null;
  statusCode: number | null;
  createdAt: number;
  startedAt: number | null;
  firstTokenAt: number | null;
  completedAt: number | null;
}

function newId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function useGenerate() {
  const [streams, setStreams] = useState<StreamRun[]>([]);
  const controllers = useRef(new Map<string, AbortController>());

  const update = useCallback((id: string, patch: Partial<StreamRun>) => {
    setStreams((prev) => prev.map((run) => (run.id === id ? { ...run, ...patch } : run)));
  }, []);

  const startGeneration = useCallback(
    (prompt: string, maxTokens = 96) => {
      const trimmed = prompt.trim();
      if (!trimmed) {
        return null;
      }
      const id = newId();
      const controller = new AbortController();
      controllers.current.set(id, controller);
      const createdAt = performance.now();
      const run: StreamRun = {
        id,
        prompt: trimmed,
        maxTokens,
        text: "",
        status: "queued",
        emittedChunks: 0,
        finalTokens: null,
        finishReason: null,
        error: null,
        statusCode: null,
        createdAt,
        startedAt: null,
        firstTokenAt: null,
        completedAt: null
      };
      setStreams((prev) => [run, ...prev].slice(0, 48));

      const onEvent = (event: GenerateEvent) => {
        if (event.type === "token") {
          const now = performance.now();
          setStreams((prev) =>
            prev.map((item) => {
              if (item.id !== id) {
                return item;
              }
              return {
                ...item,
                status: "streaming",
                startedAt: item.startedAt ?? now,
                firstTokenAt: item.firstTokenAt ?? now,
                emittedChunks: item.emittedChunks + 1,
                text: `${item.text}${event.text}`
              };
            })
          );
        } else if (event.type === "done") {
          update(id, {
            status: "done",
            finalTokens: event.tokens,
            finishReason: event.finish_reason,
            completedAt: performance.now()
          });
        } else {
          update(id, {
            status: "error",
            error: event.message,
            completedAt: performance.now()
          });
        }
      };

      const runStream = async () => {
        update(id, { startedAt: performance.now(), status: "streaming" });
        try {
          await generate({ prompt: trimmed, max_tokens: maxTokens }, onEvent, controller.signal);
        } catch (exc) {
          if (controller.signal.aborted) {
            update(id, { status: "cancelled", completedAt: performance.now() });
          } else if (exc instanceof ApiError) {
            update(id, {
              status: "error",
              error: exc.detail,
              statusCode: exc.status,
              completedAt: performance.now()
            });
          } else {
            update(id, {
              status: "error",
              error: exc instanceof Error ? exc.message : String(exc),
              completedAt: performance.now()
            });
          }
        } finally {
          controllers.current.delete(id);
        }
      };

      void runStream();
      return id;
    },
    [update]
  );

  const stopGeneration = useCallback(
    (id: string) => {
      controllers.current.get(id)?.abort();
      update(id, { status: "cancelled", completedAt: performance.now() });
    },
    [update]
  );

  const clearTerminal = useCallback(() => {
    setStreams((prev) => prev.filter((run) => run.status === "streaming" || run.status === "queued"));
  }, []);

  const startDemoLoad = useCallback(
    (count: number, maxTokens = 96) => {
      for (let i = 0; i < count; i += 1) {
        window.setTimeout(() => {
          startGeneration(PROMPTS[i % PROMPTS.length], maxTokens);
        }, i * 90);
      }
    },
    [startGeneration]
  );

  const active = useMemo(
    () => streams.filter((run) => run.status === "streaming" || run.status === "queued"),
    [streams]
  );

  return {
    streams,
    active,
    startGeneration,
    stopGeneration,
    clearTerminal,
    startDemoLoad,
    samplePrompts: PROMPTS
  };
}

