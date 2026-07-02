export interface GenerateRequest {
  prompt: string;
  max_tokens: number;
}

export interface MetricsResponse {
  waiting_sequences: number;
  active_sequences: number;
  completed_sequences: number;
  failed_sequences: number;
  admitted_sequences: number;
  evicted_sequences: number;
  iterations: number;
  total_generated_tokens: number;
  used_blocks: number;
  free_blocks: number;
  max_blocks_used: number;
  tokens_per_second: number;
  last_ttft_s: number | null;
  peak_vram_mb: number;
  uptime_s: number;
  model: string;
}

export interface HealthResponse {
  status: "ok" | "degraded" | string;
  model: string;
  engine_alive: boolean;
  device: string;
}

export type GenerateEvent =
  | { type: "token"; text: string }
  | { type: "done"; finish_reason: string; tokens: number }
  | { type: "error"; message: string };

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const API_BASE = (import.meta.env.VITE_INFERD_API ?? "").replace(/\/$/, "");

function apiUrl(path: string) {
  if (!API_BASE) {
    return path;
  }
  return `${API_BASE}${path}`;
}

function assertNumber(value: unknown, key: string): asserts value is number {
  if (typeof value !== "number" || Number.isNaN(value)) {
    throw new Error(`/metrics schema mismatch: ${key} is not a number`);
  }
}

function assertString(value: unknown, key: string): asserts value is string {
  if (typeof value !== "string") {
    throw new Error(`schema mismatch: ${key} is not a string`);
  }
}

export function validateMetrics(value: unknown): MetricsResponse {
  if (!value || typeof value !== "object") {
    throw new Error("/metrics schema mismatch: expected object");
  }
  const obj = value as Record<string, unknown>;
  const numericKeys: Array<keyof MetricsResponse> = [
    "waiting_sequences",
    "active_sequences",
    "completed_sequences",
    "failed_sequences",
    "admitted_sequences",
    "evicted_sequences",
    "iterations",
    "total_generated_tokens",
    "used_blocks",
    "free_blocks",
    "max_blocks_used",
    "tokens_per_second",
    "peak_vram_mb",
    "uptime_s"
  ];
  for (const key of numericKeys) {
    assertNumber(obj[key], key);
  }
  if (obj.last_ttft_s !== null) {
    assertNumber(obj.last_ttft_s, "last_ttft_s");
  }
  assertString(obj.model, "model");
  return obj as unknown as MetricsResponse;
}

export function validateHealth(value: unknown): HealthResponse {
  if (!value || typeof value !== "object") {
    throw new Error("/healthz schema mismatch: expected object");
  }
  const obj = value as Record<string, unknown>;
  assertString(obj.status, "status");
  assertString(obj.model, "model");
  assertString(obj.device, "device");
  if (typeof obj.engine_alive !== "boolean") {
    throw new Error("/healthz schema mismatch: engine_alive is not a boolean");
  }
  return obj as unknown as HealthResponse;
}

async function parseError(response: Response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      return body.detail;
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText || "request failed";
  }
}

export async function fetchMetrics(signal?: AbortSignal): Promise<MetricsResponse> {
  const response = await fetch(apiUrl("/metrics"), { signal });
  if (!response.ok) {
    throw new ApiError(response.status, await parseError(response));
  }
  return validateMetrics(await response.json());
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(apiUrl("/healthz"), { signal });
  if (!response.ok) {
    throw new ApiError(response.status, await parseError(response));
  }
  return validateHealth(await response.json());
}

function dispatchSseChunk(chunk: string, onEvent: (event: GenerateEvent) => void) {
  const line = chunk.split("\n").find((part) => part.startsWith("data: "));
  if (!line) {
    return;
  }
  onEvent(JSON.parse(line.slice("data: ".length)) as GenerateEvent);
}

function parseSseFrames(
  buffer: string,
  onEvent: (event: GenerateEvent) => void
): string {
  const chunks = buffer.split("\n\n");
  const remainder = chunks.pop() ?? "";
  for (const chunk of chunks) {
    dispatchSseChunk(chunk, onEvent);
  }
  return remainder;
}

export async function generate(
  body: GenerateRequest,
  onEvent: (event: GenerateEvent) => void,
  signal?: AbortSignal
) {
  const response = await fetch(apiUrl("/generate"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal
  });
  if (!response.ok) {
    throw new ApiError(response.status, await parseError(response));
  }
  if (!response.body) {
    throw new Error("/generate returned no response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSseFrames(buffer, onEvent);
    }
    if (done) {
      buffer += decoder.decode();
      // A terminal frame may arrive without a trailing blank line on close.
      parseSseFrames(`${buffer}\n\n`, onEvent);
      break;
    }
  }
}
