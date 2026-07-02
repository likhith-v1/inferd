export function compact(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    notation: Math.abs(value) >= 10000 ? "compact" : "standard",
    maximumFractionDigits: Math.abs(value) >= 1000 ? 1 : 0
  }).format(value);
}

export function fixed(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

export function rate(value: number | null | undefined, digits = 1) {
  const formatted = fixed(value, digits);
  return formatted === "—" ? formatted : `${formatted} tok/s`;
}

export function seconds(value: number | null | undefined, digits = 2) {
  const formatted = fixed(value, digits);
  return formatted === "—" ? formatted : `${formatted}s`;
}

export function mibToGib(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `${(value / 1024).toFixed(digits)} GiB`;
}

export function percent(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(digits)}%`;
}

export function elapsed(secondsValue: number | null | undefined) {
  if (!secondsValue) {
    return "0s";
  }
  const minutes = Math.floor(secondsValue / 60);
  const seconds = Math.floor(secondsValue % 60);
  if (minutes <= 0) {
    return `${seconds}s`;
  }
  return `${minutes}m ${seconds}s`;
}

export function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

