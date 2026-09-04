/** Effective value of a parameter: the user override if set, else the preset baseline. */
export function resolveValue<T>(override: T | undefined, base: T): T {
  return override !== undefined ? override : base;
}

/** Whether a numeric value differs from its default beyond float noise. */
export function isModified(value: number, defaultValue: number, epsilon = 1e-6): boolean {
  return Math.abs(value - defaultValue) > epsilon;
}

/** Signed, formatted delta of value from its default (e.g. "+0.33"). */
export function formatDelta(
  value: number,
  defaultValue: number,
  format: (n: number) => string,
): string {
  const d = value - defaultValue;
  return (d > 0 ? '+' : '') + format(d);
}
