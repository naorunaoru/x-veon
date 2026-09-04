/** Pure hashing, report-building, and comparison helpers for the golden route. */

export interface RunEntry {
  key: string;
  run: number;
  display: string;
  displayDark: string;
  elapsedMs: number;
  error?: string;
}

export interface ExportRun {
  key: string;
  bytes: number;
  sha256: string;
  error?: string;
}

export interface GoldenEntry {
  display: string;
  displayDark: string;
  stable: boolean;
  runs: number;
  elapsedMs: number[];
  error?: string;
}

export interface GoldenExportEntry {
  bytes: number;
  sha256: string;
  error?: string;
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
}

export type GoldenMode = 'quick' | 'full';

export interface GoldenReport {
  mode: GoldenMode;
  adapter: AdapterInfo;
  commit: string;
  recordedAt: string;
  entries: Record<string, GoldenEntry>;
  exports: Record<string, GoldenExportEntry>;
}

export interface GoldenBaseline {
  adapter: AdapterInfo;
  commit: string;
  recordedAt: string;
  entries: Record<string, { display: string; displayDark: string }>;
  exports: Record<string, { bytes: number; sha256: string }>;
}

export interface Expectation {
  keys: string[];
  exportKeys: string[];
  runs: number;
}

export type CompareStatus = 'PASS' | 'FAIL' | 'NEW' | 'UNSTABLE';

export interface CompareResult {
  key: string;
  status: CompareStatus;
  reason?: string;
}

export function methodKey(sample: string, method: string, size?: string): string {
  return size ? `${sample}|${method}:${size}` : `${sample}|${method}`;
}

export function exportKey(sample: string, format: string): string {
  return `${sample}|${format}`;
}

export async function hashBytes(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', copy.buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export function hashFloat32(data: Float32Array): Promise<string> {
  return hashBytes(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}

export function buildReport(
  mode: GoldenMode,
  runs: RunEntry[],
  exports: ExportRun[],
  adapter: AdapterInfo,
  commit: string,
  recordedAt: string,
): GoldenReport {
  const entries: Record<string, GoldenEntry> = {};
  for (const run of runs) {
    const entry = entries[run.key];
    if (!entry) {
      entries[run.key] = {
        display: run.display,
        displayDark: run.displayDark,
        stable: true,
        runs: 1,
        elapsedMs: [run.elapsedMs],
        ...(run.error ? { error: run.error } : {}),
      };
      continue;
    }

    entry.runs += 1;
    entry.elapsedMs.push(run.elapsedMs);
    if (run.error) entry.error = run.error;
    if (run.display !== entry.display || run.displayDark !== entry.displayDark) entry.stable = false;
  }

  const exportEntries: Record<string, GoldenExportEntry> = {};
  for (const entry of exports) {
    exportEntries[entry.key] = {
      bytes: entry.bytes,
      sha256: entry.sha256,
      ...(entry.error ? { error: entry.error } : {}),
    };
  }

  return { mode, adapter, commit, recordedAt, entries, exports: exportEntries };
}

export function compareToBaseline(
  report: GoldenReport,
  baseline: GoldenBaseline | null,
  expected: Expectation,
): CompareResult[] {
  const results: CompareResult[] = [];

  for (const key of expected.keys) {
    const entry = report.entries[key];
    if (!entry) {
      results.push({ key, status: 'FAIL', reason: 'missing from report' });
      continue;
    }
    if (entry.error) {
      results.push({ key, status: 'FAIL', reason: entry.error });
      continue;
    }
    if (entry.runs < expected.runs) {
      results.push({ key, status: 'FAIL', reason: `runs ${entry.runs} < ${expected.runs}` });
      continue;
    }
    if (!entry.stable) {
      results.push({ key, status: 'UNSTABLE' });
      continue;
    }
    if (!baseline) {
      results.push({ key, status: 'NEW' });
      continue;
    }

    const baselineEntry = baseline.entries[key];
    if (!baselineEntry) {
      results.push({ key, status: 'FAIL', reason: 'missing from baseline' });
      continue;
    }
    if (baselineEntry.display !== entry.display) {
      results.push({ key, status: 'FAIL', reason: 'display differs' });
      continue;
    }
    if (baselineEntry.displayDark !== entry.displayDark) {
      results.push({ key, status: 'FAIL', reason: 'displayDark differs' });
      continue;
    }
    results.push({ key, status: 'PASS' });
  }

  for (const key of expected.exportKeys) {
    const entry = report.exports[key];
    if (!entry) {
      results.push({ key, status: 'FAIL', reason: 'missing from report' });
      continue;
    }
    if (entry.error) {
      results.push({ key, status: 'FAIL', reason: entry.error });
      continue;
    }
    if (!baseline) {
      results.push({ key, status: 'NEW' });
      continue;
    }

    const baselineEntry = baseline.exports[key];
    if (!baselineEntry) {
      results.push({ key, status: 'FAIL', reason: 'missing from baseline' });
      continue;
    }
    results.push(baselineEntry.sha256 === entry.sha256
      ? { key, status: 'PASS' }
      : { key, status: 'FAIL', reason: 'bytes differ' });
  }

  return results;
}

export function overallStatus(results: CompareResult[]): 'PASS' | 'FAIL' | 'RECORDED' | 'UNSTABLE' {
  if (results.length === 0) return 'FAIL';
  if (results.some((result) => result.status === 'FAIL')) return 'FAIL';
  if (results.some((result) => result.status === 'UNSTABLE')) return 'UNSTABLE';
  if (results.every((result) => result.status === 'NEW')) return 'RECORDED';
  if (results.every((result) => result.status === 'PASS')) return 'PASS';
  return 'FAIL';
}
