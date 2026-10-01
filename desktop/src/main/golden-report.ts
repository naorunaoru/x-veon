import { app } from 'electron';
import { writeFile } from 'node:fs/promises';
type GoldenWindow = { webContents: { executeJavaScript(script: string): Promise<unknown> } };
export async function watchGoldenReport(win: GoldenWindow, file: string, opts?: { timeoutMs?: number }) {
  const terminal = new Set(['PASS', 'FAIL', 'UNSTABLE', 'ERROR', 'RECORDED', 'NEW']);
  const deadline = Date.now() + (opts?.timeoutMs ?? 15 * 60_000);
  let output: unknown = { status: 'TIMEOUT' };
  while (Date.now() < deadline) {
    const remaining = deadline - Date.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = await Promise.race([
      win.webContents.executeJavaScript('window.__goldenRun').catch(() => undefined),
      new Promise<undefined>(resolve => { timer = setTimeout(() => resolve(undefined), remaining); }),
    ]).finally(() => clearTimeout(timer));
    if (run && typeof run === 'object' && 'status' in run && terminal.has(String(run.status))) {
      const result = run as { status: string; totalMs?: number; report?: unknown };
      output = { status: result.status, totalMs: result.totalMs, report: result.report }; break;
    }
    const delay = Math.min(250, deadline - Date.now());
    if (delay > 0) await new Promise(resolve => setTimeout(resolve, delay));
  }
  try { await writeFile(file, JSON.stringify(output, null, 2)); } finally { app.quit(); }
}
