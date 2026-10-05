import { afterEach, expect, it, vi } from 'vitest';
import { watchGoldenReport } from './golden-report';
const mocks = vi.hoisted(() => ({ writeFile: vi.fn(async (_file: string, _contents: string) => {}), quit: vi.fn() }));
vi.mock('electron', () => ({ app: { quit: mocks.quit } }));
vi.mock('node:fs/promises', () => ({ writeFile: mocks.writeFile }));
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks(); });
it.each(['BENCH', 'PASS', 'FAIL', 'UNSTABLE', 'ERROR', 'RECORDED', 'NEW'])('ignores titles and missing state until %s', async status => {
  vi.useFakeTimers(); let run: unknown; const executeJavaScript = vi.fn(async (_script: string) => run);
  const pending = watchGoldenReport({ webContents: { executeJavaScript } }, '/report');
  await vi.advanceTimersByTimeAsync(1000); expect(mocks.writeFile).not.toHaveBeenCalled();
  run = { status: 'RUNNING' }; await vi.advanceTimersByTimeAsync(1000); expect(mocks.quit).not.toHaveBeenCalled();
  run = { status, totalMs: 12, report: { tests: 10 } }; await vi.advanceTimersByTimeAsync(1000); await pending;
  expect(JSON.parse(mocks.writeFile.mock.calls[0][1] as string)).toEqual(run); expect(mocks.quit).toHaveBeenCalledOnce();
  expect(executeJavaScript.mock.calls[0][0]).toContain('__goldenRun');
});
it('writes TIMEOUT and quits when no terminal status arrives', async () => {
  vi.useFakeTimers(); const pending = watchGoldenReport({ webContents: { executeJavaScript: async () => undefined } }, '/report', { timeoutMs: 100 });
  await vi.advanceTimersByTimeAsync(100); await pending; expect(JSON.parse(mocks.writeFile.mock.calls[0][1] as string)).toEqual({ status: 'TIMEOUT' }); expect(mocks.quit).toHaveBeenCalledOnce();
});
