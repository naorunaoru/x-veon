import { startApp } from '@/startApp';
import { runGolden } from '@/dev/golden';
import { discardResult } from '@/app/services/processing';
import { useAppStore } from '@/app/store';
import { createGoldenHost } from './host/golden-fixtures';
const terminal = new Set(['PASS', 'FAIL', 'UNSTABLE', 'ERROR', 'RECORDED', 'NEW']);
export async function runGoldenApp(root: HTMLElement): Promise<void> {
  const fixture = createGoldenHost(window.xveon);
  startApp(root, fixture.host);
  const started = performance.now();
  await runGolden(async id => {
    discardResult(id); useAppStore.getState().removeFile(id); fixture.releaseFixture(id);
  }, { encoder: 'native' });
  const report = (window as unknown as { __golden?: { status: string } }).__golden;
  if (report && terminal.has(report.status)) {
    (window as unknown as { __goldenRun: unknown }).__goldenRun = { status: report.status, totalMs: performance.now() - started, report };
  }
}
