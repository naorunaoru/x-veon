import { startApp } from '@/startApp';
import { createWebHost } from './host';
export function boot(root: HTMLElement): void {
  const golden =
    __XV_GOLDEN__ && new URLSearchParams(location.search).has('golden');
  startApp(root, createWebHost(golden ? { deliver: () => {} } : {}));
  if (
    __XV_GOLDEN__ &&
    new URLSearchParams(location.search).get('spike') === 'timing'
  )
    void publishSpikeTiming();
  if (
    __XV_GOLDEN__ &&
    new URLSearchParams(location.search).get('spike') === 'view'
  )
    void import('@/dev/spike-app')
      .then((module) => module.showSpikeFixture())
      .catch(console.error);
  if (golden)
    void import('@/dev/golden')
      .then((module) => module.runGolden())
      .catch((error) => console.error('[golden] failed:', error));
}

// Spike timing uses exactly the same pipeline implementation as the desktop host.
export async function publishSpikeTiming(): Promise<void> {
  const { runTiming } = await import('@/dev/spike-timing');
  try {
    const payload = {
      status: 'PASS',
      runtime: navigator.userAgent,
      ...(await runTiming()),
    };
    (window as any).__spike = payload;
    document.title = 'spike timing: complete';
    const report = document.createElement('pre');
    report.textContent = JSON.stringify(payload, null, 2);
    document.body.append(report);
  } catch (error) {
    (window as any).__spike = { status: 'ERROR', error: String(error) };
    document.title = 'spike timing: ERROR';
  }
}
