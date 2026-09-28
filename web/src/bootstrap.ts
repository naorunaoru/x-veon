import { startApp } from '@/startApp';
import { createWebHost } from './host';
export function boot(root: HTMLElement): void {
  const golden = __XV_GOLDEN__ && new URLSearchParams(location.search).has('golden');
  startApp(root, createWebHost(golden ? { deliver: () => {} } : {}));
  if (golden)
    void import('@/dev/golden')
      .then((module) => module.runGolden())
      .catch((error) => console.error('[golden] failed:', error));
}
