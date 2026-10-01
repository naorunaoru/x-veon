import 'fake-indexeddb/auto';
import { beforeEach, expect, it, vi } from 'vitest';
import { fakeHost } from '@/test/fake-host';
import { setHost } from './host';
import { useAppStore } from '@/app/store';
import { transact } from '@/app/storage/database';
import type { CfaType, ModelSize } from '@/lib/types';
const m = vi.hoisted(() => ({ availableSizes: vi.fn() }));
vi.mock('@/pipeline', () => ({
  initPipeline: async () => ({ models: { backend: 'webgpu', availableSizes: m.availableSizes } }),
}));
vi.mock('./processing', () => ({ setPipeline: vi.fn() }));
vi.mock('./library', () => ({ matchLensFor: vi.fn(), folderSwitchVersion: () => 0 }));
import { initApp } from './bootstrap';
let serial = 0;
beforeEach(() => {
  setHost({ ...fakeHost(), settingsDbName: `legacy-default-${++serial}` });
  useAppStore.setState({ files: [], modelSize: 'S', initialized: false, initError: null });
  m.availableSizes.mockReturnValue(new Set<ModelSize>(['S']));
});
it('ignores the pre-M1 M default when the loaded manifest only ships S', async () => {
  await transact(`legacy-default-${serial}`, 'settings', 'readwrite', (store) =>
    store.put({ key: 'modelSize', value: 'M' }),
  );
  await initApp({ cancelled: false });
  expect(useAppStore.getState()).toMatchObject({ modelSize: 'S', initialized: true, initError: null });
});
it('retains a restored size when at least one CFA offers it', async () => {
  m.availableSizes.mockImplementation(
    (cfa: CfaType) => new Set<ModelSize>(cfa === 'bayer' ? ['S', 'M'] : ['S']),
  );
  await transact(`legacy-default-${serial}`, 'settings', 'readwrite', (store) =>
    store.put({ key: 'modelSize', value: 'M' }),
  );
  await initApp({ cancelled: false });
  expect(useAppStore.getState().modelSize).toBe('M');
});
