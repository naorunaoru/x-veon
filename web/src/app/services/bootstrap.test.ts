import { beforeEach, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({ initPipeline: vi.fn(), restore: vi.fn(), probeHdrDisplay: vi.fn(), setPipeline: vi.fn(), matchLensFor: vi.fn(), cleanupOrphans: vi.fn() }));
vi.mock('@/pipeline', () => ({ initPipeline: m.initPipeline }));
vi.mock('./persistence', () => ({ restore: m.restore }));
vi.mock('./processing', () => ({ setPipeline: m.setPipeline }));
vi.mock('./library', () => ({ matchLensFor: m.matchLensFor, cleanupOrphans: m.cleanupOrphans }));
vi.mock('@/renderer/hdr-display', () => ({ probeHdrDisplay: m.probeHdrDisplay, hasWindowManagementApi: () => true }));
import { useAppStore } from '@/app/store';
import { initApp } from './bootstrap';
beforeEach(() => {
  vi.resetAllMocks();
  m.initPipeline.mockResolvedValue({ models: { backend: 'webgpu' } });
  m.restore.mockResolvedValue({ files: [], settings: {} });
  m.probeHdrDisplay.mockResolvedValue({ supported: true, accurate: false, headroom: 2 });
  useAppStore.setState({ files: [], initialized: false, initError: null, displayHdr: false, hdrPermissionNeeded: false });
});
it('initializes the pipeline and publishes HDR capability', async () => {
  await initApp({ cancelled: false });
  expect(useAppStore.getState()).toMatchObject({ initialized: true, backend: 'webgpu', displayHdr: true, displayHdrHeadroom: 2, hdrPermissionNeeded: true });
  expect(m.setPipeline).toHaveBeenCalledTimes(1); expect(m.cleanupOrphans).toHaveBeenCalledWith(new Set());
});
it('does not publish a late display probe after cancellation', async () => {
  let finish!: (value: object) => void;
  m.probeHdrDisplay.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const signal = { cancelled: false }; const run = initApp(signal);
  await vi.waitFor(() => expect(m.probeHdrDisplay).toHaveBeenCalledTimes(1));
  signal.cancelled = true; finish({ supported: true, accurate: false, headroom: 3 }); await run;
  expect(useAppStore.getState()).toMatchObject({ initialized: false, displayHdr: false, hdrPermissionNeeded: false });
  expect(m.cleanupOrphans).not.toHaveBeenCalled();
});
it('releases restored thumbnail URLs when initialization is cancelled', async () => {
  m.restore.mockResolvedValue({ files: [{ thumbnailUrl: 'blob:restored' }], settings: {} });
  URL.revokeObjectURL = vi.fn();
  await initApp({ cancelled: true });
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:restored');
  expect(m.setPipeline).not.toHaveBeenCalled(); expect(m.probeHdrDisplay).not.toHaveBeenCalled();
});
it('reports an initialization failure', async () => {
  m.initPipeline.mockRejectedValue(new Error('device unavailable'));
  await initApp({ cancelled: false });
  expect(useAppStore.getState()).toMatchObject({ initialized: false, initError: 'device unavailable' });
});
