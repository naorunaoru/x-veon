import { beforeEach, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({ initPipeline: vi.fn(), restore: vi.fn(), probeHdrDisplay: vi.fn(), setPipeline: vi.fn(), matchLensFor: vi.fn(), cleanupOrphans: vi.fn() }));
vi.mock('@/pipeline', () => ({ initPipeline: m.initPipeline }));
vi.mock('./persistence', () => ({ restore: m.restore }));
vi.mock('./processing', () => ({ setPipeline: m.setPipeline }));
vi.mock('./library', () => ({ matchLensFor: m.matchLensFor, cleanupOrphans: m.cleanupOrphans }));
vi.mock('@/renderer/hdr-display', () => ({ probeHdrDisplay: m.probeHdrDisplay, hasWindowManagementApi: () => true }));
import { useAppStore, type QueuedFile } from '@/app/store';
import { initApp } from './bootstrap';
beforeEach(() => {
  vi.resetAllMocks();
  m.initPipeline.mockResolvedValue({ models: { backend: 'webgpu' } });
  m.restore.mockResolvedValue({ files: [], settings: {}, complete: true });
  m.probeHdrDisplay.mockResolvedValue({ supported: true, accurate: false, headroom: 2 });
  useAppStore.setState({ files: [], initialized: false, initError: null, displayHdr: false, hdrPermissionNeeded: false });
});
it('initializes the pipeline and publishes HDR capability', async () => {
  await initApp({ cancelled: false });
  expect(useAppStore.getState()).toMatchObject({ initialized: true, backend: 'webgpu', displayHdr: true, displayHdrHeadroom: 2, hdrPermissionNeeded: true });
  expect(m.setPipeline).toHaveBeenCalledTimes(1); expect(m.cleanupOrphans).toHaveBeenCalledTimes(1);
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
  m.restore.mockResolvedValue({ files: [{ thumbnailUrl: 'blob:restored' }], settings: {}, complete: true });
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
it('skips orphan cleanup when the stored library could not be read', async () => {
  m.restore.mockResolvedValue({ files: [], settings: {}, complete: false });
  await initApp({ cancelled: false });
  expect(useAppStore.getState().initialized).toBe(true);
  expect(m.cleanupOrphans).not.toHaveBeenCalled();
});
it('keeps photos imported while startup was running', async () => {
  const dropped = { id: 'dropped', thumbnailUrl: null } as QueuedFile;
  const stored = { id: 'stored', thumbnailUrl: null } as QueuedFile;
  let finish!: (value: object) => void;
  m.restore.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const run = initApp({ cancelled: false });
  useAppStore.getState().addFiles([dropped]);
  finish({ files: [stored], settings: { selectedFileId: 'stored' }, complete: true });
  await run;
  expect(useAppStore.getState().files.map((f) => f.id)).toEqual(['stored', 'dropped']);
  expect(useAppStore.getState().selectedFileId).toBe('dropped');
});
