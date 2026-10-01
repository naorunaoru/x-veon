import { setHost } from './host';
import { fakeHost } from '@/test/fake-host';
import { beforeEach, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({
  initPipeline: vi.fn(),
  restore: vi.fn(),
  probeHdrDisplay: vi.fn(),
  setPipeline: vi.fn(),
  matchLensFor: vi.fn(),
  cleanupOrphans: vi.fn(),
  folderSwitchVersion: vi.fn(() => 0),
  openFolder: vi.fn(), flushPersistence: vi.fn(), onUnsavedChange: vi.fn(), unsavedEdits: vi.fn(() => []),
}));
vi.mock('@/pipeline', () => ({ initPipeline: m.initPipeline }));
vi.mock('./persistence', () => ({ restore: m.restore, flushPersistence: m.flushPersistence, onUnsavedChange: m.onUnsavedChange, unsavedEdits: m.unsavedEdits }));
vi.mock('./processing', () => ({ setPipeline: m.setPipeline }));
vi.mock('./library', () => ({ matchLensFor: m.matchLensFor, folderSwitchVersion: m.folderSwitchVersion, openFolder: m.openFolder, cleanupOrphans: m.cleanupOrphans }));

import { useAppStore, type QueuedFile } from '@/app/store';
import { initApp, startHostCoordination } from './bootstrap';
beforeEach(() => {
  vi.resetAllMocks();
  m.openFolder.mockResolvedValue(undefined);
  const host = fakeHost();
  host.display.probe = m.probeHdrDisplay;
  host.display.requestAccurateHeadroom = vi.fn();
  host.library.release = (photos) =>
    photos.forEach((p) => {
      if (p.thumbnailUrl) URL.revokeObjectURL(p.thumbnailUrl);
    });
  setHost(host);
  m.initPipeline.mockResolvedValue({ models: { backend: 'webgpu' } });
  m.restore.mockResolvedValue({ files: [], settings: {}, complete: true });
  m.probeHdrDisplay.mockResolvedValue({ supported: true, accurate: false, headroom: 2 });
  useAppStore.setState({
    files: [],
    initialized: false,
    initError: null,
    displayHdr: false,
    hdrPermissionNeeded: false,
  });
});
it('initializes the pipeline and publishes HDR capability', async () => {
  await initApp({ cancelled: false });
  expect(useAppStore.getState()).toMatchObject({
    initialized: true,
    backend: 'webgpu',
    displayHdr: true,
    displayHdrHeadroom: 2,
    hdrPermissionNeeded: true,
  });
  expect(m.setPipeline).toHaveBeenCalledTimes(1);
});
it('does not publish a late display probe after cancellation', async () => {
  let finish!: (value: object) => void;
  m.probeHdrDisplay.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const signal = { cancelled: false };
  const run = initApp(signal);
  await vi.waitFor(() => expect(m.probeHdrDisplay).toHaveBeenCalledTimes(1));
  signal.cancelled = true;
  finish({ supported: true, accurate: false, headroom: 3 });
  await run;
  expect(useAppStore.getState()).toMatchObject({
    initialized: false,
    displayHdr: false,
    hdrPermissionNeeded: false,
  });
  expect(m.cleanupOrphans).not.toHaveBeenCalled();
});
it('releases restored thumbnail URLs when initialization is cancelled', async () => {
  m.restore.mockResolvedValue({ files: [{ thumbnailUrl: 'blob:restored' }], settings: {}, complete: true });
  URL.revokeObjectURL = vi.fn();
  await initApp({ cancelled: true });
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:restored');
  expect(m.setPipeline).not.toHaveBeenCalled();
  expect(m.probeHdrDisplay).not.toHaveBeenCalled();
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
  m.restore.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const run = initApp({ cancelled: false });
  useAppStore.getState().addFiles([dropped]);
  finish({ files: [stored], settings: { selectedFileId: 'stored' }, complete: true });
  await run;
  expect(useAppStore.getState().files.map((f) => f.id)).toEqual(['stored', 'dropped']);
  expect(useAppStore.getState().selectedFileId).toBe('dropped');
});

it('does not request storage persistence', async () => {
  const persist = vi.fn();
  Object.defineProperty(navigator, 'storage', { configurable: true, value: { persist } });
  await initApp({ cancelled: false });
  expect(persist).not.toHaveBeenCalled();
});
it('does not show an HDR permission dialog without the capability', async () => {
  const host = fakeHost();
  host.display.probe = m.probeHdrDisplay;
  setHost(host);
  await initApp({ cancelled: false });
  expect(useAppStore.getState().hdrPermissionNeeded).toBe(false);
});

it('wires folder and flush requests, reports summaries and releases subscriptions', async () => {
  const host = fakeHost();
  let request!: (folder?: { id: string; name: string }) => void;
  let flush!: () => Promise<void>;
  const stopFolder = vi.fn(), stopFlush = vi.fn(), stopUnsaved = vi.fn();
  host.library.onFolderRequest = cb => { request = cb; return stopFolder; };
  host.library.onFlushRequest = cb => { flush = cb; return stopFlush; };
  host.library.reportUnsaved = vi.fn();
  m.onUnsavedChange.mockReturnValue(stopUnsaved);
  setHost(host);
  const stop = startHostCoordination();
  expect(host.library.reportUnsaved).toHaveBeenCalledWith([]);
  request({ id: 'A', name: 'A' });
  expect(m.openFolder).toHaveBeenCalledWith({ id: 'A', name: 'A' });
  await flush();
  expect(m.flushPersistence).toHaveBeenCalledOnce();
  m.onUnsavedChange.mock.calls[0][0]([{ id: 'a', name: 'a', folder: { id: 'A', name: 'A' }, error: 'disk full', revision: 1, edit: {}, facts: {}, deferred: false }]);
  expect(host.library.reportUnsaved).toHaveBeenLastCalledWith([{ id: 'a', name: 'a', folder: { id: 'A', name: 'A' }, error: 'disk full' }]);
  stop();
  expect(stopFolder).toHaveBeenCalledOnce(); expect(stopFlush).toHaveBeenCalledOnce(); expect(stopUnsaved).toHaveBeenCalledOnce();
});
it('coordination supports a host without folder capabilities', () => {
  const stop = startHostCoordination();
  expect(() => stop()).not.toThrow();
});

it('does not merge a late initial folder restore into a newer folder switch', async () => {
  const host = fakeHost({ openFolder: vi.fn() });
  host.display.probe = m.probeHdrDisplay;
  setHost(host);
  let finish!: (value: object) => void;
  m.restore.mockImplementation(() => new Promise(r => { finish = r; }));
  const run = initApp({ cancelled: false });
  m.folderSwitchVersion.mockReturnValue(1);
  useAppStore.setState({ files: [{ id: 'B' } as QueuedFile], selectedFileId: 'B', folder: { id: 'B', name: 'B' } });
  finish({ files: [{ id: 'A' } as QueuedFile], settings: {}, complete: true, folder: { id: 'A', name: 'A' } });
  await run;
  expect(useAppStore.getState()).toMatchObject({ files: [{ id: 'B' }], folder: { id: 'B' }, selectedFileId: 'B' });
});
