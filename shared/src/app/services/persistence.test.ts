import { afterEach, beforeEach, expect, it, vi } from 'vitest';
vi.mock('@/app/storage/settings-storage', () => ({
  getSetting: vi.fn(),
  putSetting: vi.fn(async () => {}),
  pauseSettings: vi.fn(),
  resumeSettings: vi.fn(),
}));
import { getSetting, putSetting } from '@/app/storage/settings-storage';
import { useAppStore } from '@/app/store';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { setHost } from './host';
import { startPersistence, restore, cancelPhotoSave } from './persistence';
let host: ReturnType<typeof fakeHost>;
let stop: () => void;
function photo() {
  const p = fakePhoto();
  p.edit.demosaicMethod = 'dht';
  return fromLibraryPhoto(p);
}
beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  host = fakeHost();
  setHost(host);
  useAppStore.setState({
    files: [photo()],
    selectedFileId: 'a',
    exportQuality: 95,
    modelSize: 'S',
    demosaicMethod: 'neural-net',
  });
  stop = startPersistence();
});
afterEach(() => {
  stop();
  vi.useRealTimers();
});
const flush = () => vi.advanceTimersByTimeAsync(301);
it('debounces edits, omits transient updates and keeps settings separate', async () => {
  const s = useAppStore.getState();
  s.setFileLookPreset('a', 'colorful');
  s.setFilePreProcessOverride('a', 'exposure', 1);
  s.updateFileProgress('a', 1, 3);
  s.setViewScale(2);
  s.setOpenPanel('scopes');
  expect(host.library.save).not.toHaveBeenCalled();
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  expect(host.library.save).toHaveBeenCalledWith(
    'a',
    expect.objectContaining({ lookPreset: 'colorful', preProcessOverrides: { exposure: 1 } }),
    expect.anything(),
  );
  s.setExportQuality(80);
  expect(putSetting).toHaveBeenCalledWith('exportQuality', 80);
});
it('does not write on hydration or selection, and restores model defaults', async () => {
  vi.mocked(host.library.load).mockResolvedValue({ photos: [fakePhoto('r')], complete: true });
  vi.mocked(getSetting).mockImplementation(
    async (key) => (({ modelSize: 'M', exportQuality: 80 }) as Record<string, unknown>)[key] as never,
  );
  const restored = await restore();
  useAppStore.getState().restoreFromDb(restored.files, restored.settings);
  useAppStore.getState().selectFile('r');
  await flush();
  expect(host.library.save).not.toHaveBeenCalled();
  expect(restored.settings.modelSize).toBe('M');
});
it('keeps a rejected edit in session and retries once on focus', async () => {
  vi.mocked(host.library.save).mockRejectedValueOnce(new Error('disk full'));
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  expect(useAppStore.getState().files[0]).toMatchObject({
    editing: 'session',
    editingNote: 'disk full',
    edit: { lookPreset: 'umbra' },
  });
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  window.dispatchEvent(new Event('focus'));
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(2);
  expect(useAppStore.getState().files[0].editing).toBe('saved');
});
it('serializes revisions and never acknowledges a newer edit with an older completion', async () => {
  let finish!: () => void;
  vi.mocked(host.library.save).mockImplementationOnce(
    () =>
      new Promise<void>((resolve) => {
        finish = resolve;
      }),
  );
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  finish();
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(2);
  expect(vi.mocked(host.library.save).mock.calls[1][1].lookPreset).toBe('colorful');
});
it('guards view-only edits and persists neither processing nor note changes', async () => {
  useAppStore.setState({ files: [{ ...photo(), editing: 'view-only', editingNote: 'newer schema' }] });
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  useAppStore.getState().setFileDemosaicMethod('a', 'bilinear');
  useAppStore.getState().updateFileStatus('a', 'processing');
  await flush();
  expect(useAppStore.getState().files[0].edit.lookPreset).toBe('default');
  expect(host.library.save).not.toHaveBeenCalled();
});
it('cancels pending saves on removal and unsubscribes cleanly', async () => {
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  await cancelPhotoSave('a');
  useAppStore.getState().removeFile('a');
  await flush();
  expect(host.library.save).not.toHaveBeenCalled();
  vi.mocked(putSetting).mockClear();
  stop();
  useAppStore.getState().setExportQuality(12);
  expect(putSetting).not.toHaveBeenCalled();
});
it('saves an error fact after processing ends and reports incomplete restore', async () => {
  useAppStore.getState().updateFileStatus('a', 'processing');
  await flush();
  expect(host.library.save).not.toHaveBeenCalled();
  useAppStore.getState().updateFileStatus('a', 'error', 'bad file');
  await flush();
  expect(host.library.save).toHaveBeenCalledWith(
    'a',
    expect.anything(),
    expect.objectContaining({ status: 'error', error: 'bad file' }),
  );
  vi.mocked(host.library.load).mockResolvedValue({ photos: [], complete: false });
  expect((await restore()).complete).toBe(false);
});
