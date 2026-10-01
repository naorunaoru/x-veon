import { beforeEach, expect, it, vi } from 'vitest';
import { useAppStore } from '@/app/store';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { setHost } from './host';
import { importFiles, removeFile, startLibraryWatching, switchFolder, openFolder } from './library';
import { fromLibraryPhoto } from '@/app/store/photo';
vi.mock('./processing', () => ({ discardResult: vi.fn() }));
vi.mock('./persistence', () => ({
  flushPersistence: vi.fn(async () => {}),
  unsavedEdits: vi.fn(() => []),
  retryUnsaved: vi.fn(),
  restoreFromLedger: vi.fn((file, entry) => entry ? { ...file, edit: entry.edit, editRevision: entry.revision, modelNeedsResolution: entry.deferred, editing: 'session', editingNote: entry.error } : file),
  cancelPhotoSave: vi.fn(async () => {}),
  pausePersistence: vi.fn(),
  resumePersistence: vi.fn(),
}));
vi.mock('@/app/lens/lensfun', () => ({ matchLens: vi.fn(async () => ({ lensModel: 'XF35' })) }));
import { matchLens } from '@/app/lens/lensfun';
import { discardResult } from './processing';
import { cancelPhotoSave, flushPersistence } from './persistence';
let host: ReturnType<typeof fakeHost>;
beforeEach(() => {
  vi.clearAllMocks();
  host = fakeHost();
  setHost(host);
  useAppStore.setState({ files: [], selectedFileId: null, folder: null });
});
it('adds host snapshots, selects the first added photo and matches its lens', async () => {
  const photo = fakePhoto();
  photo.facts.metadata = { camera: 'Fuji', lensModel: 'XF35', focalLength: 35, fNumber: 2 };
  vi.mocked(host.library.addFiles).mockResolvedValue({ photos: [photo], selectedIds: ['a'], complete: true });
  await importFiles([new File(['x'], 'a.raf')]);
  expect(useAppStore.getState().selectedFileId).toBe('a');
  await vi.waitFor(() =>
    expect(useAppStore.getState().files[0].lensProfile).toMatchObject({ lensModel: 'XF35' }),
  );
  expect(matchLens).toHaveBeenCalledWith('Fuji', 'XF35');
});
it('removes only when the host supports it, draining saves before host deletion', async () => {
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())], selectedFileId: 'a' });
  await removeFile('a');
  expect(useAppStore.getState().files).toHaveLength(1);
  host.library.remove = vi.fn(async () => {});
  await removeFile('a');
  expect(useAppStore.getState().files).toEqual([]);
  expect(discardResult).toHaveBeenCalledWith('a');
  expect(cancelPhotoSave).toHaveBeenCalledWith('a');
  expect(host.library.remove).toHaveBeenCalledWith('a');
  expect(vi.mocked(cancelPhotoSave).mock.invocationCallOrder[0]).toBeLessThan(
    vi.mocked(host.library.remove).mock.invocationCallOrder[0],
  );
});
it('applies thumbnail facts without replacing a local edit and releases its subscription', () => {
  let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
  const stop = vi.fn();
  host.library.onChange = (listener) => {
    publish = listener;
    return stop;
  };
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())] });
  const unsubscribe = startLibraryWatching();
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  publish({
    kind: 'facts',
    snapshot: { complete: true, photos: [{ ...fakePhoto(), thumbnailUrl: 'blob:thumb' }] },
  });
  expect(useAppStore.getState().files[0]).toMatchObject({
    thumbnailUrl: 'blob:thumb',
    edit: { lookPreset: 'umbra' },
  });
  unsubscribe();
  expect(stop).toHaveBeenCalledOnce();
});

const snapshot = (id: string) => ({ photos: [fakePhoto(id)], folder: { id, name: id }, complete: true });
it('flushes before opening, replaces photos and disposes old results; cancellation keeps state', async () => {
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto('old'))] });
  host.library.openFolder = vi.fn(async () => snapshot('A'));
  await openFolder({ id: 'A', name: 'A' });
  expect(vi.mocked(flushPersistence).mock.invocationCallOrder[0]).toBeLessThan(vi.mocked(host.library.openFolder).mock.invocationCallOrder[0]);
  expect(host.library.openFolder).toHaveBeenCalledWith({ id: 'A', name: 'A' });
  expect(useAppStore.getState()).toMatchObject({ folder: { id: 'A' }, selectedFileId: 'A', files: [{ id: 'A' }] });
  expect(discardResult).toHaveBeenCalledWith('old');
  const files = useAppStore.getState().files;
  vi.mocked(host.library.openFolder).mockResolvedValue(null);
  await openFolder();
  expect(useAppStore.getState().files).toBe(files);
});
it('ignores a late A snapshot after B, including returning to A', async () => {
  let finish!: (s: ReturnType<typeof snapshot>) => void;
  const first = switchFolder(() => new Promise(r => { finish = r; }));
  await Promise.resolve();
  await switchFolder(async () => snapshot('B'));
  await switchFolder(async () => ({ ...snapshot('A'), selectedIds: ['second'], photos: [fakePhoto('second')] }));
  finish(snapshot('A'));
  await first;
  expect(useAppStore.getState().selectedFileId).toBe('second');
});
it('never loads an older request held in its flush step', async () => {
  let finish!: () => void;
  vi.mocked(flushPersistence).mockImplementationOnce(() => new Promise(r => { finish = r; }));
  const load = vi.fn(async () => snapshot('A'));
  const a = switchFolder(load);
  await switchFolder(async () => snapshot('B'));
  finish();
  await a;
  expect(load).not.toHaveBeenCalled();
  expect(useAppStore.getState().folder?.id).toBe('B');
});
it('replaces on a folder-host drop and appends on a web drop', async () => {
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto('old'))] });
  host.library.openFolder = vi.fn();
  vi.mocked(host.library.addFiles).mockResolvedValue({ ...snapshot('A'), photos: [fakePhoto('a'), fakePhoto('b')], selectedIds: ['b'] });
  await importFiles([]);
  expect(useAppStore.getState().files.map(f => f.id)).toEqual(['a', 'b']);
  expect(useAppStore.getState().selectedFileId).toBe('b');
  delete host.library.openFolder;
  vi.mocked(host.library.addFiles).mockResolvedValue(snapshot('web'));
  await importFiles([]);
  expect(useAppStore.getState().files.map(f => f.id)).toEqual(['a', 'b', 'web']);
});
it('ignores facts and replacement snapshots from an earlier folder', () => {
  let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
  host.library.onChange = listener => { publish = listener; return () => {}; };
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto('a'))], folder: { id: 'A', name: 'A' } });
  const stop = startLibraryWatching();
  publish({ kind: 'replace', snapshot: snapshot('B') });
  publish({ kind: 'facts', snapshot: { ...snapshot('B'), photos: [{ ...fakePhoto('a'), thumbnailUrl: 'stale' }] } });
  expect(useAppStore.getState().files).toMatchObject([{ id: 'a', thumbnailUrl: null }]);
  stop();
});
