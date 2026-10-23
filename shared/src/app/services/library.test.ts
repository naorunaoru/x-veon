import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useAppStore } from '@/app/store';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { setHost } from './host';
import { importFiles, matchLensFor, removeFile, startLibraryWatching, switchFolder, openFolder } from './library';
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
afterEach(() => vi.restoreAllMocks());
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
/** A photo that reports a lens (or none), carrying the given stored profile; the profile is untyped so an old shape can be seeded. */
const lensPhoto = (id: string, { lens = true, profile = null }: { lens?: boolean; profile?: unknown } = {}) => {
  const photo = fakePhoto(id);
  photo.facts.metadata = lens ? { camera: 'Fuji', lensModel: 'XF35', focalLength: 35, fNumber: 2 } : null;
  photo.facts.lensProfile = profile as typeof photo.facts.lensProfile;
  return fromLibraryPhoto(photo);
};
const fileOf = (id: string) => useAppStore.getState().files.find((f) => f.id === id)!;
/** A decode result as processing publishes it; the lens model arrives here on a web import. */
const decoded = (lensModel: string) => ({
  exportData: { width: 4, height: 2, xyzToCam: null, wbCoeffs: new Float32Array(3), camToXyz: new Float32Array(12), orientation: 'Normal' },
  metadata: { make: 'Fuji', model: 'X-T5', width: 4, height: 2, tileCount: 1, inferenceTime: 0, backend: 'webgpu', exposureBias: 0, lensModel, focalLength: 35, fNumber: 2, colorTemp: 0, tint: 0 },
});
it('marks the photo matched when the match settles, and when there is nothing to match', async () => {
  useAppStore.setState({ files: [lensPhoto('a'), lensPhoto('bare', { lens: false })] });
  matchLensFor('a');
  expect(fileOf('a').lensMatched).toBe(false);
  await vi.waitFor(() => expect(fileOf('a').lensMatched).toBe(true));
  expect(fileOf('a').lensProfile).toMatchObject({ lensModel: 'XF35' });
  // A photo whose lens is not known yet is not "nothing to match": the decode may still name one.
  matchLensFor('bare');
  expect(fileOf('bare').lensMatched).toBe(false);
  useAppStore.getState().setFileResult('bare', decoded(''), 'dht');
  matchLensFor('bare');
  expect(fileOf('bare').lensMatched).toBe(true);
  expect(matchLens).toHaveBeenCalledTimes(1);
});
it('a fresh import is not matched until its decode names a lens and the match settles', async () => {
  // Web imports start with no metadata and RAF quick metadata has no lens model.
  vi.mocked(host.library.addFiles).mockResolvedValue({ photos: [fakePhoto()], selectedIds: ['a'], complete: true });
  await importFiles([new File(['x'], 'a.raf')]);
  expect(fileOf('a').lensMatched).toBe(false);
  expect(matchLens).not.toHaveBeenCalled();
  let settle!: (profile: Awaited<ReturnType<typeof matchLens>>) => void;
  vi.mocked(matchLens).mockReturnValueOnce(new Promise((resolve) => { settle = resolve; }));
  useAppStore.getState().setFileResult('a', decoded('XF35'), 'dht');
  matchLensFor('a');
  expect(matchLens).toHaveBeenCalledWith('Fuji X-T5', 'XF35');
  expect(fileOf('a').lensMatched).toBe(false);
  settle({ lensModel: 'XF35' } as never);
  await vi.waitFor(() => expect(fileOf('a').lensMatched).toBe(true));
  expect(fileOf('a').lensProfile).toMatchObject({ lensModel: 'XF35' });
});
it('marks the photo matched when the match fails', async () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.mocked(matchLens).mockRejectedValueOnce(new Error('no lens data'));
  useAppStore.setState({ files: [lensPhoto('a')] });
  matchLensFor('a');
  await vi.waitFor(() => expect(fileOf('a').lensMatched).toBe(true));
  expect(fileOf('a').lensProfile).toBeNull();
  expect(warn).toHaveBeenCalledWith('Lens match failed:', expect.any(Error));
});
it('re-matches a stored profile that predates the calibration fields', async () => {
  const stored = { lensModel: 'XF35', mount: 'Fujifilm X', cropfactor: 1.5, distortion: [], tca: [], vignetting: [] };
  useAppStore.setState({ files: [
    lensPhoto('old', { profile: stored }),
    lensPhoto('new', { profile: { ...stored, calibCropfactor: 1.5, calibAspectRatio: 1.5 } }),
  ] });
  matchLensFor('new');
  expect(fileOf('new').lensMatched).toBe(true);
  expect(matchLens).not.toHaveBeenCalled();
  matchLensFor('old');
  await vi.waitFor(() => expect(fileOf('old').lensMatched).toBe(true));
  expect(matchLens).toHaveBeenCalledTimes(1);
  expect(fileOf('old').lensProfile).toEqual({ lensModel: 'XF35' });
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
it('dispatches host loads in request order even when an earlier flush is held', async () => {
  let finish!: () => void;
  vi.mocked(flushPersistence).mockImplementationOnce(() => new Promise(r => { finish = r; }));
  const load = vi.fn(async () => snapshot('A'));
  const a = switchFolder(load);
  const newer = vi.fn(async () => snapshot('B'));
  const b = switchFolder(newer);
  await vi.waitFor(() => expect(flushPersistence).toHaveBeenCalled());
  expect(load).not.toHaveBeenCalled(); expect(newer).not.toHaveBeenCalled();
  finish(); await Promise.all([a, b]);
  expect(load.mock.invocationCallOrder[0]).toBeLessThan(newer.mock.invocationCallOrder[0]);
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

it.each(['loading', 'flushing'])('cancellation preserves the earlier %s choice and bootstrap version', async phase => {
 const { folderSwitchVersion } = await import('./library'); const version = folderSwitchVersion();
 let finish!: () => void; const gate = new Promise<void>(r => { finish = r; });
 if (phase === 'flushing') vi.mocked(flushPersistence).mockImplementationOnce(() => gate);
 const first = switchFolder(async () => { if (phase === 'loading') await gate; return snapshot('kept'); });
 await Promise.resolve(); const cancelled = switchFolder(async () => null);
 if (phase === 'loading') await cancelled;
 expect(folderSwitchVersion()).toBe(version);
 finish(); await Promise.all([first, cancelled]); expect(useAppStore.getState().folder?.id).toBe('kept');
 expect(folderSwitchVersion()).toBeGreaterThan(version);
});
