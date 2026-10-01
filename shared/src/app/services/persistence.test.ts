import { afterEach, beforeEach, expect, it, vi } from 'vitest';
vi.mock('./processing', () => ({ discardResult: vi.fn() }));
vi.mock('@/app/lens/lensfun', () => ({ matchLens: vi.fn(async () => null) }));
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
import { switchFolder, startLibraryWatching } from './library';
import { setHost } from './host';
import { startPersistence, restore, cancelPhotoSave, unsavedEdits, onUnsavedChange, flushPersistence, retryUnsaved, isPhotoDirty, pausePersistence, resumePersistence } from './persistence';
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
    folder: { id: 'A', name: 'A' },
    exportQuality: 95,
    modelSize: 'S',
    demosaicMethod: 'neural-net',
  });
  stop = startPersistence();
});
afterEach(async () => {
  if (typeof unsavedEdits === 'function')
    await Promise.all(unsavedEdits().map(e => cancelPhotoSave(e.id)));
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
  expect(host.library.saveFacts).toHaveBeenCalledWith(
    'a',
    expect.objectContaining({ status: 'error', error: 'bad file' }),
  );
  vi.mocked(host.library.load).mockResolvedValue({ photos: [], complete: false });
  expect((await restore()).complete).toBe(false);
});

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(r => { resolve = r; });
  return { promise, resolve };
}
it('records a processing edit immediately and saves it after processing ends', async () => {
  useAppStore.getState().updateFileStatus('a', 'processing');
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  expect(unsavedEdits()).toEqual([expect.objectContaining({ id: 'a', folder: { id: 'A', name: 'A' }, edit: expect.objectContaining({ lookPreset: 'umbra' }) })]);
  expect(unsavedEdits()[0].revision).toBe(useAppStore.getState().files[0].editRevision);
  await flush();
  expect(host.library.save).not.toHaveBeenCalled();
  useAppStore.getState().updateFileStatus('a', 'done');
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  expect(unsavedEdits()).toEqual([]);
});
it('reports deferred edits without sending them, and completes the same revision', async () => {
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())] });
  const inventories: unknown[] = [];
  const unsubscribe = onUnsavedChange(entries => inventories.push(entries));
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  const revision = useAppStore.getState().files[0].editRevision;
  expect(unsavedEdits()[0]).toMatchObject({ deferred: true, revision });
  await flushPersistence();
  await flush();
  expect(host.library.save).not.toHaveBeenCalled();
  useAppStore.setState(state => ({ files: state.files.map(f => ({ ...f, modelNeedsResolution: false, edit: { ...f.edit, model: { size: 'S', sha256: 'resolved' } } })) }));
  expect(unsavedEdits()[0]).toMatchObject({ deferred: false, revision });
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  expect(host.library.save).toHaveBeenCalledWith('a', expect.objectContaining({ model: { size: 'S', sha256: 'resolved' } }), expect.anything());
  expect(inventories).toEqual(expect.arrayContaining([[expect.objectContaining({ deferred: true, revision })], []]));
  unsubscribe();
});
it('does not acknowledge edit B when in-flight edit A completes', async () => {
  const a = deferred(), b = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => a.promise).mockImplementationOnce(() => b.promise);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  const revision = useAppStore.getState().files[0].editRevision;
  a.resolve();
  await flush();
  expect(unsavedEdits()[0]).toMatchObject({ revision, edit: { lookPreset: 'colorful' } });
  expect(useAppStore.getState().files[0].editing).toBe('session');
  b.resolve();
  await flush();
  expect(unsavedEdits()).toEqual([]);
});
it('serializes fact writes after an edit without carrying or clearing an edit', async () => {
  const a = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => a.promise);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().updateFileStatus('a', 'error', 'bad raw');
  await flush();
  expect(host.library.saveFacts).not.toHaveBeenCalled();
  expect(unsavedEdits()).toHaveLength(1);
  a.resolve();
  await flush();
  expect(host.library.saveFacts).toHaveBeenCalledWith('a', expect.objectContaining({ error: 'bad raw' }));
  expect(vi.mocked(host.library.saveFacts).mock.calls[0]).toHaveLength(2);
  expect(host.library.save).toHaveBeenCalledTimes(1);
});
it('keeps rejected entries across folders, flushes both, and retries only on request', async () => {
  vi.mocked(host.library.save).mockRejectedValue(new Error('RAW vanished'));
  const changed = vi.fn();
  const unsubscribe = onUnsavedChange(changed);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  expect(changed).toHaveBeenLastCalledWith([expect.objectContaining({ error: 'RAW vanished' })]);
  useAppStore.setState({ files: [{ ...photo(), id: 'b' }], folder: { id: 'B', name: 'B' } });
  useAppStore.getState().setFileLookPreset('b', 'colorful');
  await flush();
  expect(unsavedEdits().map(e => e.id)).toEqual(['a', 'b']);
  const count = vi.mocked(host.library.save).mock.calls.length;
  await vi.advanceTimersByTimeAsync(10000);
  expect(host.library.save).toHaveBeenCalledTimes(count);
  const a = deferred(), b = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => a.promise).mockImplementationOnce(() => b.promise);
  let finished = false;
  const flushing = flushPersistence().then(() => { finished = true; });
  await Promise.resolve();
  expect(host.library.save).toHaveBeenCalledTimes(count + 2);
  a.resolve();
  await Promise.resolve();
  expect(finished).toBe(false);
  b.resolve();
  await flushing;
  expect(unsavedEdits()).toEqual([]);
  useAppStore.getState().setFileLookPreset('b', 'umbra');
  await flush();
  retryUnsaved();
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(count + 4);
  unsubscribe();
});

it('restores deferred edits across a reopen with the same revision and resolves exactly once', async () => {
  useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())] });
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  const revision = useAppStore.getState().files[0].editRevision;
  await switchFolder(async () => ({ photos: [], folder: { id: 'B', name: 'B' }, complete: true }));
  expect(unsavedEdits()).toHaveLength(1);
  await switchFolder(async () => ({ photos: [fakePhoto()], folder: { id: 'A', name: 'A' }, complete: true }));
  expect(useAppStore.getState().files[0]).toMatchObject({ editRevision: revision, modelNeedsResolution: true, editing: 'session', edit: { lookPreset: 'umbra' } });
  useAppStore.setState(state => ({ files: state.files.map(f => ({ ...f, modelNeedsResolution: false, edit: { ...f.edit, model: { size: 'S', sha256: 'resolved' } } })) }));
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  expect(host.library.save).toHaveBeenCalledWith('a', expect.objectContaining({ model: { size: 'S', sha256: 'resolved' } }), expect.anything());
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  expect(useAppStore.getState().files[0].editRevision).toBeGreaterThan(revision);
  const next = useAppStore.getState().files[0].editRevision;
  useAppStore.getState().addFiles([{ ...photo(), id: 'new' }]);
  useAppStore.getState().setFileLookPreset('new', 'umbra');
  expect(useAppStore.getState().files[1].editRevision).toBeGreaterThan(next);
});
it('reloads clean watched photos, overlays dirty ones, resends them and retains missing entries', async () => {
  vi.mocked(host.library.save).mockRejectedValue(new Error('permission denied'));
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  await switchFolder(async () => ({ photos: [fakePhoto()], folder: { id: 'A', name: 'A' }, complete: true }));
  expect(useAppStore.getState().files[0]).toMatchObject({ editingNote: 'permission denied', edit: { lookPreset: 'umbra' } });
  useAppStore.getState().addFiles([{ ...photo(), id: 'clean' }]);
  let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
  host.library.onChange = cb => { publish = cb; return () => {}; };
  const stopWatch = startLibraryWatching();
  const before = vi.mocked(host.library.save).mock.calls.length;
  const clean = fakePhoto('clean'); clean.edit.lookPreset = 'colorful';
  publish({ kind: 'replace', snapshot: { photos: [fakePhoto(), clean, fakePhoto('new')], folder: { id: 'A', name: 'A' }, complete: true } });
  expect(useAppStore.getState().files.map(f => f.edit.lookPreset)).toEqual(['umbra', 'colorful', 'default']);
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(before + 1);
  publish({ kind: 'replace', snapshot: { photos: [clean], folder: { id: 'A', name: 'A' }, complete: true } });
  expect(useAppStore.getState().files.map(f => f.id)).toEqual(['clean']);
  expect(unsavedEdits()[0]).toMatchObject({ id: 'a', error: 'permission denied' });
  stopWatch();
});

it('flush waits for edit B queued behind in-flight A without needing its debounce timer', async () => {
  const a = deferred(), b = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => a.promise).mockImplementationOnce(() => b.promise);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  let done = false;
  const flushing = flushPersistence().then(() => { done = true; });
  a.resolve();
  for (let i = 0; i < 12; ++i) await Promise.resolve();
  expect(host.library.save).toHaveBeenCalledTimes(2);
  expect(done).toBe(false);
  b.resolve();
  await flushing;
  expect(isPhotoDirty('a')).toBe(false);
});
it('a facts change neither retries nor clears a rejected edit', async () => {
  vi.mocked(host.library.save).mockRejectedValueOnce(new Error('denied'));
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().updateFileStatus('a', 'error', 'bad raw');
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
  expect(host.library.saveFacts).toHaveBeenCalledTimes(1);
  expect(unsavedEdits()[0]).toMatchObject({ error: 'denied', edit: { lookPreset: 'umbra' } });
});
it('saves only facts for a view-only photo', async () => {
  useAppStore.setState({ files: [{ ...photo(), editing: 'view-only', editingNote: 'newer schema' }] });
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  useAppStore.getState().updateFileStatus('a', 'error', 'bad raw');
  await flush();
  expect(host.library.save).not.toHaveBeenCalled();
  expect(host.library.saveFacts).toHaveBeenCalledWith('a', expect.objectContaining({ error: 'bad raw' }));
  expect(isPhotoDirty('a')).toBe(false);
});
it('focus retries failures from every folder without a timer retry loop', async () => {
  vi.mocked(host.library.save).mockRejectedValue(new Error('denied'));
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.setState({ files: [{ ...photo(), id: 'b' }], folder: { id: 'B', name: 'B' } });
  useAppStore.getState().setFileLookPreset('b', 'umbra');
  await flush();
  vi.mocked(host.library.save).mockClear();
  window.dispatchEvent(new Event('focus'));
  await flush();
  expect(vi.mocked(host.library.save).mock.calls.map(call => call[0])).toEqual(['a', 'b']);
  await vi.advanceTimersByTimeAsync(10000);
  expect(host.library.save).toHaveBeenCalledTimes(2);
});
it('clear drains writes and forgets the ledger before resuming', async () => {
  const a = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => a.promise);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  let paused = false;
  const pause = pausePersistence().then(() => { paused = true; });
  expect(unsavedEdits()).toEqual([]);
  await Promise.resolve();
  expect(paused).toBe(false);
  a.resolve();
  await pause;
  useAppStore.setState({ files: [] });
  resumePersistence();
  window.dispatchEvent(new Event('focus'));
  await flush();
  expect(host.library.save).toHaveBeenCalledTimes(1);
});

it('keeps the latest edit and reports an older in-flight save rejection', async () => {
  let reject!: (error: Error) => void;
  const b = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => new Promise<void>((_resolve, fail) => { reject = fail; })).mockImplementationOnce(() => b.promise);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  reject(new Error('RAW vanished'));
  await flush();
  try {
    expect(unsavedEdits()[0]).toMatchObject({ edit: { lookPreset: 'colorful' }, error: 'RAW vanished' });
    expect(useAppStore.getState().files[0].editingNote).toBe('RAW vanished');
  } finally { b.resolve(); }
  await flush();
  expect(unsavedEdits()).toEqual([]);
});

it('keeps edit B flushable when persistence remounts while save A is in flight', async () => {
  const a = deferred();
  vi.mocked(host.library.save).mockImplementationOnce(() => a.promise);
  useAppStore.getState().setFileLookPreset('a', 'umbra');
  await flush();
  useAppStore.getState().setFileLookPreset('a', 'colorful');
  stop();
  stop = startPersistence();
  const flushing = flushPersistence();
  a.resolve();
  await flushing;
  expect(host.library.save).toHaveBeenCalledTimes(2);
  expect(isPhotoDirty('a')).toBe(false);
});
