import { expect, it, vi } from 'vitest';
import { createWebLibrary } from './library';
import { createFileStorage } from './idb-storage';
import * as database from '@/app/storage/database';
import { setHost } from '@/app/services/host';
import { fakeHost } from '@/test/fake-host';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { startPersistence } from '@/app/services/persistence';
import { clearLibrary, importFiles } from '@/app/services/library';
const storage = vi.hoisted(() => ({
  writeRaw: vi.fn(async () => {}),
  writeThumbnail: vi.fn(async () => {}),
  clear: vi.fn(async () => {}),
  readThumbnail: vi.fn(),
  listRawFileIds: vi.fn(async () => new Set()),
}));
vi.mock('./opfs-storage', () => ({ createOpfsStorage: () => storage }));
vi.mock('@/pipeline/decode/raf-thumbnail', () => ({
  extractRafThumbnail: () => null,
  extractRafQuickMetadata: () => null,
}));
vi.mock('@/app/services/processing', () => ({ discardResult: vi.fn() }));
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function raw(name: string, wait = Promise.resolve()) {
  const file = new File(['raw'], name);
  Object.defineProperty(file, 'arrayBuffer', {
    value: async () => {
      await wait;
      return new ArrayBuffer(4);
    },
  });
  return file;
}
it('drains an import, settings write and in-flight save while cancelling the debounced save before Clear', async () => {
  const dbName = 'clear-all-writers';
  const library = createWebLibrary({ dbName, opfsRoot: 'beta' });
  setHost({ ...fakeHost(), library, settingsDbName: dbName });
  const snapshot = await library.addFiles([raw('a.raf'), raw('b.raf')]);
  await vi.waitFor(async () => expect(await createFileStorage(dbName).getAllFiles()).toHaveLength(2));
  useAppStore.setState({
    files: snapshot.photos.map((p) =>
      fromLibraryPhoto({ ...p, edit: { ...p.edit, demosaicMethod: 'bilinear' } }),
    ),
    selectedFileId: snapshot.photos[0].id,
    exportQuality: 95,
  });
  const stop = startPersistence();
  const saving = deferred(),
    setting = deferred(),
    importing = deferred();
  const originalSave = library.save.bind(library);
  const save = vi.spyOn(library, 'save').mockImplementationOnce(async (...args) => {
    await saving.promise;
    await originalSave(...args);
  });
  const originalTransact = database.transact;
  try {
    useAppStore.getState().setFilePreProcessOverride(snapshot.photos[0].id, 'exposure', 1);
    await vi.waitFor(() => expect(save).toHaveBeenCalledOnce());
    const transaction = vi.spyOn(database, 'transact').mockImplementationOnce(async (...args) => {
      await setting.promise;
      return originalTransact(...args);
    });
    useAppStore.getState().setExportQuality(80);
    expect(transaction).toHaveBeenCalledWith(dbName, 'settings', 'readwrite', expect.any(Function));
    useAppStore.getState().setFilePreProcessOverride(snapshot.photos[1].id, 'exposure', 2);
    await importFiles([raw('pending.raf', importing.promise)]);
    const clearing = clearLibrary();
    let finished = false;
    void clearing.then(() => {
      finished = true;
    });
    setting.resolve();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(finished).toBe(false);
    expect(storage.clear).not.toHaveBeenCalled();
    saving.resolve();
    await vi.waitFor(() => expect(transaction).toHaveBeenCalledTimes(2));
    expect(finished).toBe(false);
    expect(storage.clear).not.toHaveBeenCalled();
    importing.resolve();
    await clearing;
    window.dispatchEvent(new Event('focus'));
    await new Promise((resolve) => setTimeout(resolve, 350));
    expect(save).toHaveBeenCalledOnce();
    expect(storage.clear).toHaveBeenCalledOnce();
    expect(useAppStore.getState().files).toEqual([]);
    expect((await indexedDB.databases()).some((db) => db.name === dbName)).toBe(false);
  } finally {
    saving.resolve();
    setting.resolve();
    importing.resolve();
    stop();
    vi.restoreAllMocks();
  }
});
