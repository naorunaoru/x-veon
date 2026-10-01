import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
const processRaw = vi.hoisted(() => vi.fn());
const storage = vi.hoisted(() => ({
  writeRaw: vi.fn(),
  writeThumbnail: vi.fn(),
  readRaw: vi.fn(),
  readThumbnail: vi.fn(),
  deleteAllForFile: vi.fn(),
  listRawFileIds: vi.fn(),
  clear: vi.fn(),
}));
vi.mock('./opfs-storage', () => ({ createOpfsStorage: () => storage }));
vi.mock('@/pipeline', () => ({ processRaw }));
vi.mock('@/app/services/library', () => ({ matchLensFor: vi.fn() }));
vi.mock('@/pipeline/decode/raf-thumbnail', () => ({
  extractRafThumbnail: () => new Blob(['t']),
  extractRafQuickMetadata: () => ({ camera: 'Fuji', lensModel: 'XF35', focalLength: 35, fNumber: 2 }),
}));
import { createWebLibrary } from './library';
import { assertDatabaseActive, closeDatabase, openDatabase } from '@/app/storage/database';
import { createFileStorage, type PersistedFile } from './idb-storage';
import { defaultEdit, fakeHost } from '@/test/fake-host';
import { fromLibraryPhoto, processingKey, factsOf } from '@/app/store/photo';
import { useAppStore } from '@/app/store';
import { setHost } from '@/app/services/host';
import { processFile, setPipeline, discardResult } from '@/app/services/processing';
import { startPersistence } from '@/app/services/persistence';
import { useAutoProcess } from '@/app/hooks/useAutoProcess';
import type { DemosaicMethod } from '@/lib/types';
let serial = 0;
function setup() {
  const dbName = `library-test-${++serial}`;
  return { dbName, host: createWebLibrary({ dbName, opfsRoot: 'dev' }) };
}
function raw(name = 'a.raf') {
  const file = new File(['raw'], name);
  Object.defineProperty(file, 'arrayBuffer', { value: async () => new ArrayBuffer(4), configurable: true });
  return file;
}
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
function processedImage(method: DemosaicMethod) {
  return {
    method,
    gpu: { texture: {} as GPUTexture, width: 4, height: 2 },
    meta: {
      exportData: { width: 4, height: 2, xyzToCam: null, wbCoeffs: new Float32Array(3), camToXyz: new Float32Array(12), orientation: 'Normal' },
      metadata: { make: 'F', model: 'X', width: 4, height: 2, tileCount: 1, inferenceTime: 0, backend: 'webgpu', exposureBias: 0, lensModel: 'L', focalLength: 0, fNumber: 0, colorTemp: 0, tint: 0, cfaType: 'bayer' as const, modelIdentity: method === 'neural-net' ? { size: 'S' as const, sha256: 'used' } : undefined },
    },
    dispose: vi.fn(),
  };
}
beforeEach(() => {
  vi.resetAllMocks();
  processRaw.mockReset();
  storage.listRawFileIds.mockResolvedValue(new Set());
  storage.readThumbnail.mockResolvedValue(null);
  storage.writeRaw.mockResolvedValue(undefined);
  storage.writeThumbnail.mockResolvedValue(undefined);
  storage.deleteAllForFile.mockResolvedValue(undefined);
  URL.createObjectURL = vi.fn(() => 'blob:thumb');
  URL.revokeObjectURL = vi.fn();
});
describe('web library', () => {
  it('processes, saves facts, reloads, and follows a changed default without creating an edit', async () => {
    const { host, dbName } = setup();
    const { photos: [photo] } = await host.addFiles([raw('first.arw')]);
    const appHost = fakeHost();
    appHost.library = host;
    setHost(appHost);
    setPipeline({} as never);
    useAppStore.setState({ files: [fromLibraryPhoto(photo)], selectedFileId: photo.id, demosaicMethod: 'markesteijn3', modelSize: 'S', processingFileId: null, initialized: true });
    processRaw.mockImplementation(async (_bytes, options: { method: DemosaicMethod }) => processedImage(options.method));
    const save = vi.spyOn(host, 'saveFacts');
    const stop = startPersistence();
    try {
      await processFile(photo.id);
      expect(processRaw).toHaveBeenCalledWith(expect.any(ArrayBuffer), expect.objectContaining({ method: 'neural-net' }), expect.anything());
      await vi.waitFor(async () => {
        expect(save).toHaveBeenCalledTimes(1);
        expect((await createFileStorage(dbName).getAllFiles())[0]).toMatchObject({ resultMethod: 'neural-net', editMethod: null, status: 'done' });
      });
    } finally {
      stop();
      discardResult(photo.id);
    }
    storage.readRaw.mockResolvedValue(new ArrayBuffer(4));
    const reloaded = createWebLibrary({ dbName, opfsRoot: 'dev' });
    const restored = (await reloaded.load()).photos[0];
    expect(restored.edit).toMatchObject({ demosaicMethod: null, model: null });
    expect(restored.facts.resultMethod).toBe('neural-net');
    const nextHost = fakeHost();
    nextHost.library = reloaded;
    setHost(nextHost);
    useAppStore.setState({ files: [fromLibraryPhoto(restored)], selectedFileId: photo.id, demosaicMethod: 'bilinear', modelSize: 'S', processingFileId: null, initialized: true });
    const { unmount } = renderHook(useAutoProcess);
    try {
      await vi.waitFor(() => expect(processRaw).toHaveBeenCalledTimes(2));
      expect(processRaw).toHaveBeenLastCalledWith(expect.any(ArrayBuffer), expect.objectContaining({ method: 'bilinear' }), expect.anything());
      await vi.waitFor(() => expect(useAppStore.getState().files[0].resultMethod).toBe('bilinear'));
      expect(useAppStore.getState().files[0].edit.demosaicMethod).toBeNull();
    } finally {
      unmount();
      discardResult(photo.id);
    }
  });
  it('reloads processing facts without turning an untouched photo into an edit', async () => {
    const { host, dbName } = setup();
    const { photos: [photo] } = await host.addFiles([raw()]);
    await host.save(photo.id, photo.edit, { ...photo.facts, resultMethod: 'neural-net' });
    const reloaded = createWebLibrary({ dbName, opfsRoot: 'dev' });
    const restored = (await reloaded.load()).photos[0];
    expect(restored.edit.demosaicMethod).toBeNull();
    expect(restored.edit.model).toBeNull();
    expect(restored.facts.resultMethod).toBe('neural-net');
    const queued = fromLibraryPhoto(restored);
    expect(processingKey(queued, { demosaicMethod: 'dht', modelSize: 'S' })).toBe('dht');
    expect(processingKey(queued, { demosaicMethod: 'neural-net', modelSize: 'S' })).toBe('neural-net:S:');
    expect(factsOf(queued).resultMethod).toBe('neural-net');
  });
  it('filters imports and retains File bytes inside the host, with selected IDs', async () => {
    const { host } = setup();
    const result = await host.addFiles([raw('one.RAF'), raw('no.txt'), raw('two.arw')]);
    expect(result.photos.map((p) => p.originalName)).toEqual(['one.RAF', 'two.arw']);
    expect(result.selectedIds).toEqual([result.photos[0].id]);
    expect(result.photos[0]).not.toHaveProperty('file');
    expect(result.photos[0].editing).toBe('saved');
    expect((await host.readRaw(result.photos[0].id)).byteLength).toBe(4);
    await vi.waitFor(() => expect(storage.writeThumbnail).toHaveBeenCalledTimes(2));
  });
  it('restores old records without rewriting metadata, order or edits', async () => {
    const { host, dbName } = setup();
    const records = createFileStorage(dbName);
    const legacy: PersistedFile = {
      id: 'old',
      name: 'old',
      originalName: 'old.raf',
      fileSize: 123,
      cfaType: 'xtrans',
      camera: 'Fuji',
      lensModel: 'XF35',
      focalLength: 35,
      fNumber: 2,
      status: 'done',
      error: null,
      resultMethod: 'dht',
      resultMeta: null,
      cachedMethods: [],
      lookPreset: 'marvelous',
      openDrtOverrides: { cwp: 0.4 },
      preProcessOverrides: { exposure: 1 },
      lensProfile: null,
      addedAt: 1234,
    };
    await records.putFile(legacy);
    const { photos, complete } = await host.load();
    expect(complete).toBe(true);
    expect(photos[0]).toMatchObject({
      fileSize: 123,
      edit: { model: null, demosaicMethod: 'dht', lookPreset: 'marvelous' },
    });
    await host.save('old', { ...photos[0].edit, preProcessOverrides: { exposure: 2 } }, photos[0].facts);
    expect((await records.getAllFiles())[0]).toMatchObject({
      fileSize: 123,
      addedAt: 1234,
      lookPreset: 'marvelous',
      preProcessOverrides: { exposure: 2 },
    });
    await closeDatabase(dbName);
  });
  it('keeps a newly imported photo while loading and cleaning orphans', async () => {
    const { host } = setup();
    const listing = deferred<Set<string>>();
    storage.listRawFileIds.mockReturnValueOnce(listing.promise);
    const loading = host.load();
    await vi.waitFor(() => expect(storage.listRawFileIds).toHaveBeenCalled());
    const {
      photos: [photo],
    } = await host.addFiles([raw()]);
    listing.resolve(new Set([photo.id, 'zombie']));
    await loading;
    expect(storage.deleteAllForFile).toHaveBeenCalledWith('zombie');
    expect(storage.deleteAllForFile).not.toHaveBeenCalledWith(photo.id);
  });
  it('removal before bytes arrive prevents writes and after a write drains then cleans it', async () => {
    const { host } = setup();
    const bytes = deferred<ArrayBuffer>();
    const file = raw();
    Object.defineProperty(file, 'arrayBuffer', { value: () => bytes.promise, configurable: true });
    const {
      photos: [photo],
    } = await host.addFiles([file]);
    const removing = host.remove!(photo.id);
    bytes.resolve(new ArrayBuffer(4));
    await removing;
    expect(storage.writeRaw).not.toHaveBeenCalled();
    expect((await host.load()).photos).toEqual([]);
    const write = deferred<void>();
    storage.writeRaw.mockReturnValueOnce(write.promise);
    const {
      photos: [next],
    } = await host.addFiles([raw('next.raf')]);
    await vi.waitFor(() => expect(storage.writeRaw).toHaveBeenCalled());
    const removeNext = host.remove!(next.id);
    write.resolve();
    await removeNext;
    expect(storage.deleteAllForFile).toHaveBeenCalledWith(next.id);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:thumb');
  });
  it('reports incomplete loads and never cleans unknown ownership', async () => {
    const logged = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { host, dbName } = setup();
    const db = await openDatabase(dbName);
    db.close();
    const result = await host.load();
    expect(result.complete).toBe(false);
    expect(storage.listRawFileIds).not.toHaveBeenCalled();
    expect(logged).toHaveBeenCalledExactlyOnceWith('Could not read the library from IndexedDB:', expect.any(DOMException));
    logged.mockRestore();
  });
  it('reports a missing restored RAW with the existing message', async () => {
    const { host } = setup();
    storage.readRaw.mockResolvedValue(null);
    await expect(host.readRaw('missing')).rejects.toThrow('RAW file not found');
  });
});

it.each(['stable', 'beta', 'dev'])(
  'clears only the %s namespace and stable legacy ownership',
  async (channel) => {
    const dbName = `clear-${channel}-${++serial}`;
    const other = `untouched-${serial}`;
    const db = await openDatabase(other);
    db.close();
    const host = createWebLibrary({ dbName, opfsRoot: channel });
    const {
      photos: [photo],
    } = await host.addFiles([raw()]);
    await host.readRaw(photo.id);
    await host.clear!();
    expect(storage.clear).toHaveBeenCalledWith(channel === 'stable');
    expect((await indexedDB.databases()).some((entry) => entry.name === dbName)).toBe(false);
    expect((await indexedDB.databases()).some((entry) => entry.name === other)).toBe(true);
  },
);
it('clear drains a pending import and never resurrects its records', async () => {
  const { host, dbName } = setup();
  const pending = deferred<void>();
  storage.writeRaw.mockReturnValueOnce(pending.promise);
  await host.addFiles([raw()]);
  await vi.waitFor(() => expect(storage.writeRaw).toHaveBeenCalled());
  const cleared = host.clear!();
  pending.resolve();
  await cleared;
  expect(await createFileStorage(dbName).getAllFiles()).toEqual([]);
});
it('clear reports a deletion blocked by another connection without reloading', async () => {
  const dbName = `blocked-${++serial}`;
  const reload = vi.fn();
  const host = createWebLibrary({ dbName, opfsRoot: 'dev', reload });
  const external = await new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(dbName, 1);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  await expect(host.clear!()).rejects.toThrow('Close other tabs');
  expect(reload).not.toHaveBeenCalled();
  external.close();
  await host.clear!();
  expect(reload).toHaveBeenCalledOnce();
});
it('does not revoke the current host-owned thumbnail when a cancelled startup releases its snapshot', async () => {
  const { host } = setup();
  const { photos } = await host.addFiles([raw()]);
  await vi.waitFor(() => expect(URL.createObjectURL).toHaveBeenCalled());
  const snapshot = await host.load();
  host.release?.(snapshot.photos);
  expect(URL.revokeObjectURL).not.toHaveBeenCalledWith(snapshot.photos[0].thumbnailUrl);
  await host.remove!(photos[0].id);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:thumb');
});

it('suspends imports and saves when another tab clears the database, then reloads', async () => {
  const dbName = `invalidated-${++serial}`;
  const reload = vi.fn();
  const host = createWebLibrary({ dbName, opfsRoot: 'beta', reload });
  await host.load();
  const bytes = deferred<ArrayBuffer>();
  const file = raw();
  Object.defineProperty(file, 'arrayBuffer', { value: () => bytes.promise });
  const { photos } = await host.addFiles([file]);
  const cleared = new Promise<void>((resolve, reject) => {
    const request = indexedDB.deleteDatabase(dbName);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  // Versionchange is delivered before the pending import can finish.
  await vi.waitFor(() => expect(() => assertDatabaseActive(dbName)).toThrow('Reload'));
  try {
    // The tab must release its connection while the import is still busy.
    await vi.waitFor(() => expect(reload).toHaveBeenCalledOnce());
    await cleared;
  } finally {
    bytes.resolve(new ArrayBuffer(4));
  }
  await expect(host.addFiles([raw('late.raf')])).rejects.toThrow('Reload');
  await expect(host.save(photos[0].id, photos[0].edit, photos[0].facts)).rejects.toThrow('Reload');
  expect(storage.writeRaw).not.toHaveBeenCalled();
  expect((await indexedDB.databases()).some((db) => db.name === dbName)).toBe(false);
});
it('reports an OPFS failure after database deletion and retries without restoring stale records', async () => {
  const dbName = `partial-clear-${++serial}`;
  const reload = vi.fn();
  const host = createWebLibrary({ dbName, opfsRoot: 'beta', reload });
  const snapshot = await host.addFiles([raw()]);
  await vi.waitFor(async () => expect(await createFileStorage(dbName).getAllFiles()).toHaveLength(1));
  storage.clear.mockRejectedValueOnce(new Error('OPFS denied')).mockResolvedValueOnce(undefined);
  await expect(host.clear!()).rejects.toThrow('OPFS denied');
  expect(reload).not.toHaveBeenCalled();
  expect((await indexedDB.databases()).some((db) => db.name === dbName)).toBe(false);
  await expect(
    host.save(snapshot.photos[0].id, snapshot.photos[0].edit, snapshot.photos[0].facts),
  ).rejects.toThrow('no longer');
  await host.clear!();
  expect(reload).toHaveBeenCalledOnce();
  expect((await indexedDB.databases()).some((db) => db.name === dbName)).toBe(false);
});

it('merges facts atomically without replacing an edit saved by another adapter', async () => {
  const { host, dbName } = setup();
  const { photos: [photo] } = await host.addFiles([raw()]);
  await host.save(photo.id, defaultEdit(), photo.facts);
  const stale = createWebLibrary({ dbName, opfsRoot: 'dev' });
  await stale.load();
  const newer = { ...defaultEdit(), lookPreset: 'umbra' as const, demosaicMethod: 'dht' as const, preProcessOverrides: { exposure: 2 } };
  const editing = host.save(photo.id, newer, photo.facts);
  const facts = { ...photo.facts, resultMethod: 'bilinear' as const, error: 'fact' };
  await Promise.all([editing, stale.saveFacts(photo.id, facts)]);
  const record = (await createFileStorage(dbName).getAllFiles())[0];
  expect(record).toMatchObject({ lookPreset: 'umbra', editMethod: 'dht', preProcessOverrides: { exposure: 2 }, resultMethod: 'bilinear', error: 'fact' });
  await closeDatabase(dbName);
});
