import { beforeEach, describe, expect, it, vi } from 'vitest';
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
vi.mock('@/pipeline/decode/raf-thumbnail', () => ({
  extractRafThumbnail: () => new Blob(['t']),
  extractRafQuickMetadata: () => ({ camera: 'Fuji', lensModel: 'XF35', focalLength: 35, fNumber: 2 }),
}));
import { createWebLibrary } from './library';
import { assertDatabaseActive, closeDatabase, openDatabase } from '@/app/storage/database';
import { createFileStorage, type PersistedFile } from './idb-storage';
import { defaultEdit } from '@/test/fake-host';
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
beforeEach(() => {
  vi.resetAllMocks();
  storage.listRawFileIds.mockResolvedValue(new Set());
  storage.readThumbnail.mockResolvedValue(null);
  storage.writeRaw.mockResolvedValue(undefined);
  storage.writeThumbnail.mockResolvedValue(undefined);
  storage.deleteAllForFile.mockResolvedValue(undefined);
  URL.createObjectURL = vi.fn(() => 'blob:thumb');
  URL.revokeObjectURL = vi.fn();
});
describe('web library', () => {
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
    const { host, dbName } = setup();
    const db = await openDatabase(dbName);
    db.close();
    const result = await host.load();
    expect(result.complete).toBe(false);
    expect(storage.listRawFileIds).not.toHaveBeenCalled();
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
  bytes.resolve(new ArrayBuffer(4));
  await cleared;
  await vi.waitFor(() => expect(reload).toHaveBeenCalledOnce());
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
