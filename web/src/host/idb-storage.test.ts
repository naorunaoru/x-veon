import { afterEach, expect, it, vi } from 'vitest';
import { IDBDatabase } from 'fake-indexeddb';
import { createFileStorage } from './idb-storage';
import { fromRecord, toRecord, type PersistedFile } from './idb-storage';
import { defaultEdit, fakePhoto } from '@/test/fake-host';
import { factsOf, fromLibraryPhoto } from '@/app/store/photo';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
const { putFile } = createFileStorage('abort-test');
afterEach(() => vi.restoreAllMocks());
it('rejects a save when its request succeeded but the transaction aborted', async () => {
  const transaction = IDBDatabase.prototype.transaction;
  vi.spyOn(IDBDatabase.prototype, 'transaction').mockImplementation(function (
    this: IDBDatabase,
    ...args: Parameters<typeof transaction>
  ) {
    const tx = transaction.apply(this, args);
    if (args[1] === 'readwrite') {
      const store = tx.objectStore('files');
      const put = store.put.bind(store);
      vi.spyOn(store, 'put').mockImplementation((...values: Parameters<typeof put>) => {
        const request = put(...values);
        request.addEventListener('success', () => tx.abort());
        return request;
      });
      vi.spyOn(tx, 'objectStore').mockReturnValue(store);
    }
    return tx;
  });
  await expect(putFile({ id: 'aborted' } as never)).rejects.toBeDefined();
});

/** A record in the shape written before the sidecar migration. */
function legacyRecord(fields: Partial<PersistedFile> = {}): PersistedFile {
  return {
    id: 'a', name: 'a', originalName: 'a.raf', fileSize: 42,
    cfaType: 'xtrans', camera: null, lensModel: null, focalLength: null, fNumber: null,
    status: 'queued', error: null, resultMethod: null, resultMeta: null,
    cachedMethods: [], lensProfile: null, addedAt: 1,
    lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {}, editMethod: null, model: null,
    ...fields,
  };
}
const LEGACY_KEYS = ['lookPreset', 'editMethod', 'openDrtOverrides', 'preProcessOverrides', 'model', 'lens'];

it('writes the sidecar shape and no legacy field', () => {
  const record = toRecord(fakePhoto(), { ...defaultEdit(), lookPreset: 'umbra' }, 2, fakePhoto().facts, 1);
  expect(record.sidecar).toEqual({ 'xveon:SchemaVersion': '1', 'xveon:Look': 'umbra', 'xmp:Rating': '2' });
  for (const key of LEGACY_KEYS) expect(Object.hasOwn(record, key)).toBe(false);
  const photo = fromRecord(record, null);
  expect(photo.edit.lookPreset).toBe('umbra');
  expect(photo.rating).toBe(2);
  expect(photo.editing).toBe('saved');
});

it('reads a legacy record as the same edit with rating 0', () => {
  const lens = { ...DEFAULT_LENS_EDIT, ca: false, source: { lensfun: { file: 'misc.json', model: '23mmF1.4XM' } } };
  const photo = fromRecord(legacyRecord({ lookPreset: 'umbra', editMethod: 'dht', preProcessOverrides: { exposure: 1 }, lens }), null);
  expect(photo.edit).toEqual({ ...defaultEdit(), lookPreset: 'umbra', demosaicMethod: 'dht', preProcessOverrides: { exposure: 1 }, lens });
  expect(photo.rating).toBe(0);
  expect(photo.editing).toBe('saved');
});

it('reads a pre-editMethod record', () => {
  const record = legacyRecord({ resultMethod: 'markesteijn3' });
  delete record.editMethod;
  const photo = fromRecord(record, null);
  expect(photo.edit.demosaicMethod).toBe('markesteijn3');
  expect(photo.facts.resultMethod).toBe('markesteijn3');
});

it('keeps unknown sidecar properties across a save', () => {
  const previous = { ...legacyRecord(), sidecar: { 'xveon:SchemaVersion': '1', 'xveon:Look': 'umbra', 'xveon:Future': 'x' } };
  const record = toRecord(fakePhoto(), defaultEdit(), 0, fakePhoto().facts, 1, previous);
  expect(record.sidecar).toEqual({ 'xveon:SchemaVersion': '1', 'xveon:Future': 'x' });
});

it('loads an unreadable sidecar as view-only', () => {
  const record = { ...toRecord(fakePhoto(), defaultEdit(), 3, fakePhoto().facts, 1), sidecar: { 'xveon:Look': 'nope', 'xmp:Rating': '3' } };
  const photo = fromRecord(record, null);
  expect(photo.editing).toBe('view-only');
  expect(photo.editingNote).toMatch(/Sidecar is unreadable: Look: invalid preset/);
  expect(photo.edit).toEqual(defaultEdit());
  expect(photo.rating).toBe(0);
});

it('stores and restores the edit method separately from the result method', async () => {
  const storage = createFileStorage('two-methods');
  const photo = fakePhoto();
  const edit = { ...defaultEdit(), demosaicMethod: 'markesteijn3' as const };
  const facts = { ...photo.facts, resultMethod: 'neural-net' as const };
  await storage.putFile(toRecord(photo, edit, 0, facts, 2));
  const record = (await storage.getAllFiles())[0];
  expect(record.sidecar).toMatchObject({ 'xveon:Demosaic': 'markesteijn3' });
  expect(record.resultMethod).toBe('neural-net');
  const restored = fromRecord(record, null);
  expect(restored.edit.demosaicMethod).toBe('markesteijn3');
  expect(restored.facts.resultMethod).toBe('neural-net');
  const queued = fromLibraryPhoto(restored);
  expect(queued.resultMethod).toBe('neural-net');
  expect(factsOf(queued).resultMethod).toBe('neural-net');
});

it.each([null, 'ahd'] as const)('facts migration preserves legacy edit method %s', async method => {
 const storage = createFileStorage('legacy-facts-' + method);
 const record = legacyRecord({ resultMethod: method }); delete record.editMethod;
 await storage.putFile(record); await storage.mergeFacts(record.id, { ...fakePhoto().facts, resultMethod: 'neural-net' });
 const restored = fromRecord((await storage.getAllFiles())[0], null);
 expect(restored.edit.demosaicMethod).toBe(method); expect(restored.facts.resultMethod).toBe('neural-net');
});

it('a facts merge leaves a sidecar record without a legacy edit method', async () => {
  const storage = createFileStorage('sidecar-facts');
  const p = fakePhoto();
  await storage.putFile(toRecord(p, defaultEdit(), 0, { ...p.facts, resultMethod: 'ahd' }, 1));
  const merged = await storage.mergeFacts(p.id, { ...p.facts, resultMethod: 'neural-net' });
  expect(Object.hasOwn(merged, 'editMethod')).toBe(false);
  expect(fromRecord(merged, null).edit.demosaicMethod).toBeNull();
});

it('round-trips the lens block and reads an old record as absent', () => {
  const lens = { ...DEFAULT_LENS_EDIT, ca: false, source: { lensfun: { file: 'misc.json', model: '23mmF1.4XM' } } };
  const record = toRecord(fakePhoto(), { ...defaultEdit(), lens }, 0, fakePhoto().facts, 1);
  expect(fromRecord(record, null).edit.lens).toEqual(lens);
  expect(fromRecord(legacyRecord(), null).edit.lens).toBeUndefined();
});
