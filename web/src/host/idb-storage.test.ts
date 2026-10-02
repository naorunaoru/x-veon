import { afterEach, expect, it, vi } from 'vitest';
import { IDBDatabase } from 'fake-indexeddb';
import { createFileStorage } from './idb-storage';
import { fromRecord, toRecord, type PersistedFile } from './idb-storage';
import { defaultEdit, fakePhoto } from '@/test/fake-host';
import { factsOf, fromLibraryPhoto } from '@/app/store/photo';
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

it('restores legacy resultMethod as both the edit and processing fact', () => {
  const record = toRecord(fakePhoto(), defaultEdit(), fakePhoto().facts, 1);
  delete (record as Partial<PersistedFile>).editMethod;
  record.resultMethod = 'markesteijn3';
  const photo = fromRecord(record, null);
  expect(photo.edit.demosaicMethod).toBe('markesteijn3');
  expect(photo.facts.resultMethod).toBe('markesteijn3');
});

it('stores and restores the edit method separately from the result method', async () => {
  const storage = createFileStorage('two-methods');
  const photo = fakePhoto();
  const edit = { ...defaultEdit(), demosaicMethod: 'markesteijn3' as const };
  const facts = { ...photo.facts, resultMethod: 'neural-net' as const };
  await storage.putFile(toRecord(photo, edit, facts, 2));
  const record = (await storage.getAllFiles())[0];
  expect(record.editMethod).toBe('markesteijn3');
  expect(record.resultMethod).toBe('neural-net');
  const restored = fromRecord(record, null);
  expect(restored.edit.demosaicMethod).toBe('markesteijn3');
  expect(restored.facts.resultMethod).toBe('neural-net');
  const queued = fromLibraryPhoto(restored);
  expect(queued.resultMethod).toBe('neural-net');
  expect(factsOf(queued).resultMethod).toBe('neural-net');
});

it.each([null, 'ahd'] as const)('facts migration preserves legacy edit method %s', async method => {
 const storage = createFileStorage('legacy-facts-' + method); const p = fakePhoto();
 const record = toRecord(p, defaultEdit(), { ...p.facts, resultMethod: method }, 1); delete record.editMethod;
 await storage.putFile(record); await storage.mergeFacts(p.id, { ...p.facts, resultMethod: 'neural-net' });
 const restored = fromRecord((await storage.getAllFiles())[0], null);
 expect(restored.edit.demosaicMethod).toBe(method); expect(restored.facts.resultMethod).toBe('neural-net');
});
