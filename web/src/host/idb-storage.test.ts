import { afterEach, expect, it, vi } from 'vitest';
import { IDBDatabase } from 'fake-indexeddb';
import { createFileStorage } from './idb-storage';
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
