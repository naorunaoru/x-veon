/** Version 1 is shared by the web library and app settings. Connections are owned here. */
const connections = new Map<string, Promise<IDBDatabase>>();
export function openDatabase(name: string): Promise<IDBDatabase> {
  const existing = connections.get(name);
  if (existing) return existing;
  const opening = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(name, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('files')) db.createObjectStore('files', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('settings')) db.createObjectStore('settings', { keyPath: 'key' });
    };
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => {
        db.close();
        connections.delete(name);
      };
      resolve(db);
    };
    request.onerror = () => reject(request.error);
  });
  connections.set(name, opening);
  void opening.catch(() => {
    if (connections.get(name) === opening) connections.delete(name);
  });
  return opening;
}
export async function closeDatabase(name: string): Promise<void> {
  const opening = connections.get(name);
  connections.delete(name);
  if (opening) (await opening.catch(() => null))?.close();
}
/** Resolve only when the transaction commits, including reads for consistent error handling. */
export async function transact<T>(
  name: string,
  store: 'files' | 'settings',
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDatabase(name);
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(store, mode);
    let value: T;
    tx.oncomplete = () => resolve(value);
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
    const request = operation(tx.objectStore(store));
    request.onsuccess = () => {
      value = request.result;
    };
    request.onerror = () => reject(request.error);
  });
}
export async function deleteDatabase(name: string): Promise<void> {
  await closeDatabase(name);
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Close other tabs using this library, then try again.'));
  });
}
