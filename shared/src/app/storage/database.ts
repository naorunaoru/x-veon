/** Version 1 is shared by the web library and app settings. Connections are owned here. */
const connections = new Map<string, Promise<IDBDatabase>>();
const invalidated = new Set<string>();
interface DatabaseLifecycle {
  suspend(): void | Promise<void>;
  closed(): void;
}
const lifecycles = new Map<string, Set<DatabaseLifecycle>>();
export function onDatabaseVersionChange(name: string, lifecycle: DatabaseLifecycle): void {
  const listeners = lifecycles.get(name) ?? new Set<DatabaseLifecycle>();
  listeners.add(lifecycle);
  lifecycles.set(name, listeners);
}
export function assertDatabaseActive(name: string): void {
  if (invalidated.has(name)) throw new Error('This library changed in another tab. Reload to continue.');
}
export function openDatabase(name: string): Promise<IDBDatabase> {
  try {
    assertDatabaseActive(name);
  } catch (error) {
    return Promise.reject(error);
  }
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
        // A stale tab must never reopen and repopulate a deleted database.
        invalidated.add(name);
        connections.delete(name);
        const listeners = [...(lifecycles.get(name) ?? [])];
        lifecycles.delete(name);
        // Suspend synchronously, then drain host operations before releasing deletion.
        const draining = listeners.map((listener) => {
          try {
            return Promise.resolve(listener.suspend());
          } catch (error) {
            return Promise.reject(error);
          }
        });
        void Promise.allSettled(draining).then(() => {
          db.close();
          for (const listener of listeners) listener.closed();
        });
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
  assertDatabaseActive(name);
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
