import type { CfaType, DemosaicMethod, ExportFormat, LookPreset } from '@/pipeline/types';
import type { SerializableResultMeta } from '@/pipeline/types';

// ── Schema ──────────────────────────────────────────────────────────────────

export interface PersistedFile {
  id: string;
  name: string;
  originalName: string;
  fileSize: number;
  cfaType: CfaType | null;
  camera: string | null;
  thumbnailBlob: Blob | null;
  status: 'queued' | 'done' | 'error';
  error: string | null;
  resultMethod: DemosaicMethod | null;
  resultMeta: SerializableResultMeta | null;
  cachedMethods: DemosaicMethod[];
  lookPreset: LookPreset;
  openDrtOverrides: Record<string, number | boolean>;
  addedAt: number;
}

interface AppSetting {
  key: string;
  value: unknown;
}

// ── Connection ──────────────────────────────────────────────────────────────

const DB_NAME = 'xtrans-demosaic';
const DB_VERSION = 1;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('files')) {
        db.createObjectStore('files', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('settings')) {
        db.createObjectStore('settings', { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => {
      dbPromise = null;
      reject(req.error);
    };
  });
  return dbPromise;
}

// ── Files CRUD ──────────────────────────────────────────────────────────────

export async function getAllFiles(): Promise<PersistedFile[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('files', 'readonly');
    const store = tx.objectStore('files');
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as PersistedFile[]);
    req.onerror = () => reject(req.error);
  });
}

export async function putFile(file: PersistedFile): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('files', 'readwrite');
    const store = tx.objectStore('files');
    const req = store.put(file);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function deleteFile(id: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('files', 'readwrite');
    const store = tx.objectStore('files');
    const req = store.delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ── Settings CRUD ───────────────────────────────────────────────────────────

export async function getSetting<T>(key: string): Promise<T | undefined> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('settings', 'readonly');
    const store = tx.objectStore('settings');
    const req = store.get(key);
    req.onsuccess = () => {
      const row = req.result as AppSetting | undefined;
      resolve(row?.value as T | undefined);
    };
    req.onerror = () => reject(req.error);
  });
}

export async function putSetting(key: string, value: unknown): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('settings', 'readwrite');
    const store = tx.objectStore('settings');
    const req = store.put({ key, value } satisfies AppSetting);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ── Debounced file persist (for rapid slider changes) ───────────────────────

const pendingTimers = new Map<string, ReturnType<typeof setTimeout>>();

export function debouncedPutFile(file: PersistedFile, delayMs = 300): void {
  const existing = pendingTimers.get(file.id);
  if (existing) clearTimeout(existing);
  pendingTimers.set(
    file.id,
    setTimeout(() => {
      pendingTimers.delete(file.id);
      putFile(file).catch((e) => console.warn('IDB persist failed:', e));
    }, delayMs),
  );
}
