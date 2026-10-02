import type { CfaType, DemosaicMethod, LookPreset, ModelIdentity, SerializableResultMeta } from '@/lib/types';
import type { LensProfile } from '@/app/lens/lensfun';
import type { LibraryPhoto, PhotoEdit, PhotoFacts } from '@/host';
import { transact, openDatabase, assertDatabaseActive } from '@/app/storage/database';
export interface PersistedFile {
  id: string;
  name: string;
  originalName: string;
  fileSize: number;
  cfaType: CfaType | null;
  camera: string | null;
  lensModel: string | null;
  focalLength: number | null;
  fNumber: number | null;
  status: 'queued' | 'done' | 'error';
  error: string | null;
  resultMethod: DemosaicMethod | null;
  editMethod?: DemosaicMethod | null;
  resultMeta: SerializableResultMeta | null;
  cachedMethods: DemosaicMethod[]; // deprecated — kept for schema compat
  lookPreset: LookPreset;
  lensProfile: LensProfile | null;
  openDrtOverrides: Record<string, number | boolean>;
  preProcessOverrides: Record<string, number>;
  addedAt: number;
  model?: ModelIdentity | null;
}

export function createFileStorage(dbName: string) {
  return {
    getAllFiles: () => transact<PersistedFile[]>(dbName, 'files', 'readonly', (store) => store.getAll()),
    putFile: (file: PersistedFile) =>
      transact(dbName, 'files', 'readwrite', (store) => store.put(file)).then(() => {}),
    mergeFacts: async (id: string, facts: PhotoFacts): Promise<PersistedFile> => {
      const db = await openDatabase(dbName);
      assertDatabaseActive(dbName);
      return new Promise((resolve, reject) => {
        const tx = db.transaction('files', 'readwrite');
        const store = tx.objectStore('files');
        let merged: PersistedFile;
        let missing = false;
        tx.oncomplete = () => resolve(merged);
        tx.onerror = () => reject(tx.error ?? new Error('Facts write failed.'));
        tx.onabort = () => reject(missing ? new Error('Photo is no longer in the library.') : tx.error ?? new Error('Facts write aborted.'));
        const request = store.get(id);
        request.onsuccess = () => {
          if (!request.result) { missing = true; tx.abort(); return; }
          const stored = request.result as PersistedFile;
          merged = { ...stored, editMethod: 'editMethod' in stored ? stored.editMethod : stored.resultMethod, ...recordFacts(facts) };
          store.put(merged);
        };
      });
    },
    deleteFile: (id: string) => transact(dbName, 'files', 'readwrite', (store) => store.delete(id)),
  };
}
export function fromRecord(p: PersistedFile, thumbnailUrl: string | null): LibraryPhoto {
  return {
    id: p.id,
    name: p.name,
    originalName: p.originalName,
    fileSize: p.fileSize,
    thumbnailUrl,
    editing: 'saved',
    editingNote: null,
    edit: {
      version: 1,
      lookPreset: p.lookPreset,
      openDrtOverrides: p.openDrtOverrides,
      preProcessOverrides: p.preProcessOverrides ?? {},
      demosaicMethod: 'editMethod' in p ? p.editMethod ?? null : p.resultMethod,
      model: p.model ?? null,
    },
    facts: {
      cfaType: p.cfaType,
      metadata: p.camera
        ? {
            camera: p.camera,
            lensModel: p.lensModel ?? '',
            focalLength: p.focalLength ?? 0,
            fNumber: p.fNumber ?? 0,
          }
        : null,
      status: p.status === 'done' && p.resultMeta ? 'done' : 'queued',
      error: null,
      resultMeta: p.resultMeta,
      resultMethod: p.resultMethod,
      lensProfile: p.lensProfile ?? null,
    },
  };
}
export function toRecord(
  photo: LibraryPhoto,
  edit: PhotoEdit,
  facts: PhotoFacts,
  addedAt: number,
): PersistedFile {
  return {
    id: photo.id,
    name: photo.name,
    originalName: photo.originalName,
    fileSize: photo.fileSize,
    ...recordFacts(facts),
    editMethod: edit.demosaicMethod,
    cachedMethods: [],
    lookPreset: edit.lookPreset,
    openDrtOverrides: edit.openDrtOverrides,
    preProcessOverrides: edit.preProcessOverrides,
    model: edit.model,
    addedAt,
  };
}

/** This explicit field list is the only record data a facts transaction may change. */
function recordFacts(facts: PhotoFacts) {
  return {
    cfaType: facts.cfaType,
    camera: facts.metadata?.camera ?? null,
    lensModel: facts.metadata?.lensModel ?? null,
    focalLength: facts.metadata?.focalLength ?? null,
    fNumber: facts.metadata?.fNumber ?? null,
    status: facts.status,
    error: facts.error,
    resultMeta: facts.resultMeta,
    resultMethod: facts.resultMethod,
    lensProfile: facts.lensProfile,
  };
}
