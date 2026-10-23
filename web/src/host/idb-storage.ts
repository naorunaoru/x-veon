import type { CfaType, DemosaicMethod, LookPreset, ModelIdentity, SerializableResultMeta } from '@/lib/types';
import type { LensProfile } from '@/app/lens/lensfun';
import type { LensEdit } from '@/lib/lens/edit';
import type { LibraryPhoto, PhotoEdit, PhotoFacts, Rating } from '@/host';
import { decodeSidecar, encodeSidecar, unknownProperties, type SidecarProperties } from '@/app/sidecar';
import { defaultPhotoEdit } from '@/app/photo-edit';
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
  resultMeta: SerializableResultMeta | null;
  cachedMethods: DemosaicMethod[]; // deprecated — kept for schema compat
  lensProfile: LensProfile | null;
  addedAt: number;
  sidecar?: SidecarProperties;
  /** Legacy shape, read only; absent once a record has been saved in the new shape. */
  editMethod?: DemosaicMethod | null;
  lookPreset?: LookPreset;
  openDrtOverrides?: Record<string, number | boolean>;
  preProcessOverrides?: Record<string, number>;
  model?: ModelIdentity | null;
  lens?: LensEdit;
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
          // A legacy record without editMethod reads its edit from resultMethod; pin it before the facts change.
          const pinned = stored.sidecar || 'editMethod' in stored ? {} : { editMethod: stored.resultMethod };
          merged = { ...stored, ...pinned, ...recordFacts(facts) };
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
    ...(p.sidecar ? readSidecar(p.sidecar) : { rating: 0, editing: 'saved', editingNote: null, edit: legacyEdit(p) }),
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
  rating: Rating,
  facts: PhotoFacts,
  addedAt: number,
  previous?: PersistedFile,
): PersistedFile {
  return {
    id: photo.id,
    name: photo.name,
    originalName: photo.originalName,
    fileSize: photo.fileSize,
    ...recordFacts(facts),
    cachedMethods: [],
    addedAt,
    sidecar: encodeSidecar({ edit, rating }, unknownProperties(previous?.sidecar ?? {})),
  };
}

function readSidecar(sidecar: SidecarProperties): Pick<LibraryPhoto, 'edit' | 'rating' | 'editing' | 'editingNote'> {
  try {
    const { edit, rating } = decodeSidecar(sidecar);
    return { edit, rating, editing: 'saved', editingNote: null };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { edit: defaultPhotoEdit(), rating: 0, editing: 'view-only', editingNote: 'Sidecar is unreadable: ' + message };
  }
}

function legacyEdit(p: PersistedFile): PhotoEdit {
  return {
    version: 1,
    lookPreset: p.lookPreset ?? 'default',
    openDrtOverrides: p.openDrtOverrides ?? {},
    preProcessOverrides: p.preProcessOverrides ?? {},
    demosaicMethod: 'editMethod' in p ? p.editMethod ?? null : p.resultMethod,
    model: p.model ?? null,
    ...(p.lens ? { lens: p.lens } : {}),
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
