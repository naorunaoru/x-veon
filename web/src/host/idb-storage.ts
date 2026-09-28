import type { CfaType, DemosaicMethod, LookPreset, ModelIdentity, SerializableResultMeta } from '@/lib/types';
import type { LensProfile } from '@/app/lens/lensfun';
import type { LibraryPhoto, PhotoEdit, PhotoFacts } from '@/host';
import { transact } from '@/app/storage/database';
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
      demosaicMethod: p.resultMethod,
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
    cfaType: facts.cfaType,
    camera: facts.metadata?.camera ?? null,
    lensModel: facts.metadata?.lensModel ?? null,
    focalLength: facts.metadata?.focalLength ?? null,
    fNumber: facts.metadata?.fNumber ?? null,
    status: facts.status,
    error: facts.error,
    resultMeta: facts.resultMeta,
    resultMethod: edit.demosaicMethod,
    cachedMethods: [],
    lensProfile: facts.lensProfile,
    lookPreset: edit.lookPreset,
    openDrtOverrides: edit.openDrtOverrides,
    preProcessOverrides: edit.preProcessOverrides,
    model: edit.model,
    addedAt,
  };
}
