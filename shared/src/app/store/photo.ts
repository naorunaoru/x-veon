import type { LibraryPhoto, PhotoEdit, PhotoFacts } from '@/host';
import { deserializeResultMeta, serializeResultMeta, type DemosaicMethod, type ModelSize } from '@/lib/types';
import type { QueuedFile } from './types';
export function fromLibraryPhoto(photo: LibraryPhoto): QueuedFile {
  const { facts, ...base } = photo;
  return {
    ...base,
    metadata: facts.metadata,
    cfaType: facts.cfaType,
    status: facts.status,
    error: facts.error,
    lensProfile: facts.lensProfile,
    result: facts.resultMeta ? deserializeResultMeta(facts.resultMeta) : null,
    resultMethod: facts.resultMeta ? photo.edit.demosaicMethod : null,
    progress: null,
    editRevision: 0,
    modelNeedsResolution: false,
    actualModel: null,
    modelNote: null,
    processedKey: null,
  };
}
export function factsOf(file: QueuedFile): PhotoFacts {
  return {
    cfaType: file.cfaType,
    metadata: file.metadata,
    status: file.status === 'processing' ? 'queued' : file.status,
    error: file.error,
    lensProfile: file.lensProfile,
    resultMeta: file.result ? serializeResultMeta(file.result) : null,
  };
}
/** A user mutation can adopt the model actually used; passive processing cannot replace an unknown hash. */
export function editPhoto(file: QueuedFile, patch: Partial<PhotoEdit>): QueuedFile {
  if (file.editing === 'view-only') return file;
  const edit = { ...file.edit, ...patch };
  if (edit.demosaicMethod && edit.demosaicMethod !== 'neural-net') edit.model = null;
  else if (!('model' in patch) && file.actualModel) edit.model = file.actualModel;
  return {
    ...file,
    edit,
    modelNeedsResolution:
      (!edit.demosaicMethod || edit.demosaicMethod === 'neural-net') &&
      !file.actualModel &&
      !('model' in patch),
    editRevision: file.editRevision + 1,
    modelNote: edit.model?.sha256 === file.actualModel?.sha256 ? null : file.modelNote,
  };
}
export function processingKey(
  file: QueuedFile,
  defaults: { demosaicMethod: DemosaicMethod; modelSize: ModelSize },
): string {
  const method = file.edit.demosaicMethod ?? defaults.demosaicMethod;
  return method === 'neural-net'
    ? `${method}:${file.edit.model?.size ?? defaults.modelSize}:${file.edit.model?.sha256 ?? ''}`
    : method;
}
