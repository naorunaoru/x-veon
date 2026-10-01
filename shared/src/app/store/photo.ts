import type { LibraryPhoto, PhotoEdit, PhotoFacts } from '@/host';
import { deserializeResultMeta, serializeResultMeta, type DemosaicMethod, type ModelSize } from '@/lib/types';
import type { QueuedFile } from './types';
import { effectiveMethod } from '@/app/photo-edit';
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
    resultMethod: facts.resultMethod,
    progress: null,
    editRevision: 0,
    modelNeedsResolution: false,
    actualModel: facts.resultMethod === 'neural-net' ? facts.resultMeta?.metadata.modelIdentity ?? null : null,
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
    resultMethod: file.resultMethod,
  };
}
/** A user mutation can adopt the model actually used; passive processing cannot replace an unknown hash. */
export function editPhoto(file: QueuedFile, patch: Partial<PhotoEdit>, defaults: { demosaicMethod: DemosaicMethod }): QueuedFile {
  if (file.editing === 'view-only') return file;
  const edit = { ...file.edit, ...patch };
  if (edit.demosaicMethod === null)
    edit.demosaicMethod = file.resultMethod ?? effectiveMethod(file.edit, defaults.demosaicMethod, file.cfaType);
  if (edit.demosaicMethod && edit.demosaicMethod !== 'neural-net') edit.model = null;
  else if (!('model' in patch) && file.actualModel) edit.model = file.actualModel;
  const adoptsCurrentResult =
    !('model' in patch) &&
    file.actualModel &&
    file.status === 'done' &&
    file.resultMethod === 'neural-net' &&
    edit.demosaicMethod === 'neural-net' &&
    file.processedKey ===
      processingKey(file, { demosaicMethod: 'neural-net', modelSize: file.actualModel.size });
  return {
    ...file,
    edit,
    // Adopting the checkpoint already rendered changes provenance, not image pixels.
    processedKey: adoptsCurrentResult
      ? processingKey({ ...file, edit }, { demosaicMethod: 'neural-net', modelSize: file.actualModel!.size })
      : file.processedKey,
    modelNeedsResolution:
      edit.demosaicMethod === 'neural-net' &&
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
  const method = effectiveMethod(file.edit, defaults.demosaicMethod, file.cfaType);
  return method === 'neural-net'
    ? `${method}:${file.edit.model?.size ?? defaults.modelSize}:${file.edit.model?.sha256 ?? ''}`
    : method;
}
