import type { LibraryPhoto, PhotoEdit, PhotoFacts } from '@/host';
import { deserializeResultMeta, serializeResultMeta, type DemosaicMethod, type ModelSize } from '@/lib/types';
import type { QueuedFile } from './types';
import { effectiveMethod } from '@/app/photo-edit';
let revision = 0;
/** Unique for every user edit throughout this window session. */
export function nextRevision(): number { return ++revision; }
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
  // A first grade edit before decode must wait for the CFA-compatible method.
  const awaitingMethod = edit.demosaicMethod === null && file.resultMethod === null && file.cfaType === null;
  if (edit.demosaicMethod === null && !awaitingMethod)
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
    modelNeedsResolution: awaitingMethod || (
      edit.demosaicMethod === 'neural-net' &&
      !file.actualModel &&
      !('model' in patch)),
    editRevision: nextRevision(),
    editing: 'session',
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

/** Preserve rendering and session history when a listing only refreshes disk/cache state. */
export function mergeLibraryPhoto(live: QueuedFile, incoming: QueuedFile, defaults: { demosaicMethod: DemosaicMethod; modelSize: ModelSize }): { file: QueuedFile; invalidate: boolean } {
  const sameRaw = live.sourceVersion === incoming.sourceVersion && live.fileSize === incoming.fileSize && live.originalName === incoming.originalName;
  const next = { ...incoming, cfaType: incoming.cfaType ?? live.cfaType };
  const invalidate = !sameRaw || (live.cfaType !== null && incoming.cfaType !== null && live.cfaType !== incoming.cfaType) || processingKey(live, defaults) !== processingKey(next, defaults);
  if (invalidate) return { file: incoming, invalidate: true };
  const flatEqual = (a: object, b: object) => Object.keys(a).length === Object.keys(b).length && Object.entries(a).every(([key, value]) => value === (b as Record<string, unknown>)[key]);
  const sameEdit = live.edit.lookPreset === incoming.edit.lookPreset && live.edit.demosaicMethod === incoming.edit.demosaicMethod
    && live.edit.model?.size === incoming.edit.model?.size && live.edit.model?.sha256 === incoming.edit.model?.sha256
    && flatEqual(live.edit.preProcessOverrides, incoming.edit.preProcessOverrides) && flatEqual(live.edit.openDrtOverrides, incoming.edit.openDrtOverrides);
  return { invalidate: false, file: {
    ...live,
    name: incoming.name, thumbnailUrl: incoming.thumbnailUrl,
    editing: incoming.editing, editingNote: incoming.editingNote,
    metadata: incoming.metadata ?? live.metadata, cfaType: next.cfaType,
    lensProfile: incoming.lensProfile ?? live.lensProfile,
    ...(sameEdit ? {} : { edit: incoming.edit, editRevision: incoming.editRevision, modelNeedsResolution: incoming.modelNeedsResolution, lookHistory: undefined }),
  } };
}
