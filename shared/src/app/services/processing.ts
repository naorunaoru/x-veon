/**
 * Owns the pipeline context and the processed images. The canvas borrows a result to display
 * it (acquireResult) and gives it back when it stops (release), so a result outlives its canvas:
 * re-showing a photo, or rebuilding the display, doesn't reprocess it. Besides the results on
 * screen and the selected photo's, RETAINED_RESULTS more are kept, least recently used evicted.
 * A result replaced by a new run or discarded with its file is disposed once nothing shows it.
 */
import { useAppStore } from '@/app/store';
import { processRaw, type PipelineContext, type ProcessedImage } from '@/pipeline';
import { processingKey } from '@/app/store/photo';
import { getHost } from './host';
import { matchLensFor } from './library';

/** Processed results kept for photos that are neither on screen nor selected. */
export const RETAINED_RESULTS = 1;

interface Entry {
  image: ProcessedImage;
  pins: number;
  lastUsed: number;
  /** Replaced or discarded: dispose when the last pin goes. */
  stale: boolean;
}

let context: PipelineContext | null = null;
const results = new Map<string, Entry>();
let clock = 0;
let inFlight: string | null = null;
let runDiscarded = false;

export function setPipeline(ctx: PipelineContext): void {
  context = ctx;
}

export function getPipeline(): PipelineContext {
  if (!context) throw new Error('pipeline not initialised');
  return context;
}

export function isProcessing(): boolean {
  return inFlight !== null;
}

export async function processFile(fileId: string): Promise<void> {
  if (inFlight) return;
  const store = useAppStore.getState();
  const entry = store.files.find((f) => f.id === fileId);
  if (!entry) return;

  const requestedKey = processingKey(entry, store);
  const method = entry.edit.demosaicMethod ?? store.demosaicMethod;
  const modelSize = store.modelSize;
  const model = entry.edit.model;
  runDiscarded = false;
  inFlight = fileId;
  store.setProcessingFileId(fileId);
  store.updateFileStatus(fileId, 'processing');

  try {
    const bytes = await getHost().library.readRaw(fileId);

    const image = await processRaw(bytes, { method, modelSize, model }, getPipeline());

    if (runDiscarded || !useAppStore.getState().files.some((f) => f.id === fileId)) {
      image.dispose();
      return;
    }
    // A method/model change during decode must not publish an obsolete result.
    const current = useAppStore.getState();
    const latest = current.files.find((f) => f.id === fileId)!;
    if (processingKey(latest, current) !== requestedKey) {
      image.dispose();
      current.updateFileStatus(fileId, 'queued');
      return;
    }
    publish(fileId, image);
    useAppStore.setState((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const actualModel = image.meta.metadata.modelIdentity ?? null;
        const canRecord = f.editing !== 'view-only';
        const edit = canRecord
          ? {
              ...f.edit,
              demosaicMethod: f.edit.demosaicMethod ?? method,
              model:
                method !== 'neural-net'
                  ? null
                  : !f.edit.model || f.modelNeedsResolution || f.editRevision > entry.editRevision
                    ? actualModel
                    : f.edit.model,
            }
          : f.edit;
        const updated = {
          ...f,
          edit,
          actualModel,
          modelNeedsResolution: false,
          cfaType: image.meta.metadata.cfaType ?? f.cfaType,
          modelNote:
            edit.model?.sha256 === actualModel?.sha256 ? null : (image.meta.metadata.modelNote ?? null),
        };
        return { ...updated, processedKey: processingKey(updated, state) };
      }),
    }));
    useAppStore.getState().setFileResult(fileId, image.meta, method);
    matchLensFor(fileId);
  } catch (e) {
    discardResult(fileId); // also release a published-but-unclaimed result if publication throws
    const detail = e instanceof Error ? e.message : typeof e === 'string' ? e : String(e);
    const msg =
      detail.trim() ||
      'This file could not be opened. It may be corrupt, or the camera or format may be unsupported by this build.';
    useAppStore.setState((state) => ({
      files: state.files.map((f) => (f.id === fileId ? { ...f, processedKey: requestedKey } : f)),
    }));
    useAppStore.getState().updateFileStatus(fileId, 'error', msg);
    console.error(e);
  } finally {
    inFlight = null;
    useAppStore.getState().setProcessingFileId(null);
  }
}

function retire(entry: Entry): void {
  entry.stale = true;
  if (entry.pins === 0) entry.image.dispose();
}

function publish(fileId: string, image: ProcessedImage): void {
  const previous = results.get(fileId);
  if (previous) retire(previous);
  results.set(fileId, { image, pins: 0, lastUsed: ++clock, stale: false });
  trim();
}

/** Evict least recently used results beyond RETAINED_RESULTS, never one on screen or selected. */
function trim(): void {
  const selected = useAppStore.getState().selectedFileId;
  const evictable = [...results.entries()]
    .filter(([id, entry]) => entry.pins === 0 && id !== selected)
    .sort((a, b) => a[1].lastUsed - b[1].lastUsed);
  for (let i = 0; i < evictable.length - RETAINED_RESULTS; i++) {
    const [id, entry] = evictable[i];
    results.delete(id);
    retire(entry);
  }
}

/** The current result for a file, without borrowing it. */
export function getResult(fileId: string): ProcessedImage | null {
  return results.get(fileId)?.image ?? null;
}

/**
 * Borrow a file's result for display. The caller must call `release` when it stops showing it;
 * until then the image stays alive even if it is replaced or discarded.
 */
export function acquireResult(fileId: string): { image: ProcessedImage; release: () => void } | null {
  const entry = results.get(fileId);
  if (!entry) return null;
  entry.pins++;
  entry.lastUsed = ++clock;
  let released = false;
  return {
    image: entry.image,
    release: () => {
      if (released) return;
      released = true;
      entry.pins--;
      entry.lastUsed = ++clock;
      if (entry.stale) {
        if (entry.pins === 0) entry.image.dispose();
      } else {
        trim();
      }
    },
  };
}

/** Drop a file's result (disposed once nothing shows it) and invalidate an in-flight completion. */
export function discardResult(fileId: string): void {
  if (inFlight === fileId) runDiscarded = true;
  const entry = results.get(fileId);
  if (!entry) return;
  results.delete(fileId);
  retire(entry);
}
