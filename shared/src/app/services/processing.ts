/**
 * Owns the pipeline context and the processed images. The canvas borrows a result to display
 * it (acquireResult) and gives it back when it stops (release), so a result outlives its canvas:
 * re-showing a photo, or rebuilding the display, doesn't reprocess it. Besides the results on
 * screen and the selected photo's, RETAINED_RESULTS more are kept, least recently used evicted.
 * A result replaced by a new run or discarded with its file is disposed once nothing shows it.
 */
import { useAppStore } from '@/app/store';
import { processRaw, type PipelineContext, type ProcessedImage } from '@/pipeline';
import { needsProcessing, processingKey } from '@/app/store/photo';
import { effectiveMethod } from '@/app/photo-edit';
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

    const image = await processRaw(bytes, { method, modelSize, model, ...(entry.edit.demosaicMethod === null ? { resolveDefault: true } : {}) }, getPipeline());

    if (runDiscarded || !useAppStore.getState().files.some((f) => f.id === fileId)) {
      image.dispose();
      return;
    }
    // A method/model change during decode must not publish an obsolete result.
    const current = useAppStore.getState();
    const latest = current.files.find((f) => f.id === fileId)!;
    const decodedMethod = effectiveMethod(latest.edit, current.demosaicMethod, image.meta.metadata.cfaType ?? latest.cfaType);
    if (processingKey(latest, current) !== requestedKey || decodedMethod !== image.method) {
      image.dispose();
      current.updateFileStatus(fileId, 'queued');
      return;
    }
    publish(fileId, image);
    useAppStore.setState((state) => ({
      files: state.files.map((f) => {
        if (f.id !== fileId) return f;
        const actualModel = image.meta.metadata.modelIdentity ?? null;
        const completingEdit = f.editing !== 'view-only' && f.modelNeedsResolution;
        const edit = completingEdit
          ? { ...f.edit, demosaicMethod: f.edit.demosaicMethod ?? image.method, model: image.method === 'neural-net' ? (actualModel ?? f.edit.model) : null }
          : f.edit;
        const updated = {
          ...f,
          edit,
          actualModel,
          modelNeedsResolution: f.modelNeedsResolution && image.method === 'neural-net' && !actualModel,
          cfaType: image.meta.metadata.cfaType ?? f.cfaType,
          modelNote:
            edit.model?.sha256 === actualModel?.sha256 ? null : (image.meta.metadata.modelNote ?? null),
        };
        return { ...updated, processedKey: processingKey(updated, state) };
      }),
    }));
    useAppStore.getState().setFileResult(fileId, image.meta, image.method);
    matchLensFor(fileId);
  } catch (e) {
    // Folder/watch replacement invalidates failures as well as successful results.
    if (runDiscarded || !useAppStore.getState().files.some((f) => f.id === fileId)) return;
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
    pump();
  }
}

interface Waiter {
  fileId: string;
  resolve(): void;
  reject(error: Error): void;
}
/** Exports waiting for a photo's processing, served first come first served. */
const waiters: Waiter[] = [];

/** A published, non-stale result. */
export function hasResult(fileId: string): boolean {
  const entry = results.get(fileId);
  return !!entry && !entry.stale;
}

/**
 * Resolves once fileId's run has settled (result published or status 'error'); rejects with
 * 'no longer in the library' when the photo is gone. Waits for any run in flight and yields to
 * the current photo whenever it needsProcessing (and the store is initialized).
 */
export function processForExport(fileId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    waiters.push({ fileId, resolve, reject });
    pump();
  });
}

function settleWaiter(waiter: Waiter): void {
  if (useAppStore.getState().files.some((f) => f.id === waiter.fileId)) waiter.resolve();
  else waiter.reject(new Error('no longer in the library'));
}

/** Runs after every processFile settles and whenever a waiter is added. */
function pump(): void {
  const state = useAppStore.getState();
  for (const waiter of [...waiters]) {
    const file = state.files.find((f) => f.id === waiter.fileId);
    if (file && !(hasResult(waiter.fileId) && !needsProcessing(file, state))) continue;
    waiters.splice(waiters.indexOf(waiter), 1);
    settleWaiter(waiter);
  }
  if (inFlight || waiters.length === 0) return;
  // The auto-process effect runs the current photo; its settlement pumps again.
  const current = state.files.find((f) => f.id === state.currentId);
  if (current && state.initialized && current.status !== 'error' && needsProcessing(current, state)) return;
  const waiter = waiters[0];
  const run = processFile(waiter.fileId);
  if (inFlight !== waiter.fileId) return; // another run won the race: wait for the next settlement
  waiters.shift();
  const settle = () => {
    const now = useAppStore.getState();
    const file = now.files.find((f) => f.id === waiter.fileId);
    // A run superseded mid-flight (key changed) published nothing and reset the photo to queued: run it again.
    if (file && file.status !== 'error' && !hasResult(waiter.fileId) && needsProcessing(file, now)) {
      waiters.unshift(waiter);
      pump();
      return;
    }
    settleWaiter(waiter);
  };
  void run.then(settle, settle);
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
  const selected = useAppStore.getState().currentId;
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
