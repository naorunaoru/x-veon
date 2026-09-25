/**
 * Owns the pipeline context and every processed image that has not been displayed yet.
 * The canvas takes a result to display it (ownership passes with it); a result that is
 * replaced by a new run or discarded with its file is disposed here.
 */
import { useAppStore } from '@/app/store';
import { processRaw, type PipelineContext, type ProcessedImage } from '@/pipeline';
import { readRaw } from '@/app/storage/opfs-storage';
import { matchLensFor } from './library';

let context: PipelineContext | null = null;
const results = new Map<string, ProcessedImage>();
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

  // Preserve the original single-slot bound, including across different files.
  for (const id of results.keys()) discardResult(id);
  runDiscarded = false;
  inFlight = fileId;
  store.setProcessingFileId(fileId);
  store.updateFileStatus(fileId, 'processing');

  try {
    // RAW bytes: the File object for fresh drops, OPFS for restored sessions
    let bytes: ArrayBuffer;
    if (entry.file) {
      bytes = await entry.file.arrayBuffer();
    } else {
      const raw = await readRaw(fileId);
      if (!raw) throw new Error('RAW file not found in storage. Please re-add this file.');
      bytes = raw;
    }

    const { demosaicMethod: method, modelSize } = useAppStore.getState();
    const ctx = getPipeline();
    // Run with the size the user chose; results are labelled with the size actually loaded.
    if (method === 'neural-net') await ctx.models.switchSize(modelSize);
    const image = await processRaw(bytes, { method, modelSize }, ctx);

    if (runDiscarded || !useAppStore.getState().files.some((f) => f.id === fileId)) {
      image.dispose();
      return;
    }
    results.set(fileId, image);
    useAppStore.getState().setFileResult(fileId, image.meta, method);
    matchLensFor(fileId);
  } catch (e) {
    discardResult(fileId); // also release a published-but-unclaimed result if publication throws
    const detail = e instanceof Error ? e.message : typeof e === 'string' ? e : String(e);
    const msg = detail.trim() || 'This file could not be opened. It may be corrupt, or the camera or format may be unsupported by this build.';
    useAppStore.getState().updateFileStatus(fileId, 'error', msg);
    console.error(e);
  } finally {
    inFlight = null;
    useAppStore.getState().setProcessingFileId(null);
  }
}

/** The undisplayed result for a file, without taking it. */
export function getResult(fileId: string): ProcessedImage | null {
  return results.get(fileId) ?? null;
}

/** Transfer ownership of a file's result to the caller (who must dispose it). */
export function takeResult(fileId: string): ProcessedImage | null {
  const image = results.get(fileId) ?? null;
  results.delete(fileId);
  return image;
}

/** Dispose an undisplayed result and invalidate an in-flight completion for this file. */
export function discardResult(fileId: string): void {
  if (inFlight === fileId) runDiscarded = true;
  results.get(fileId)?.dispose();
  results.delete(fileId);
}
