/**
 * One export batch at a time: today's single-photo export job run for a list of photos, in order,
 * with at most two jobs past queued (the two plane slots). A photo without a current result is
 * processed on demand, behind the photo the user is looking at.
 */
import { create } from 'zustand';
import { useAppStore, type AppState } from '@/app/store';
import { needsProcessing } from '@/app/store/photo';
import { exportFormatInfo } from '@/lib/catalog';
import type { ExportFormat } from '@/lib/types';
import type { ExportTarget, OnExists, SaveTo } from '@/host';
import { enqueueExport, type ExportJobHandle } from './export';
import { hasResult, processForExport } from './processing';
import { getHost } from './host';

export type BatchJobState = 'queued' | 'processing' | 'rendering' | 'encoding' | 'written' | 'skipped' | 'failed' | 'cancelled';
export interface BatchJob { id: string; fileId: string; name: string; state: BatchJobState; error: string | null }
export interface ExportBatch {
  id: string; jobs: BatchJob[]; target: ExportTarget | null;
  settings: { format: ExportFormat; quality: number; saveTo: SaveTo; onExists: OnExists };
  status: 'resolving' | 'running' | 'done' | 'failed';
  /** Closed to appends by cancelQueuedExports (a folder switch or clear): its running jobs finish, new exports start a fresh batch. */
  closed: boolean;
}
export const useExportBatch = create<{ batch: ExportBatch | null }>(() => ({ batch: null }));

const MAX_ACTIVE = 2;
/** Jobs past queued, by job id. */
const active = new Set<string>();
const handles = new Map<string, ExportJobHandle>();
let dismissTimer: ReturnType<typeof setTimeout> | undefined;

const pending = (job: BatchJob) => job.state === 'queued' || job.state === 'processing' || job.state === 'rendering' || job.state === 'encoding';
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const currentJob = (jobId: string) => useExportBatch.getState().batch?.jobs.find((job) => job.id === jobId);

function updateBatch(change: (batch: ExportBatch) => Partial<ExportBatch>): void {
  useExportBatch.setState(({ batch }) => batch ? { batch: { ...batch, ...change(batch) } } : {});
}
function patchJobs(patch: (job: BatchJob) => Partial<BatchJob> | null): void {
  updateBatch((batch) => ({ jobs: batch.jobs.map((job) => { const next = patch(job); return next ? { ...job, ...next } : job; }) }));
}
/** Moves a pending job on; a job already ended (cancelled meanwhile) ignores late news. */
function advance(jobId: string, next: Pick<BatchJob, 'state'> & Partial<BatchJob>): void {
  const job = currentJob(jobId);
  if (!job || !pending(job)) return;
  patchJobs((other) => other.id === jobId ? next : null);
  if (pending({ ...job, ...next })) return;
  active.delete(jobId);
  handles.delete(jobId);
  pump();
}
function cancelJobs(which: (job: BatchJob) => boolean): void {
  const batch = useExportBatch.getState().batch;
  if (!batch) return;
  const ids = new Set(batch.jobs.filter((job) => pending(job) && which(job)).map((job) => job.id));
  for (const id of ids) {
    handles.get(id)?.cancel();
    handles.delete(id);
    active.delete(id);
  }
  patchJobs((job) => ids.has(job.id) ? { state: 'cancelled' } : null);
  pump();
}

async function runJob(jobId: string): Promise<void> {
  const { batch } = useExportBatch.getState();
  const job = currentJob(jobId);
  if (!batch || !job) return;
  const { fileId } = job;
  try {
    const store = useAppStore.getState();
    const file = store.files.find((f) => f.id === fileId);
    if (!file) throw new Error('no longer in the library');
    if (!hasResult(fileId) || needsProcessing(file, store)) {
      advance(jobId, { state: 'processing' });
      await processForExport(fileId);
      if (currentJob(jobId)?.state !== 'processing') return;
      if (!hasResult(fileId)) throw new Error(useAppStore.getState().files.find((f) => f.id === fileId)?.error ?? 'Processing failed');
    }
    const { format, quality, onExists } = batch.settings;
    const handle = enqueueExport(fileId, {
      format, quality, onExists, target: batch.target, name: job.name,
      observer: { state: (state) => { if (state === 'rendering' || state === 'encoding') advance(jobId, { state }); } },
    });
    handles.set(jobId, handle);
    const result = await handle.promise;
    advance(jobId, { state: result === null ? 'cancelled' : result.outcome });
  } catch (error) {
    advance(jobId, { state: 'failed', error: message(error) });
  }
}

/** Starts queued jobs up to the limit, then ends the batch once nothing is pending. */
function pump(): void {
  for (;;) {
    const batch = useExportBatch.getState().batch;
    if (!batch) return;
    if (batch.status === 'running' && active.size < MAX_ACTIVE) {
      const next = batch.jobs.find((job) => job.state === 'queued' && !active.has(job.id));
      if (next) {
        active.add(next.id);
        void runJob(next.id);
        continue;
      }
    }
    if ((batch.status !== 'running' && batch.status !== 'resolving') || batch.jobs.some(pending)) return;
    const summary = batchSummary(batch);
    if (summary.total === 0) { dismissBatch(); return; }
    // Resolving with no pending jobs and some failures cannot happen: a failed resolve ends the batch itself.
    const status = summary.failed > 0 ? 'failed' : 'done';
    useExportBatch.setState({ batch: { ...batch, status } });
    if (status === 'done') dismissTimer = setTimeout(dismissBatch, 10_000);
    return;
  }
}

async function resolveTarget(batchId: string, saveTo: SaveTo, fileIds: string[]): Promise<void> {
  const ours = () => useExportBatch.getState().batch?.id === batchId;
  let target: ExportTarget | null;
  try {
    const exporter = getHost().exporter;
    const availability = await exporter.status();
    if (!availability.available) throw new Error(availability.reason);
    if (!ours()) return;
    target = await exporter.resolveTarget(saveTo, fileIds);
  } catch (error) {
    if (!ours()) return;
    const reason = message(error);
    patchJobs((job) => pending(job) ? { state: 'failed', error: reason } : null);
    updateBatch(() => ({ status: 'failed' }));
    return;
  }
  if (!ours()) return;
  updateBatch(() => ({ target, status: 'running' }));
  pump();
}

/** [] → no-op; appends to a resolving/running batch (skipping fileIds already pending), else starts a new one (dismissing an ended one). */
export function startExport(fileIds: string[]): void {
  if (fileIds.length === 0) return;
  const unique = [...new Set(fileIds)];
  const files = useAppStore.getState().files;
  const makeJob = (fileId: string, format: ExportFormat): BatchJob => ({
    id: crypto.randomUUID(), fileId, state: 'queued', error: null,
    name: `${files.find((f) => f.id === fileId)?.name ?? fileId}.${exportFormatInfo(format).ext}`,
  });
  const current = useExportBatch.getState().batch;
  if (current && !current.closed && (current.status === 'resolving' || current.status === 'running')) {
    const waiting = new Set(current.jobs.filter(pending).map((job) => job.fileId));
    const added = unique.filter((id) => !waiting.has(id)).map((id) => makeJob(id, current.settings.format));
    if (added.length === 0) return;
    useExportBatch.setState({ batch: { ...current, jobs: [...current.jobs, ...added] } });
    pump();
    return;
  }
  // An ended or closed batch is replaced; a closed batch's running jobs finish unseen.
  release();
  const { exportFormat: format, exportQuality: quality, exportSaveTo: saveTo, exportOnExists: onExists } = useAppStore.getState();
  const batch: ExportBatch = {
    id: crypto.randomUUID(), jobs: unique.map((id) => makeJob(id, format)), target: null,
    settings: { format, quality, saveTo, onExists }, status: 'resolving', closed: false,
  };
  useExportBatch.setState({ batch });
  void resolveTarget(batch.id, saveTo, unique);
}

/** queued/processing → cancelled; active handles cancelled; then settles. */
export function cancelBatch(): void {
  cancelJobs(() => true);
}

/** Only queued/processing jobs → cancelled; running ones finish. */
export function cancelQueuedExports(): void {
  updateBatch(() => ({ closed: true }));
  cancelJobs((job) => job.state === 'queued' || job.state === 'processing');
}

export function dismissBatch(): void {
  for (const handle of handles.values()) handle.cancel();
  release();
}
/** Forgets the batch; late settlements of its jobs no longer find them and are ignored. */
function release(): void {
  clearTimeout(dismissTimer);
  dismissTimer = undefined;
  handles.clear();
  active.clear();
  useExportBatch.setState({ batch: null });
}

/** Removes the row and startExport([fileId]). */
export function retryJob(jobId: string): void {
  const job = currentJob(jobId);
  if (!job || pending(job)) return;
  updateBatch((batch) => ({ jobs: batch.jobs.filter((other) => other.id !== jobId) }));
  startExport([job.fileId]);
}

/** selectedIds, else [currentId], else []. */
export function exportTargets(state: Pick<AppState, 'selectedIds' | 'currentId'>): string[] {
  if (state.selectedIds.length > 0) return state.selectedIds;
  return state.currentId ? [state.currentId] : [];
}

export type ExportMark = 'queued' | 'running' | 'done' | 'failed';

/** The latest job for the file. */
export function exportMark(batch: ExportBatch | null, fileId: string): ExportMark | null {
  return markOf(batch?.jobs.filter((other) => other.fileId === fileId).at(-1)?.state);
}

/** exportMark for every file in one pass: the latest job per file wins; files without a mark are absent. */
export function exportMarks(batch: ExportBatch | null): Map<string, ExportMark> {
  const latest = new Map<string, BatchJob['state']>();
  for (const job of batch?.jobs ?? []) latest.set(job.fileId, job.state);
  const marks = new Map<string, ExportMark>();
  for (const [fileId, state] of latest) {
    const mark = markOf(state);
    if (mark) marks.set(fileId, mark);
  }
  return marks;
}

function markOf(state: BatchJob['state'] | undefined): ExportMark | null {
  switch (state) {
    case 'queued': return 'queued';
    case 'processing': case 'rendering': case 'encoding': return 'running';
    case 'written': case 'skipped': return 'done';
    case 'failed': return 'failed';
    default: return null;
  }
}

/** total = non-cancelled jobs; completed = written+skipped+failed; done = written+skipped. */
export function batchSummary(batch: ExportBatch): { total: number; completed: number; done: number; failed: number } {
  let total = 0, done = 0, failed = 0;
  for (const job of batch.jobs) {
    if (job.state === 'cancelled') continue;
    total++;
    if (job.state === 'written' || job.state === 'skipped') done++;
    else if (job.state === 'failed') failed++;
  }
  return { total, completed: done + failed, done, failed };
}
