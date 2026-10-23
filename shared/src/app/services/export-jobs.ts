import { create } from 'zustand';
import { useAppStore } from '@/app/store';
import { exportFormatInfo } from '@/lib/catalog';
import type { ExportResult, ExportTarget } from '@/host';
import { enqueueExport, type ExportJobHandle, type ExportState } from './export';
import { getHost } from './host';

export interface ExportJobView {
  id: string;
  fileId: string;
  label: string;
  state: ExportState;
  error: string | null;
  result: ExportResult | null;
  /** The resolved target, kept for Reveal. */
  destination: ExportTarget | null;
}
export const useExportJobs = create<{ jobs: ExportJobView[] }>(() => ({ jobs: [] }));
const handles = new Map<string, ExportJobHandle>();
const timers = new Map<string, ReturnType<typeof setTimeout>>();
function update(id: string, patch: Partial<ExportJobView>) {
  useExportJobs.setState((state) => ({ jobs: state.jobs.map((job) => job.id === id ? { ...job, ...patch } : job) }));
}
export function dismissExport(id: string): void {
  clearTimeout(timers.get(id));
  timers.delete(id);
  handles.delete(id);
  useExportJobs.setState((state) => ({ jobs: state.jobs.filter((job) => job.id !== id) }));
}
export function cancelExport(id: string): void {
  handles.get(id)?.cancel();
  dismissExport(id);
}
export function startExport(fileId: string): string {
  const id = crypto.randomUUID();
  const snapshot = useAppStore.getState();
  const file = snapshot.files.find((photo) => photo.id === fileId);
  const { exportFormat: format, exportQuality: quality, exportSaveTo, exportOnExists: onExists } = snapshot;
  const label = `${file?.name ?? fileId}.${exportFormatInfo(format).ext}`;
  const listed = () => useExportJobs.getState().jobs.some((job) => job.id === id);
  useExportJobs.setState((state) => ({ jobs: [...state.jobs, { id, fileId, label, state: 'queued', error: null, result: null, destination: null }] }));
  let handle: ExportJobHandle | undefined;
  void getHost().exporter.resolveTarget(exportSaveTo, [fileId]).then((target) => {
    // Cancelling while the target resolves removes the job; nothing is enqueued.
    if (!listed()) return null;
    update(id, { destination: target });
    handle = enqueueExport(fileId, { format, quality, target, name: label, onExists, observer: { state: (state) => update(id, { state }) } });
    handles.set(id, handle);
    return handle.promise;
  }).then((result) => {
    handles.delete(id);
    if (!handle || handle.state === 'cancelled') { dismissExport(id); return; }
    update(id, { state: 'done', result, ...(result?.name ? { label: result.name } : {}) });
    if (listed()) {
      timers.set(id, setTimeout(() => dismissExport(id), 10_000));
    }
  }, (error: unknown) => {
    handles.delete(id);
    update(id, { state: 'failed', error: error instanceof Error ? error.message : String(error) });
  });
  return id;
}
