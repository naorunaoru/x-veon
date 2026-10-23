import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto, processingKey } from '@/app/store/photo';
import { useAppStore } from '@/app/store';
import type { ExportResult } from '@/host';
import { setHost } from './host';
import {
  batchSummary, cancelBatch, cancelQueuedExports, dismissBatch, exportMark, exportTargets, retryJob, startExport, useExportBatch,
  type BatchJob, type ExportBatch,
} from './export-batch';
const m = vi.hoisted(() => ({
  readback: vi.fn(), dispose: vi.fn(), release: vi.fn(), setImage: vi.fn(), setLens: vi.fn(),
  hasResult: vi.fn(), processForExport: vi.fn(),
}));
vi.mock('@/renderer', () => ({
  createRenderer: vi.fn(async () => ({ readback: m.readback, setImage: m.setImage, setLens: m.setLens, dispose: m.dispose })),
}));
vi.mock('./processing', () => ({
  acquireResult: () => ({ image: { gpu: {} }, release: m.release }),
  hasResult: m.hasResult,
  processForExport: m.processForExport,
}));
let host: ReturnType<typeof fakeHost>;
const batch = () => useExportBatch.getState().batch!;
const states = () => batch().jobs.map((job) => `${job.fileId}:${job.state}`);
const encodedNames = () => vi.mocked(host.exporter.encode).mock.calls.map((call) => call[1].name);
const deferred = <T,>() => {
  let resolve!: (value: T) => void, reject!: (error: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
};
function processed(id: string) {
  const photo = fromLibraryPhoto(fakePhoto(id));
  photo.status = 'done';
  photo.result = {
    exportData: { width: 1, height: 1, orientation: 'Normal', xyzToCam: null, wbCoeffs: new Float32Array(3), camToXyz: new Float32Array(12) },
    metadata: {} as never,
  };
  photo.processedKey = processingKey(photo, useAppStore.getState());
  return photo;
}
beforeEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
  m.readback.mockResolvedValue(new Float32Array([1, 2, 3, 1]));
  m.hasResult.mockReturnValue(true);
  m.processForExport.mockResolvedValue(undefined);
  host = fakeHost();
  setHost(host);
  useAppStore.setState({ demosaicMethod: 'dht', modelSize: 'S', exportFormat: 'tiff', exportQuality: 95, exportSaveTo: { mode: 'beside' }, exportOnExists: 'skip' });
  useAppStore.setState({ files: [processed('a'), processed('b'), processed('c')], currentId: 'a', selectedIds: [] });
});
afterEach(() => {
  cancelBatch();
  dismissBatch();
  vi.useRealTimers();
});

it('exports the selection in order and ends done', async () => {
  startExport(['b', 'a']);
  expect(batch()).toMatchObject({ status: 'resolving', settings: { format: 'tiff', quality: 95, saveTo: { mode: 'beside' }, onExists: 'skip' } });
  expect(batch().jobs.map((job) => job.name)).toEqual(['b.tif', 'a.tif']);
  await vi.waitFor(() => expect(batch().status).toBe('done'));
  expect(host.exporter.resolveTarget).toHaveBeenCalledOnce();
  expect(host.exporter.resolveTarget).toHaveBeenCalledWith({ mode: 'beside' }, ['b', 'a']);
  expect(batch().target).toEqual({ token: 'test' });
  expect(encodedNames()).toEqual(['b.tif', 'a.tif']);
  expect(host.exporter.encode).toHaveBeenCalledWith(expect.objectContaining({ format: 'tiff', quality: 95 }), { target: { token: 'test' }, name: 'b.tif', onExists: 'skip' });
  expect(states()).toEqual(['b:written', 'a:written']);
  expect(batchSummary(batch())).toEqual({ total: 2, completed: 2, done: 2, failed: 0 });
});

it('keeps at most two jobs past queued', async () => {
  const gates = [deferred<ExportResult>(), deferred<ExportResult>()];
  vi.mocked(host.exporter.encode).mockImplementationOnce(() => gates[0].promise).mockImplementationOnce(() => gates[1].promise);
  startExport(['a', 'b', 'c']);
  await vi.waitFor(() => expect(states()).toEqual(['a:encoding', 'b:encoding', 'c:queued']));
  gates[0].resolve({ outcome: 'written' });
  await vi.waitFor(() => expect(states()).toEqual(['a:written', 'b:encoding', 'c:written']));
  gates[1].resolve({ outcome: 'written' });
  await vi.waitFor(() => expect(batch().status).toBe('done'));
});

it('processes a photo without a result first and reports processing', async () => {
  let ready = false;
  m.hasResult.mockImplementation((id: string) => id !== 'a' || ready);
  const gate = deferred<void>();
  m.processForExport.mockImplementation(() => gate.promise.then(() => { ready = true; }));
  startExport(['a']);
  await vi.waitFor(() => expect(states()).toEqual(['a:processing']));
  expect(host.exporter.encode).not.toHaveBeenCalled();
  gate.resolve();
  await vi.waitFor(() => expect(batch().status).toBe('done'));
  expect(m.processForExport).toHaveBeenCalledOnce();
  expect(m.processForExport).toHaveBeenCalledWith('a');
  expect(states()).toEqual(['a:written']);
});

it('fails a job whose photo still has no result after processing, with the processing error', async () => {
  m.hasResult.mockImplementation((id: string) => id !== 'a');
  m.processForExport.mockImplementation(async () => {
    useAppStore.setState((state) => ({ files: state.files.map((f) => f.id === 'a' ? { ...f, status: 'error' as const, error: 'broken RAW' } : f) }));
  });
  m.processForExport.mockImplementationOnce(async () => { throw new Error('pipeline lost'); });
  startExport(['b', 'a']);
  await vi.waitFor(() => expect(batch().status).toBe('failed'));
  expect(batch().jobs.map((job) => [job.state, job.error])).toEqual([['written', null], ['failed', 'pipeline lost']]);
  retryJob(batch().jobs[1].id);
  await vi.waitFor(() => expect(batch().status).toBe('failed'));
  expect(batch().jobs.map((job) => [job.fileId, job.state, job.error])).toEqual([['a', 'failed', 'broken RAW']]);
});

it('appends to a running batch and does not queue a pending photo twice', async () => {
  const gate = deferred<ExportResult>();
  vi.mocked(host.exporter.encode).mockImplementationOnce(() => gate.promise);
  useAppStore.setState({ exportFormat: 'tiff' });
  startExport(['a']);
  startExport(['a', 'b']); // appended while resolving
  expect(batch().jobs.map((job) => job.fileId)).toEqual(['a', 'b']);
  await vi.waitFor(() => expect(states()).toEqual(['a:encoding', 'b:written']));
  useAppStore.setState({ exportFormat: 'avif' });
  startExport(['a', 'c', 'c']); // appended while running: the batch keeps its settings
  expect(batch().jobs.map((job) => job.fileId)).toEqual(['a', 'b', 'c']);
  gate.resolve({ outcome: 'written' });
  await vi.waitFor(() => expect(batch().status).toBe('done'));
  expect(encodedNames()).toEqual(['a.tif', 'b.tif', 'c.tif']);
  expect(host.exporter.resolveTarget).toHaveBeenCalledOnce();
  startExport([]);
  expect(batch().jobs).toHaveLength(3);
});

it.each(['unavailable', 'unresolved'] as const)('fails every job when export is unavailable or the target cannot be resolved: %s', async (cause) => {
  if (cause === 'unavailable') vi.mocked(host.exporter.status).mockResolvedValue({ available: false, reason: 'addon unavailable' });
  else vi.mocked(host.exporter.resolveTarget).mockRejectedValue(new Error('Photos must be in one folder'));
  const reason = cause === 'unavailable' ? 'addon unavailable' : 'Photos must be in one folder';
  startExport(['a', 'b']);
  await vi.waitFor(() => expect(batch().status).toBe('failed'));
  expect(batch().jobs.map((job) => [job.state, job.error])).toEqual([['failed', reason], ['failed', reason]]);
  expect(host.exporter.encode).not.toHaveBeenCalled();
  if (cause === 'unavailable') expect(host.exporter.resolveTarget).not.toHaveBeenCalled();
});

it('fails a queued job whose photo was removed', async () => {
  startExport(['a', 'b']);
  useAppStore.setState((state) => ({ files: state.files.filter((f) => f.id !== 'b') }));
  await vi.waitFor(() => expect(batch().status).toBe('failed'));
  expect(batch().jobs.map((job) => [job.fileId, job.state, job.error])).toEqual([['a', 'written', null], ['b', 'failed', 'no longer in the library']]);
  expect(batchSummary(batch())).toEqual({ total: 2, completed: 2, done: 1, failed: 1 });
});

it('cancelQueuedExports leaves the running job alone', async () => {
  const gate = deferred<ExportResult>();
  vi.mocked(host.exporter.encode).mockImplementationOnce(() => gate.promise);
  m.hasResult.mockImplementation((id: string) => id !== 'b');
  m.processForExport.mockImplementation(() => new Promise(() => {}));
  startExport(['a', 'b', 'c']);
  await vi.waitFor(() => expect(states()).toEqual(['a:encoding', 'b:processing', 'c:queued']));
  cancelQueuedExports();
  expect(states()).toEqual(['a:encoding', 'b:cancelled', 'c:cancelled']);
  expect(batch().status).toBe('running');
  gate.resolve({ outcome: 'written' });
  await vi.waitFor(() => expect(batch().status).toBe('done'));
  expect(states()).toEqual(['a:written', 'b:cancelled', 'c:cancelled']);
  expect(batchSummary(batch())).toEqual({ total: 1, completed: 1, done: 1, failed: 0 });
  expect(encodedNames()).toEqual(['a.tif']);
});

it('cancelBatch cancels the running job and drops the rest; an all-cancelled batch dismisses', async () => {
  const gate = deferred<ExportResult>();
  vi.mocked(host.exporter.encode).mockImplementation(() => gate.promise);
  m.hasResult.mockImplementation((id: string) => id !== 'b');
  m.processForExport.mockImplementation(() => new Promise(() => {}));
  startExport(['a', 'b', 'c']);
  await vi.waitFor(() => expect(states()).toEqual(['a:encoding', 'b:processing', 'c:queued']));
  const signal = vi.mocked(host.exporter.encode).mock.calls[0][0].signal!;
  cancelBatch();
  expect(signal.aborted).toBe(true);
  expect(useExportBatch.getState().batch).toBeNull();
  gate.resolve({ outcome: 'written' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(useExportBatch.getState().batch).toBeNull();
  expect(m.release).toHaveBeenCalled();
});

it('cancelBatch while the target resolves never starts a job', async () => {
  const target = deferred<{ token: string }>();
  vi.mocked(host.exporter.resolveTarget).mockImplementationOnce(() => target.promise);
  startExport(['a']);
  await vi.waitFor(() => expect(host.exporter.resolveTarget).toHaveBeenCalled());
  cancelBatch();
  expect(useExportBatch.getState().batch).toBeNull();
  target.resolve({ token: 'late' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(useExportBatch.getState().batch).toBeNull();
  expect(m.readback).not.toHaveBeenCalled();
});

it('a done batch dismisses after ten seconds; a failed one stays', async () => {
  vi.useFakeTimers();
  startExport(['a']);
  await vi.advanceTimersByTimeAsync(0);
  expect(batch().status).toBe('done');
  await vi.advanceTimersByTimeAsync(9_999);
  expect(batch().status).toBe('done');
  await vi.advanceTimersByTimeAsync(1);
  expect(useExportBatch.getState().batch).toBeNull();
  vi.mocked(host.exporter.encode).mockRejectedValue(new Error('encode broke'));
  startExport(['a']);
  await vi.advanceTimersByTimeAsync(20_000);
  expect(batch()).toMatchObject({ status: 'failed', jobs: [{ state: 'failed', error: 'encode broke' }] });
  expect(vi.getTimerCount()).toBe(0);
});

it('a new batch clears the previous batch’s dismissal timer', async () => {
  vi.useFakeTimers();
  startExport(['a']);
  await vi.advanceTimersByTimeAsync(0);
  expect(batch().status).toBe('done');
  vi.mocked(host.exporter.encode).mockRejectedValue(new Error('encode broke'));
  startExport(['b']);
  expect(vi.getTimerCount()).toBe(0);
  await vi.advanceTimersByTimeAsync(20_000);
  expect(batch()).toMatchObject({ status: 'failed', jobs: [{ fileId: 'b' }] });
});

it('retryJob re-queues one photo with the current settings', async () => {
  vi.mocked(host.exporter.encode).mockRejectedValueOnce(new Error('disk full'));
  startExport(['a', 'b']);
  await vi.waitFor(() => expect(batch().status).toBe('failed'));
  const failed = batch().jobs.find((job) => job.state === 'failed')!;
  expect(failed).toMatchObject({ fileId: 'a', error: 'disk full' });
  useAppStore.setState({ exportFormat: 'avif' });
  const before = batch().id;
  retryJob(failed.id);
  expect(batch().id).not.toBe(before);
  expect(batch().jobs.map((job) => job.name)).toEqual(['a.avif']);
  expect(batch().settings.format).toBe('avif');
  await vi.waitFor(() => expect(batch().status).toBe('done'));
  expect(encodedNames().at(-1)).toBe('a.avif');
});

it('skipped counts as done and marks the cell done; failed marks it failed', () => {
  const job = (fileId: string, state: BatchJob['state']): BatchJob => ({ id: `${fileId}-${state}`, fileId, name: `${fileId}.tif`, state, error: null });
  const sample: ExportBatch = {
    id: 'x', target: null, status: 'running',
    settings: { format: 'tiff', quality: 95, saveTo: { mode: 'beside' }, onExists: 'skip' },
    jobs: [job('a', 'skipped'), job('b', 'failed'), job('c', 'queued'), job('d', 'encoding'), job('e', 'cancelled'), job('f', 'written'), job('g', 'processing'), job('h', 'rendering'),
      job('b', 'cancelled'), job('f', 'failed')],
  };
  expect(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'z'].map((id) => exportMark(sample, id)))
    .toEqual(['done', null, 'queued', 'running', null, 'failed', 'running', 'running', null]);
  expect(exportMark(null, 'a')).toBeNull();
  expect(batchSummary(sample)).toEqual({ total: 8, completed: 4, done: 2, failed: 2 });
});

it('exportTargets prefers the selection over the current photo', () => {
  expect(exportTargets({ selectedIds: ['c', 'a'], currentId: 'b' })).toEqual(['c', 'a']);
  expect(exportTargets({ selectedIds: [], currentId: 'b' })).toEqual(['b']);
  expect(exportTargets({ selectedIds: [], currentId: null })).toEqual([]);
});
