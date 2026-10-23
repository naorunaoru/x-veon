import { beforeEach, expect, it, vi } from 'vitest';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { useAppStore } from '@/app/store';
import { setHost } from './host';
import { startExport, cancelExport, dismissExport, useExportJobs } from './export-jobs';
const m = vi.hoisted(() => ({ readback: vi.fn(), dispose: vi.fn(), release: vi.fn(), setImage: vi.fn(), setLens: vi.fn() }));
vi.mock('@/renderer', () => ({
  createRenderer: vi.fn(async () => ({ readback: m.readback, setImage: m.setImage, setLens: m.setLens, dispose: m.dispose })),
}));
vi.mock('./processing', () => ({ acquireResult: () => ({ image: { gpu: {} }, release: m.release }) }));
let host: ReturnType<typeof fakeHost>;
beforeEach(() => {
  for (const job of useExportJobs.getState().jobs) cancelExport(job.id);
  vi.useRealTimers();
  vi.clearAllMocks();
  m.readback.mockResolvedValue(new Float32Array([1, 2, 3, 1]));
  host = fakeHost();
  setHost(host);
  const photo = fromLibraryPhoto(fakePhoto());
  photo.result = {
    exportData: {
      width: 1,
      height: 1,
      orientation: 'Rotate90',
      xyzToCam: null,
      wbCoeffs: new Float32Array(3),
      camToXyz: new Float32Array(12),
    },
    metadata: {} as never,
  };
  useAppStore.setState({ files: [photo], currentId: 'a', exportQuality: 95, exportFormat: 'tiff' });
});

it('adds a queued named job, resolves the target first, then keeps the result name and target', async () => {
  vi.mocked(host.exporter.encode).mockResolvedValue({ outcome: 'written', name: 'chosen.tif' });
  const id = startExport('a');
  expect(useExportJobs.getState().jobs[0]).toMatchObject({ id, label: 'a.tif', state: 'queued' });
  await vi.waitFor(() => expect(useExportJobs.getState().jobs[0]).toMatchObject({ label: 'chosen.tif', state: 'done', destination: { token: 'test' } }));
  expect(host.exporter.resolveTarget).toHaveBeenCalledWith({ mode: 'beside' }, ['a']);
  expect(vi.mocked(host.exporter.resolveTarget).mock.invocationCallOrder[0]).toBeLessThan(vi.mocked(host.exporter.status).mock.invocationCallOrder[0]);
  expect(host.exporter.encode).toHaveBeenCalledWith(expect.objectContaining({ format: 'tiff' }), { target: { token: 'test' }, name: 'a.tif', onExists: 'skip' });
});
it('fails the job when the target cannot be resolved', async () => {
  vi.mocked(host.exporter.resolveTarget).mockRejectedValue(new Error('no folder'));
  startExport('a');
  await vi.waitFor(() => expect(useExportJobs.getState().jobs[0]).toMatchObject({ state: 'failed', error: 'no folder' }));
  expect(host.exporter.status).not.toHaveBeenCalled();
});
it('never enqueues a job cancelled while its target resolves', async () => {
  let resolve!: () => void;
  vi.mocked(host.exporter.resolveTarget).mockImplementationOnce(() => new Promise((done) => { resolve = () => done({ token: 'test' }); }));
  const id = startExport('a');
  cancelExport(id);
  resolve();
  await new Promise((done) => setTimeout(done, 0));
  expect(host.exporter.status).not.toHaveBeenCalled();
  expect(useExportJobs.getState().jobs).toEqual([]);
});
it('keeps encode failures until dismissed', async () => {
  vi.mocked(host.exporter.encode).mockRejectedValue(new Error('encode broke'));
  const id = startExport('a');
  await vi.waitFor(() => expect(useExportJobs.getState().jobs[0]).toMatchObject({ state: 'failed', error: 'encode broke' }));
  dismissExport(id);
  expect(useExportJobs.getState().jobs).toEqual([]);
});
it('aborts encoding and ignores late events after removing a cancelled job', async () => {
  let finish!: () => void;
  vi.mocked(host.exporter.encode).mockImplementation((job) => new Promise((resolve) => {
    finish = () => resolve({ outcome: 'written', name: 'late.tif' });
    expect(job.signal!.aborted).toBe(false);
  }));
  const id = startExport('a');
  await vi.waitFor(() => expect(host.exporter.encode).toHaveBeenCalledOnce());
  const signal = vi.mocked(host.exporter.encode).mock.calls[0][0].signal;
  cancelExport(id);
  expect(signal!.aborted).toBe(true);
  finish();
  await Promise.resolve(); await Promise.resolve();
  expect(useExportJobs.getState().jobs).toEqual([]);
});
it('expires done jobs after ten seconds and keeps failed jobs', async () => {
  vi.useFakeTimers();
  const id = startExport('a');
  await vi.advanceTimersByTimeAsync(0);
  expect(useExportJobs.getState().jobs[0].state).toBe('done');
  await vi.advanceTimersByTimeAsync(9999);
  expect(useExportJobs.getState().jobs[0].id).toBe(id);
  await vi.advanceTimersByTimeAsync(1);
  expect(useExportJobs.getState().jobs).toEqual([]);
  vi.mocked(host.exporter.encode).mockRejectedValue(new Error('broken'));
  startExport('a');
  await vi.advanceTimersByTimeAsync(20000);
  expect(useExportJobs.getState().jobs[0].state).toBe('failed');
});
it('runs exports for different photos concurrently', async () => {
  useAppStore.setState((s) => ({ files: [...s.files, { ...s.files[0], id: 'b', name: 'b' }] }));
  let finish!: () => void;
  vi.mocked(host.exporter.encode).mockImplementationOnce(() => new Promise((resolve) => { finish = () => resolve({ outcome: 'written' }); }));
  startExport('a'); startExport('b');
  await vi.waitFor(() => expect(useExportJobs.getState().jobs.map((job) => job.state)).toEqual(['encoding', 'done']));
  finish();
  await vi.waitFor(() => expect(useExportJobs.getState().jobs.every((job) => job.state === 'done')).toBe(true));
});
