import { beforeEach, expect, it, vi } from 'vitest';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { useAppStore } from '@/app/store';
import { setHost } from './host';
import { enqueueExport } from './export';
const m = vi.hoisted(() => ({ readback: vi.fn(), dispose: vi.fn(), release: vi.fn(), setImage: vi.fn() }));
vi.mock('@/renderer', () => ({
  createRenderer: vi.fn(async () => ({ readback: m.readback, setImage: m.setImage, dispose: m.dispose })),
}));
vi.mock('./processing', () => ({ acquireResult: () => ({ image: { gpu: {} }, release: m.release }) }));
let host: ReturnType<typeof fakeHost>;
beforeEach(() => {
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
  useAppStore.setState({ files: [photo], selectedFileId: 'a', exportQuality: 95, exportFormat: 'jpeg-hdr' });
});
it('asks availability and destination before rendering, and cancellation does no GPU work', async () => {
  vi.mocked(host.exporter.chooseDestination).mockResolvedValue(null);
  const job = enqueueExport('a');
  await expect(job.promise).resolves.toBeNull();
  expect(m.readback).not.toHaveBeenCalled();
  expect(host.exporter.encode).not.toHaveBeenCalled();
  expect(job.state).toBe('cancelled');
  vi.mocked(host.exporter.status).mockResolvedValue({ available: false, reason: 'addon unavailable' });
  await expect(enqueueExport('a').promise).rejects.toThrow('addon unavailable');
});
it.each(['jpeg-hdr', 'avif', 'tiff'] as const)(
  'renders %s through the host with unchanged parameters',
  async (format) => {
    const job = enqueueExport('a', format, 88);
    await job.promise;
    expect(host.exporter.encode).toHaveBeenCalledWith(
      expect.objectContaining({ format, width: 1, height: 1, orientation: 'Rotate90', quality: 88 }),
      { token: 'test' },
    );
    const calls = m.readback.mock.calls;
    expect(calls.map((c) => c[2])).toEqual(
      format === 'jpeg-hdr' ? ['rec709', 'rec2020'] : [format === 'avif' ? 'rec2020' : 'rec709'],
    );
    if (format !== 'tiff') expect(vi.mocked(host.exporter.encode).mock.calls[0][0].peakLuminance).toBe(1000);
    expect(job.state).toBe('done');
    expect(m.dispose).toHaveBeenCalled();
    expect(m.release).toHaveBeenCalled();
  },
);
it('serializes renders while letting the next render proceed during host encoding', async () => {
  let done!: () => void;
  vi.mocked(host.exporter.encode).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        done = () => resolve({});
      }),
  );
  const first = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(first.state).toBe('encoding'));
  const next = enqueueExport('a', 'avif');
  await next.promise;
  expect(first.state).toBe('encoding');
  expect(next.state).toBe('done');
  done();
  await first.promise;
});
it('cancels after readback and releases the image without encoding', async () => {
  let finish!: (data: Float32Array) => void;
  m.readback.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const job = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(m.readback).toHaveBeenCalled());
  job.cancel();
  finish(new Float32Array(4));
  await job.promise;
  expect(job.state).toBe('cancelled');
  expect(host.exporter.encode).not.toHaveBeenCalled();
  expect(m.release).toHaveBeenCalled();
});

it('drops a queued job immediately when cancelled, without waiting for another readback', async () => {
  let finish!: (data: Float32Array) => void;
  m.readback.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const first = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(first.state).toBe('rendering'));
  const queued = enqueueExport('a', 'tiff');
  queued.cancel();
  const stateAfterCancel = queued.state;
  finish(new Float32Array(4));
  await first.promise;
  expect(stateAfterCancel).toBe('cancelled');
  await expect(queued.promise).resolves.toBeNull();
  expect(m.readback).toHaveBeenCalledTimes(1);
});
it('does not render the second JPEG readback after cancellation of the first', async () => {
  let finish!: (data: Float32Array) => void;
  m.readback.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const handle = enqueueExport('a', 'jpeg-hdr');
  await vi.waitFor(() => expect(m.readback).toHaveBeenCalled());
  handle.cancel();
  finish(new Float32Array(4));
  await handle.promise;
  expect(m.readback).toHaveBeenCalledTimes(1);
  expect(host.exporter.encode).not.toHaveBeenCalled();
});
it('waits for the chosen photo method or model to finish processing before export', async () => {
  useAppStore.setState((state) => ({
    files: state.files.map((file) => ({
      ...file,
      processedKey: 'bilinear',
      edit: { ...file.edit, demosaicMethod: 'neural-net' as const },
    })),
  }));
  await expect(enqueueExport('a').promise).rejects.toThrow('Wait for this photo to finish processing');
  expect(host.exporter.chooseDestination).not.toHaveBeenCalled();
  expect(m.readback).not.toHaveBeenCalled();
  expect(m.release).toHaveBeenCalledOnce();
});

it('passes the photo identity and resolves a hash result without a blob', async () => {
  const result = { bytes: 42, sha256: 'abc', encodeMs: 12, name: 'a.avif' };
  vi.mocked(host.exporter.encode).mockResolvedValue(result);
  await expect(enqueueExport('a', 'avif').promise).resolves.toEqual(result);
  expect(host.exporter.chooseDestination).toHaveBeenCalledWith('a', 'a.avif', 'avif');
});

it('observes rendering, encoding and done, and the chosen destination once', async () => {
  const state = vi.fn();
  const destination = vi.fn();
  await enqueueExport('a', 'tiff', 95, { state, destination }).promise;
  expect(state.mock.calls.flat()).toEqual(['rendering', 'encoding', 'done']);
  expect(destination).toHaveBeenCalledExactlyOnceWith({ token: 'test' });
});
it('observes a cancelled destination without rendering', async () => {
  vi.mocked(host.exporter.chooseDestination).mockResolvedValue(null);
  const state = vi.fn();
  await enqueueExport('a', 'tiff', 95, { state }).promise;
  expect(state.mock.calls.flat()).toEqual(['cancelled']);
});
it('bounds rendered planes to two and frees capacity after encoding', async () => {
  const finishes: Array<() => void> = [];
  vi.mocked(host.exporter.encode).mockImplementation(() => new Promise((resolve) => finishes.push(() => resolve({}))));
  const first = enqueueExport('a', 'tiff');
  const second = enqueueExport('a', 'tiff');
  const third = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(host.exporter.encode).toHaveBeenCalledTimes(2));
  expect(m.readback).toHaveBeenCalledTimes(2);
  expect(third.state).toBe('queued');
  const cancelled = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(host.exporter.chooseDestination).toHaveBeenCalledTimes(4));
  cancelled.cancel();
  await cancelled.promise;
  finishes[0]();
  await first.promise;
  await vi.waitFor(() => expect(host.exporter.encode).toHaveBeenCalledTimes(3));
  finishes[1](); finishes[2]();
  await Promise.all([second.promise, third.promise]);
  vi.mocked(host.exporter.encode).mockResolvedValue({});
  await enqueueExport('a', 'tiff').promise;
  expect(m.readback).toHaveBeenCalledTimes(4);
});
it.each(['queued', 'rendering', 'encoding'] as const)('retains its snapshot through a folder switch while %s', async (phase) => {
  let resume!: () => void;
  const gate = new Promise<void>((resolve) => { resume = resolve; });
  if (phase === 'queued') vi.mocked(host.exporter.chooseDestination).mockImplementationOnce(async () => { await gate; return { token: 'test' }; });
  if (phase === 'rendering') m.readback.mockImplementationOnce(async () => { await gate; return new Float32Array(4); });
  if (phase === 'encoding') vi.mocked(host.exporter.encode).mockImplementationOnce(async () => { await gate; return {}; });
  const job = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(phase === 'queued' ? vi.mocked(host.exporter.chooseDestination).mock.calls.length : job.state).toBe(phase === 'queued' ? 1 : phase));
  useAppStore.setState({ files: [], selectedFileId: null });
  if (phase === 'rendering') expect(m.release).not.toHaveBeenCalled();
  resume();
  await job.promise;
  expect(job.state).toBe('done');
  expect(host.exporter.chooseDestination).toHaveBeenCalledOnce();
  expect(m.release).toHaveBeenCalledOnce();
});

it.each(['failed', 'cancelled'] as const)('returns plane capacity when an encode ends %s', async (ending) => {
  let settle!: () => void;
  vi.mocked(host.exporter.encode).mockImplementationOnce(() => new Promise((resolve, reject) => {
    settle = ending === 'failed' ? () => reject(new Error('broken encode')) : () => resolve({});
  }));
  let finishSecond!: () => void;
  vi.mocked(host.exporter.encode).mockImplementationOnce(() => new Promise((resolve) => { finishSecond = () => resolve({}); }));
  const first = enqueueExport('a', 'tiff');
  // Attach the rejection handler before settling the fake encode.
  const firstOutcome = first.promise.catch((error: unknown) => error);
  const second = enqueueExport('a', 'tiff');
  const third = enqueueExport('a', 'tiff');
  await vi.waitFor(() => expect(host.exporter.encode).toHaveBeenCalledTimes(2));
  if (ending === 'cancelled') first.cancel();
  expect(m.readback).toHaveBeenCalledTimes(2);
  settle();
  await firstOutcome;
  await third.promise;
  expect(first.state).toBe(ending);
  expect(m.readback).toHaveBeenCalledTimes(3);
  finishSecond();
  await second.promise;
});
