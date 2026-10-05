import { beforeEach, expect, it, vi } from 'vitest';
const encode = vi.hoisted(() => vi.fn());
vi.mock('./encoders', () => ({ encoderFor: () => ({ encode }) }));
import { createExporter } from './exporter';
import type { EncodeJob } from '@/host';
const job = (): EncodeJob => ({
  format: 'tiff',
  data: new Float32Array(4),
  hdrData: null,
  width: 1,
  height: 1,
  orientation: 'Normal',
  quality: 95,
  peakLuminance: 100,
});
beforeEach(() => {
  encode.mockReset().mockResolvedValue({ blob: new Blob(['encoded']), encodeMs: 12 });
});
it('serializes worker requests and delivers each output under its own name', async () => {
  let finish!: (result: { blob: Blob; encodeMs: number }) => void;
  encode.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const deliver = vi.fn();
  const exporter = createExporter(deliver);
  const first = exporter.encode(job(), { token: 'a.tif' });
  const second = exporter.encode(job(), { token: 'b.tif' });
  await vi.waitFor(() => expect(encode).toHaveBeenCalledTimes(1));
  finish({ blob: new Blob(['first']), encodeMs: 12 });
  await Promise.all([first, second]);
  expect(deliver.mock.calls.map((c) => c[1])).toEqual(['a.tif', 'b.tif']);
});
it('returns encoded bytes without downloading for the golden delivery and discards aborted output', async () => {
  const delivery = vi.fn();
  const exporter = createExporter(delivery);
  const result = await exporter.encode(job(), { token: 'golden' });
  expect(result.blob?.size).toBe(7);
  let finish!: (result: { blob: Blob; encodeMs: number }) => void;
  encode.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const controller = new AbortController();
  const aborted = exporter.encode({ ...job(), signal: controller.signal }, { token: 'cancelled' });
  await vi.waitFor(() => expect(encode).toHaveBeenCalledTimes(2));
  controller.abort();
  finish({ blob: new Blob(['cancelled']), encodeMs: 12 });
  await expect(aborted).rejects.toThrow();
  expect(delivery).toHaveBeenCalledTimes(1);
});

it('chooses a download name and returns size, name and timing', async () => {
 const exporter = createExporter(vi.fn());
 await expect(exporter.chooseDestination('id', 'a.avif', 'avif')).resolves.toEqual({ token: 'a.avif' });
 await expect(exporter.encode(job(), { token: 'a.avif' })).resolves.toEqual({ blob: expect.any(Blob), bytes: 7, name: 'a.avif', encodeMs: 12 });
});
