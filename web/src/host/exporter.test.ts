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
  encode.mockReset().mockResolvedValue(new Blob(['encoded']));
});
it('serializes worker requests and delivers each output under its own name', async () => {
  let finish!: (blob: Blob) => void;
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
  finish(new Blob(['first']));
  await Promise.all([first, second]);
  expect(deliver.mock.calls.map((c) => c[1])).toEqual(['a.tif', 'b.tif']);
});
it('returns encoded bytes without downloading for the golden delivery and discards aborted output', async () => {
  const delivery = vi.fn();
  const exporter = createExporter(delivery);
  const result = await exporter.encode(job(), { token: 'golden' });
  expect(result.blob?.size).toBe(7);
  let finish!: (blob: Blob) => void;
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
  finish(new Blob(['cancelled']));
  await expect(aborted).rejects.toThrow();
  expect(delivery).toHaveBeenCalledTimes(1);
});
