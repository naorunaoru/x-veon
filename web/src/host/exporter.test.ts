import { beforeEach, expect, it, vi } from 'vitest';
const encode = vi.hoisted(() => vi.fn());
vi.mock('./encoders', () => ({ encoderFor: () => ({ encode }) }));
import { createExporter } from './exporter';
import type { EncodeJob, ExportOutput } from '@/host';
const out = (name: string): ExportOutput => ({ target: null, name, onExists: 'skip' });
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
  const first = exporter.encode(job(), out('a.tif'));
  const second = exporter.encode(job(), out('b.tif'));
  await vi.waitFor(() => expect(encode).toHaveBeenCalledTimes(1));
  finish({ blob: new Blob(['first']), encodeMs: 12 });
  await Promise.all([first, second]);
  expect(deliver.mock.calls.map((c) => c[1])).toEqual(['a.tif', 'b.tif']);
});
it('returns encoded bytes without downloading for the golden delivery and discards aborted output', async () => {
  const delivery = vi.fn();
  const exporter = createExporter(delivery);
  const result = await exporter.encode(job(), out('golden'));
  expect(result.blob?.size).toBe(7);
  let finish!: (result: { blob: Blob; encodeMs: number }) => void;
  encode.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const controller = new AbortController();
  const aborted = exporter.encode({ ...job(), signal: controller.signal }, out('cancelled'));
  await vi.waitFor(() => expect(encode).toHaveBeenCalledTimes(2));
  controller.abort();
  finish({ blob: new Blob(['cancelled']), encodeMs: 12 });
  await expect(aborted).rejects.toThrow();
  expect(delivery).toHaveBeenCalledTimes(1);
});

it('encodes under the output name and reports written', async () => {
 const deliver = vi.fn();
 const exporter = createExporter(deliver);
 await expect(exporter.resolveTarget({ mode: 'beside' }, ['id'])).resolves.toBeNull();
 await expect(exporter.encode(job(), out('a.avif'))).resolves.toEqual({ outcome: 'written', blob: expect.any(Blob), bytes: 7, name: 'a.avif', encodeMs: 12 });
 expect(deliver).toHaveBeenCalledExactlyOnceWith(expect.any(Blob), 'a.avif');
});
