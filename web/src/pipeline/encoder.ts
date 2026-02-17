import type { ExportFormat } from './types';

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./encoder-worker.ts', import.meta.url), { type: 'module' });
  }
  return worker;
}

function encodeViaWorker(
  hwc: Float32Array, width: number, height: number,
  xyzToCam: Float32Array, wbCoeffs: Float32Array, orientation: string,
  format: string, quality: number, odrtConfig: Float32Array,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const hwcCopy = hwc.slice();
    const xyzToCamCopy = xyzToCam.slice();
    const wbCopy = wbCoeffs.slice();
    const odrtCopy = odrtConfig.slice();

    w.onmessage = (e) => {
      if (e.data.type === 'done') {
        resolve(new Uint8Array(e.data.data));
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message));
      }
    };
    w.onerror = (e) => reject(new Error(e.message));

    w.postMessage({
      type: 'encode',
      hwc: hwcCopy.buffer,
      width, height,
      xyzToCam: xyzToCamCopy.buffer,
      wbCoeffs: wbCopy.buffer,
      orientation, format, quality,
      odrtConfig: odrtCopy.buffer,
    }, [hwcCopy.buffer, xyzToCamCopy.buffer, wbCopy.buffer, odrtCopy.buffer]);
  });
}

export async function encodeImage(
  hwc: Float32Array, width: number, height: number,
  xyzToCam: Float32Array | null, wbCoeffs: Float32Array,
  orientation: string,
  format: ExportFormat, quality: number,
  odrtConfig: Float32Array,
): Promise<{ blob: Blob; ext: string }> {
  const encoded = await encodeViaWorker(
    hwc, width, height,
    xyzToCam ?? new Float32Array(9), wbCoeffs, orientation,
    format, quality, odrtConfig,
  );

  const mimeTypes: Record<string, string> = { avif: 'image/avif', 'jpeg-hdr': 'image/jpeg', tiff: 'image/tiff' };
  const extensions: Record<string, string> = { avif: 'avif', 'jpeg-hdr': 'jpg', tiff: 'tif' };

  const blob = new Blob([encoded.buffer as ArrayBuffer], { type: mimeTypes[format] });
  return { blob, ext: extensions[format] };
}
