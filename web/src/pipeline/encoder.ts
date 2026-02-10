import { isWebCodecsAvifSupported, encodeAvifWebCodecs } from './encode-avif-webcodecs';
import type { ExportFormat } from './types';

let worker: Worker | null = null;
let webCodecsAvif: boolean | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./encoder-worker.ts', import.meta.url), { type: 'module' });
  }
  return worker;
}

function encodeViaWorker(
  hwc: Float32Array, width: number, height: number,
  xyzToCam: Float32Array, orientation: string,
  format: string, quality: number,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const hwcCopy = hwc.slice();
    const xyzToCamCopy = xyzToCam.slice();

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
      orientation, format, quality,
    }, [hwcCopy.buffer, xyzToCamCopy.buffer]);
  });
}

export async function encodeImage(
  hwc: Float32Array, width: number, height: number,
  xyzToCam: Float32Array | null, orientation: string,
  format: ExportFormat, quality: number,
  canvas?: HTMLCanvasElement | null,
): Promise<{ blob: Blob; ext: string }> {
  // AVIF: try WebCodecs first (captures from the already-rendered HDR canvas)
  if (format === 'avif' && canvas) {
    if (webCodecsAvif === null) {
      webCodecsAvif = await isWebCodecsAvifSupported(canvas.width, canvas.height);
      console.log(`WebCodecs AV1: ${webCodecsAvif ? 'supported' : 'not available, using rav1e WASM'}`);
    }
    if (webCodecsAvif) {
      const blob = await encodeAvifWebCodecs(canvas, quality);
      return { blob, ext: 'avif' };
    }
  }

  // JPEG, TIFF, or AVIF fallback — encode in Web Worker
  const encoded = await encodeViaWorker(
    hwc, width, height,
    xyzToCam ?? new Float32Array(9), orientation,
    format, quality,
  );

  const mimeTypes: Record<string, string> = { avif: 'image/avif', jpeg: 'image/jpeg', tiff: 'image/tiff' };
  const extensions: Record<string, string> = { avif: 'avif', jpeg: 'jpg', tiff: 'tif' };

  const blob = new Blob([encoded.buffer as ArrayBuffer], { type: mimeTypes[format] });
  return { blob, ext: extensions[format] };
}
