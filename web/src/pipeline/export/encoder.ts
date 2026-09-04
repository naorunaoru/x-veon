import type { ExportFormat } from '@/lib/types';
import { exportFormatInfo } from '@/lib/catalog';

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./encoder-worker.ts', import.meta.url), { type: 'module' });
  }
  return worker;
}

function encodeViaWorker(
  data: Float32Array, hdrData: Float32Array,
  width: number, height: number,
  orientation: string, format: string,
  quality: number, peakLuminance: number,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const dataCopy = data.slice();
    const hdrCopy = hdrData.slice();

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
      data: dataCopy.buffer,
      hdrData: hdrCopy.buffer,
      width, height,
      orientation, format, quality,
      peakLuminance,
    }, [dataCopy.buffer, hdrCopy.buffer]);
  });
}

export async function encodeImage(
  data: Float32Array,
  hdrData: Float32Array | null,
  width: number, height: number,
  orientation: string,
  format: ExportFormat,
  quality: number,
  peakLuminance: number,
): Promise<{ blob: Blob; ext: string }> {
  const encoded = await encodeViaWorker(
    data, hdrData ?? new Float32Array(0),
    width, height,
    orientation, format, quality, peakLuminance,
  );

  const info = exportFormatInfo(format);
  const blob = new Blob([encoded.buffer as ArrayBuffer], { type: info.mime });
  return { blob, ext: info.ext };
}
