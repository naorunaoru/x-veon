import type { ExportFormat } from '@/lib/types';

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./encoder-worker.ts', import.meta.url), { type: 'module' });
  }
  return worker;
}

export function encodeViaWorker(
  data: Float32Array, hdrData: Float32Array,
  width: number, height: number,
  orientation: string, format: ExportFormat,
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
