import type { ExportFormat } from '@/lib/types';

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./encoder-worker.ts', import.meta.url), { type: 'module' });
  }
  return worker;
}

/** The array's buffer for transfer — itself when the view spans all of it, else a copy. */
function transferable(a: Float32Array): ArrayBuffer {
  return a.byteOffset === 0 && a.byteLength === a.buffer.byteLength
    ? a.buffer as ArrayBuffer
    : a.slice().buffer;
}

/**
 * Encode in the worker. `data` and `hdrData` are transferred, not copied: at 24 MP each is
 * ~290 MB, so the caller hands them over and must not touch them afterwards (they detach).
 */
export function encodeViaWorker(
  data: Float32Array, hdrData: Float32Array,
  width: number, height: number,
  orientation: string, format: ExportFormat,
  quality: number, peakLuminance: number,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const dataBuf = transferable(data);
    const hdrBuf = transferable(hdrData);

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
      data: dataBuf,
      hdrData: hdrBuf,
      width, height,
      orientation, format, quality,
      peakLuminance,
    }, [dataBuf, hdrBuf]);
  });
}
