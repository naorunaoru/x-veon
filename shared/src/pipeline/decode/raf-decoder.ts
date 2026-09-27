import type { PreparedCfa, RawMeta } from '../types';

type Reply =
  | { type: 'done'; raw: RawMeta; prepared: PreparedCfa | null; prepareError: string | null }
  | { type: 'pong' } | { type: 'error'; message: string };

/** Factory for the decode worker; replaceable in tests. */
export let createDecodeWorker = (): Worker =>
  new Worker(new URL('./decode-worker.ts', import.meta.url), { type: 'module' });

export function setDecodeWorkerFactory(factory: () => Worker): void {
  createDecodeWorker = factory;
  worker?.terminate();
  worker = null;
}

let worker: Worker | null = null;
let queue: Promise<unknown> = Promise.resolve();

/** Send one request to the worker. After any failure the worker is discarded, never reused. */
function request(message: object, transfer: Transferable[] = []): Promise<Reply> {
  const run = () => new Promise<Reply>((resolve, reject) => {
    const w = worker ??= createDecodeWorker();
    const discard = () => {
      w.terminate();
      if (worker === w) worker = null;
    };
    w.onmessage = (e: MessageEvent<Reply>) => {
      if (e.data.type === 'error') {
        discard();
        reject(new Error(e.data.message));
      } else {
        resolve(e.data);
      }
    };
    w.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      discard();
      reject(new Error(e.message || 'RAW decoder crashed'));
    };
    w.postMessage(message, transfer);
  });
  // One decode at a time: each request owns the worker's handlers until it settles.
  const result = queue.then(run, run);
  queue = result.catch(() => {});
  return result;
}

/** Start the decoder and wait until its wasm module has loaded. */
export async function initWasm(): Promise<void> {
  await request({ type: 'ping' });
}

/**
 * Decode a RAW file and prepare its CFA in the decoder worker. `bytes` is transferred
 * (detached) to the worker. A file that crashes the decoder only fails its own decode; the next
 * one gets a fresh instance. `prepareError` is set, and `prepared` null, when the file decoded but
 * its CFA can't be laid out.
 */
export async function decodeRaw(bytes: ArrayBuffer): Promise<{
  raw: RawMeta; prepared: PreparedCfa | null; prepareError: string | null;
}> {
  const reply = await request({ type: 'decode', bytes }, [bytes]);
  if (reply.type !== 'done') throw new Error('Unexpected reply from the RAW decoder');
  return { raw: reply.raw, prepared: reply.prepared, prepareError: reply.prepareError };
}
