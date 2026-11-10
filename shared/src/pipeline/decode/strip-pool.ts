/**
 * The strips of a compressed Fuji RAF decode independently of each other, so the decode worker
 * hands them to strip workers of its own, each with its own instance of the decoder (nothing is
 * shared between them, so the page needs no cross-origin isolation). The workers are kept for the
 * life of the decode worker. A strip that fails takes the whole decode with it: every strip
 * worker is terminated, so no instance that saw a failure is used again.
 */

/** One strip, as `RawDecode.layout` describes it, with its bytes cut out of the file. */
export interface StripJob {
  index: number;
  /** The 16 header bytes of the compressed data. */
  header: Uint8Array;
  data: Uint8Array;
  /** The strip's row of q_bases; empty for a lossless file. */
  qBases: Uint8Array;
  stripWidth: number;
  cols: number;
  height: number;
}

export type StripInit = { module: WebAssembly.Module };
export type StripReply = { index: number; block: Uint16Array } | { index: number; error: string };

const LAYOUT_HEAD = 2;
const LAYOUT_STRIP = 6;

/** Cut each strip's bytes out of `file`, as copies a transfer can take. */
export function stripJobs(file: Uint8Array, layout: Uint32Array): StripJob[] {
  const [headerAt, height] = layout;
  const header = file.slice(headerAt, headerAt + 16);
  const jobs: StripJob[] = [];
  for (let at = LAYOUT_HEAD; at < layout.length; at += LAYOUT_STRIP) {
    const [dataStart, dataEnd, qStart, qEnd, stripWidth, cols] = layout.subarray(at, at + LAYOUT_STRIP);
    jobs.push({
      index: jobs.length, header, data: file.slice(dataStart, dataEnd), qBases: file.slice(qStart, qEnd), stripWidth, cols, height,
    });
  }
  return jobs;
}

const createStripWorker = (): Worker =>
  new Worker(new URL('./strip-worker.ts', import.meta.url), { type: 'module' });

export class StripPool {
  private workers: Worker[] = [];

  /** `module` is the compiled decoder; each worker only instantiates it. */
  constructor(
    private module: WebAssembly.Module,
    private concurrency = navigator.hardwareConcurrency || 4,
    private create = createStripWorker,
  ) {}

  /**
   * Decode every job, passing each block to `put` as it arrives. One run at a time. On the first
   * failure (a strip error, a crashed worker, a refused block) the workers are terminated and
   * the promise rejects.
   */
  run(jobs: StripJob[], put: (index: number, block: Uint16Array) => void): Promise<void> {
    // Largest first: with fewer workers than strips, the big ones don't end up queued last.
    const queue = [...jobs].sort((a, b) => b.data.length - a.data.length);
    const size = Math.min(queue.length, this.concurrency);
    while (this.workers.length < size) {
      const worker = this.create();
      worker.postMessage({ module: this.module } satisfies StripInit);
      this.workers.push(worker);
    }
    return new Promise<void>((resolve, reject) => {
      let left = queue.length;
      if (left === 0) { resolve(); return; }
      const fail = (message: string) => {
        left = -1;
        this.terminate();
        reject(new Error(message));
      };
      const next = (worker: Worker) => {
        const job = queue.shift();
        if (job) worker.postMessage(job, [job.data.buffer, job.qBases.buffer]);
      };
      for (const worker of this.workers.slice(0, size)) {
        worker.onmessage = (e: MessageEvent<StripReply>) => {
          if (left < 0) return;
          if ('error' in e.data) { fail(e.data.error); return; }
          try {
            put(e.data.index, e.data.block);
          } catch (err) {
            fail(err instanceof Error ? err.message : String(err));
            return;
          }
          if (--left === 0) resolve();
          else next(worker);
        };
        worker.onerror = (e: ErrorEvent) => {
          e.preventDefault();
          if (left >= 0) fail(e.message || 'RAW strip decoder crashed');
        };
        next(worker);
      }
    });
  }

  terminate(): void {
    for (const worker of this.workers) worker.terminate();
    this.workers = [];
  }
}
