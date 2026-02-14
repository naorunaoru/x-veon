type Algorithm = 'bilinear' | 'markesteijn1' | 'markesteijn3' | 'dht';

const STRIP_OVERLAP = 18; // 3 × X-Trans period (6), safe for 3-pass refinement + Markesteijn border
const MAX_WORKERS = 8;
const MIN_STRIP_HEIGHT = 128; // Don't split if strips would be smaller than this

export class DemosaicPool {
  private workers: Worker[] = [];
  private size: number;

  constructor() {
    this.size = Math.min(navigator.hardwareConcurrency ?? 4, MAX_WORKERS);
  }

  private ensureWorkers(): void {
    if (this.workers.length > 0) return;
    for (let i = 0; i < this.size; i++) {
      this.workers.push(
        new Worker(new URL('./demosaic-worker.ts', import.meta.url), { type: 'module' }),
      );
    }
  }

  async run(
    cfa: Float32Array,
    width: number,
    height: number,
    dy: number,
    dx: number,
    algorithm: Algorithm,
  ): Promise<Float32Array> {
    this.ensureWorkers();

    // Fast path: small image or single core — no splitting
    const effectiveWorkers = Math.min(
      this.size,
      Math.floor(height / MIN_STRIP_HEIGHT),
    );
    if (effectiveWorkers <= 1) {
      return this.runOne(0, cfa, width, height, dy, dx, algorithm);
    }

    // Split into horizontal strips
    const baseHeight = Math.ceil(height / effectiveWorkers);
    const strips: Array<{
      startRow: number;
      stripHeight: number;
      stripDy: number;
      innerStart: number; // first owned row within strip output
      innerEnd: number;   // last+1 owned row within strip output
      ownedStart: number; // first owned row in full image
    }> = [];

    for (let i = 0; i < effectiveWorkers; i++) {
      const ownedStart = i * baseHeight;
      const ownedEnd = Math.min((i + 1) * baseHeight, height);
      const startRow = Math.max(0, ownedStart - STRIP_OVERLAP);
      const endRow = Math.min(height, ownedEnd + STRIP_OVERLAP);

      strips.push({
        startRow,
        stripHeight: endRow - startRow,
        stripDy: (dy + startRow) % 6,
        innerStart: ownedStart - startRow,
        innerEnd: ownedEnd - startRow,
        ownedStart,
      });
    }

    // Farm strips to workers in parallel
    const promises = strips.map((strip, i) => {
      const stripCfa = cfa.subarray(
        strip.startRow * width,
        (strip.startRow + strip.stripHeight) * width,
      );
      return this.runOne(i, stripCfa, width, strip.stripHeight, strip.stripDy, dx, algorithm);
    });

    const results = await Promise.all(promises);

    // Stitch: copy inner (non-overlap) rows from each strip into final output
    const npix = width * height;
    const output = new Float32Array(3 * npix);

    for (let i = 0; i < results.length; i++) {
      const strip = strips[i];
      const result = results[i];
      const stripPixels = width * strip.stripHeight;
      const innerRows = strip.innerEnd - strip.innerStart;

      for (let c = 0; c < 3; c++) {
        const srcOff = c * stripPixels + strip.innerStart * width;
        const dstOff = c * npix + strip.ownedStart * width;
        output.set(result.subarray(srcOff, srcOff + innerRows * width), dstOff);
      }
    }

    return output;
  }

  private runOne(
    workerIdx: number,
    stripCfa: Float32Array,
    width: number,
    height: number,
    dy: number,
    dx: number,
    algorithm: Algorithm,
  ): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const w = this.workers[workerIdx];
      const cfaCopy = stripCfa.slice();

      w.onmessage = (e) => {
        if (e.data.type === 'done') {
          resolve(new Float32Array(e.data.data));
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      w.onerror = (e) => reject(new Error(e.message));

      w.postMessage({
        type: 'demosaic',
        cfa: cfaCopy.buffer,
        width, height, dy, dx, algorithm,
      }, [cfaCopy.buffer]);
    });
  }

  destroy(): void {
    for (const w of this.workers) {
      w.terminate();
    }
    this.workers = [];
  }
}
