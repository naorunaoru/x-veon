import type { DemosaicMethod } from '@/lib/types';
import { normalizeRows } from '../preprocess/preprocessor';
import type { DemosaicInput } from './strategy';

type Algorithm = Exclude<DemosaicMethod, 'neural-net'>;

/** Map (dy%2, dx%2) → Bayer variant string for demosaic_bayer WASM */
const BAYER_VARIANT_MAP = ['rggb', 'grbg', 'gbrg', 'bggr'] as const;
function bayerVariantForShift(dy: number, dx: number): string {
  return BAYER_VARIANT_MAP[(dy % 2) * 2 + (dx % 2)];
}

const STRIP_OVERLAP_FACTOR = 3; // 3 × CFA period, safe for 3-pass refinement + Markesteijn border
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

  /**
   * Demosaic the canonically aligned u16 CFA in `input` and return the visible image as cropped
   * HWC. Each strip is normalised as it is cut out (the copy a transfer needs anyway), and each
   * strip's owned rows are written straight into the cropped output.
   */
  async run(input: DemosaicInput, algorithm: Algorithm): Promise<Float32Array> {
    this.ensureWorkers();

    const { cfa, lut, width, height, period, pattern } = input;
    const isBayer = period === 2;
    const stripOverlap = STRIP_OVERLAP_FACTOR * period;

    // Small image or single core — one strip
    const effectiveWorkers = Math.max(1, Math.min(
      this.size,
      Math.floor(height / MIN_STRIP_HEIGHT),
    ));

    // Split into horizontal strips
    const baseHeight = Math.ceil(height / effectiveWorkers);
    const strips: Array<{
      startRow: number;
      stripHeight: number;
      stripDy: number;
      ownedStart: number; // first owned row in full image
      ownedEnd: number;   // last+1 owned row in full image
    }> = [];

    for (let i = 0; i < effectiveWorkers; i++) {
      const ownedStart = i * baseHeight;
      const ownedEnd = Math.min((i + 1) * baseHeight, height);
      const startRow = effectiveWorkers === 1 ? 0 : Math.max(0, ownedStart - stripOverlap);
      const endRow = effectiveWorkers === 1 ? height : Math.min(height, ownedEnd + stripOverlap);

      strips.push({
        startRow,
        stripHeight: endRow - startRow,
        stripDy: startRow % period,
        ownedStart,
        ownedEnd,
      });
    }

    // Farm strips to workers in parallel
    const results = await Promise.all(strips.map((strip, i) => this.runOne(
      i,
      normalizeRows(cfa, width, strip.startRow, strip.startRow + strip.stripHeight, pattern, period, lut),
      width, strip.stripHeight, strip.stripDy, 0, algorithm, isBayer,
    )));

    // Stitch: each strip's owned rows that fall in the visible area, planar → cropped HWC
    const { padTop, padLeft, visibleWidth: visW, visibleHeight: visH } = input;
    const hwc = new Float32Array(visW * visH * 3);
    for (let i = 0; i < results.length; i++) {
      const strip = strips[i];
      const result = results[i];
      const plane = width * strip.stripHeight;
      const from = Math.max(strip.ownedStart, padTop);
      const to = Math.min(strip.ownedEnd, padTop + visH);
      for (let y = from; y < to; y++) {
        const src = (y - strip.startRow) * width + padLeft;
        let dst = (y - padTop) * visW * 3;
        for (let x = 0; x < visW; x++, dst += 3) {
          hwc[dst] = result[src + x];
          hwc[dst + 1] = result[plane + src + x];
          hwc[dst + 2] = result[2 * plane + src + x];
        }
      }
    }
    return hwc;
  }

  /** `stripCfa` is transferred to the worker. */
  private runOne(
    workerIdx: number,
    stripCfa: Float32Array,
    width: number,
    height: number,
    dy: number,
    dx: number,
    algorithm: Algorithm,
    isBayer: boolean,
  ): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const w = this.workers[workerIdx];

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
        cfa: stripCfa.buffer,
        width, height, dy, dx, algorithm,
        bayerVariant: isBayer ? bayerVariantForShift(dy, dx) : undefined,
      }, [stripCfa.buffer]);
    });
  }

  destroy(): void {
    for (const w of this.workers) {
      w.terminate();
    }
    this.workers = [];
  }
}
