import { initDemosaicGpu, gpuAvailable, runBilinearGpu, runDhtGpu } from './demosaic-gpu';
import { DemosaicPool } from './demosaic-pool';
import type { DemosaicMethod } from './types';

type TraditionalMethod = Exclude<DemosaicMethod, 'neural-net'>;

export async function initDemosaicGpuSafe(): Promise<boolean> {
  return initDemosaicGpu();
}

let pool: DemosaicPool | null = null;

function getPool(): DemosaicPool {
  if (!pool) {
    pool = new DemosaicPool();
  }
  return pool;
}

export function destroyDemosaicPool(): void {
  if (pool) {
    pool.destroy();
    pool = null;
  }
}

export async function runDemosaic(
  cfa: Float32Array,
  width: number,
  height: number,
  dy: number,
  dx: number,
  algorithm: TraditionalMethod,
  cfaPattern: Uint32Array,
  period: number,
): Promise<Float32Array> {
  // GPU paths
  if (algorithm === 'bilinear' && gpuAvailable()) {
    try {
      console.time('[demosaic] gpu bilinear');
      const result = await runBilinearGpu(cfa, width, height, dy, dx, cfaPattern, period);
      console.timeEnd('[demosaic] gpu bilinear');
      return result;
    } catch (e) {
      console.warn('[demosaic] GPU bilinear failed, falling back to worker:', e);
    }
  }

  if (algorithm === 'dht' && gpuAvailable()) {
    try {
      console.time('[demosaic] gpu dht');
      const result = await runDhtGpu(cfa, width, height, dy, dx, cfaPattern, period);
      console.timeEnd('[demosaic] gpu dht');
      return result;
    } catch (e) {
      console.warn('[demosaic] GPU DHT failed, falling back to WASM worker:', e);
    }
  }

  // WASM worker pool
  console.time(`[demosaic] pool ${algorithm}`);
  const result = await getPool().run(cfa, width, height, dy, dx, algorithm, period, period === 2);
  console.timeEnd(`[demosaic] pool ${algorithm}`);
  return result;
}
