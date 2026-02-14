import { initDemosaicGpu, gpuAvailable, runBilinearGpu, runDhtGpu } from './demosaic-gpu';
import { DemosaicPool } from './demosaic-pool';

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
  algorithm: 'bilinear' | 'markesteijn1' | 'markesteijn3' | 'dht',
  cfaPattern: Uint32Array,
  period: number,
): Promise<Float32Array> {
  const isBayer = period === 2;

  // Markesteijn is X-Trans only — fall back to bilinear for Bayer
  if (isBayer && (algorithm === 'markesteijn1' || algorithm === 'markesteijn3')) {
    console.warn(`[demosaic] ${algorithm} is X-Trans only, falling back to bilinear`);
    algorithm = 'bilinear';
  }

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

  if (algorithm === 'dht') {
    if (gpuAvailable()) {
      try {
        console.time('[demosaic] gpu dht');
        const result = await runDhtGpu(cfa, width, height, dy, dx, cfaPattern, period);
        console.timeEnd('[demosaic] gpu dht');
        return result;
      } catch (e) {
        console.warn('[demosaic] GPU DHT failed, falling back to WASM worker:', e);
      }
    } else {
      console.warn('[demosaic] DHT GPU unavailable, using WASM worker');
    }
    // Fall through to worker pool with algorithm='dht'
  }

  // WASM worker pool (X-Trans only — hardcoded pattern in WASM module)
  if (isBayer) {
    throw new Error('Traditional demosaic for Bayer requires GPU. Please use Neural Network method.');
  }

  console.time(`[demosaic] pool ${algorithm}`);
  const result = await getPool().run(cfa, width, height, dy, dx, algorithm, period);
  console.timeEnd(`[demosaic] pool ${algorithm}`);
  return result;
}
