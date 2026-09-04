import { DemosaicPool } from './demosaic-pool';
import { cropPlanar, type DemosaicInput, type DemosaicStrategy, type TraditionalMethod } from './strategy';

let pool: DemosaicPool | null = null;

function getPool(): DemosaicPool {
  if (!pool) {
    pool = new DemosaicPool();
  }
  return pool;
}

/** Terminate the workers. Called after every processing run, as before. */
export function destroyDemosaicPool(): void {
  if (pool) {
    pool.destroy();
    pool = null;
  }
}

/** Run a traditional method in the WASM worker pool on the canonically aligned CFA. */
export async function runInPool(input: DemosaicInput, algorithm: TraditionalMethod): Promise<Float32Array> {
  console.time(`[demosaic] pool ${algorithm}`);
  const result = await getPool().run(
    input.cfa, input.width, input.height, 0, 0, algorithm, input.period, input.period === 2,
  );
  console.timeEnd(`[demosaic] pool ${algorithm}`);
  return result;
}

/** A method that only exists in the WASM pool (markesteijn1/3, ahd, ppg, mhc). */
export function poolStrategy(id: TraditionalMethod): DemosaicStrategy {
  return {
    id,
    async run(input) {
      return cropPlanar(await runInPool(input, id), input);
    },
  };
}
