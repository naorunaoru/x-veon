import { gpuAvailable, runDhtGpu } from './demosaic-gpu';
import { runInPool } from './wasm-pool';
import { cropPlanar, type DemosaicStrategy } from './strategy';

export const dhtStrategy: DemosaicStrategy = {
  id: 'dht',
  async run(input) {
    if (gpuAvailable()) {
      try {
        console.time('[demosaic] gpu dht');
        const result = await runDhtGpu(
          input.cfa, input.width, input.height, 0, 0, input.pattern, input.period,
        );
        console.timeEnd('[demosaic] gpu dht');
        return cropPlanar(result, input);
      } catch (e) {
        console.warn('[demosaic] GPU DHT failed, falling back to WASM worker:', e);
      }
    }
    return cropPlanar(await runInPool(input, 'dht'), input);
  },
};
