import { gpuAvailable, runBilinearGpu } from './demosaic-gpu';
import { runInPool } from './wasm-pool';
import { cropPlanar, type DemosaicStrategy } from './strategy';

export const bilinearStrategy: DemosaicStrategy = {
  id: 'bilinear',
  async run(input) {
    if (gpuAvailable()) {
      try {
        console.time('[demosaic] gpu bilinear');
        const result = await runBilinearGpu(
          input.cfa, input.width, input.height, 0, 0, input.pattern, input.period,
        );
        console.timeEnd('[demosaic] gpu bilinear');
        return cropPlanar(result, input);
      } catch (e) {
        console.warn('[demosaic] GPU bilinear failed, falling back to worker:', e);
      }
    }
    return cropPlanar(await runInPool(input, 'bilinear'), input);
  },
};
