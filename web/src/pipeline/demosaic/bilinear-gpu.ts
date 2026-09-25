import { runBilinearGpu } from './demosaic-gpu';
import { uploadNormalizedCfa } from './cfa-gpu';
import { runInPool } from './wasm-pool';
import type { DemosaicStrategy } from './strategy';

export const bilinearStrategy: DemosaicStrategy = {
  id: 'bilinear',
  async run(input, ctx) {
    try {
      console.time('[demosaic] gpu bilinear');
      const cfa = uploadNormalizedCfa(ctx.device, input);
      let buffer: GPUBuffer;
      try {
        buffer = runBilinearGpu(ctx.device, cfa, input.width, input.height, input.pattern, input.period);
      } finally {
        cfa.destroy();
      }
      console.timeEnd('[demosaic] gpu bilinear');
      return {
        rgb: { buffer, stride: input.width, offsetX: input.padLeft, offsetY: input.padTop },
        tileCount: 1,
      };
    } catch (e) {
      console.warn('[demosaic] GPU bilinear failed, falling back to worker:', e);
    }
    return { rgb: await runInPool(input, 'bilinear'), tileCount: 1 };
  },
};
