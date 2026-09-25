import { runDhtGpu } from './demosaic-gpu';
import { uploadNormalizedCfa } from './cfa-gpu';
import { runInPool } from './wasm-pool';
import type { DemosaicStrategy } from './strategy';

export const dhtStrategy: DemosaicStrategy = {
  id: 'dht',
  async run(input, ctx) {
    try {
      console.time('[demosaic] gpu dht');
      const cfa = uploadNormalizedCfa(ctx.device, input);
      let buffer: GPUBuffer;
      try {
        buffer = runDhtGpu(ctx.device, cfa, input.width, input.height, input.pattern, input.period);
      } finally {
        cfa.destroy();
      }
      console.timeEnd('[demosaic] gpu dht');
      return {
        rgb: { buffer, stride: input.width, offsetX: input.padLeft, offsetY: input.padTop },
        tileCount: 1,
      };
    } catch (e) {
      console.warn('[demosaic] GPU DHT failed, falling back to WASM worker:', e);
    }
    return { rgb: await runInPool(input, 'dht'), tileCount: 1 };
  },
};
