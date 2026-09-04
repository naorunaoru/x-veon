import { generateTiles } from '../preprocess/preprocessor';
import { createGpuNNPipeline } from './tile-blend-gpu';
import { PATCH_SIZE, OVERLAP, TILE_BATCH } from '../constants';
import type { DemosaicStrategy } from './strategy';

/** Fully GPU-resident path: extract tiles → infer → blend → finalize+crop, all on ctx.device. */
export const neuralNetStrategy: DemosaicStrategy = {
  id: 'neural-net',
  async run(input, ctx, opts) {
    const tileGrid = generateTiles(input.width, input.height, PATCH_SIZE, OVERLAP);
    const { tiles, hPad, wPad } = tileGrid;

    const gpu = createGpuNNPipeline(
      ctx.device, input.cfa, input.width, input.height,
      input.masks, input.clipNorm, tiles,
      hPad, wPad, PATCH_SIZE, OVERLAP,
      input.padTop, input.padLeft, input.visibleHeight, input.visibleWidth, TILE_BATCH,
    );

    try {
      // Process batches — everything stays on GPU
      for (let b = 0; b < tiles.length; ) {
        const end = Math.min(b + TILE_BATCH, tiles.length);
        const count = end - b;

        // GPU: extract tiles from CFA → 5ch NCHW buffer
        const inputBuf = gpu.extractBatch(b, count);

        // GPU: inference (GPU buffer in → GPU buffer out)
        const { buffer: inferBuf, dispose } = await ctx.models.runBatchGpu(
          input.cfaType, inputBuf, count, PATCH_SIZE,
        );

        // GPU: accumulate inference output into blend buffer
        try {
          gpu.accumulateBatch(inferBuf, b, count);
        } finally {
          dispose();
        }

        b = end;
        opts.onProgress?.(end, tiles.length);
      }

      // Finalize+crop on GPU → GPUBuffer passed directly to postprocess
      const hwc = await gpu.finalize();
      return { hwc, tileCount: tiles.length };
    } catch (error) {
      gpu.destroy();
      throw error;
    }
  },
};
