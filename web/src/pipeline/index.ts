import { decodeRaw, initWasm } from './decode/raf-decoder';
import type { PreparedCfa, RawMeta } from './types';
import { channelClips, makeChannelMasks } from './preprocess/preprocessor';
import { models } from './inference';
import { strategyFor, destroyDemosaicPool, type DemosaicInput } from './demosaic';
import { buildColorMatrix } from './postprocess/postprocessor';
import { gpuPostprocess } from './postprocess/postprocess-gpu';
import { estimateColorTemperature } from './color-temperature';
import { PATCH_SIZE } from './constants';
import { getDevice, setSharedDevice } from '@/gpu/device';
import type { GpuImage, ModelSize, ProcessingResultMeta } from '@/lib/types';
import type { PipelineContext, ProcessOptions } from './context';

export type { PipelineContext, ProcessOptions } from './context';

/** A processed RAW: the RGBA32F texture on the GPU plus its metadata. The owner must dispose it. */
export interface ProcessedImage {
  gpu: GpuImage;
  meta: ProcessingResultMeta;
  dispose(): void;
}

/**
 * Initialise the wasm decoder and the models in parallel, then share ONNX Runtime's WebGPU
 * device with everything else (post-process, GPU demosaic, renderer) through gpu/device.ts.
 */
export async function initPipeline(opts: { modelSize: ModelSize }): Promise<PipelineContext> {
  await Promise.all([initWasm(), models.init(opts.modelSize)]);
  // Share ORT's WebGPU device with the renderer and post-processing for zero-copy buffer interop
  if (models.device) setSharedDevice(models.device);
  return { device: await getDevice(), models };
}

/** Flatten a 2D pattern array into a Uint32Array for GPU/demosaic use */
function flattenPattern(pattern: readonly (readonly number[])[], period: number): Uint32Array {
  const flat = new Uint32Array(period * period);
  for (let y = 0; y < period; y++) {
    for (let x = 0; x < period; x++) {
      flat[y * period + x] = pattern[y][x];
    }
  }
  return flat;
}

export async function processRaw(
  bytes: ArrayBuffer, options: ProcessOptions, ctx: PipelineContext,
): Promise<ProcessedImage> {
  let ownedOutput: GpuImage | null = null;
  try {
    // 1. Decode RAW, and prepare the CFA in the decoder worker: crop to the visible area, find
    //    the CFA pattern shift, calibrate white levels per colour (actual sensor saturation),
    //    pad for phase alignment, and build the normalisation table (no WB — the model is
    //    trained on raw CFA data). The CFA stays u16 until a demosaic consumes it.
    let raw: RawMeta;
    let prepared: PreparedCfa | null;
    let prepareError: string | null;
    try {
      ({ raw, prepared, prepareError } = await decodeRaw(bytes));
    } catch (e) {
      const detail = (e instanceof Error ? e.message : String(e)).trim();
      throw new Error(
        `Couldn't decode this RAW file. The camera or format may not be supported by this build of the decoder${detail ? ` — ${detail}` : ''}.`,
      );
    }
    console.log(`RAW: ${raw.make} ${raw.model} (${raw.width}x${raw.height}, cfa=${raw.cfaWidth}x${raw.cfaStr.length / raw.cfaWidth})`);
    if (!prepared) throw new Error(prepareError ?? 'The CFA of this file could not be prepared.');

    const { pattern, period, dy, dx, cfaType } = prepared.cfa;
    const visWidth = prepared.visibleWidth;
    const visHeight = prepared.visibleHeight;
    console.log(`CFA: ${cfaType} (period=${period}, shift=dy${dy} dx${dx})`);
    console.log(`WP calibration: metadata=[${Array.from(raw.whiteLevels)}] calibrated=[${Array.from(prepared.whiteLevels)}] black=[${Array.from(raw.blackLevels)}]`);

    // 6. WB coefficients (normalize to G=1, applied post-demosaic on GPU)
    const wb = new Float32Array([
      raw.wbCoeffs[0] / raw.wbCoeffs[1],
      1.0,
      raw.wbCoeffs[2] / raw.wbCoeffs[1],
    ]);

    // 7. Per-channel clip thresholds (all 0.96 after per-colour normalization)
    const clipNorm = channelClips();
    // WB-scaled clips for GPU postprocessor (HL recovery operates on WB'd data)
    const clipsWb: [number, number, number] = [
      clipNorm[0] * wb[0], clipNorm[1] * wb[1], clipNorm[2] * wb[2],
    ];

    // 9. Demosaic through the strategy for the requested method
    const startTime = Date.now();
    // Label the result with the models actually loaded, not with the requested size.
    const modelSize = ctx.models.size;
    const input: DemosaicInput = {
      cfa: prepared.data,
      lut: prepared.lut,
      width: prepared.width,
      height: prepared.height,
      padTop: prepared.padTop,
      padLeft: prepared.padLeft,
      visibleWidth: visWidth,
      visibleHeight: visHeight,
      pattern: flattenPattern(pattern, period),
      period,
      cfaType,
      masks: makeChannelMasks(PATCH_SIZE, pattern, period),
      clipNorm,
    };
    prepared = null;
    const demosaiced = await strategyFor(options.method).run(input, ctx, options);
    // The GPU methods only submit work. Wait for it (the neural net's finalize and the GPU
    // methods' readback used to), so the reported time covers the demosaic, as before.
    if (!(demosaiced.rgb instanceof Float32Array)) {
      try {
        await ctx.device.queue.onSubmittedWorkDone();
      } catch (error) {
        demosaiced.rgb.buffer.destroy();
        throw error;
      }
    }

    // 10. GPU postprocess: WB → highlight recovery → CC → DR → RGBA (+ clip ratio in alpha)
    // If matrix construction fails before transfer, release the strategy output.
    let ccMatrix: Float32Array | null;
    try {
      ccMatrix = raw.xyzToCam ? buildColorMatrix(raw.xyzToCam) : null;
    } catch (error) {
      if (!(demosaiced.rgb instanceof Float32Array)) demosaiced.rgb.buffer.destroy();
      throw error;
    }
    // gpuPostprocess consumes a GPU input on both success and failure.
    const gpuResult = await gpuPostprocess(
      ctx.device, demosaiced.rgb, visWidth, visHeight,
      wb, clipsWb, ccMatrix, raw.drGain,
    );
    ownedOutput = { texture: gpuResult.texture, width: visWidth, height: visHeight };
    const inferenceTime = (Date.now() - startTime) / 1000;

    // 11. Compute final display dimensions (after orientation)
    const orientation = raw.orientation;
    const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
    const finalWidth = swap ? visHeight : visWidth;
    const finalHeight = swap ? visWidth : visHeight;

    // 12. Estimate illuminant color temperature and tint from WB + color matrix
    const { temp: colorTemp, tint } = estimateColorTemperature(wb, raw.camToXyz);

    const meta: ProcessingResultMeta = {
      exportData: {
        width: visWidth,
        height: visHeight,
        xyzToCam: null,  // CC already applied on GPU
        wbCoeffs: wb,
        camToXyz: raw.camToXyz,
        orientation,
      },
      metadata: {
        make: raw.make,
        model: raw.model,
        width: finalWidth,
        height: finalHeight,
        tileCount: demosaiced.tileCount,
        inferenceTime,
        backend: options.method === 'neural-net' ? (ctx.models.backend ?? 'unknown') : options.method,
        exposureBias: raw.exposureBias,
        lensModel: raw.lensModel,
        focalLength: raw.focalLength,
        fNumber: raw.fNumber,
        colorTemp,
        tint,
        modelSize: options.method === 'neural-net' ? modelSize : undefined,
      },
    };

    const gpu: GpuImage = { texture: gpuResult.texture, width: visWidth, height: visHeight };
    const image = { gpu, meta, dispose: () => gpu.texture.destroy() };
    ownedOutput = null; // transfer to the caller only after metadata construction succeeds
    return image;
  } finally {
    ownedOutput?.texture.destroy();
    destroyDemosaicPool();
  }
}
