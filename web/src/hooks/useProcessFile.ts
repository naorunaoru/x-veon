import { useCallback, useRef, useState } from 'react';
import { useAppStore } from '@/store';
import { decodeRaw } from '@/pipeline/raf-decoder';
import type { RawImage } from '@/pipeline/types';
import {
  cropToVisible,
  findPatternShift,
  calibrateWhiteLevels,
  normalizeRawCfa,
  channelClips,
  padToAlignment,
  generateTiles,
  makeChannelMasks,
} from '@/pipeline/preprocessor';
import { runBatchGpu, getBackend, getInferenceDevice } from '@/pipeline/inference';
import { runDemosaic, destroyDemosaicPool } from '@/pipeline/demosaic';
import { cropToHWC, buildColorMatrix } from '@/pipeline/postprocessor';
import { createGpuNNPipeline } from '@/pipeline/tile-blend-gpu';
import { gpuPostprocess } from '@/pipeline/postprocess-gpu';
import { getDevice } from '@/gl/renderer';
import { PATCH_SIZE, OVERLAP, TILE_BATCH } from '@/pipeline/constants';
import type { DemosaicMethod, ProcessingResultMeta } from '@/pipeline/types';
import { estimateColorTemperature } from '@/pipeline/color-temperature';
import { readRaw } from '@/lib/opfs-storage';
import { setGpuResult } from '@/lib/hwc-handoff';

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


export function useProcessFile() {
  const [isProcessing, setIsProcessing] = useState(false);
  const lockRef = useRef(false);

  const processFile = useCallback(async (fileId: string) => {
    if (lockRef.current) return;
    const store = useAppStore.getState();
    const fileEntry = store.files.find((f) => f.id === fileId);
    if (!fileEntry) return;

    lockRef.current = true;
    setIsProcessing(true);
    useAppStore.getState().updateFileStatus(fileId, 'processing');

    try {
      // 1. Decode RAW (File object for fresh drops, OPFS for restored sessions)
      let arrayBuffer: ArrayBuffer | null;
      if (fileEntry.file) {
        arrayBuffer = await fileEntry.file.arrayBuffer();
      } else {
        const raw = await readRaw(fileEntry.id);
        if (!raw) throw new Error('RAW file not found in storage. Please re-add this file.');
        arrayBuffer = raw;
      }
      let raw: RawImage;
      try {
        raw = decodeRaw(arrayBuffer);
      } catch (e) {
        const detail = (e instanceof Error ? e.message : String(e)).trim();
        throw new Error(
          `Couldn't decode this RAW file. The camera or format may not be supported by this build of the decoder${detail ? ` — ${detail}` : ''}.`,
        );
      }
      arrayBuffer = null;
      console.log(`RAW: ${raw.make} ${raw.model} (${raw.width}x${raw.height}, cfa=${raw.cfaWidth}x${raw.cfaStr.length / raw.cfaWidth})`);

      // 2. Crop to visible area
      let visible = cropToVisible(raw.data, raw.width, raw.height, raw.crops);
      const visWidth = visible.width;
      const visHeight = visible.height;

      // 3. White-point calibration: detect actual sensor saturation
      const whiteLevels = calibrateWhiteLevels(
        visible.data, visWidth, visHeight, raw.whiteLevels,
      );
      console.log(`WP calibration: metadata=[${Array.from(raw.whiteLevels)}] calibrated=[${Array.from(whiteLevels)}] black=[${Array.from(raw.blackLevels)}]`);

      // 4. Normalize (no WB — model trained on raw CFA data)
      let cfa: Float32Array | null = normalizeRawCfa(
        visible.data, visWidth, visHeight, raw.blackLevels, whiteLevels,
      );
      visible = null!;

      // 5. WB coefficients (normalize to G=1, applied post-demosaic on GPU)
      const wb = new Float32Array([
        raw.wbCoeffs[0] / raw.wbCoeffs[1],
        1.0,
        raw.wbCoeffs[2] / raw.wbCoeffs[1],
      ]);

      // 6. Find CFA pattern shift and type
      const cfaInfo = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);
      const { pattern, period, dy, dx, cfaType } = cfaInfo;
      console.log(`CFA: ${cfaType} (period=${period}, shift=dy${dy} dx${dx})`);

      // 7. Per-channel clip thresholds (all 0.987 after per-CFA-position normalization)
      const clipNorm = channelClips();
      // WB-scaled clips for GPU postprocessor (HL recovery operates on WB'd data)
      const clipsWb: [number, number, number] = [
        clipNorm[0] * wb[0], clipNorm[1] * wb[1], clipNorm[2] * wb[2],
      ];

      // 8. Pad for alignment
      const method: DemosaicMethod = useAppStore.getState().demosaicMethod;
      let padded = padToAlignment(cfa, visWidth, visHeight, dy, dx);
      const padTop = padded.padTop;
      const padLeft = padded.padLeft;
      if (padded.data !== cfa) cfa = null;

      // 9. Demosaic
      const startTime = Date.now();
      let gpuResult: import('@/pipeline/postprocess-gpu').PostprocessResult;
      let hPad: number;
      let wPad: number;
      let tileCount: number;

      const flatCfa = flattenPattern(pattern, period);

      if (method === 'neural-net') {
        // Fully GPU-resident NN path: extract → infer → blend, all on GPU
        const cfaData = padded.data;
        const cfaW = padded.width;
        const cfaH = padded.height;
        const tileGrid = generateTiles(cfaW, cfaH, PATCH_SIZE, OVERLAP);
        hPad = tileGrid.hPad;
        wPad = tileGrid.wPad;
        tileCount = tileGrid.tiles.length;
        const tiles = tileGrid.tiles;

        const masks = makeChannelMasks(PATCH_SIZE, pattern, period);
        const device = getInferenceDevice() ?? await getDevice();
        const gpu = createGpuNNPipeline(
          device, cfaData, cfaW, cfaH,
          masks, clipNorm as [number, number, number], tiles,
          hPad, wPad, PATCH_SIZE, OVERLAP,
          padTop, padLeft, visHeight, visWidth, TILE_BATCH,
        );
        padded = null!; cfa = null;

        // Process batches — everything stays on GPU
        for (let b = 0; b < tiles.length; ) {
          const end = Math.min(b + TILE_BATCH, tiles.length);
          const count = end - b;

          // GPU: extract tiles from CFA → 5ch NCHW buffer
          const inputBuf = gpu.extractBatch(b, count);

          // GPU: inference (GPU buffer in → GPU buffer out)
          const { buffer: inferBuf, dispose } = await runBatchGpu(
            cfaType, inputBuf, count, PATCH_SIZE,
          );

          // GPU: accumulate inference output into blend buffer
          gpu.accumulateBatch(inferBuf, b, count);
          dispose();

          b = end;
        }

        // Finalize+crop on GPU → GPUBuffer passed directly to postprocess
        const hwcBuf = await gpu.finalize();

        const ccMatrix = raw.xyzToCam ? buildColorMatrix(raw.xyzToCam) : null;
        gpuResult = await gpuPostprocess(
          device, hwcBuf, visWidth, visHeight,
          wb, clipsWb, ccMatrix, raw.drGain,
        );
      } else {
        // Traditional demosaic: process full image at once (no tile progress)
        const algorithm = method;
        hPad = padded.height;
        wPad = padded.width;
        tileCount = 1;

        // After padToAlignment, the CFA is shifted to canonical (0,0) alignment
        const blended = await runDemosaic(padded.data, padded.width, padded.height, 0, 0, algorithm, flatCfa, period);
        padded = null!; cfa = null;

        // Crop to original size (HWC, raw demosaic output, no WB)
        const rawHwc = cropToHWC(blended, hPad, wPad, padTop, padLeft, visHeight, visWidth);

        // GPU postprocess: WB → highlight recovery → CC → DR → clip mask
        const ccMatrix = raw.xyzToCam ? buildColorMatrix(raw.xyzToCam) : null;
        const device = await getDevice();
        gpuResult = await gpuPostprocess(
          device, rawHwc, visWidth, visHeight,
          wb, clipsWb, ccMatrix, raw.drGain,
        );
      }

      const inferenceTime = (Date.now() - startTime) / 1000;

      // 11. Compute final display dimensions (after orientation)
      const orientation = raw.orientation;
      const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
      const finalWidth = swap ? visHeight : visWidth;
      const finalHeight = swap ? visWidth : visHeight;

      // 12. Estimate illuminant color temperature and tint from WB + color matrix
      const { temp: colorTemp, tint } = estimateColorTemperature(wb, raw.camToXyz);

      // Hand off GPU buffer for immediate display (zero-copy)
      setGpuResult(fileId, gpuResult);

      const resultMeta: ProcessingResultMeta = {
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
          tileCount,
          inferenceTime,
          backend: method === 'neural-net' ? (getBackend() ?? 'unknown') : method,
          exposureBias: raw.exposureBias,
          lensModel: raw.lensModel,
          focalLength: raw.focalLength,
          fNumber: raw.fNumber,
          colorTemp,
          tint,
          modelSize: method === 'neural-net' ? useAppStore.getState().modelSize : undefined,
        },
      };

      useAppStore.getState().setFileResult(fileId, resultMeta, method);
    } catch (e) {
      const detail = e instanceof Error ? e.message : typeof e === 'string' ? e : String(e);
      const msg = detail.trim() || 'This file could not be opened. It may be corrupt, or the camera or format may be unsupported by this build.';
      useAppStore.getState().updateFileStatus(fileId, 'error', msg);
      console.error(e);
    } finally {
      destroyDemosaicPool();
      lockRef.current = false;
      setIsProcessing(false);
    }
  }, []);

  return { processFile, isProcessing };
}
