import { useCallback, useRef, useState } from 'react';
import { useAppStore } from '@/store';
import { decodeRaw } from '@/pipeline/raf-decoder';
import {
  cropToVisible,
  findPatternShift,
  normalizeRawCfa,
  channelClips,
  padToAlignment,
  generateTiles,
  makeChannelMasks,
  prefillBatchMasks,
  fillBatchCfa,
} from '@/pipeline/preprocessor';
import { runBatch, getBackend } from '@/pipeline/inference';
import { runDemosaic, destroyDemosaicPool } from '@/pipeline/demosaic';
import { cropToHWC, buildColorMatrix } from '@/pipeline/postprocessor';
import { createGpuTileBlender } from '@/pipeline/tile-blend-gpu';
import { gpuPostprocess } from '@/pipeline/postprocess-gpu';
import { getDevice } from '@/gl/renderer';
import { PATCH_SIZE, OVERLAP, TILE_BATCH } from '@/pipeline/constants';
import type { DemosaicMethod, ProcessingResultMeta } from '@/pipeline/types';
import { estimateColorTemperature } from '@/pipeline/color-temperature';
import { writeHwc, hwcKey, readRaw } from '@/lib/opfs-storage';
import { setHwc, setClipMask } from '@/lib/hwc-handoff';

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
      const raw = decodeRaw(arrayBuffer);
      arrayBuffer = null;
      console.log(`RAW: ${raw.make} ${raw.model} (${raw.width}x${raw.height}, cfa=${raw.cfaWidth}x${raw.cfaStr.length / raw.cfaWidth})`);

      // 2. Crop to visible area
      let visible = cropToVisible(raw.data, raw.width, raw.height, raw.crops);
      const visWidth = visible.width;
      const visHeight = visible.height;

      // 3. Normalize (no WB — model trained on raw CFA data)
      let cfa: Float32Array | null = normalizeRawCfa(
        visible.data, visWidth, visHeight, raw.blackLevels, raw.whiteLevels,
      );
      visible = null!;

      // 4. WB coefficients (normalize to G=1, applied post-demosaic on GPU)
      const wb = new Float32Array([
        raw.wbCoeffs[0] / raw.wbCoeffs[1],
        1.0,
        raw.wbCoeffs[2] / raw.wbCoeffs[1],
      ]);

      // 5. Find CFA pattern shift and type
      const cfaInfo = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);
      const { pattern, period, dy, dx, cfaType } = cfaInfo;
      console.log(`CFA: ${cfaType} (period=${period}, shift=dy${dy} dx${dx})`);

      // 6. Per-channel clip thresholds
      const black = raw.blackLevels[0];
      const range = raw.whiteLevels[0] - black;
      const clipNorm = channelClips(raw.cfaStr, raw.cfaWidth, raw.whiteLevels, black, range);
      // WB-scaled clips for GPU postprocessor (HL recovery operates on WB'd data)
      const clipsWb: [number, number, number] = [
        clipNorm[0] * wb[0], clipNorm[1] * wb[1], clipNorm[2] * wb[2],
      ];

      // 7. Pad for alignment
      const method: DemosaicMethod = useAppStore.getState().demosaicMethod;
      let padded = padToAlignment(cfa, visWidth, visHeight, dy, dx);
      const padTop = padded.padTop;
      const padLeft = padded.padLeft;
      if (padded.data !== cfa) cfa = null;

      // 8. Demosaic
      const startTime = Date.now();
      let hwcResult: Float32Array;
      let clipMaskResult: Float32Array;
      let hPad: number;
      let wPad: number;
      let tileCount: number;

      const flatCfa = flattenPattern(pattern, period);

      if (method === 'neural-net') {
        // NN path: tile → inference → GPU blend (no CPU round-trip)
        const cfaData = padded.data;
        const cfaW = padded.width;
        const cfaH = padded.height;
        const tileGrid = generateTiles(cfaW, cfaH, PATCH_SIZE, OVERLAP);
        padded = null!; cfa = null;
        hPad = tileGrid.hPad;
        wPad = tileGrid.wPad;
        tileCount = tileGrid.tiles.length;

        const masks = makeChannelMasks(PATCH_SIZE, pattern, period);
        const device = await getDevice();
        const gpuBlender = createGpuTileBlender(
          device, hPad, wPad, PATCH_SIZE, OVERLAP,
          padTop, padLeft, visHeight, visWidth, TILE_BATCH,
        );
        const tiles = tileGrid.tiles;
        const tileSize = 5 * PATCH_SIZE * PATCH_SIZE;

        // Pre-allocate two batch buffers with masks baked in (double-buffer)
        const bufs = [new Float32Array(TILE_BATCH * tileSize), new Float32Array(TILE_BATCH * tileSize)];
        prefillBatchMasks(bufs[0], masks, TILE_BATCH, PATCH_SIZE);
        prefillBatchMasks(bufs[1], masks, TILE_BATCH, PATCH_SIZE);

        let slot = 0;
        fillBatchCfa(bufs[0], cfaData, cfaW, cfaH, tiles, 0, Math.min(TILE_BATCH, tiles.length), PATCH_SIZE);

        let b = 0;
        while (b < tiles.length) {
          const end = Math.min(b + TILE_BATCH, tiles.length);
          const count = end - b;
          const cur = bufs[slot];
          const inferPromise = runBatch(cfaType, cur.subarray(0, count * tileSize), count, PATCH_SIZE);

          // Fill next batch in alternate buffer while GPU is busy
          const nextB = end;
          const nextEnd = Math.min(nextB + TILE_BATCH, tiles.length);
          if (nextB < tiles.length) {
            slot ^= 1;
            fillBatchCfa(bufs[slot], cfaData, cfaW, cfaH, tiles, nextB, nextEnd, PATCH_SIZE);
          }

          const batchOut = await inferPromise;
          gpuBlender.accumulateBatch(batchOut, tiles, b, count);
          useAppStore.getState().updateFileProgress(fileId, end, tiles.length);
          b = nextB;
        }

        // Finalize+crop on GPU → GPUBuffer passed directly to postprocess
        const hwcBuf = await gpuBlender.finalize();
        gpuBlender.destroy();

        // Skip CPU cropToHWC — already cropped on GPU
        const ccMatrix = raw.xyzToCam ? buildColorMatrix(raw.xyzToCam) : null;
        const { hwc, clipMask } = await gpuPostprocess(
          device, hwcBuf, visWidth, visHeight,
          wb, clipsWb, ccMatrix, raw.drGain,
        );

        hwcResult = hwc;
        clipMaskResult = clipMask;
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
        const { hwc, clipMask } = await gpuPostprocess(
          device, rawHwc, visWidth, visHeight,
          wb, clipsWb, ccMatrix, raw.drGain,
        );
        hwcResult = hwc;
        clipMaskResult = clipMask;
      }

      const inferenceTime = (Date.now() - startTime) / 1000;
      const hwc = hwcResult;
      const clipMask = clipMaskResult;

      // 11. Compute final display dimensions (after orientation)
      const orientation = raw.orientation;
      const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
      const finalWidth = swap ? visHeight : visWidth;
      const finalHeight = swap ? visWidth : visHeight;

      // 12. Estimate illuminant color temperature and tint from WB + color matrix
      const { temp: colorTemp, tint } = estimateColorTemperature(wb, raw.camToXyz);

      // Hand off for immediate display (avoids OPFS round-trip)
      const key = hwcKey(fileId, method);
      setHwc(key, hwc);
      setClipMask(key, clipMask);

      // Persist NN results to OPFS for session recovery / file revisit
      if (method === 'neural-net') {
        writeHwc(hwcKey(fileId, method), hwc);
      }

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
      const msg = e instanceof Error ? e.message : typeof e === 'string' ? e : String(e);
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
