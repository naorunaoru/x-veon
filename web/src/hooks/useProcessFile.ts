import { useCallback, useRef, useState } from 'react';
import { useAppStore } from '@/store';
import { decodeRaw } from '@/pipeline/raf-decoder';
import {
  cropToVisible,
  findPatternShift,
  normalizeRawCfa,
  channelClips,
  applyWhiteBalance,
  padToAlignment,
  generateTiles,
  makeChannelMasks,
  prefillBatchMasks,
  fillBatchCfa,
} from '@/pipeline/preprocessor';
import { runBatch, getBackend } from '@/pipeline/inference';
import { runDemosaic, destroyDemosaicPool } from '@/pipeline/demosaic';
import { createTileBlender, cropToHWC, buildColorMatrix, applyColorCorrection } from '@/pipeline/postprocessor';
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

      // 3. Normalize
      let cfa: Float32Array | null = normalizeRawCfa(
        visible.data, visWidth, visHeight, raw.blackLevels, raw.whiteLevels,
      );
      visible = null!;

      // 4. WB coefficients (normalize to G=1)
      const wb = new Float32Array([
        raw.wbCoeffs[0] / raw.wbCoeffs[1],
        1.0,
        raw.wbCoeffs[2] / raw.wbCoeffs[1],
      ]);

      // 5. Find CFA pattern shift and type
      const cfaInfo = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);
      const { pattern, period, dy, dx, cfaType } = cfaInfo;
      console.log(`CFA: ${cfaType} (period=${period}, shift=dy${dy} dx${dx})`);

      // 6. Apply white balance
      applyWhiteBalance(cfa, visWidth, visHeight, wb, pattern, period, dy, dx);

      // 7. Per-channel clip thresholds (for clip mask input to model)
      const black = raw.blackLevels[0];
      const range = raw.whiteLevels[0] - black;
      const clipNorm = channelClips(raw.cfaStr, raw.cfaWidth, raw.whiteLevels, black, range);
      const clips: [number, number, number] = [
        clipNorm[0] * wb[0], clipNorm[1] * wb[1], clipNorm[2] * wb[2],
      ];

      // 7b. Compute full-image clip ratio (for overlay visualization)
      // Ratio = cfa / clip_level, smooth 0→1. At 1.0 the pixel is clipped.
      const clipMask = new Float32Array(visWidth * visHeight);
      for (let y = 0; y < visHeight; y++) {
        const patY = ((y + dy) % period + period) % period;
        const row = y * visWidth;
        for (let x = 0; x < visWidth; x++) {
          const ch = pattern[patY][((x + dx) % period + period) % period];
          const val = cfa[row + x];
          const cl = clips[ch];
          clipMask[row + x] = Math.min(val / cl, 1);
        }
      }

      // 8. Pad for alignment
      let padded = padToAlignment(cfa, visWidth, visHeight, dy, dx);
      const padTop = padded.padTop;
      const padLeft = padded.padLeft;
      if (padded.data !== cfa) cfa = null;

      // 9. Demosaic
      const method: DemosaicMethod = useAppStore.getState().demosaicMethod;
      const startTime = Date.now();
      let blended: Float32Array | null;
      let hPad: number;
      let wPad: number;
      let tileCount: number;

      const flatCfa = flattenPattern(pattern, period);

      if (method === 'neural-net') {
        // NN path: tile → inference → incremental blend
        const cfaData = padded.data;
        const cfaW = padded.width;
        const cfaH = padded.height;
        const tileGrid = generateTiles(cfaW, cfaH, PATCH_SIZE, OVERLAP);
        padded = null!; cfa = null;
        hPad = tileGrid.hPad;
        wPad = tileGrid.wPad;
        tileCount = tileGrid.tiles.length;

        const masks = makeChannelMasks(PATCH_SIZE, pattern, period);
        const blender = createTileBlender(hPad, wPad, PATCH_SIZE, OVERLAP);
        const tiles = tileGrid.tiles;
        const tileSize = 5 * PATCH_SIZE * PATCH_SIZE;
        const outSize = 3 * PATCH_SIZE * PATCH_SIZE;

        // getCh for padded CFA (shifts are 0,0 after padToAlignment)
        const getCh = (y: number, x: number) => pattern[y % period][x % period];

        // When disabled, fillBatchCfa fills channel 4 with zeros
        const useClip = useAppStore.getState().useClipMaskInference;
        const inferClips = useClip ? clips : undefined;
        const inferGetCh = useClip ? getCh : undefined;

        // Pre-allocate two batch buffers with masks baked in (double-buffer)
        const bufs = [new Float32Array(TILE_BATCH * tileSize), new Float32Array(TILE_BATCH * tileSize)];
        prefillBatchMasks(bufs[0], masks, TILE_BATCH, PATCH_SIZE);
        prefillBatchMasks(bufs[1], masks, TILE_BATCH, PATCH_SIZE);

        let slot = 0;
        fillBatchCfa(bufs[0], cfaData, cfaW, cfaH, tiles, 0, Math.min(TILE_BATCH, tiles.length), PATCH_SIZE, inferClips, inferGetCh);

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
            fillBatchCfa(bufs[slot], cfaData, cfaW, cfaH, tiles, nextB, nextEnd, PATCH_SIZE, inferClips, inferGetCh);
          }

          const batchOut = await inferPromise;
          for (let i = 0; i < count; i++) {
            blender.accumulate(batchOut.subarray(i * outSize, (i + 1) * outSize), tiles[b + i].x, tiles[b + i].y);
          }
          useAppStore.getState().updateFileProgress(fileId, end, tiles.length);
          b = nextB;
        }

        blended = blender.finalize();
      } else {
        // Traditional demosaic: process full image at once (no tile progress)
        const algorithm = method;
        hPad = padded.height;
        wPad = padded.width;
        tileCount = 1;

        // After padToAlignment, the CFA is shifted to canonical (0,0) alignment
        blended = await runDemosaic(padded.data, padded.width, padded.height, 0, 0, algorithm, flatCfa, period);
        padded = null!; cfa = null;
      }

      const inferenceTime = (Date.now() - startTime) / 1000;

      // 10. Crop to original size (CHW -> HWC)
      const hwc = cropToHWC(blended, hPad, wPad, padTop, padLeft, visHeight, visWidth);
      blended = null;

      // 11. Apply camera → sRGB color correction
      if (raw.xyzToCam) {
        const ccMatrix = buildColorMatrix(raw.xyzToCam);
        applyColorCorrection(hwc, visWidth * visHeight, ccMatrix);
      }

      // 11b. Fuji DR compensation — undo deliberate underexposure
      if (raw.drGain > 1.0) {
        for (let i = 0; i < hwc.length; i++) hwc[i] *= raw.drGain;
      }

      // 12. Compute final display dimensions (after orientation)
      const orientation = raw.orientation;
      const swap = orientation === 'Rotate90' || orientation === 'Rotate270';
      const finalWidth = swap ? visHeight : visWidth;
      const finalHeight = swap ? visWidth : visHeight;

      // 13. Estimate illuminant color temperature and tint from WB + color matrix
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
          xyzToCam: null,  // CC already applied
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
