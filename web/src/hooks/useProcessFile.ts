import { useCallback, useRef, useState } from 'react';
import { useAppStore } from '@/store';
import { decodeRaw } from '@/pipeline/raf-decoder';
import {
  cropToVisible,
  findPatternShift,
  normalizeRawCfa,
  channelClips,
  reconstructHighlightsCfa,
  applyWhiteBalance,
  padToAlignment,
  generateTiles,
  makeChannelMasks,
  buildTileInput,
} from '@/pipeline/preprocessor';
import { reconstructHighlightsSegmented } from '@/pipeline/highlight-segments';
import { runTile, getBackend } from '@/pipeline/inference';
import { runDemosaic, destroyDemosaicPool } from '@/pipeline/demosaic';
import { createTileBlender, cropToHWC, buildColorMatrix, applyColorCorrection } from '@/pipeline/postprocessor';
import { PATCH_SIZE, OVERLAP } from '@/pipeline/constants';
import type { DemosaicMethod, ProcessingResultMeta } from '@/pipeline/types';
import { writeHwc } from '@/lib/opfs-storage';

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
      // 1. Decode RAW
      let arrayBuffer: ArrayBuffer | null = await fileEntry.file.arrayBuffer();
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

      // 6. Apply white balance (before HL reconstruction so channels are balanced)
      applyWhiteBalance(cfa, visWidth, visHeight, wb, pattern, period, dy, dx);

      // 7. Highlight reconstruction: opposed inpainting then segmentation
      // Per-channel clip in normalized space: (whiteLevel[c] - black) / range,
      // then multiplied by WB. When all white levels are the same this equals
      // wb[c]; when they differ (common on Sony/Canon) it gives the actual
      // per-channel saturation.
      const black = raw.blackLevels[0];
      const range = raw.whiteLevels[0] - black;
      const clipNorm = channelClips(raw.cfaStr, raw.cfaWidth, raw.whiteLevels, black, range);
      const clips: [number, number, number] = [
        clipNorm[0] * wb[0], clipNorm[1] * wb[1], clipNorm[2] * wb[2],
      ];
      const originalCfa = new Float32Array(cfa);
      reconstructHighlightsCfa(cfa, visWidth, visHeight, pattern, period, dy, dx, clips);
      reconstructHighlightsSegmented(cfa, visWidth, visHeight, pattern, period, dy, dx, 2, 0.5, originalCfa, clips);

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
        const tileGrid = generateTiles(
          padded.data, padded.width, padded.height, PATCH_SIZE, OVERLAP,
        );
        padded = null!; cfa = null;
        hPad = tileGrid.hPad;
        wPad = tileGrid.wPad;
        tileCount = tileGrid.tiles.length;

        const masks = makeChannelMasks(PATCH_SIZE, pattern, period);
        const blender = createTileBlender(hPad, wPad, PATCH_SIZE, OVERLAP);

        for (let i = 0; i < tileGrid.tiles.length; i++) {
          const { x, y } = tileGrid.tiles[i];
          const input = buildTileInput(tileGrid.paddedCfa, wPad, x, y, PATCH_SIZE, masks);
          const output = await runTile(cfaType, input, PATCH_SIZE);
          blender.accumulate(output, x, y);
          useAppStore.getState().updateFileProgress(fileId, i + 1, tileGrid.tiles.length);
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

      // Persist large hwc buffer to OPFS (off JS heap)
      await writeHwc(fileId, hwc);

      const resultMeta: ProcessingResultMeta = {
        exportData: {
          width: visWidth,
          height: visHeight,
          xyzToCam: null,  // CC already applied
          wbCoeffs: wb,
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
        },
      };

      useAppStore.getState().setFileResult(fileId, resultMeta);
    } catch (e) {
      useAppStore.getState().updateFileStatus(fileId, 'error', (e as Error).message);
      console.error(e);
    } finally {
      destroyDemosaicPool();
      lockRef.current = false;
      setIsProcessing(false);
    }
  }, []);

  return { processFile, isProcessing };
}
