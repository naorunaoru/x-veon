import { useCallback, useState } from 'react';
import { useAppStore } from '@/store';
import { decodeRaf } from '@/pipeline/raf-decoder';
import {
  cropToVisible,
  findPatternShift,
  normalizeRawCfa,
  reconstructHighlightsCfa,
  applyWhiteBalance,
  padToAlignment,
  generateTiles,
  makeChannelMasks,
  buildTileInput,
} from '@/pipeline/preprocessor';
import { runTile, getBackend } from '@/pipeline/inference';
import {
  blendTiles,
  cropToHWC,
  buildColorMatrix,
  applyColorCorrection,
  applyExifRotation,
  toImageData,
} from '@/pipeline/postprocessor';
import { processHdr } from '@/pipeline/hdr-encoder';
import { PATCH_SIZE, OVERLAP } from '@/pipeline/constants';
import type { ProcessingResult } from '@/pipeline/types';

export function useProcessFile() {
  const [isProcessing, setIsProcessing] = useState(false);

  const processFile = useCallback(async (fileId: string) => {
    const store = useAppStore.getState();
    const fileEntry = store.files.find((f) => f.id === fileId);
    if (!fileEntry || isProcessing) return;

    setIsProcessing(true);
    useAppStore.getState().updateFileStatus(fileId, 'processing');

    try {
      const arrayBuffer = await fileEntry.file.arrayBuffer();

      // 1. Decode RAF
      const raw = decodeRaf(arrayBuffer);
      console.log(`RAF: ${raw.make} ${raw.model} (${raw.width}x${raw.height})`);

      // 2. Crop to visible area
      const visible = cropToVisible(raw.data, raw.width, raw.height, raw.crops);

      // 3. Normalize
      const cfa = normalizeRawCfa(
        visible.data, visible.width, visible.height, raw.blackLevels, raw.whiteLevels,
      );

      // 4. WB coefficients (normalize to G=1)
      const wb = new Float32Array([
        raw.wbCoeffs[0] / raw.wbCoeffs[1],
        1.0,
        raw.wbCoeffs[2] / raw.wbCoeffs[1],
      ]);

      // 5. Find CFA pattern shift
      const { dy, dx } = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);

      // 6. LCh highlight reconstruction
      reconstructHighlightsCfa(cfa, visible.width, visible.height, dy, dx);

      // 7. Apply white balance
      applyWhiteBalance(cfa, visible.width, visible.height, wb, dy, dx);

      // 8. Pad for alignment
      const padded = padToAlignment(cfa, visible.width, visible.height, dy, dx);

      // 9. Tile
      const { tiles, paddedCfa, hPad, wPad } = generateTiles(
        padded.data, padded.width, padded.height, PATCH_SIZE, OVERLAP,
      );

      // 10. Precompute masks
      const masks = makeChannelMasks(PATCH_SIZE);

      // 11. Inference
      const tileOutputs: Float32Array[] = [];
      const startTime = Date.now();

      for (let i = 0; i < tiles.length; i++) {
        const { x, y } = tiles[i];
        const input = buildTileInput(paddedCfa, wPad, x, y, PATCH_SIZE, masks);
        const output = await runTile(input, PATCH_SIZE);
        tileOutputs.push(output);
        useAppStore.getState().updateFileProgress(fileId, i + 1, tiles.length);
      }

      const inferenceTime = (Date.now() - startTime) / 1000;

      // 12. Blend tiles
      const blended = blendTiles(tileOutputs, tiles, hPad, wPad, PATCH_SIZE, OVERLAP);

      // 13. Crop to original size (CHW -> HWC)
      const hwc = cropToHWC(
        blended, hPad, wPad, padded.padTop, padded.padLeft, visible.height, visible.width,
      );

      // 14. Color correction + display
      const xyzToCam3x3 = raw.xyzToCam.length >= 9
        ? raw.xyzToCam.slice(0, 9)
        : null;
      const numPixels = visible.width * visible.height;

      // Clone linear camera RGB for export
      const hwcForExport = hwc.slice();

      const hdrSupported = useAppStore.getState().hdrSupported;
      let imageData: ImageData;
      let finalWidth = visible.width;
      let finalHeight = visible.height;

      if (hdrSupported) {
        imageData = processHdr(hwc, numPixels, xyzToCam3x3, visible.width, visible.height, raw.orientation);
        finalWidth = imageData.width;
        finalHeight = imageData.height;
      } else {
        if (xyzToCam3x3) {
          const colorMatrix = buildColorMatrix(xyzToCam3x3);
          applyColorCorrection(hwc, numPixels, colorMatrix);
        }
        const rotated = applyExifRotation(hwc, visible.width, visible.height, raw.orientation);
        imageData = toImageData(rotated.data, rotated.width, rotated.height);
        finalWidth = rotated.width;
        finalHeight = rotated.height;
      }

      const result: ProcessingResult = {
        imageData,
        exportData: {
          hwc: hwcForExport,
          width: visible.width,
          height: visible.height,
          xyzToCam: xyzToCam3x3,
          orientation: raw.orientation,
        },
        isHdr: hdrSupported,
        metadata: {
          make: raw.make,
          model: raw.model,
          width: finalWidth,
          height: finalHeight,
          tileCount: tiles.length,
          inferenceTime,
          backend: getBackend() ?? 'unknown',
        },
      };

      useAppStore.getState().setFileResult(fileId, result);
    } catch (e) {
      useAppStore.getState().updateFileStatus(fileId, 'error', (e as Error).message);
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  return { processFile, isProcessing };
}
