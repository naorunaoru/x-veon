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
import { runDemosaic, destroyDemosaicPool } from '@/pipeline/demosaic';
import {
  createTileBlender,
  cropToHWC,
  buildColorMatrix,
  toImageDataWithCC,
} from '@/pipeline/postprocessor';
import { processHdr } from '@/pipeline/hdr-encoder';
import { PATCH_SIZE, OVERLAP } from '@/pipeline/constants';
import type { DemosaicMethod, ProcessingResult } from '@/pipeline/types';

export function useProcessFile() {
  const [isProcessing, setIsProcessing] = useState(false);

  const processFile = useCallback(async (fileId: string) => {
    const store = useAppStore.getState();
    const fileEntry = store.files.find((f) => f.id === fileId);
    if (!fileEntry || isProcessing) return;

    setIsProcessing(true);
    useAppStore.getState().updateFileStatus(fileId, 'processing');

    try {
      // 1. Decode RAF
      let arrayBuffer: ArrayBuffer | null = await fileEntry.file.arrayBuffer();
      const raw = decodeRaf(arrayBuffer);
      arrayBuffer = null;
      console.log(`RAF: ${raw.make} ${raw.model} (${raw.width}x${raw.height})`);

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

      // 5. Find CFA pattern shift
      const { dy, dx } = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);

      // 6. LCh highlight reconstruction
      reconstructHighlightsCfa(cfa, visWidth, visHeight, dy, dx);

      // 7. Apply white balance
      applyWhiteBalance(cfa, visWidth, visHeight, wb, dy, dx);

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

      if (method === 'neural-net') {
        // NN path: tile → inference → incremental blend
        const tileGrid = generateTiles(
          padded.data, padded.width, padded.height, PATCH_SIZE, OVERLAP,
        );
        padded = null!; cfa = null;
        hPad = tileGrid.hPad;
        wPad = tileGrid.wPad;
        tileCount = tileGrid.tiles.length;

        const masks = makeChannelMasks(PATCH_SIZE);
        const blender = createTileBlender(hPad, wPad, PATCH_SIZE, OVERLAP);

        for (let i = 0; i < tileGrid.tiles.length; i++) {
          const { x, y } = tileGrid.tiles[i];
          const input = buildTileInput(tileGrid.paddedCfa, wPad, x, y, PATCH_SIZE, masks);
          const output = await runTile(input, PATCH_SIZE);
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

        blended = await runDemosaic(padded.data, padded.width, padded.height, dy, dx, algorithm);
        padded = null!; cfa = null;
      }

      const inferenceTime = (Date.now() - startTime) / 1000;

      // 10. Crop to original size (CHW -> HWC)
      const hwc = cropToHWC(blended, hPad, wPad, padTop, padLeft, visHeight, visWidth);
      blended = null;

      // 11. Color correction + display (fused — hwc is never mutated)
      const xyzToCam3x3 = raw.xyzToCam.length >= 9
        ? raw.xyzToCam.slice(0, 9)
        : null;
      const numPixels = visWidth * visHeight;
      const hdrSupported = useAppStore.getState().hdrSupported;
      let imageData: ImageData;
      let finalWidth: number;
      let finalHeight: number;

      if (hdrSupported) {
        imageData = processHdr(hwc, numPixels, xyzToCam3x3, visWidth, visHeight, raw.orientation);
        finalWidth = imageData.width;
        finalHeight = imageData.height;
      } else {
        const colorMatrix = xyzToCam3x3 ? buildColorMatrix(xyzToCam3x3) : null;
        imageData = toImageDataWithCC(hwc, visWidth, visHeight, colorMatrix, raw.orientation);
        finalWidth = imageData.width;
        finalHeight = imageData.height;
      }

      const result: ProcessingResult = {
        imageData,
        exportData: {
          hwc,
          width: visWidth,
          height: visHeight,
          xyzToCam: xyzToCam3x3,
          orientation: raw.orientation,
        },
        isHdr: hdrSupported,
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

      useAppStore.getState().setFileResult(fileId, result);
    } catch (e) {
      useAppStore.getState().updateFileStatus(fileId, 'error', (e as Error).message);
      console.error(e);
    } finally {
      destroyDemosaicPool();
      setIsProcessing(false);
    }
  }, [isProcessing]);

  return { processFile, isProcessing };
}
