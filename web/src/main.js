import { initWasm, decodeRaf } from './raf-decoder.js';
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
} from './preprocessor.js';
import { initModel, runTile, getBackend } from './inference.js';
import {
  blendTiles,
  cropToHWC,
  buildColorMatrix,
  applyColorCorrection,
  applyExifRotation,
  toImageData,
} from './postprocessor.js';
import {
  initCanvas, isHdrSupported,
  renderToCanvas, exportCanvas,
  showProgress, updateProgress, setStatus,
  showDownloadButton, hideDownloadButton,
} from './display.js';
import { processHdr } from './hdr-encoder.js';
import { PATCH_SIZE, OVERLAP } from './constants.js';

let ready = false;

async function init() {
  setStatus('Loading WASM decoder and ONNX model\u2026');

  initCanvas();

  try {
    await Promise.all([
      initWasm(),
      initModel('./model.onnx'),
    ]);
  } catch (e) {
    setStatus(`Init failed: ${e.message}`);
    console.error(e);
    return;
  }

  ready = true;
  const backend = getBackend();
  const hdr = isHdrSupported() ? ', HDR' : '';
  setStatus(`Ready. Drop a RAF file. (inference: ${backend}${hdr})`);
}

async function processFile(arrayBuffer, baseName) {
  if (!ready) return;
  ready = false; // prevent concurrent runs

  hideDownloadButton();

  try {
    // 1. Decode RAF
    setStatus('Decoding RAF\u2026');
    const raw = decodeRaf(arrayBuffer);
    console.log(`RAF: ${raw.make} ${raw.model} (${raw.width}x${raw.height})`);
    console.log(`  WB: [${Array.from(raw.wbCoeffs).map(v => v.toFixed(3)).join(', ')}]`);
    console.log(`  Black: ${raw.blackLevels[0]}, White: ${raw.whiteLevels[0]}`);
    console.log(`  Crops: [${Array.from(raw.crops).join(', ')}] (top, right, bottom, left)`);

    // 2. Crop to visible area (rawloader returns full sensor incl. optical black borders)
    const visible = cropToVisible(raw.data, raw.width, raw.height, raw.crops);
    console.log(`  Visible: ${visible.width}x${visible.height}`);

    // 3. Normalize
    setStatus('Normalizing CFA\u2026');
    const cfa = normalizeRawCfa(
      visible.data, visible.width, visible.height, raw.blackLevels, raw.whiteLevels
    );

    // 4. WB coefficients (normalize to G=1)
    const wb = new Float32Array([
      raw.wbCoeffs[0] / raw.wbCoeffs[1],
      1.0,
      raw.wbCoeffs[2] / raw.wbCoeffs[1],
    ]);

    // 5. Find CFA pattern shift from rawloader's metadata + crop offset.
    // This matches the visible CFA against our reference X-Trans pattern.
    const { dy, dx } = findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops);
    console.log(`  CFA: ${raw.cfaStr} (${raw.cfaWidth}x${raw.cfaStr.length / raw.cfaWidth})`);
    console.log(`  Pattern shift: dy=${dy}, dx=${dx}`);

    // 6. LCh highlight reconstruction at CFA level (before WB, clip = 1.0)
    setStatus('Reconstructing highlights\u2026');
    reconstructHighlightsCfa(cfa, visible.width, visible.height, dy, dx);

    // 7. Apply white balance
    applyWhiteBalance(cfa, visible.width, visible.height, wb, dy, dx);

    // 8. Pad for alignment
    const padded = padToAlignment(cfa, visible.width, visible.height, dy, dx);

    // 9. Tile
    const { tiles, paddedCfa, hPad, wPad } = generateTiles(
      padded.data, padded.width, padded.height, PATCH_SIZE, OVERLAP
    );
    console.log(`  Tiles: ${tiles.length} (${PATCH_SIZE}px, overlap ${OVERLAP}px)`);

    // 10. Precompute masks
    const masks = makeChannelMasks(PATCH_SIZE);

    // 11. Inference
    setStatus(`Running inference (${tiles.length} tiles)\u2026`);
    showProgress(true);
    const tileOutputs = [];
    const startTime = Date.now();

    for (let i = 0; i < tiles.length; i++) {
      const { x, y } = tiles[i];
      const input = buildTileInput(paddedCfa, wPad, x, y, PATCH_SIZE, masks);
      const output = await runTile(input, PATCH_SIZE);
      tileOutputs.push(output);
      updateProgress(i + 1, tiles.length, startTime);
    }

    const inferTime = ((Date.now() - startTime) / 1000).toFixed(1);
    showProgress(false);

    // 12. Blend tiles
    setStatus('Blending tiles\u2026');
    const blended = blendTiles(tileOutputs, tiles, hPad, wPad, PATCH_SIZE, OVERLAP);

    // 13. Crop to original size (also converts CHW -> HWC)
    const hwc = cropToHWC(
      blended, hPad, wPad, padded.padTop, padded.padLeft, visible.height, visible.width
    );

    // 14. Color correction + display
    const xyzToCam3x3 = raw.xyzToCam.length >= 9
      ? raw.xyzToCam.slice(0, 9)
      : null;
    const numPixels = visible.width * visible.height;

    if (isHdrSupported()) {
      // HDR path: BT.2020 color correction → HLG → HDR canvas
      setStatus('Applying HDR color correction\u2026');
      const hdrImageData = processHdr(hwc, numPixels, xyzToCam3x3, visible.width, visible.height, raw.orientation);
      renderToCanvas(hdrImageData);

      const statusBase =
        `${raw.make} ${raw.model} \u2014 ${hdrImageData.width}\u00d7${hdrImageData.height} \u2014 ` +
        `${tiles.length} tiles in ${inferTime}s (${getBackend()})`;
      setStatus(`${statusBase} \u2014 Encoding HDR AVIF\u2026`);

      try {
        const { blob, ext } = await exportCanvas();
        const filename = `${baseName || raw.model || 'photo'}_hdr.${ext}`;
        showDownloadButton(blob, filename);
        setStatus(statusBase);
      } catch (err) {
        console.error('AVIF export failed:', err);
        setStatus(statusBase);
      }
    } else {
      // sRGB fallback
      setStatus('Applying color correction\u2026');
      if (xyzToCam3x3) {
        const colorMatrix = buildColorMatrix(xyzToCam3x3);
        applyColorCorrection(hwc, numPixels, colorMatrix);
      }
      const rotated = applyExifRotation(hwc, visible.width, visible.height, raw.orientation);
      const imageData = toImageData(rotated.data, rotated.width, rotated.height);
      renderToCanvas(imageData);

      setStatus(
        `${raw.make} ${raw.model} \u2014 ${rotated.width}\u00d7${rotated.height} \u2014 ` +
        `${tiles.length} tiles in ${inferTime}s (${getBackend()})`
      );
    }
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    console.error(e);
  } finally {
    ready = true;
  }
}

// --- UI wiring ---

function handleFile(file) {
  if (!file) return;
  const name = file.name.toLowerCase();
  if (!name.endsWith('.raf')) {
    setStatus('Please drop a Fujifilm .RAF file.');
    return;
  }
  const baseName = file.name.replace(/\.[^.]+$/, '');
  file.arrayBuffer().then(buf => processFile(buf, baseName));
}

document.addEventListener('DOMContentLoaded', () => {
  init();

  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
  });
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
});
