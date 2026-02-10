import { isWebCodecsAvifSupported, encodeAvifWebCodecs } from './encode-avif-webcodecs.js';

let worker = null;
let webCodecsAvif = null; // null = untested, true/false = cached result

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./encoder-worker.js', import.meta.url), { type: 'module' });
  }
  return worker;
}

/**
 * Encode via Web Worker (WASM path). Clones buffers for transfer
 * so currentExport stays intact for re-exports in different formats.
 */
function encodeViaWorker(hwc, width, height, xyzToCam, orientation, format, quality) {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const hwcCopy = hwc.slice();
    const xyzToCamCopy = xyzToCam.slice();

    w.onmessage = (e) => {
      if (e.data.type === 'done') {
        resolve(new Uint8Array(e.data.data));
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message));
      }
    };
    w.onerror = (e) => reject(new Error(e.message));

    w.postMessage({
      type: 'encode',
      hwc: hwcCopy.buffer,
      width, height,
      xyzToCam: xyzToCamCopy.buffer,
      orientation, format, quality,
    }, [hwcCopy.buffer, xyzToCamCopy.buffer]);
  });
}

/**
 * Encode linear camera RGB to the specified format.
 *
 * For AVIF: tries WebCodecs (native/hardware AV1) first, capturing directly
 * from the HDR canvas. Falls back to WASM rav1e in a Web Worker.
 *
 * For JPEG/TIFF: runs WASM encoder in a Web Worker.
 *
 * @param {Float32Array} hwc - Linear camera RGB in HWC layout (not mutated)
 * @param {number} width
 * @param {number} height
 * @param {Float32Array} xyzToCam - 9-element XYZ->Camera matrix (row-major)
 * @param {string} orientation - EXIF orientation string
 * @param {string} format - "avif", "jpeg", or "tiff"
 * @param {number} quality - 1-100
 * @returns {Promise<{blob: Blob, ext: string}>}
 */
export async function encodeImage(hwc, width, height, xyzToCam, orientation, format, quality) {
  // AVIF: try WebCodecs first (captures from the already-rendered HDR canvas)
  if (format === 'avif') {
    const canvas = document.getElementById('output');
    if (webCodecsAvif === null) {
      webCodecsAvif = await isWebCodecsAvifSupported(canvas.width, canvas.height);
      console.log(`WebCodecs AV1: ${webCodecsAvif ? 'supported' : 'not available, using rav1e WASM'}`);
    }
    if (webCodecsAvif) {
      const blob = await encodeAvifWebCodecs(canvas, quality);
      return { blob, ext: 'avif' };
    }
  }

  // JPEG, TIFF, or AVIF fallback — encode in Web Worker
  const encoded = await encodeViaWorker(hwc, width, height, xyzToCam, orientation, format, quality);

  const mimeTypes = { avif: 'image/avif', jpeg: 'image/jpeg', tiff: 'image/tiff' };
  const extensions = { avif: 'avif', jpeg: 'jpg', tiff: 'tif' };

  const blob = new Blob([encoded], { type: mimeTypes[format] });
  return { blob, ext: extensions[format] };
}
