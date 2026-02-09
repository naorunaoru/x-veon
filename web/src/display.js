const progressContainer = () => document.getElementById('progress-container');
const progressBar = () => document.getElementById('progress-bar');
const progressText = () => document.getElementById('progress-text');
const statusEl = () => document.getElementById('status');
const canvasEl = () => document.getElementById('output');
const downloadEl = () => document.getElementById('download-link');

let hdrMode = false;

/**
 * Initialize the canvas context. Tries rec2100-hlg for HDR display;
 * falls back to sRGB if unsupported.
 */
export function initCanvas() {
  const canvas = canvasEl();
  const ctx = canvas.getContext('2d', { colorSpace: 'rec2100-hlg' });
  const attrs = ctx.getContextAttributes?.();
  hdrMode = attrs?.colorSpace === 'rec2100-hlg';
  console.log(`Canvas HDR: ${hdrMode ? 'rec2100-hlg' : 'srgb (fallback)'}`);
}

export function isHdrSupported() {
  return hdrMode;
}

/**
 * Render ImageData to the output canvas.
 * Works with both sRGB and rec2100-hlg ImageData.
 */
export function renderToCanvas(imageData) {
  const canvas = canvasEl();
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  // getContext returns the same context initialized by initCanvas
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  canvas.classList.add('visible');
}

/**
 * Export the current canvas content as an AVIF blob.
 */
export function exportCanvasAsAvif() {
  return new Promise((resolve, reject) => {
    canvasEl().toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error('toBlob returned null')),
      'image/avif',
      0.9,
    );
  });
}

/**
 * Show/hide the progress bar.
 */
export function showProgress(visible) {
  progressContainer().hidden = !visible;
  if (!visible) {
    progressBar().style.width = '0%';
    progressText().textContent = '';
  }
}

/**
 * Update the progress bar.
 *
 * @param {number} current - Current tile index (1-based)
 * @param {number} total - Total tile count
 * @param {number} startTime - Date.now() when processing started
 */
export function updateProgress(current, total, startTime) {
  const pct = (current / total) * 100;
  progressBar().style.width = `${pct.toFixed(1)}%`;

  const elapsed = (Date.now() - startTime) / 1000;
  const rate = current / elapsed;
  const remaining = total > current ? ((total - current) / rate) : 0;

  progressText().textContent =
    `${current}/${total} tiles (${remaining.toFixed(0)}s remaining)`;
}

/**
 * Set the status message.
 */
export function setStatus(msg) {
  statusEl().textContent = msg;
}

/**
 * Show the HDR AVIF download button with a blob URL.
 */
export function showDownloadButton(blob, filename) {
  const link = downloadEl();
  if (link.href && link.href.startsWith('blob:')) {
    URL.revokeObjectURL(link.href);
  }
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.hidden = false;
}

/**
 * Hide the download button (e.g. when starting a new file).
 */
export function hideDownloadButton() {
  const link = downloadEl();
  if (link.href && link.href.startsWith('blob:')) {
    URL.revokeObjectURL(link.href);
  }
  link.removeAttribute('href');
  link.hidden = true;
}
