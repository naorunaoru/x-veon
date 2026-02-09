const progressContainer = () => document.getElementById('progress-container');
const progressBar = () => document.getElementById('progress-bar');
const progressText = () => document.getElementById('progress-text');
const statusEl = () => document.getElementById('status');
const canvasEl = () => document.getElementById('output');

/**
 * Render ImageData to the output canvas.
 */
export function renderToCanvas(imageData) {
  const canvas = canvasEl();
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  canvas.classList.add('visible');
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
