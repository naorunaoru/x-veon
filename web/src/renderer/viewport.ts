import { U_VIEW_X, U_VIEW_Y } from './uniforms';

export interface DisplayViewport {
  /** Host size and pan in CSS pixels; scale is CSS pixels per image pixel. */
  width: number;
  height: number;
  dpr: number;
  scale: number;
  offsetX: number;
  offsetY: number;
  orientation: string;
}

/** Resize only at draw time and map the original texture's UVs into clip space. */
export function applyViewport(
  canvas: HTMLCanvasElement, data: Float32Array, view: DisplayViewport | null,
  imageWidth: number, imageHeight: number, maxDimension: number,
): void {
  if (!view) {
    data.set([2, 0, -1, 0], U_VIEW_X);
    data.set([0, -2, 1, 0], U_VIEW_Y);
    return;
  }
  const width = Math.max(1, view.width);
  const height = Math.max(1, view.height);
  const density = Math.min(view.dpr, maxDimension / width, maxDimension / height);
  const pixelWidth = Math.max(1, Math.round(width * density));
  const pixelHeight = Math.max(1, Math.round(height * density));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;

  const w = imageWidth * view.scale;
  const h = imageHeight * view.scale;
  let x = [w, 0, view.offsetX];
  let y = [0, h, view.offsetY];
  switch (view.orientation) {
    case 'Rotate90': x = [0, -h, view.offsetX + h]; y = [w, 0, view.offsetY]; break;
    case 'Rotate180': x = [-w, 0, view.offsetX + w]; y = [0, -h, view.offsetY + h]; break;
    case 'Rotate270': x = [0, h, view.offsetX]; y = [-w, 0, view.offsetY + w]; break;
  }
  data.set([2 * x[0] / width, 2 * x[1] / width, 2 * x[2] / width - 1, view.scale <= 1 ? 1 : 0], U_VIEW_X);
  data.set([-2 * y[0] / height, -2 * y[1] / height, 1 - 2 * y[2] / height, 0], U_VIEW_Y);
}
