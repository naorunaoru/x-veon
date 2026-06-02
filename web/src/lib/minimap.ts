export interface MinimapInput {
  scale: number;
  panX: number;
  panY: number;
  contentW: number;
  contentH: number;
  containerW: number;
  containerH: number;
  boxW: number;
  boxH: number;
}

export interface MinimapLayout {
  /** Scale from image px → minimap box px. */
  k: number;
  /** Displayed image box within the minimap (letterboxed, centered). */
  imgX: number; imgY: number; imgW: number; imgH: number;
  /** Viewport rectangle within the minimap (clamped to the displayed image). */
  rectX: number; rectY: number; rectW: number; rectH: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Map the current pan/zoom to the minimap's displayed-image box + viewport rect. */
export function computeMinimap(i: MinimapInput): MinimapLayout {
  const k = Math.min(i.boxW / i.contentW, i.boxH / i.contentH);
  const imgW = i.contentW * k;
  const imgH = i.contentH * k;
  const imgX = (i.boxW - imgW) / 2;
  const imgY = (i.boxH - imgH) / 2;

  // Visible image region (image px): top-left = -pan/scale, size = container/scale.
  const visLeft = -i.panX / i.scale;
  const visTop = -i.panY / i.scale;
  const visW = i.containerW / i.scale;
  const visH = i.containerH / i.scale;

  // Into box coords, clamped inside the displayed image.
  let rectX = imgX + visLeft * k;
  let rectY = imgY + visTop * k;
  let rectW = visW * k;
  let rectH = visH * k;
  rectW = Math.min(rectW, imgW);
  rectH = Math.min(rectH, imgH);
  rectX = clamp(rectX, imgX, imgX + imgW - rectW);
  rectY = clamp(rectY, imgY, imgY + imgH - rectH);

  return { k, imgX, imgY, imgW, imgH, rectX, rectY, rectW, rectH };
}

export interface MinimapDragInput {
  scale: number;
  startPanX: number;
  startPanY: number;
  dxBox: number;
  dyBox: number;
  contentW: number;
  contentH: number;
  boxW: number;
  boxH: number;
}

/** A drag of (dxBox, dyBox) box-pixels → the new absolute pan offset. */
export function minimapDragToPan(i: MinimapDragInput): { x: number; y: number } {
  const k = Math.min(i.boxW / i.contentW, i.boxH / i.contentH);
  // Moving the rect +dxBox box-px → image visible-left += dxBox/k image-px → offset -= (dxBox/k)*scale.
  return {
    x: i.startPanX - (i.dxBox / k) * i.scale,
    y: i.startPanY - (i.dyBox / k) * i.scale,
  };
}
