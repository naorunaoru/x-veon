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
  /** Viewport rectangle in minimap-box px; may be negative or exceed the box
   *  (clipped by CSS overflow:hidden) so it keeps the true viewport aspect + pan. */
  rectX: number; rectY: number; rectW: number; rectH: number;
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

  // Into box coords. NOT clamped to the image: when the viewport extends past the
  // image (aspect mismatch, or panned off an edge onto empty stage) the rect must
  // keep the viewport's aspect (rectW/rectH === containerW/containerH for all
  // scale/pan) and its true position, simply overflowing the box — CSS
  // overflow:hidden clips it and the box-shadow dim mask stays correct.
  const rectX = imgX + visLeft * k;
  const rectY = imgY + visTop * k;
  const rectW = visW * k;
  const rectH = visH * k;

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
