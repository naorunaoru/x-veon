import { describe, it, expect } from 'vitest';
import { computeMinimap, minimapDragToPan } from './minimap';

// A 4000×3000 image, 800×600 viewport, box 160×120.
// Scale 1, pan 0 → at scale 1 the image (4000×3000) far overflows the 800×600 viewport.
const base = { contentW: 4000, contentH: 3000, containerW: 800, containerH: 600, boxW: 160, boxH: 120 };

describe('computeMinimap', () => {
  it('fits the image into the box preserving aspect (here exactly filling)', () => {
    const m = computeMinimap({ ...base, scale: 1, panX: 0, panY: 0 });
    // 4000×3000 fits 160×120 at k = min(160/4000, 120/3000) = 0.04 → 160×120, centered at 0,0.
    expect(m.imgW).toBeCloseTo(160, 3);
    expect(m.imgH).toBeCloseTo(120, 3);
    expect(m.imgX).toBeCloseTo(0, 3);
    expect(m.imgY).toBeCloseTo(0, 3);
  });
  it('viewport rect covers the visible fraction of the image', () => {
    // pan 0, scale 1: visible region top-left = (0,0), size = container/scale = 800×600 image px.
    const m = computeMinimap({ ...base, scale: 1, panX: 0, panY: 0 });
    // rect in box coords = visibleRegion * k = 800*0.04 × 600*0.04 = 32×24 at (0,0).
    expect(m.rectX).toBeCloseTo(0, 3);
    expect(m.rectY).toBeCloseTo(0, 3);
    expect(m.rectW).toBeCloseTo(32, 3);
    expect(m.rectH).toBeCloseTo(24, 3);
  });
  it('panning the photo moves the rect right/down', () => {
    // offset -400 (image shifted left by 400 screen px at scale 1) → visible left = 400 image px.
    const m = computeMinimap({ ...base, scale: 1, panX: -400, panY: -300 });
    expect(m.rectX).toBeCloseTo(400 * 0.04, 3); // 16
    expect(m.rectY).toBeCloseTo(300 * 0.04, 3); // 12
  });
  it('letterboxes when box aspect differs from content (production 160x106 box)', () => {
    // 4000x3000 into a 160x106 box: k = min(160/4000, 106/3000) = 106/3000 = 0.035333.
    // imgH fills 106, imgW = 141.33 → letterboxed horizontally, imgX = (160-141.33)/2 = 9.33.
    const m = computeMinimap({ ...base, boxH: 106, scale: 1, panX: -400, panY: -300 });
    const k = 106 / 3000;
    expect(m.imgX).toBeCloseTo(9.333, 2);
    expect(m.imgY).toBeCloseTo(0, 3);
    // rectX must include the letterbox offset: imgX + visibleLeft * k.
    expect(m.rectX).toBeCloseTo(9.333 + 400 * k, 2);
    expect(m.rectY).toBeCloseTo(300 * k, 2);
  });
  it('does NOT clamp: a far-off pan drives the rect past the box (CSS clips it)', () => {
    // Was the old "clamps the rect inside the displayed image" test. The rect now
    // reports the true off-image position instead of pinning to the image edge.
    const m = computeMinimap({ ...base, scale: 1, panX: 99999, panY: 99999 });
    // -99999/1 * k(0.04) = -3999.96, far outside [imgX, imgX+imgW].
    expect(m.rectX).toBeCloseTo(-3999.96, 2);
    expect(m.rectY).toBeCloseTo(-3999.96, 2);
    expect(m.rectX).toBeLessThan(m.imgX);
  });

  it('keeps the VIEWPORT aspect (not the photo aspect) when the viewport overflows the image', () => {
    // Regression for the reported bug. Square 1000x1000 viewport over a wide
    // 6000x3000 (2:1) image, zoomed so the viewport spills above/below the image.
    const m = computeMinimap({
      contentW: 6000, contentH: 3000, containerW: 1000, containerH: 1000,
      boxW: 160, boxH: 106, scale: 0.3, panX: -400, panY: 50,
    });
    // rect must be SQUARE (1:1 viewport aspect), NOT the image's 2:1.
    expect(m.rectW / m.rectH).toBeCloseTo(1, 5);
    expect(m.rectW).toBeCloseTo(88.889, 2);
    expect(m.rectH).toBeCloseTo(88.889, 2);
    // and it straddles the displayed image vertically (overflows top + bottom).
    expect(m.rectY).toBeLessThan(m.imgY);
    expect(m.rectY + m.rectH).toBeGreaterThan(m.imgY + m.imgH);
  });

  it('reflects the true pan when panned off the left edge (negative rectX, not pinned)', () => {
    // Regression for "minimap does not reflect the actual pan position".
    const m = computeMinimap({
      contentW: 6000, contentH: 4000, containerW: 1200, containerH: 800,
      boxW: 160, boxH: 106, scale: 4, panX: 800, panY: -2000,
    });
    // viewport extends left of the image → rectX negative (old code pinned to imgX).
    expect(m.rectX).toBeCloseTo(-4.8, 2);
    expect(m.rectX).toBeLessThan(m.imgX);
    expect(m.rectW / m.rectH).toBeCloseTo(1200 / 800, 5);
  });

  it('rect aspect always equals the viewport aspect regardless of scale', () => {
    // rectW/rectH = (containerW/scale·k)/(containerH/scale·k) = containerW/containerH.
    for (const scale of [0.3, 0.5, 1, 2, 7]) {
      const m = computeMinimap({
        contentW: 6000, contentH: 4000, containerW: 1900, containerH: 1100,
        boxW: 160, boxH: 106, scale, panX: 0, panY: 0,
      });
      expect(m.rectW / m.rectH).toBeCloseTo(1900 / 1100, 9);
    }
  });
});

describe('minimapDragToPan', () => {
  it('moving the rect by box-pixels maps back to an offset (inverse of k·scale)', () => {
    // k = 0.04, scale = 1. Move rect +16 box-px in X → image moves +400 px → offsetX -= 400.
    const pan = minimapDragToPan({ ...base, scale: 1, startPanX: 0, startPanY: 0, dxBox: 16, dyBox: 12 });
    expect(pan.x).toBeCloseTo(-400, 3);
    expect(pan.y).toBeCloseTo(-300, 3);
  });
});
