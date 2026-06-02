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
  it('clamps the rect inside the displayed image', () => {
    const m = computeMinimap({ ...base, scale: 1, panX: 99999, panY: 99999 });
    expect(m.rectX).toBeGreaterThanOrEqual(m.imgX - 0.001);
    expect(m.rectX + m.rectW).toBeLessThanOrEqual(m.imgX + m.imgW + 0.001);
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
