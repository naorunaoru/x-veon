import { describe, expect, it } from 'vitest';
import { applyViewport, type DisplayViewport } from './viewport';
import { U_VIEW_X, U_VIEW_Y, UNIFORM_FLOATS } from './uniforms';

const view: DisplayViewport = { width: 800, height: 600, dpr: 2, scale: 0.1, offsetX: 100, offsetY: 100, orientation: 'Normal' };

function fixture(v = view, max = 8192) {
  const canvas = document.createElement('canvas');
  const data = new Float32Array(UNIFORM_FLOATS);
  applyViewport(canvas, data, v, 6000, 4000, max);
  const point = (u: number, w: number) => [
    (data[U_VIEW_X] * u + data[U_VIEW_X + 1] * w + data[U_VIEW_X + 2] + 1) * v.width / 2,
    (1 - data[U_VIEW_Y] * u - data[U_VIEW_Y + 1] * w - data[U_VIEW_Y + 2]) * v.height / 2,
  ];
  return { canvas, data, point };
}

describe('display viewport', () => {
  it('allocates viewport × DPR pixels for a 24 MP source regardless of zoom', () => {
    const { canvas } = fixture();
    expect([canvas.width, canvas.height]).toEqual([1600, 1200]);
    expect(canvas.width * canvas.height).toBe(1_920_000);
    expect(fixture({ ...view, scale: 32 }).canvas.width).toBe(1600);
  });

  it.each([
    ['Normal', [100, 100], [700, 100], [100, 500]],
    ['Rotate90', [500, 100], [500, 700], [100, 100]],
    ['Rotate180', [700, 500], [100, 500], [700, 100]],
    ['Rotate270', [100, 700], [100, 100], [500, 700]],
  ] as const)('maps source corners through %s orientation, scale and pan', (orientation, tl, tr, bl) => {
    const { point } = fixture({ ...view, orientation });
    for (const [actual, expected] of [[point(0, 0), tl], [point(1, 0), tr], [point(0, 1), bl]]) {
      expect(actual[0]).toBeCloseTo(expected[0], 3);
      expect(actual[1]).toBeCloseTo(expected[1], 3);
    }
  });

  it('keeps offscreen coordinates unclamped so zoom crops instead of stretching', () => {
    const { point, data } = fixture({ ...view, scale: 2, offsetX: -200, offsetY: -300 });
    expect(point(0, 0)[0]).toBeCloseTo(-200);
    expect(point(1, 1)[0]).toBeCloseTo(11800);
    expect(data[U_VIEW_X + 3]).toBe(0); // nearest texels during pixel inspection
  });

  it('handles zero-sized hosts and caps backing size to the device limit', () => {
    const hidden = fixture({ ...view, width: 0, height: 0 });
    expect([...hidden.data].every(Number.isFinite)).toBe(true);
    expect(hidden.canvas.width).toBeGreaterThan(0);
    const { canvas } = fixture({ ...view, width: 10000, height: 5000 }, 8192);
    expect([canvas.width, canvas.height]).toEqual([8192, 4096]);
  });
});
