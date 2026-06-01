import { describe, it, expect } from 'vitest';
import { sliderToZoom, zoomToSlider, ZOOM_MIN, ZOOM_MAX } from './zoom';

describe('zoom slider mapping (log over [ZOOM_MIN, ZOOM_MAX])', () => {
  it('maps slider ends to the zoom range', () => {
    expect(sliderToZoom(0)).toBeCloseTo(ZOOM_MIN, 5);
    expect(sliderToZoom(1)).toBeCloseTo(ZOOM_MAX, 5);
  });
  it('maps the midpoint to the geometric mean (log scale)', () => {
    expect(sliderToZoom(0.5)).toBeCloseTo(Math.sqrt(ZOOM_MIN * ZOOM_MAX), 5);
  });
  it('round-trips', () => {
    for (const z of [0.1, 0.25, 0.5, 1, 2, 4]) {
      expect(sliderToZoom(zoomToSlider(z))).toBeCloseTo(z, 5);
    }
  });
  it('clamps zoomToSlider to [0,1]', () => {
    expect(zoomToSlider(0.001)).toBe(0);
    expect(zoomToSlider(99)).toBe(1);
  });
});
