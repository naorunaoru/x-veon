import { describe, expect, it } from 'vitest';
import { NO_CURVES, type LensCurves } from '@/lib/lens/curves';
import { LENS_TABLE_BYTES, LENS_TABLE_DOMAIN, LENS_TABLE_SIZE, autoscale, buildLensTable, lensSourceUv } from './lens-table';

const ON = { distortion: true, ca: true, vignetting: true };
const frame = { width: 3000, height: 2000 };
const barrel: LensCurves = { ...NO_CURVES, distortion: (r) => 1 - 0.1 * r * r };
const pincushion: LensCurves = { ...NO_CURVES, distortion: (r) => 1 + 0.1 * r * r };

describe('lens radial table', () => {
  it('has the layout the shader reads', () => {
    expect([LENS_TABLE_SIZE, LENS_TABLE_DOMAIN, LENS_TABLE_BYTES]).toEqual([1024, 1.25, 16384]);
  });

  it('autoscale zooms out for barrel, bound by the short-edge midpoints', () => expect(autoscale(barrel, ON, frame)).toBeCloseTo(0.968, 2));
  it('autoscale zooms in for pincushion, bound by the corners', () => expect(autoscale(pincushion, ON, frame)).toBeCloseTo(1.086, 2));
  it('autoscale of the identity is 1.001, and 1 with a finite table when the solve cannot converge', () => {
    expect(autoscale(NO_CURVES, ON, frame)).toBeCloseTo(1.001, 6);
    const extreme: LensCurves = { ...NO_CURVES, distortion: () => 0.1 };
    expect(autoscale(extreme, ON, frame)).toBe(1);
    expect(buildLensTable(extreme, ON, frame)!.table.every(Number.isFinite)).toBe(true);
  });
  it('autoscale of a square frame is bound by the four edge midpoints, at 1/√2 of the half-diagonal', () => {
    // Midpoint root of r(1 − 0.1r²) = 0.7071 is r ≈ 0.7491, so 0.7071 / 0.7491 × 1.001; the corners alone would give 0.867.
    expect(autoscale(barrel, ON, { width: 2000, height: 2000 })).toBeCloseTo(0.9448, 4);
  });
  it('autoscale counts a correction only when its toggle is on', () => {
    const curves: LensCurves = { distortion: (r) => 1 - 0.1 * r * r, caRed: () => 1.05, caBlue: () => 0.95, vignetting: null };
    // The red channel samples further out than green, so CA on zooms in beyond the distortion-only scale (0.968).
    expect(autoscale(curves, { ...ON, ca: false }, frame)).toBeCloseTo(0.968, 2);
    expect(autoscale(curves, ON, frame)).toBeGreaterThan(1);
    expect(autoscale(curves, { ...ON, distortion: false, ca: false }, frame)).toBeCloseTo(1.001, 6);
    // CA alone on the identity geometry: red reaches 1.05 × the output radius.
    expect(autoscale(curves, { ...ON, distortion: false }, frame)).toBeCloseTo(1.05 * 1.001, 6);
  });
  it('autoscale falls back to 1 for a curve that is not a number at the edge of the domain', () => {
    expect(autoscale({ ...NO_CURVES, distortion: () => NaN }, ON, frame)).toBe(1);
  });

  it('samples the domain into 1024 rgba entries with the gain indexed by source radius', () => {
    const curves: LensCurves = { distortion: (r) => 1 - 0.1 * r * r, caRed: () => 1.001, caBlue: () => 0.999, vignetting: (r) => 1 + r };
    const remap = buildLensTable(curves, ON, frame)!;
    expect(remap.table).toHaveLength(4096);
    const at = (i: number) => Array.from(remap.table.subarray(i * 4, i * 4 + 4));
    expect(at(0)).toEqual([1.001, 1, 0.999, 1].map(Math.fround));   // the table is f32
    const i = 1023, rho = 1.25;
    const m = 1 - 0.1 * rho * rho;
    expect(at(i)[1]).toBeCloseTo(m, 6); expect(at(i)[0]).toBeCloseTo(m * 1.001, 6); expect(at(i)[3]).toBeCloseTo(1 + rho, 6);
    expect(remap.centre).toEqual([1500, 1000]); expect(remap.halfDiagPx).toBeCloseTo(1802.776, 3);
  });
  it('takes the optical centre and half-diagonal from the frame when it gives them', () => {
    const remap = buildLensTable(barrel, ON, { ...frame, centre: [1600, 1100], halfDiagPx: 2000 })!;
    expect([remap.width, remap.height, remap.centre, remap.halfDiagPx]).toEqual([3000, 2000, [1600, 1100], 2000]);
    expect(remap.scale).toBe(autoscale(barrel, ON, { ...frame, centre: [1600, 1100], halfDiagPx: 2000 }));
  });
  it('toggles neutralise their channels and nothing enabled yields null', () => {
    const curves: LensCurves = { distortion: (r) => 1 - 0.1 * r * r, caRed: () => 1.001, caBlue: () => 0.999, vignetting: (r) => 1 + r };
    const noCa = buildLensTable(curves, { ...ON, ca: false }, frame)!.table;
    expect(noCa[4092]).toBe(noCa[4093]); expect(noCa[4094]).toBe(noCa[4093]);
    const noDist = buildLensTable(curves, { ...ON, distortion: false }, frame)!.table;
    expect(noDist[4093]).toBe(1); expect(noDist[4092]).toBeCloseTo(1.001, 6);
    expect(buildLensTable(curves, { ...ON, vignetting: false }, frame)!.table[4095]).toBe(1);
    expect(buildLensTable(curves, { distortion: false, ca: false, vignetting: false }, frame)).toBeNull();
    expect(buildLensTable(NO_CURVES, ON, frame)).toBeNull();
  });
  it('is null when the enabled corrections have no curve, even if a disabled one has', () => {
    expect(buildLensTable(barrel, { ...ON, distortion: false }, frame)).toBeNull();
    expect(buildLensTable({ ...NO_CURVES, vignetting: (r) => 1 + r }, { ...ON, vignetting: false }, frame)).toBeNull();
  });

  it('the CPU twin maps uv through centre, scale and table', () => {
    const remap = buildLensTable({ ...NO_CURVES, distortion: () => 0.5 }, ON, { ...frame, centre: [1600, 1000] })!;
    const s = remap.scale;
    const out = lensSourceUv(remap, [1, 0.5]);   // right edge midpoint, 1400 px from the shifted centre
    expect(out.g[0]).toBeCloseTo((1600 + 1400 / s * 0.5) / 3000, 6); expect(out.g[1]).toBeCloseTo(0.5, 6);
    expect(out.gain).toBe(1);
    expect(lensSourceUv(remap, [1600 / 3000, 0.5]).g).toEqual([1600 / 3000, 0.5]);
  });
  it('the CPU twin offsets the channels by their own multipliers and reads the gain at the source radius', () => {
    const curves: LensCurves = { distortion: () => 0.9, caRed: () => 1.01, caBlue: () => 0.99, vignetting: (r) => 1 + r };
    const remap = buildLensTable(curves, ON, frame)!;
    const s = remap.scale;
    const uv: [number, number] = [0.9, 0.2];
    const dx = (0.9 - 0.5) * 3000 / s, dy = (0.2 - 0.5) * 2000 / s;
    const out = lensSourceUv(remap, uv);
    expect(out.g[0]).toBeCloseTo((1500 + dx * 0.9) / 3000, 6); expect(out.g[1]).toBeCloseTo((1000 + dy * 0.9) / 2000, 6);
    expect(out.r[0]).toBeCloseTo((1500 + dx * 0.9 * 1.01) / 3000, 6); expect(out.r[1]).toBeCloseTo((1000 + dy * 0.9 * 1.01) / 2000, 6);
    expect(out.b[0]).toBeCloseTo((1500 + dx * 0.9 * 0.99) / 3000, 6); expect(out.b[1]).toBeCloseTo((1000 + dy * 0.9 * 0.99) / 2000, 6);
    expect(out.gain).toBeCloseTo(1 + Math.hypot(dx, dy) / remap.halfDiagPx * 0.9, 4);
  });
  it('the CPU twin holds the table ends beyond the domain', () => {
    const remap = buildLensTable({ ...NO_CURVES, distortion: (r) => 1 - 0.1 * r * r }, ON, frame)!;
    // Far outside the frame: the radius is past the domain, so the multiplier is the last entry's.
    const out = lensSourceUv(remap, [10, 0.5]);
    const last = remap.table[(LENS_TABLE_SIZE - 1) * 4 + 1];
    expect(out.g[0]).toBeCloseTo((1500 + (9.5 * 3000) / remap.scale * last) / 3000, 6);
  });
});
