import { describe, expect, it } from 'vitest';
import { NO_CURVES } from '@/lib/lens/curves';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
import { XT3 } from '@/lib/lens/fuji-fixtures';
import { parseFujiTables } from '@/lib/lens/fuji-tables';
import type { LensProfile } from '@/lib/lensfun-types';
import { lensReadouts, lensStatus, resolveLens } from './resolve';

const xt3 = parseFujiTables(XT3)!;
const profile: LensProfile = {
  lensModel: 'XF35mmF2 R WR', mount: 'Fujifilm X', cropfactor: 1.5, calibCropfactor: 1.5, calibAspectRatio: 1.5,
  distortion: [{ model: 'poly3', focal: 35, k1: -0.01 }], tca: [],
  vignetting: [{ model: 'pa', focal: 35, aperture: 2, distance: 1000, k1: -0.3, k2: 0, k3: 0 }],
};
const base = { edit: DEFAULT_LENS_EDIT, focal: 35, fNumber: 2, halfDiagPx: 3749 };
const picked = { ...DEFAULT_LENS_EDIT, source: { lensfun: { file: 'f', model: 'm' } } };

describe('resolveLens', () => {
  it('auto takes the camera table per correction and falls back to lensfun where the camera has none', () => {
    const r = resolveLens({ ...base, camera: xt3, lensfun: profile, picked: null });
    expect(r.sources).toEqual({ distortion: 'camera', ca: 'camera', vignetting: 'lensfun' });
    expect(r.curves.vignetting!(1)).toBeCloseTo(1 / 0.7, 6); expect(r.manualLive).toBe(false);
  });
  it('auto with nothing goes manual-live with manual curves', () => {
    const r = resolveLens({ ...base, camera: null, lensfun: null, picked: null, edit: { ...DEFAULT_LENS_EDIT, manual: { ...DEFAULT_LENS_EDIT.manual, distortion: 10 } } });
    expect(r.sources).toEqual({ distortion: 'manual', ca: 'none', vignetting: 'none' }); expect(r.manualLive).toBe(true);
    expect(r.curves.distortion!(1)).toBeCloseTo(0.9, 9);
  });
  it('a hand-picked lens uses lensfun only; an unloaded pick corrects nothing', () => {
    expect(resolveLens({ ...base, edit: picked, camera: xt3, lensfun: null, picked: profile }).sources.distortion).toBe('lensfun');
    const pending = resolveLens({ ...base, edit: picked, camera: xt3, lensfun: null, picked: undefined });
    expect(pending.curves).toEqual(NO_CURVES); expect(pending.manualLive).toBe(false);
  });
  it('a hand-picked lens ignores the camera tables and the automatic profile, and is never manual-live', () => {
    const r = resolveLens({ ...base, edit: picked, camera: xt3, lensfun: profile, picked: { ...profile, distortion: [], vignetting: [] } });
    expect(r.sources).toEqual({ distortion: 'none', ca: 'none', vignetting: 'none' });
    const none = resolveLens({ ...base, edit: picked, camera: xt3, lensfun: profile, picked: null });
    expect(none.curves).toEqual(NO_CURVES); expect(none.manualLive).toBe(false);
  });
  it('manual source ignores both profiles', () => {
    const r = resolveLens({ ...base, edit: { ...DEFAULT_LENS_EDIT, source: 'manual' }, camera: xt3, lensfun: profile, picked: null });
    expect(r.sources.distortion).toBe('none'); expect(r.manualLive).toBe(true);
  });
  it('manual source applies the sliders, and auto does too once it has nothing', () => {
    const manual = { ...DEFAULT_LENS_EDIT.manual, caRed: 2, vignetting: 1 };
    for (const source of ['manual', 'auto'] as const) {
      const r = resolveLens({ ...base, edit: { ...DEFAULT_LENS_EDIT, source, manual }, camera: null, lensfun: null, picked: null });
      expect(r.sources).toEqual({ distortion: 'none', ca: 'manual', vignetting: 'manual' });
      expect(r.curves.caRed!(1)).toBeCloseTo(1 + 2 / 3749, 9);
    }
  });
  it('auto does not go manual-live while any correction has a source', () => {
    const r = resolveLens({ ...base, edit: { ...DEFAULT_LENS_EDIT, manual: { ...DEFAULT_LENS_EDIT.manual, vignetting: 1 } }, camera: xt3, lensfun: null, picked: null });
    expect(r.sources).toEqual({ distortion: 'camera', ca: 'camera', vignetting: 'none' });
    expect(r.manualLive).toBe(false); expect(r.curves.vignetting).toBeNull();
  });
  it('treats a stored profile without a calibration crop factor as no profile', () => {
    const { calibCropfactor: _, ...stale } = profile;
    const old = stale as LensProfile;
    const r = resolveLens({ ...base, camera: null, lensfun: old, picked: null });
    expect(r.sources).toEqual({ distortion: 'none', ca: 'none', vignetting: 'none' }); expect(r.manualLive).toBe(true);
    expect(resolveLens({ ...base, edit: picked, camera: null, lensfun: null, picked: old }).curves).toEqual(NO_CURVES);
  });
});

describe('lensReadouts', () => {
  it('reads the corner', () => {
    const r = lensReadouts({ distortion: () => 0.92, caRed: () => 1.001, caBlue: () => 0.999, vignetting: () => 2 }, 1000);
    expect(r).toEqual({ distortion: expect.closeTo(8, 9), caRed: expect.closeTo(0.92, 9), caBlue: expect.closeTo(-0.92, 9), vignetting: 1 });
  });
  it('counts a missing curve as no correction', () => {
    expect(lensReadouts(NO_CURVES, 3749)).toEqual({ distortion: 0, caRed: 0, caBlue: 0, vignetting: 0 });
  });
});

describe('lensStatus', () => {
  const exif = 'XF35mmF2 R WR';
  const status = (over: Partial<Parameters<typeof lensStatus>[0]>) =>
    lensStatus({ resolved: null, edit: DEFAULT_LENS_EDIT, exifLens: exif, lensfunModel: null, matching: false, ...over });

  it('names a mix of camera and lensfun with the source of each correction', () => {
    const resolved = resolveLens({ ...base, camera: xt3, lensfun: profile, picked: null });
    expect(status({ resolved, lensfunModel: exif }))
      .toEqual({ line: 'Camera + Lensfun', detail: 'Distortion: camera · CA: camera · Vignetting: lensfun' });
  });
  it('waits for the decode, then for the match', () => {
    const resolved = resolveLens({ ...base, camera: xt3, lensfun: profile, picked: null });
    expect(status({ resolved: null }).line).toBe('Waiting for decode');
    expect(status({ resolved, exifLens: 'X', matching: true }).line).toBe('Matching lens');
  });
  it('says there is no profile when auto finds nothing, whatever the sliders do', () => {
    const none = resolveLens({ ...base, camera: null, lensfun: null, picked: null });
    expect(status({ resolved: none, exifLens: 'TTArtisan 27mm' })).toEqual({ line: 'No profile for TTArtisan 27mm', detail: null });
    const sliders = resolveLens({ ...base, camera: null, lensfun: null, picked: null, edit: { ...DEFAULT_LENS_EDIT, manual: { ...DEFAULT_LENS_EDIT.manual, distortion: 5 } } });
    expect(status({ resolved: sliders }).line).toBe(`No profile for ${exif}`);
  });
  it('says Manual for the manual source', () => {
    const edit = { ...DEFAULT_LENS_EDIT, source: 'manual' as const };
    const resolved = resolveLens({ ...base, edit, camera: xt3, lensfun: profile, picked: null });
    expect(status({ resolved, edit, exifLens: 'X' }).line).toBe('Manual');
  });
  it('names the camera when it supplies every correction that is used', () => {
    const resolved = resolveLens({ ...base, camera: xt3, lensfun: null, picked: null });
    expect(status({ resolved })).toEqual({ line: `Camera · ${exif}`, detail: null });
  });
  it('names lensfun for an automatic match with only lensfun curves and for a hand-picked lens', () => {
    const resolved = resolveLens({ ...base, camera: null, lensfun: profile, picked: null });
    expect(status({ resolved, lensfunModel: profile.lensModel })).toEqual({ line: `Lensfun · ${profile.lensModel}`, detail: null });
    const hand = resolveLens({ ...base, edit: picked, camera: xt3, lensfun: null, picked: profile });
    expect(status({ resolved: hand, edit: picked, lensfunModel: 'm' })).toEqual({ line: 'Lensfun · m', detail: null });
  });
  it('keeps a hand-picked lens named while its profile is still loading', () => {
    const pending = resolveLens({ ...base, edit: picked, camera: xt3, lensfun: null, picked: undefined });
    expect(status({ resolved: pending, edit: picked, lensfunModel: null }).line).toBe('Lensfun · m');
  });
  it('leaves out the lens name when EXIF has none', () => {
    const cam = resolveLens({ ...base, camera: xt3, lensfun: null, picked: null });
    expect(status({ resolved: cam, exifLens: '' }).line).toBe('Camera');
    const none = resolveLens({ ...base, camera: null, lensfun: null, picked: null });
    expect(status({ resolved: none, exifLens: '' }).line).toBe('No profile');
  });
});
