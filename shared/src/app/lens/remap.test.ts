import { describe, expect, it } from 'vitest';
import { fromLibraryPhoto } from '@/app/store/photo';
import { lensPickKey } from '@/app/store/lens';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
import { XT3 } from '@/lib/lens/fuji-fixtures';
import type { ProcessingMetadata, ProcessingResultMeta } from '@/lib/types';
import type { LensProfile } from '@/lib/lensfun-types';
import { fakePhoto } from '@/test/fake-host';
import { lensFrameFor, lensRemapForFile, resolvedLensFor } from './remap';

const profile: LensProfile = {
  lensModel: 'XF35mmF2 R WR', mount: 'Fujifilm X', cropfactor: 1.5, calibCropfactor: 1.5, calibAspectRatio: 1.5,
  distortion: [{ model: 'poly3', focal: 35, k1: -0.01 }], tca: [],
  vignetting: [{ model: 'pa', focal: 35, aperture: 2, distance: 1000, k1: -0.3, k2: 0, k3: 0 }],
};

function resultOf(metadata: Partial<ProcessingMetadata>): ProcessingResultMeta {
  return {
    exportData: { width: 6384, height: 4182, xyzToCam: null, wbCoeffs: new Float32Array(3), camToXyz: new Float32Array(12), orientation: 'Normal' },
    metadata: { focalLength: 35, fNumber: 3.6, ...metadata } as ProcessingMetadata,
  };
}
const withResult = (metadata: Partial<ProcessingMetadata>) => ({ ...fromLibraryPhoto(fakePhoto()), result: resultOf(metadata) });
const file = withResult({ fujiCorrection: { ...XT3, centre: [3126, 2093] } });
const handPicked = { ...file.edit, lens: { ...DEFAULT_LENS_EDIT, source: { lensfun: { file: 'f', model: 'm' } } } };

describe('lensRemapForFile', () => {
  it('builds a remap for a photo with camera tables, centred on the camera frame', () => {
    const remap = lensRemapForFile(file, {})!;
    expect(remap.centre).toEqual([3126, 2093]); expect(remap.halfDiagPx).toBeCloseTo(3749, 0);
    expect(remap.scale).toBeLessThan(1);           // barrel
    expect([remap.width, remap.height]).toEqual([6384, 4182]);
  });
  it('is null before a result exists, and when every toggle is off', () => {
    expect(lensRemapForFile({ ...file, result: null }, {})).toBeNull();
    expect(lensRemapForFile({ ...file, edit: { ...file.edit, lens: { ...DEFAULT_LENS_EDIT, distortion: false, ca: false, vignetting: false } } }, {})).toBeNull();
  });
  it('reads a hand-picked profile from the picks map', () => {
    expect(lensRemapForFile({ ...file, edit: handPicked }, {})).toBeNull();
    expect(lensRemapForFile({ ...file, edit: handPicked }, { [lensPickKey('f', 'm')]: profile })).not.toBeNull();
  });
  it('is null when the automatic profile is the only source and it predates the calibration fields', () => {
    const { calibCropfactor: _, ...stale } = profile;
    const plain = withResult({});
    expect(lensRemapForFile({ ...plain, lensProfile: profile }, {})).not.toBeNull();
    expect(lensRemapForFile({ ...plain, lensProfile: stale as LensProfile }, {})).toBeNull();
  });
  it('centres on the texture when the photo carries no camera tables', () => {
    const remap = lensRemapForFile({ ...withResult({}), lensProfile: profile }, {})!;
    expect(remap.centre).toEqual([3192, 2091]); expect(remap.halfDiagPx).toBeCloseTo(Math.hypot(6384, 4182) / 2, 6);
  });
});

describe('resolvedLensFor', () => {
  it('is null before a result exists', () => {
    expect(resolvedLensFor({ ...file, result: null }, {})).toBeNull();
  });
  it('resolves per correction from the camera tables and the automatic profile', () => {
    expect(resolvedLensFor({ ...file, lensProfile: profile }, {})!.sources).toEqual({ distortion: 'camera', ca: 'camera', vignetting: 'lensfun' });
  });
  it('is manual-live when nothing matches', () => {
    expect(resolvedLensFor(withResult({}), {})).toMatchObject({ manualLive: true });
  });
  it('takes the focal length and f-number from the quick metadata when the result has none', () => {
    const two: LensProfile = { ...profile, distortion: [{ model: 'poly3', focal: 18, k1: -0.01 }, { model: 'poly3', focal: 55, k1: -0.02 }] };
    const bare = { ...withResult({ focalLength: 0, fNumber: 0 }), lensProfile: two };
    expect(resolvedLensFor(bare, {})!.sources.distortion).toBe('none');
    const quick = { ...bare, metadata: { camera: 'X-T3', lensModel: 'XF35mmF2 R WR', focalLength: 35, fNumber: 2 } };
    expect(resolvedLensFor(quick, {})!.sources).toMatchObject({ distortion: 'lensfun', vignetting: 'lensfun' });
  });
});

describe('lensFrameFor', () => {
  it('uses the texture size, the camera centre and the camera half-diagonal', () => {
    expect(lensFrameFor(file)).toEqual({ width: 6384, height: 4182, centre: [3126, 2093], halfDiagPx: expect.closeTo(3749, 0) });
  });
  it('leaves out what the photo does not carry', () => {
    expect(lensFrameFor(withResult({}))).toEqual({ width: 6384, height: 4182 });
    const noCentre = withResult({ fujiCorrection: { ...XT3, centre: null } });
    expect(lensFrameFor(noCentre)).toEqual({ width: 6384, height: 4182, halfDiagPx: expect.closeTo(3749, 0) });
    const unknownDiagonal = withResult({ fujiCorrection: { ...XT3, distortion: [0, ...XT3.distortion.slice(1)], centre: [1, 2] } });
    expect(lensFrameFor(unknownDiagonal)).toEqual({ width: 6384, height: 4182, centre: [1, 2] });
  });
  it('is null before a result exists', () => {
    expect(lensFrameFor({ ...file, result: null })).toBeNull();
  });
});
