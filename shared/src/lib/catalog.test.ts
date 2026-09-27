import { describe, it, expect } from 'vitest';
import {
  DEMOSAIC_METHODS,
  demosaicMethodsFor,
  isMethodValidForCfa,
  EXPORT_FORMATS,
  exportFormatInfo,
  MODEL_SIZES,
} from './catalog';

describe('demosaic catalogue', () => {
  it('lists the eight methods in the Settings order', () => {
    expect(DEMOSAIC_METHODS.map((m) => m.id)).toEqual([
      'neural-net', 'markesteijn3', 'markesteijn1', 'dht', 'ahd', 'ppg', 'mhc', 'bilinear',
    ]);
  });

  it('filters by sensor type like the Settings panel did', () => {
    expect(demosaicMethodsFor('xtrans').map((m) => m.id)).toEqual([
      'neural-net', 'markesteijn3', 'markesteijn1', 'dht', 'bilinear',
    ]);
    expect(demosaicMethodsFor('bayer').map((m) => m.id)).toEqual([
      'neural-net', 'ahd', 'ppg', 'mhc', 'bilinear',
    ]);
    expect(demosaicMethodsFor(null)).toHaveLength(8);
  });

  it('validates methods per sensor like useAutoProcess did', () => {
    expect(isMethodValidForCfa('dht', 'bayer')).toBe(false);
    expect(isMethodValidForCfa('ahd', 'xtrans')).toBe(false);
    expect(isMethodValidForCfa('neural-net', 'bayer')).toBe(true);
    expect(isMethodValidForCfa('bilinear', 'xtrans')).toBe(true);
    expect(isMethodValidForCfa('markesteijn3', 'xtrans')).toBe(true);
  });
});

describe('export catalogue', () => {
  it('lists the three formats with their extensions, MIME types and HDR needs', () => {
    expect(EXPORT_FORMATS.map((f) => [f.id, f.ext, f.mime, f.needsHdr])).toEqual([
      ['jpeg-hdr', 'jpg', 'image/jpeg', true],
      ['avif', 'avif', 'image/avif', true],
      ['tiff', 'tif', 'image/tiff', false],
    ]);
  });

  it('looks up a format and rejects unknown ids', () => {
    expect(exportFormatInfo('tiff').label).toBe('TIFF (Linear sRGB)');
    expect(() => exportFormatInfo('bmp' as never)).toThrow();
  });

  it('has the three model sizes', () => {
    expect(MODEL_SIZES).toEqual(['S', 'M', 'L']);
  });
});
