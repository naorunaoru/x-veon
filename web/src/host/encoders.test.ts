import { describe, it, expect } from 'vitest';
import { ENCODERS, encoderFor } from './encoders';
import { EXPORT_FORMATS } from '@/lib/catalog';

describe('encoder registry', () => {
  it('has exactly one encoder per catalogue format', () => {
    expect(ENCODERS.map((e) => e.format).sort()).toEqual(EXPORT_FORMATS.map((f) => f.id).sort());
    expect(new Set(ENCODERS.map((e) => e.format)).size).toBe(ENCODERS.length);
  });

  it('looks an encoder up and rejects unknown formats', () => {
    expect(encoderFor('tiff').format).toBe('tiff');
    expect(() => encoderFor('bmp' as never)).toThrow(/unknown export format/);
  });
});
