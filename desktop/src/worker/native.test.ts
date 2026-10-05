import { describe, expect, it, vi } from 'vitest';
import { addonFile, loadNativeEncoder } from './native';

describe('native loader', () => {
  it('names supported addons', () => {
    expect(addonFile('darwin', 'arm64')).toBe('xveon-native.darwin-arm64.node');
    expect(addonFile('win32', 'x64')).toBe('xveon-native.win32-x64-msvc.node');
  });
  it('reports load failures', () => {
    expect(loadNativeEncoder(() => { throw new Error('missing'); })).toEqual({ ok: false, reason: expect.stringContaining("couldn't be loaded") });
  });
  it('names an invalid addon', () => {
    expect(loadNativeEncoder(() => ({}), '/test')).toEqual({ ok: false, reason: expect.stringContaining(addonFile()) });
  });
  it('returns bytes and converts null HDR to undefined', async () => {
    const encode = vi.fn(async () => Buffer.from([1]));
    const result = loadNativeEncoder(() => ({ encode }));
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(result.reason);
    const data = new Float32Array(3);
    const input = { format: 'avif' as const, width: 1, height: 1, orientation: '', quality: 80, peakLuminance: 1000 };
    expect(Array.from(await result.encoder.encode(data, null, input))).toEqual([1]);
    expect(encode).toHaveBeenCalledWith(data, undefined, input);
  });
});
