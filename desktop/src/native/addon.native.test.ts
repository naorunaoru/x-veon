import { createRequire } from 'node:module';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { addonFile, loadNativeEncoder, type NativeEncodeInput } from '../worker/native';

function image(width: number, height: number): Float32Array {
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 3;
    data[i] = x / width; data[i + 1] = y / height; data[i + 2] = 0.25;
  }
  return data;
}
const loaded = loadNativeEncoder(createRequire(import.meta.url), path.resolve(__dirname, '../../native'));
const nativeDir = path.resolve(__dirname, '../../native');
function encoder() {
  expect(loaded.ok).toBe(true);
  if (!loaded.ok) throw new Error(loaded.reason);
  return loaded.encoder;
}
const input: NativeEncodeInput = { format: 'avif', width: 64, height: 48, orientation: 'Normal', quality: 50, peakLuminance: 1000, threads: 2 };
describe('native addon', () => {
  it('loads', () => { encoder(); });
  it('reads the main screen, and treats a short or null handle as no handle', () => {
    const addon = createRequire(import.meta.url)(path.join(nativeDir, addonFile())) as { displayReadings(handle?: Buffer): Record<string, unknown> | null };
    const readings = addon.displayReadings();
    expect(addon.displayReadings(Buffer.alloc(2))).toEqual(readings);
    expect(addon.displayReadings(Buffer.alloc(8))).toEqual(readings);
    if (readings === null) return; // A runner without a screen has nothing to read.
    if (process.platform === 'darwin') {
      expect(readings.currentEdr).toBeGreaterThanOrEqual(1);
      expect(readings.potentialEdr).toBeGreaterThanOrEqual(1);
      expect(readings.referenceEdr).toBeGreaterThanOrEqual(0);
      expect(readings.hdrEnabled).toBeUndefined();
    }
    if (process.platform === 'win32') {
      expect(typeof readings.hdrEnabled).toBe('boolean');
      expect(readings.maxLuminance).toBeGreaterThanOrEqual(0);
      expect(readings.potentialEdr).toBeUndefined();
    }
  });
  it.each(['avif', 'jpeg-hdr', 'tiff'] as const)('encodes %s', async format => {
    const data = image(input.width, input.height);
    const bytes = await encoder().encode(data, format === 'jpeg-hdr' ? data : null, { ...input, format });
    expect(bytes.length).toBeGreaterThan(0);
    if (format === 'avif') expect(Buffer.from(bytes.subarray(4, 8)).toString()).toBe('ftyp');
    if (format === 'jpeg-hdr') expect(Array.from(bytes.subarray(0, 2))).toEqual([0xff, 0xd8]);
    if (format === 'tiff') expect(Array.from(bytes.subarray(0, 4))).toEqual([0x49, 0x49, 0x2a, 0]);
  });
  it('rejects a length mismatch', async () => {
    await expect(encoder().encode(new Float32Array(3), null, input)).rejects.toThrow('data length mismatch');
  });
  it('keeps the event loop free', async () => {
    const width = 2048, height = 1536;
    const data = image(width, height);
    let ticks = 0;
    let running = true;
    const tick = () => { if (running) { ticks++; setTimeout(tick, 1); } };
    setTimeout(tick, 1);
    try { await encoder().encode(data, null, { ...input, width, height }); }
    finally { running = false; }
    expect(ticks).toBeGreaterThan(3);
  });
});
