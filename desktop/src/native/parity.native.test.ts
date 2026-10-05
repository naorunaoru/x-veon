import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { afterAll, beforeAll, expect, it } from 'vitest';
import { loadNativeEncoder, type NativeEncodeInput } from '../worker/native';
import { syntheticImage } from './images';

const evidenceDir = path.resolve(__dirname, '../../../.m3-evidence');
const variant = process.env.M3_PARITY_VARIANT ?? 'default';
const results: object[] = [];
const loaded = loadNativeEncoder(createRequire(import.meta.url), path.resolve(__dirname, '../../native'));
if (!loaded.ok) throw new Error(loaded.reason);
const native = loaded.encoder;
let wasm: { initSync(input: { module: Buffer }): unknown; encode_image(data: Float32Array, hdr: Float32Array, w: number, h: number, orientation: string, format: string, quality: number, peak: number): Uint8Array };
beforeAll(async () => {
  const pkg = path.resolve(__dirname, '../../../web/crates/encode-wasm/pkg');
  wasm = await import(/* @vite-ignore */ pathToFileURL(path.join(pkg, 'xtrans_encoder_wasm.js')).href);
  wasm.initSync({ module: fs.readFileSync(path.join(pkg, 'xtrans_encoder_wasm_bg.wasm')) });
});
afterAll(() => {
  fs.mkdirSync(evidenceDir, { recursive: true });
  fs.writeFileSync(path.join(evidenceDir, variant === 'default' ? 'task4-parity.json' : `task4-parity-${variant}.json`), JSON.stringify({ variant, node: process.version, results }, null, 2));
});
const formats = ['avif', 'jpeg-hdr', 'tiff'] as const;
const cases: NativeEncodeInput[] = [];
for (const quality of [95, 40]) {
  for (const format of formats) {
    for (const orientation of ['Normal', 'Rotate90', 'Rotate180', 'Rotate270']) cases.push({ format, width: 97, height: 61, orientation, quality, peakLuminance: 1000 });
    cases.push({ format, width: 640, height: 427, orientation: 'Normal', quality, peakLuminance: 1000 });
    if (format !== 'jpeg-hdr') cases.push({ format, width: 4200, height: 128, orientation: 'Normal', quality, peakLuminance: 1000 });
  }
}
cases.push({ format: 'avif', width: 3000, height: 3200, orientation: 'Normal', quality: 95, peakLuminance: 1000 });
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
for (const input of cases) {
  const { format, width: w, height: h, orientation, quality, peakLuminance: peak } = input;
  const name = `${format} ${w}x${h} orientation=${orientation} q${quality}${format === 'avif' ? ' (rav1e assembly off)' : ''}`;
  it(name, async () => {
    const data = syntheticImage(w, h, 20261005, format === 'jpeg-hdr' ? 1 : 6);
    const hdr = format === 'jpeg-hdr' ? syntheticImage(w, h, 20261006, 6) : null;
    let nativeBytes: Uint8Array;
    if (format === 'avif') {
      process.env.RAV1E_CPU_TARGET = 'rust';
      try { nativeBytes = await native.encode(data, hdr, input); }
      finally { delete process.env.RAV1E_CPU_TARGET; }
    } else nativeBytes = await native.encode(data, hdr, input);
    const wasmBytes = wasm.encode_image(data, hdr ?? new Float32Array(0), w, h, orientation, format, quality, peak);
    const equal = Buffer.from(nativeBytes).equals(Buffer.from(wasmBytes));
    results.push({ name, ...input, equal, nativeLength: nativeBytes.length, wasmLength: wasmBytes.length, nativeSha256: hash(nativeBytes), wasmSha256: hash(wasmBytes) });
    expect(equal).toBe(true);
  });
}
for (const quality of [95, 40]) it(`avif 4200x128 q${quality} threads=1 equals default`, async () => {
  const data = syntheticImage(4200, 128, 20261005, 6);
  const input: NativeEncodeInput = { format: 'avif', width: 4200, height: 128, orientation: 'Normal', quality, peakLuminance: 1000 };
  const defaultBytes = await native.encode(data, null, input);
  const singleBytes = await native.encode(data, null, { ...input, threads: 1 });
  const equal = Buffer.from(defaultBytes).equals(Buffer.from(singleBytes));
  results.push({ name: `avif threads q${quality}`, equal, defaultSha256: hash(defaultBytes), singleSha256: hash(singleBytes) });
  expect(equal).toBe(true);
});
it('informational 6240x4160 AVIF q95 speed', async () => {
  const data = syntheticImage(6240, 4160, 20261005, 6);
  const startWasm = performance.now();
  wasm.encode_image(data, new Float32Array(0), 6240, 4160, 'Normal', 'avif', 95, 1000);
  const wasmMs = performance.now() - startWasm;
  const startNative = performance.now();
  await native.encode(data, null, { format: 'avif', width: 6240, height: 4160, orientation: 'Normal', quality: 95, peakLuminance: 1000 });
  const nativeMs = performance.now() - startNative;
  fs.mkdirSync(evidenceDir, { recursive: true });
  fs.writeFileSync(path.join(evidenceDir, variant === 'default' ? 'task4-speed.json' : `task4-speed-${variant}.json`), JSON.stringify({ wasmMs, nativeMs, ratio: wasmMs / nativeMs }, null, 2));
});
