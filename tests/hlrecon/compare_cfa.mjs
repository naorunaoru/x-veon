#!/usr/bin/env node
/**
 * Compare the CFA data produced by the web pipeline's WASM decoder
 * with the Python-extracted fixture (rawpy-based).
 *
 * Usage:
 *   node tests/hlrecon/compare_cfa.mjs <raw_file> <fixture.bin>
 */

import { readFileSync } from 'fs';
import { resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmPkgDir = resolve(__dirname, '../../web/wasm-pkg');

// Load WASM module synchronously
const wasmBytes = readFileSync(join(wasmPkgDir, 'rawloader_bg.wasm'));
const { initSync, decode_image } = await import(join(wasmPkgDir, 'rawloader.js'));
initSync(wasmBytes);

const [,, rawPath, fixturePath] = process.argv;
if (!rawPath || !fixturePath) {
  console.error('Usage: node compare_cfa.mjs <raw_file> <fixture.bin>');
  process.exit(1);
}

// ---- Decode RAW via WASM (same as web pipeline) ----
const rawBytes = new Uint8Array(readFileSync(rawPath));
const img = decode_image(rawBytes);

const fullW = img.get_width();
const fullH = img.get_height();
const rawData = img.get_data();       // Uint16Array
const crops = img.get_crops();        // [top, right, bottom, left]
const blackLevels = img.get_blacklevels();
const whiteLevels = img.get_whitelevels();
const cfaStr = img.get_cfastr();
const cfaWidth = img.get_cfawidth();
const wbCoeffs = img.get_wb_coeffs();

console.log(`WASM: ${img.get_make()} ${img.get_model()}`);
console.log(`  Full: ${fullW}x${fullH}`);
console.log(`  Crops: top=${crops[0]} right=${crops[1]} bottom=${crops[2]} left=${crops[3]}`);
console.log(`  Black: [${blackLevels}]  White: [${whiteLevels}]`);
console.log(`  CFA: "${cfaStr}" (width=${cfaWidth})`);
console.log(`  WB: [${wbCoeffs}]`);

// Crop to visible (same as preprocessor.ts cropToVisible)
const top = crops[0], right = crops[1], bottom = crops[2], left = crops[3];
const visW = fullW - left - right;
const visH = fullH - top - bottom;
let visData;
if (top === 0 && right === 0 && bottom === 0 && left === 0) {
  visData = rawData;
} else {
  visData = new Uint16Array(visW * visH);
  for (let y = 0; y < visH; y++) {
    const srcOff = (y + top) * fullW + left;
    visData.set(rawData.subarray(srcOff, srcOff + visW), y * visW);
  }
}
console.log(`  Visible: ${visW}x${visH}`);

// Normalize (same as preprocessor.ts normalizeRawCfa)
const black = blackLevels[0];
const white = whiteLevels[0];
const range = white - black;
const wasmCfa = new Float32Array(visW * visH);
for (let i = 0; i < visW * visH; i++) {
  wasmCfa[i] = (visData[i] - black) / range;
}

// Find pattern shift (same as preprocessor.ts findPatternShift)
const BAYER_PATTERN = [[0, 1], [1, 2]];
const period = cfaWidth;
function parseCfa(str, w) {
  const h = str.length / w;
  const pat = [];
  for (let y = 0; y < h; y++) {
    pat[y] = [];
    for (let x = 0; x < w; x++) {
      const ch = str[y * w + x];
      pat[y][x] = ch === 'R' ? 0 : ch === 'G' ? 1 : 2;
    }
  }
  return pat;
}

const rawPattern = parseCfa(cfaStr, cfaWidth);
// Apply crop offset to visible pattern
const vis = [];
for (let y = 0; y < period; y++) {
  vis[y] = [];
  for (let x = 0; x < period; x++) {
    vis[y][x] = rawPattern[(y + top) % period][(x + left) % period];
  }
}

// Find shift
let wasmDy = -1, wasmDx = -1;
for (let dy = 0; dy < period; dy++) {
  for (let dx = 0; dx < period; dx++) {
    let match = true;
    for (let y = 0; y < period && match; y++)
      for (let x = 0; x < period && match; x++)
        if (BAYER_PATTERN[(y + dy) % period][(x + dx) % period] !== vis[y][x])
          match = false;
    if (match) { wasmDy = dy; wasmDx = dx; break; }
  }
  if (wasmDy >= 0) break;
}
console.log(`  Pattern shift: dy=${wasmDy} dx=${wasmDx}`);

// ---- Read Python fixture ----
const fixBuf = readFileSync(fixturePath);
const hdr = new DataView(fixBuf.buffer, fixBuf.byteOffset, 256);
const fW = hdr.getUint32(0, true);
const fH = hdr.getUint32(4, true);
const fPeriod = hdr.getUint32(8, true);
const fDy = hdr.getUint32(12, true);
const fDx = hdr.getUint32(16, true);

const fPatFlat = [];
for (let i = 0; i < fPeriod * fPeriod; i++) fPatFlat.push(fixBuf[20 + i]);
console.log(`\nFixture: ${fW}x${fH}, period=${fPeriod}, shift=dy${fDy} dx${fDx}`);
console.log(`  Pattern: [${fPatFlat}]`);

const fixCfaBytes = new Uint8Array(fixBuf.buffer, fixBuf.byteOffset + 256, fW * fH * 4);
const fixCfa = new Float32Array(fixCfaBytes.buffer, fixCfaBytes.byteOffset, fW * fH);

// ---- Compare ----
console.log('\n--- Dimension check ---');
console.log(`  WASM: ${visW}x${visH}  Fixture: ${fW}x${fH}  Match: ${visW === fW && visH === fH}`);
console.log(`  WASM shift: dy=${wasmDy} dx=${wasmDx}  Fixture shift: dy=${fDy} dx=${fDx}  Match: ${wasmDy === fDy && wasmDx === fDx}`);

if (visW !== fW || visH !== fH) {
  console.log('  ** DIMENSION MISMATCH **');
}

// Compare overlapping region
const cmpW = Math.min(visW, fW);
const cmpH = Math.min(visH, fH);
const npix = cmpW * cmpH;
console.log(`  Comparing overlapping region: ${cmpW}x${cmpH} (${npix} pixels)`);

// Stats on WASM CFA (full image)
let wMin = Infinity, wMax = -Infinity, wSum = 0, wClipped = 0;
for (let i = 0; i < visW * visH; i++) {
  const v = wasmCfa[i];
  if (v < wMin) wMin = v;
  if (v > wMax) wMax = v;
  wSum += v;
  if (v >= 1.0) wClipped++;
}

// Stats on fixture CFA (full image)
let fMin = Infinity, fMax = -Infinity, fSum = 0, fClipped = 0;
for (let i = 0; i < fW * fH; i++) {
  const v = fixCfa[i];
  if (v < fMin) fMin = v;
  if (v > fMax) fMax = v;
  fSum += v;
  if (v >= 1.0) fClipped++;
}

console.log('\n--- WASM CFA stats (full) ---');
console.log(`  Min: ${wMin.toFixed(6)}  Max: ${wMax.toFixed(6)}  Mean: ${(wSum / (visW * visH)).toFixed(6)}  Clipped: ${wClipped}`);

console.log('\n--- Fixture CFA stats (full) ---');
console.log(`  Min: ${fMin.toFixed(6)}  Max: ${fMax.toFixed(6)}  Mean: ${(fSum / (fW * fH)).toFixed(6)}  Clipped: ${fClipped}`);

// Pixel-by-pixel diff over overlapping region (need to use per-row access)
let maxDiff = 0, sumDiff = 0, nDiff = 0;
let maxDiffY = 0, maxDiffX = 0;
let negCount = 0;

for (let y = 0; y < cmpH; y++) {
  for (let x = 0; x < cmpW; x++) {
    const wVal = wasmCfa[y * visW + x];
    const fVal = fixCfa[y * fW + x];
    const d = Math.abs(wVal - fVal);
    if (d > maxDiff) { maxDiff = d; maxDiffY = y; maxDiffX = x; }
    sumDiff += d;
    if (d > 1e-6) nDiff++;
    if (wVal < 0 && fVal >= 0) negCount++;
  }
}

console.log('\n--- Pixel diff (overlapping region) ---');
console.log(`  MAE:  ${(sumDiff / npix).toFixed(9)}`);
console.log(`  Max:  ${maxDiff.toFixed(9)} at (y=${maxDiffY}, x=${maxDiffX})`);
console.log(`  Pixels with |diff| > 1e-6: ${nDiff} (${(100 * nDiff / npix).toFixed(2)}%)`);
console.log(`  WASM negative / fixture non-negative: ${negCount}`);

if (maxDiff > 1e-6) {
  const wVal = wasmCfa[maxDiffY * visW + maxDiffX];
  const fVal = fixCfa[maxDiffY * fW + maxDiffX];
  console.log(`\n  Worst pixel: WASM=${wVal.toFixed(9)}  Fixture=${fVal.toFixed(9)}`);

  // Show first 10 differing pixels
  console.log('\n  First 10 differing pixels:');
  let shown = 0;
  outer: for (let y = 0; y < cmpH; y++) {
    for (let x = 0; x < cmpW; x++) {
      const wv = wasmCfa[y * visW + x];
      const fv = fixCfa[y * fW + x];
      const d = Math.abs(wv - fv);
      if (d > 1e-6) {
        console.log(`    [${y},${x}] WASM=${wv.toFixed(6)}  Fix=${fv.toFixed(6)}  diff=${d.toFixed(6)}`);
        if (++shown >= 10) break outer;
      }
    }
  }
}

// Check extra columns in WASM (columns cmpW..visW-1)
if (visW > fW) {
  console.log(`\n--- Extra WASM columns (${fW}..${visW - 1}) ---`);
  let extraMin = Infinity, extraMax = -Infinity, extraSum = 0;
  const extraPix = (visW - fW) * visH;
  for (let y = 0; y < visH; y++) {
    for (let x = fW; x < visW; x++) {
      const v = wasmCfa[y * visW + x];
      if (v < extraMin) extraMin = v;
      if (v > extraMax) extraMax = v;
      extraSum += v;
    }
  }
  console.log(`  Min: ${extraMin.toFixed(6)}  Max: ${extraMax.toFixed(6)}  Mean: ${(extraSum / extraPix).toFixed(6)}`);
}
