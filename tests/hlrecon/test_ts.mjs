#!/usr/bin/env node
/**
 * Node.js test runner for the TypeScript segmentation-based highlight
 * reconstruction.  Reads the same binary fixture as the C harness, runs the
 * TS implementation, and writes the result as flat float32.
 *
 * Usage:
 *   npx tsx tests/hlrecon/test_ts.mjs fixture.bin output_ts.bin
 */

import { readFileSync, writeFileSync } from 'fs';
import { reconstructHighlightsCfa } from '../../web/src/pipeline/preprocessor.ts';

const [,, fixturePath, outputPath] = process.argv;
if (!fixturePath || !outputPath) {
  console.error('Usage: npx tsx tests/hlrecon/test_ts.mjs <fixture.bin> <output.bin>');
  process.exit(1);
}

// Read fixture
const buf = readFileSync(fixturePath);
const header = new DataView(buf.buffer, buf.byteOffset, 256);

const width = header.getUint32(0, true);
const height = header.getUint32(4, true);
const period = header.getUint32(8, true);
const dy = header.getUint32(12, true);
const dx = header.getUint32(16, true);

// Read pattern
const pattern = [];
for (let y = 0; y < period; y++) {
  const row = [];
  for (let x = 0; x < period; x++) {
    row.push(buf[20 + y * period + x]);
  }
  pattern.push(row);
}

console.log(`Fixture: ${width}x${height}, period=${period}, shift=dy${dy} dx${dx}`);
console.log('Pattern:', pattern);

// Read CFA data
const cfaBytes = new Uint8Array(buf.buffer, buf.byteOffset + 256, width * height * 4);
const cfa = new Float32Array(cfaBytes.buffer, cfaBytes.byteOffset, width * height);

// Make a working copy; keep original for refavg
const original = new Float32Array(cfa);
const working = new Float32Array(cfa);

// Phase 1: Run opposed inpainting first (same as C harness)
reconstructHighlightsCfa(working, width, height, pattern, period, dy, dx);

// Phase 2: Run segmentation-based reconstruction (original for refavg)
const { _reconstructDebug } = await import('../../web/src/pipeline/highlight-segments.ts');
const dbg = _reconstructDebug(working, width, height, pattern, period, dy, dx, 2, 0.5, original);
console.log(`Clipped superpixels: ${dbg.anyClipped}`);
console.log(`Segments: R=${dbg.segCounts[0]} G=${dbg.segCounts[1]} B=${dbg.segCounts[2]}`);
for (let c = 0; c < 3; c++) {
  for (const { id, val1, val2 } of dbg.candidates[c]) {
    console.log(`  [${'RGB'[c]}] seg ${id}: val1=${val1.toFixed(6)} val2=${val2.toFixed(6)}`);
  }
}

// Write output
const outBuf = Buffer.from(working.buffer, working.byteOffset, working.byteLength);
writeFileSync(outputPath, outBuf);
console.log(`Output written to ${outputPath}`);
