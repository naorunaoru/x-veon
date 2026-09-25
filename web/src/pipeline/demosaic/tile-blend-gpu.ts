// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-present X-Veon contributors
/**
 * Fully GPU-resident neural demosaic pipeline:
 *   0. Extract:    tile the CFA image → 5-channel NCHW batch for inference
 *   1. Accumulate: weighted scatter of each tile into the padded output buffer
 *   2. Finalize:   divide accumulated values by weight sums (in-place)
 *   3. Crop:       extract the original (unpadded) region as HWC
 *
 * Everything stays on GPU — no CPU↔GPU transfers per batch.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WG = 16;
const BUF_ALIGN = 256; // WebGPU min offset alignment (storage + uniform)

function roundUp(x: number, a: number): number {
  return Math.ceil(x / a) * a;
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

/**
 * Extract tiles from a CFA image into 5-channel NCHW batch format.
 * Dispatched as (ceil(ps/WG), ceil(ps/WG), count) — one z-slice per tile.
 * Channels: [CFA value, R mask, G mask, B mask, clip ratio].
 * ORT requires the output buffer to have STORAGE | COPY_SRC usage.
 */
const SHADER_EXTRACT = /* wgsl */ `
struct ExtractParams {
  ps: u32,
  cfa_w: u32,
  cfa_h: u32,
  count: u32,
  tile_stride: u32,   // floats between consecutive tiles (aligned)
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> cfa: array<f32>;
@group(0) @binding(1) var<storage, read> masks: array<f32>;     // 3 × ps² (R, G, B)
@group(0) @binding(2) var<storage, read> tile_pos: array<u32>;  // count × 2 (x, y)
@group(0) @binding(3) var<storage, read> clips: array<f32>;     // [clipR, clipG, clipB]
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: ExtractParams;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  let tile_idx = gid.z;
  if (px >= params.ps || py >= params.ps || tile_idx >= params.count) { return; }

  let n = params.ps * params.ps;
  let ti = py * params.ps + px;

  // Tile origin in CFA image
  let tx = tile_pos[tile_idx * 2u];
  let ty = tile_pos[tile_idx * 2u + 1u];
  let src_x = tx + px;
  let src_y = ty + py;

  // Read CFA value (zero if out-of-bounds)
  var val = 0.0;
  if (src_x < params.cfa_w && src_y < params.cfa_h) {
    val = cfa[src_y * params.cfa_w + src_x];
  }

  // Read channel masks
  let mr = masks[ti];
  let mg = masks[n + ti];
  let mb = masks[2u * n + ti];

  // Clip threshold for this pixel's channel (exactly one mask is 1.0)
  let cl = mr * clips[0] + mg * clips[1] + mb * clips[2];

  // Clip ratio: 0 below 50% of clip, ramps 0→1 from 50%→100%
  let ratio = max(min(val / cl, 1.0) * 2.0 - 1.0, 0.0);

  // Write 5 channels in NCHW layout
  let base = tile_idx * params.tile_stride;
  out[base + ti]           = val;
  out[base + n + ti]       = mr;
  out[base + 2u * n + ti]  = mg;
  out[base + 3u * n + ti]  = mb;
  out[base + 4u * n + ti]  = ratio;
}
`;

/**
 * Accumulate one tile into the padded output buffer.
 * Dispatched once per tile — each thread handles one pixel in the tile.
 * No atomics needed: dispatches are serialized within the command buffer.
 */
const SHADER_ACCUMULATE = /* wgsl */ `
struct TileParams {
  tx: u32,
  ty: u32,
  patch_size: u32,
  w_pad: u32,
}

@group(0) @binding(0) var<storage, read> tile: array<f32>;
@group(0) @binding(1) var<storage, read> w2d: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@group(0) @binding(4) var<uniform> params: TileParams;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.patch_size || py >= params.patch_size) { return; }

  let pp = params.patch_size * params.patch_size;
  let ti = py * params.patch_size + px;
  let w = w2d[ti];

  let oy = params.ty + py;
  let ox = params.tx + px;
  let oi = (oy * params.w_pad + ox) * 3u;
  let wi = oy * params.w_pad + ox;

  output[oi]      += tile[ti] * w;
  output[oi + 1u] += tile[pp + ti] * w;
  output[oi + 2u] += tile[2u * pp + ti] * w;
  weights[wi] += w;
}
`;

/**
 * Finalize: divide each pixel by accumulated weight.
 * Dispatched over the full padded image.
 */
const SHADER_FINALIZE_BLEND = /* wgsl */ `
struct Params {
  w_pad: u32,
  h_pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.w_pad || gid.y >= params.h_pad) { return; }
  let idx = gid.y * params.w_pad + gid.x;
  let w = weights[idx];
  if (w > 1e-8) {
    let inv = 1.0 / w;
    let i3 = idx * 3u;
    output[i3]      *= inv;
    output[i3 + 1u] *= inv;
    output[i3 + 2u] *= inv;
  }
}
`;

/**
 * Crop padded HWC buffer to original dimensions.
 * Output is directly consumable by gpuPostprocess.
 */
const SHADER_CROP = /* wgsl */ `
struct CropParams {
  w_pad: u32,
  pad_top: u32,
  pad_left: u32,
  w_orig: u32,
  h_orig: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: CropParams;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.w_orig || gid.y >= params.h_orig) { return; }
  let src_idx = ((gid.y + params.pad_top) * params.w_pad + gid.x + params.pad_left) * 3u;
  let dst_idx = (gid.y * params.w_orig + gid.x) * 3u;
  dst[dst_idx]      = src[src_idx];
  dst[dst_idx + 1u] = src[src_idx + 1u];
  dst[dst_idx + 2u] = src[src_idx + 2u];
}
`;

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------

interface Pipelines {
  extract: GPUComputePipeline;
  accumulate: GPUComputePipeline;
  finalizeBlend: GPUComputePipeline;
  crop: GPUComputePipeline;
}

const pipelineCache = new WeakMap<GPUDevice, Pipelines>();

function makePipeline(device: GPUDevice, code: string): GPUComputePipeline {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
}

function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (p) return p;
  p = {
    extract: makePipeline(device, SHADER_EXTRACT),
    accumulate: makePipeline(device, SHADER_ACCUMULATE),
    finalizeBlend: makePipeline(device, SHADER_FINALIZE_BLEND),
    crop: makePipeline(device, SHADER_CROP),
  };
  pipelineCache.set(device, p);
  return p;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buf(device: GPUDevice, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ size: Math.max(size, 4), usage });
}

function asGpu(data: ArrayBufferView): GPUAllowSharedBufferSource {
  return data as unknown as GPUAllowSharedBufferSource;
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface GpuNNPipeline {
  /**
   * Extract a batch of tiles from the CFA image on the GPU.
   * Returns a GPUBuffer in [count, 5, ps, ps] NCHW layout ready for ORT.
   * The buffer has STORAGE | COPY_SRC usage as required by ort.Tensor.fromGpuBuffer.
   */
  extractBatch(startIdx: number, count: number): GPUBuffer;
  /**
   * Accumulate inference output (CHW, 3×ps² per tile) into the blend buffer.
   * @param inferOut  GPUBuffer from ORT inference output (count × 3 × ps²)
   * @param startIdx  Index of first tile in this batch
   * @param count     Number of tiles in this batch
   */
  accumulateBatch(inferOut: GPUBuffer, startIdx: number, count: number): void;
  /**
   * Finalize blending, crop to original size, and return a GPUBuffer
   * containing the cropped HWC float32 data ready for gpuPostprocess.
   */
  finalize(): Promise<GPUBuffer>;
  /** Release all GPU resources. */
  destroy(): void;
}

/**
 * 1-D tile blend weights: linear ramps over the overlap, 1 elsewhere. Every weight is positive
 * (a ramp of i / overlap gave the tile edge weight 0, and since tiles start at 0 the image's
 * first row and column had no weight at all and came out black). Opposing ramps sum to 1.
 */
export function blendWeights1d(patchSize: number, overlap: number): Float32Array {
  const w = new Float32Array(patchSize).fill(1);
  for (let i = 0; i < overlap; i++) {
    w[i] = (i + 1) / (overlap + 1);
    w[patchSize - 1 - i] = (i + 1) / (overlap + 1);
  }
  return w;
}

export function createGpuNNPipeline(
  device: GPUDevice,
  cfa: Float32Array, cfaW: number, cfaH: number,
  masks: { r: Float32Array; g: Float32Array; b: Float32Array },
  clips: readonly [number, number, number],
  tiles: ReadonlyArray<{ x: number; y: number }>,
  hPad: number, wPad: number,
  patchSize: number, overlap: number,
  padTop: number, padLeft: number,
  hOrig: number, wOrig: number,
  maxBatch: number,
): GpuNNPipeline {
  const pipes = getPipelines(device);

  const S = GPUBufferUsage.STORAGE;
  const D = GPUBufferUsage.COPY_DST;
  const C = GPUBufferUsage.COPY_SRC;
  const U = GPUBufferUsage.UNIFORM;

  const pp = patchSize * patchSize;
  const padPixels = hPad * wPad;

  // --- Precompute 2D weight grid (same logic as CPU version) ---
  const w2dCpu = new Float32Array(pp);
  const w1d = blendWeights1d(patchSize, overlap);
  for (let py = 0; py < patchSize; py++) {
    const wy = w1d[py];
    const row = py * patchSize;
    for (let px = 0; px < patchSize; px++) {
      w2dCpu[row + px] = wy * w1d[px];
    }
  }

  // --- Buffer strides (aligned for WebGPU offset requirements) ---
  // ORT output: 3 channels per tile (CHW)
  const outTileBytes = pp * 3 * 4;
  const outTileStride = roundUp(outTileBytes, BUF_ALIGN);
  // Extract output: 5 channels per tile (NCHW) — tile_stride in floats
  const extractTileFloats = 5 * pp;
  const extractTileBytes = extractTileFloats * 4;
  const extractTileStride = roundUp(extractTileBytes, BUF_ALIGN);
  const extractTileStrideFloats = extractTileStride / 4;
  const paramStride = roundUp(16, BUF_ALIGN);

  // --- Allocation tracking ---
  // Every buffer this pipeline owns, so a partial construction failure releases what was
  // already created (instead of leaking it — the pipeline object is never returned to the
  // caller in that case), and so finalize()/destroy() each release a buffer at most once.
  const owned = new Set<GPUBuffer>();

  function alloc(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const b = buf(device, size, usage);
    owned.add(b);
    return b;
  }

  /** Destroy one tracked buffer and stop tracking it. No-op if already released. */
  function release(b: GPUBuffer): void {
    if (!owned.delete(b)) return;
    try { b.destroy(); } catch { /* already destroyed */ }
  }

  const {
    cfaBuf, masksBuf, tilePosCpu, tilePosBuf, clipsBuf, extractParamBuf,
    blendOutBuf, weightsBuf, w2dBuf, batchParamBuf, cropOutBuf,
  } = (() => {
    try {
      // --- One-time GPU uploads ---
      const cfaBuf = alloc(cfaW * cfaH * 4, S | D);
      device.queue.writeBuffer(cfaBuf, 0, asGpu(cfa));

      // Channel masks: pack R, G, B contiguously (3 × ps²)
      const masksCpu = new Float32Array(3 * pp);
      masksCpu.set(masks.r, 0);
      masksCpu.set(masks.g, pp);
      masksCpu.set(masks.b, 2 * pp);
      const masksBuf = alloc(3 * pp * 4, S | D);
      device.queue.writeBuffer(masksBuf, 0, asGpu(masksCpu));

      // Tile positions: flat array of [x0, y0, x1, y1, ...]
      const tilePosCpu = new Uint32Array(tiles.length * 2);
      for (let i = 0; i < tiles.length; i++) {
        tilePosCpu[i * 2] = tiles[i].x;
        tilePosCpu[i * 2 + 1] = tiles[i].y;
      }
      const tilePosBuf = alloc(tilePosCpu.byteLength, S | D);
      device.queue.writeBuffer(tilePosBuf, 0, asGpu(tilePosCpu));

      // Clip thresholds
      const clipsBuf = alloc(12, S | D);
      device.queue.writeBuffer(clipsBuf, 0, asGpu(new Float32Array(clips)));

      // Extract params (uniform, rewritten per batch)
      const extractParamBuf = alloc(32, U | D);

      // --- Blend buffers ---
      const blendOutBuf = alloc(padPixels * 3 * 4, S);
      const weightsBuf = alloc(padPixels * 4, S);
      const w2dBuf = alloc(pp * 4, S | D);
      device.queue.writeBuffer(w2dBuf, 0, asGpu(w2dCpu));

      const batchParamBuf = alloc(maxBatch * paramStride, U | D);
      const cropOutBuf = alloc(hOrig * wOrig * 3 * 4, S);

      return {
        cfaBuf, masksBuf, tilePosCpu, tilePosBuf, clipsBuf, extractParamBuf,
        blendOutBuf, weightsBuf, w2dBuf, batchParamBuf, cropOutBuf,
      };
    } catch (e) {
      for (const b of owned) {
        try { b.destroy(); } catch { /* already destroyed */ }
      }
      owned.clear();
      throw e;
    }
  })();

  const accWg = Math.ceil(patchSize / WG);

  return {
    extractBatch(startIdx: number, count: number): GPUBuffer {
      // Allocate output buffer with STORAGE | COPY_SRC (ORT requirement)
      const extractOut = alloc(count * extractTileStride, S | C);

      // Write extract params
      device.queue.writeBuffer(extractParamBuf, 0, asGpu(new Uint32Array([
        patchSize, cfaW, cfaH, count, extractTileStrideFloats, 0, 0, 0,
      ])));

      // Write tile positions for this batch (as an offset view into tilePosBuf)
      // The shader indexes tile_pos[tile_idx * 2], so we pass a view starting at startIdx
      const batchPosCpu = tilePosCpu.subarray(startIdx * 2, (startIdx + count) * 2);
      const batchPosBuf = alloc(batchPosCpu.byteLength, S | D);
      device.queue.writeBuffer(batchPosBuf, 0, asGpu(batchPosCpu));

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipes.extract);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipes.extract.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: cfaBuf } },
          { binding: 1, resource: { buffer: masksBuf } },
          { binding: 2, resource: { buffer: batchPosBuf } },
          { binding: 3, resource: { buffer: clipsBuf } },
          { binding: 4, resource: { buffer: extractOut } },
          { binding: 5, resource: { buffer: extractParamBuf } },
        ],
      }));
      pass.dispatchWorkgroups(accWg, accWg, count);
      pass.end();
      device.queue.submit([enc.finish()]);

      return extractOut;
    },

    accumulateBatch(inferOut: GPUBuffer, startIdx: number, count: number): void {
      // Upload tile params for accumulate
      const paramData = new ArrayBuffer(count * paramStride);
      for (let i = 0; i < count; i++) {
        new Uint32Array(paramData, i * paramStride, 4).set([
          tiles[startIdx + i].x, tiles[startIdx + i].y, patchSize, wPad,
        ]);
      }
      device.queue.writeBuffer(batchParamBuf, 0, paramData, 0, count * paramStride);

      // ORT output is [count, 3, ps, ps] — contiguous CHW, outTileBytes per tile.
      // Bind each tile's slice via offset into inferOut.
      const enc = device.createCommandEncoder();
      for (let i = 0; i < count; i++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipes.accumulate);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipes.accumulate.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inferOut, offset: i * outTileStride, size: outTileBytes } },
            { binding: 1, resource: { buffer: w2dBuf } },
            { binding: 2, resource: { buffer: blendOutBuf } },
            { binding: 3, resource: { buffer: weightsBuf } },
            { binding: 4, resource: { buffer: batchParamBuf, offset: i * paramStride, size: 16 } },
          ],
        }));
        pass.dispatchWorkgroups(accWg, accWg);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
    },

    async finalize(): Promise<GPUBuffer> {
      // --- Finalize pass: divide by weights ---
      const finParamBuf = alloc(8, U | D);
      device.queue.writeBuffer(finParamBuf, 0, asGpu(new Uint32Array([wPad, hPad])));

      // --- Crop pass: padded → original size ---
      const cropParamBuf = alloc(24, U | D);
      device.queue.writeBuffer(cropParamBuf, 0, asGpu(new Uint32Array([
        wPad, padTop, padLeft, wOrig, hOrig, 0,
      ])));

      const enc = device.createCommandEncoder();

      let pass = enc.beginComputePass();
      pass.setPipeline(pipes.finalizeBlend);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipes.finalizeBlend.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blendOutBuf } },
          { binding: 1, resource: { buffer: weightsBuf } },
          { binding: 2, resource: { buffer: finParamBuf } },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(wPad / WG), Math.ceil(hPad / WG));
      pass.end();

      pass = enc.beginComputePass();
      pass.setPipeline(pipes.crop);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipes.crop.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blendOutBuf } },
          { binding: 1, resource: { buffer: cropOutBuf } },
          { binding: 2, resource: { buffer: cropParamBuf } },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(wOrig / WG), Math.ceil(hOrig / WG));
      pass.end();

      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      // Free every tracked buffer except cropOutBuf, whose ownership transfers to the caller.
      // Releasing removes each buffer from `owned` as it's freed, so a later destroy() call
      // (e.g. from an unrelated failure elsewhere) finds nothing left to double-free.
      for (const b of [...owned]) {
        if (b !== cropOutBuf) release(b);
      }
      owned.delete(cropOutBuf);

      return cropOutBuf;
    },

    destroy(): void {
      for (const b of owned) {
        try { b.destroy(); } catch { /* already destroyed */ }
      }
      owned.clear();
    },
  };
}
