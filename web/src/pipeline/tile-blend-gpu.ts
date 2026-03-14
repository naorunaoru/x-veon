// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-present X-Veon contributors
/**
 * GPU compute pipeline for tile blending (overlap-add after neural demosaic).
 *
 * Replaces the CPU createTileBlender with three GPU compute passes:
 *   1. Accumulate: weighted scatter of each tile into the padded output buffer
 *   2. Finalize:   divide accumulated values by weight sums (in-place)
 *   3. Crop:       extract the original (unpadded) region as HWC
 *
 * The crop output is a GPUBuffer that can be passed directly to gpuPostprocess,
 * eliminating a GPU→CPU→GPU round-trip.
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

interface BlendPipelines {
  accumulate: GPUComputePipeline;
  finalizeBlend: GPUComputePipeline;
  crop: GPUComputePipeline;
}

const blendPipelineCache = new WeakMap<GPUDevice, BlendPipelines>();

function makePipeline(device: GPUDevice, code: string): GPUComputePipeline {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
}

function getBlendPipelines(device: GPUDevice): BlendPipelines {
  let p = blendPipelineCache.get(device);
  if (p) return p;
  p = {
    accumulate: makePipeline(device, SHADER_ACCUMULATE),
    finalizeBlend: makePipeline(device, SHADER_FINALIZE_BLEND),
    crop: makePipeline(device, SHADER_CROP),
  };
  blendPipelineCache.set(device, p);
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

export interface GpuTileBlender {
  /**
   * Upload and accumulate a batch of tiles in a single GPU submit.
   * @param batchOut  Contiguous CHW tile data (count × 3 × ps² floats)
   * @param tiles     Full tile position array
   * @param startIdx  Index of first tile in this batch
   * @param count     Number of tiles in this batch
   */
  accumulateBatch(
    batchOut: Float32Array,
    tiles: ReadonlyArray<{ x: number; y: number }>,
    startIdx: number,
    count: number,
  ): void;
  /**
   * Finalize blending, crop to original size, and return a GPUBuffer
   * containing the cropped HWC float32 data ready for gpuPostprocess.
   */
  finalize(): Promise<GPUBuffer>;
  /** Release all GPU resources. */
  destroy(): void;
}

export function createGpuTileBlender(
  device: GPUDevice,
  hPad: number, wPad: number,
  patchSize: number, overlap: number,
  padTop: number, padLeft: number,
  hOrig: number, wOrig: number,
  maxBatch: number,
): GpuTileBlender {
  const pipes = getBlendPipelines(device);

  const S = GPUBufferUsage.STORAGE;
  const D = GPUBufferUsage.COPY_DST;
  const U = GPUBufferUsage.UNIFORM;

  const pp = patchSize * patchSize;
  const padPixels = hPad * wPad;

  // --- Precompute 2D weight grid (same logic as CPU version) ---
  const w2dCpu = new Float32Array(pp);
  const w1d = new Float32Array(patchSize);
  w1d.fill(1);
  for (let i = 0; i < overlap; i++) {
    w1d[i] = i / overlap;
    w1d[patchSize - 1 - i] = i / overlap;
  }
  for (let py = 0; py < patchSize; py++) {
    const wy = w1d[py];
    const row = py * patchSize;
    for (let px = 0; px < patchSize; px++) {
      w2dCpu[row + px] = wy * w1d[px];
    }
  }

  // --- Buffer strides (aligned for WebGPU offset requirements) ---
  const tileByteSize = pp * 3 * 4;
  const tileStride = roundUp(tileByteSize, BUF_ALIGN);
  const paramStride = roundUp(16, BUF_ALIGN); // 4 × u32 = 16 bytes, padded to 256

  // --- GPU buffers ---
  const outputBuf = buf(device, padPixels * 3 * 4, S);       // zero-initialized by WebGPU
  const weightsBuf = buf(device, padPixels * 4, S);           // zero-initialized by WebGPU
  const w2dBuf = buf(device, pp * 4, S | D);
  device.queue.writeBuffer(w2dBuf, 0, asGpu(w2dCpu));

  const batchTileBuf = buf(device, maxBatch * tileStride, S | D);
  const batchParamBuf = buf(device, maxBatch * paramStride, U | D);
  const cropOutBuf = buf(device, hOrig * wOrig * 3 * 4, S);

  const accWg = Math.ceil(patchSize / WG);
  const allBufs = [outputBuf, weightsBuf, w2dBuf, batchTileBuf, batchParamBuf, cropOutBuf];

  return {
    accumulateBatch(
      batchOut: Float32Array,
      tiles: ReadonlyArray<{ x: number; y: number }>,
      startIdx: number,
      count: number,
    ): void {
      // Upload all tile data into the batch buffer (one writeBuffer per tile
      // to handle alignment gaps; the actual GPU work is a single submit).
      for (let i = 0; i < count; i++) {
        device.queue.writeBuffer(
          batchTileBuf, i * tileStride,
          asGpu(batchOut.subarray(i * pp * 3, (i + 1) * pp * 3)),
        );
      }

      // Upload all tile params (256-byte aligned per entry)
      const paramData = new ArrayBuffer(count * paramStride);
      for (let i = 0; i < count; i++) {
        new Uint32Array(paramData, i * paramStride, 4).set([
          tiles[startIdx + i].x, tiles[startIdx + i].y, patchSize, wPad,
        ]);
      }
      device.queue.writeBuffer(batchParamBuf, 0, paramData, 0, count * paramStride);

      // Record all dispatches in one command buffer (serialized — no atomics needed)
      const enc = device.createCommandEncoder();
      for (let i = 0; i < count; i++) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipes.accumulate);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipes.accumulate.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: batchTileBuf, offset: i * tileStride, size: tileByteSize } },
            { binding: 1, resource: { buffer: w2dBuf } },
            { binding: 2, resource: { buffer: outputBuf } },
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
      const finParamBuf = buf(device, 8, U | D);
      device.queue.writeBuffer(finParamBuf, 0, asGpu(new Uint32Array([wPad, hPad])));
      allBufs.push(finParamBuf);

      // --- Crop pass: padded → original size ---
      const cropParamBuf = buf(device, 24, U | D);
      device.queue.writeBuffer(cropParamBuf, 0, asGpu(new Uint32Array([
        wPad, padTop, padLeft, wOrig, hOrig, 0,
      ])));
      allBufs.push(cropParamBuf);

      const enc = device.createCommandEncoder();

      let pass = enc.beginComputePass();
      pass.setPipeline(pipes.finalizeBlend);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipes.finalizeBlend.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outputBuf } },
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
          { binding: 0, resource: { buffer: outputBuf } },
          { binding: 1, resource: { buffer: cropOutBuf } },
          { binding: 2, resource: { buffer: cropParamBuf } },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(wOrig / WG), Math.ceil(hOrig / WG));
      pass.end();

      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();

      // Free intermediate buffers, keep cropOutBuf for caller
      outputBuf.destroy();
      weightsBuf.destroy();
      w2dBuf.destroy();
      batchTileBuf.destroy();
      batchParamBuf.destroy();
      finParamBuf.destroy();
      cropParamBuf.destroy();

      // Remove cropOutBuf from allBufs — ownership transfers to the caller.
      // Without this, destroy() would invalidate the returned buffer.
      const idx = allBufs.indexOf(cropOutBuf);
      if (idx !== -1) allBufs.splice(idx, 1);

      return cropOutBuf;
    },

    destroy(): void {
      for (const b of allBufs) {
        if (!b.mapState || b.mapState === 'unmapped') {
          try { b.destroy(); } catch { /* already destroyed */ }
        }
      }
    },
  };
}
