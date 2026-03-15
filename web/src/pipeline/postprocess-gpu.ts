// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-present X-Veon contributors
/**
 * GPU compute pipeline for post-demosaic processing.
 *
 * Runs in a single command buffer:
 *   1. White balance (per-pixel multiply, in-place)
 *   2–5. Inpaint-opposed highlight recovery (Pass 1 only, no segmentation)
 *   6. Finalize: HL extension + color correction + DR gain → RGBA output
 *
 * Pipeline: raw CFA → neural demosaic → [this module] → display renderer.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WG = 16;

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

const SHADER_WB = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  wb_r: f32,
  wb_g: f32,
  wb_b: f32,
  _pad: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let base = (gid.y * params.width + gid.x) * 3u;
  data[base] *= params.wb_r;
  data[base + 1u] *= params.wb_g;
  data[base + 2u] *= params.wb_b;
}
`;

const SHADER_REFAVG_CLIP = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  clip_r: f32,
  clip_g: f32,
  clip_b: f32,
  _pad: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> refavg: array<f32>;
@group(0) @binding(2) var<storage, read_write> clip_mask: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn cbrt(x: f32) -> f32 {
  if (x <= 0.0) { return 0.0; }
  return pow(x, 1.0 / 3.0);
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let w = params.width;
  let h = params.height;
  var sr: f32 = 0.0; var sg: f32 = 0.0; var sb: f32 = 0.0;

  for (var ky: i32 = -1; ky <= 1; ky++) {
    let ny = u32(clamp(i32(y) + ky, 0, i32(h) - 1));
    for (var kx: i32 = -1; kx <= 1; kx++) {
      let nx = u32(clamp(i32(x) + kx, 0, i32(w) - 1));
      let b = (ny * w + nx) * 3u;
      sr += max(input[b], 0.0);
      sg += max(input[b + 1u], 0.0);
      sb += max(input[b + 2u], 0.0);
    }
  }

  let cr_r = cbrt(sr / 9.0);
  let cr_g = cbrt(sg / 9.0);
  let cr_b = cbrt(sb / 9.0);

  let opp_r = 0.5 * (cr_g + cr_b);
  let opp_g = 0.5 * (cr_r + cr_b);
  let opp_b = 0.5 * (cr_r + cr_g);

  let idx = y * w + x;
  let base = idx * 3u;
  refavg[base]      = opp_r * opp_r * opp_r;
  refavg[base + 1u] = opp_g * opp_g * opp_g;
  refavg[base + 2u] = opp_b * opp_b * opp_b;

  var mask: u32 = 0u;
  if (input[base] >= params.clip_r) { mask |= 1u; }
  if (input[base + 1u] >= params.clip_g) { mask |= 2u; }
  if (input[base + 2u] >= params.clip_b) { mask |= 4u; }
  clip_mask[idx] = mask;
}
`;

const SHADER_DOWNSAMPLE_CLIP = /* wgsl */ `
struct Params {
  full_width: u32,
  full_height: u32,
  ds_width: u32,
  ds_height: u32,
}

@group(0) @binding(0) var<storage, read> clip_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> mask_ds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mx = gid.x;
  let my = gid.y;
  if (mx >= params.ds_width || my >= params.ds_height) { return; }

  var result: u32 = 0u;
  for (var ky: u32 = 0u; ky < 3u; ky++) {
    let fy = my * 3u + ky;
    if (fy >= params.full_height) { continue; }
    for (var kx: u32 = 0u; kx < 3u; kx++) {
      let fx = mx * 3u + kx;
      if (fx >= params.full_width) { continue; }
      result |= clip_mask[fy * params.full_width + fx];
    }
  }

  if (mx == 0u || mx >= params.ds_width - 1u || my == 0u || my >= params.ds_height - 1u) {
    result = 0u;
  }
  mask_ds[my * params.ds_width + mx] = result;
}
`;

const SHADER_DILATE = /* wgsl */ `
struct Params {
  ds_width: u32,
  ds_height: u32,
  radius_r: u32,
  radius_g: u32,
  radius_b: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> mask_ds: array<u32>;
@group(0) @binding(1) var<storage, read_write> dilated_ds: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mx = gid.x;
  let my = gid.y;
  if (mx >= params.ds_width || my >= params.ds_height) { return; }

  let radii = array<i32, 3>(i32(params.radius_r), i32(params.radius_g), i32(params.radius_b));
  var result: u32 = 0u;

  for (var ch: u32 = 0u; ch < 3u; ch++) {
    let bit = 1u << ch;
    let r = radii[ch];
    let r2 = r * r;
    var found = false;
    for (var ky: i32 = -r; ky <= r; ky++) {
      if (found) { break; }
      let ny = i32(my) + ky;
      if (ny < 0 || ny >= i32(params.ds_height)) { continue; }
      let xr = i32(sqrt(f32(r2 - ky * ky)));
      for (var kx: i32 = -xr; kx <= xr; kx++) {
        let nx = i32(mx) + kx;
        if (nx < 0 || nx >= i32(params.ds_width)) { continue; }
        if ((mask_ds[u32(ny) * params.ds_width + u32(nx)] & bit) != 0u) {
          found = true;
          break;
        }
      }
    }
    if (found) { result |= bit; }
  }
  dilated_ds[my * params.ds_width + mx] = result;
}
`;

const SHADER_CHROMA = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  ds_width: u32,
  ds_height: u32,
  clip_r: f32,
  clip_g: f32,
  clip_b: f32,
  lo_clip_r: f32,
  lo_clip_g: f32,
  lo_clip_b: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> refavg: array<f32>;
@group(0) @binding(2) var<storage, read> clip_mask: array<u32>;
@group(0) @binding(3) var<storage, read> dilated_ds: array<u32>;
@group(0) @binding(4) var<storage, read_write> chroma_buf: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> sh_sum: array<f32, ${WG * WG * 3}>;
var<workgroup> sh_cnt: array<u32, ${WG * WG * 3}>;

fn atomicAddF32(p: ptr<storage, atomic<u32>, read_write>, val: f32) {
  var old_bits = atomicLoad(p);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
    let result = atomicCompareExchangeWeak(p, old_bits, new_bits);
    if (result.exchanged) { break; }
    old_bits = result.old_value;
  }
}

@compute @workgroup_size(${WG}, ${WG})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  for (var c: u32 = 0u; c < 3u; c++) {
    sh_sum[lid * 3u + c] = 0.0;
    sh_cnt[lid * 3u + c] = 0u;
  }

  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;
  let clips = array<f32, 3>(params.clip_r, params.clip_g, params.clip_b);
  let lo = array<f32, 3>(params.lo_clip_r, params.lo_clip_g, params.lo_clip_b);

  if (x >= 3u && x < w - 3u && y >= 3u && y < h - 3u) {
    let idx = y * w + x;
    let base = idx * 3u;
    let mask = clip_mask[idx];
    let my = min(y / 3u, params.ds_height - 1u);
    let mx = min(x / 3u, params.ds_width - 1u);
    let dil = dilated_ds[my * params.ds_width + mx];

    for (var c: u32 = 0u; c < 3u; c++) {
      let bit = 1u << c;
      let val = input[base + c];
      if ((mask & bit) == 0u && val > lo[c] && (dil & bit) != 0u) {
        sh_sum[lid * 3u + c] = val - refavg[base + c];
        sh_cnt[lid * 3u + c] = 1u;
      }
    }
  }

  workgroupBarrier();

  for (var s: u32 = ${(WG * WG) >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var c: u32 = 0u; c < 3u; c++) {
        sh_sum[lid * 3u + c] += sh_sum[(lid + s) * 3u + c];
        sh_cnt[lid * 3u + c] += sh_cnt[(lid + s) * 3u + c];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var c: u32 = 0u; c < 3u; c++) {
      if (sh_cnt[c] > 0u) {
        atomicAddF32(&chroma_buf[c], sh_sum[c]);
        atomicAdd(&chroma_buf[3u + c], sh_cnt[c]);
      }
    }
  }
}
`;

/** Finalize with HL extension + CC + DR → RGBA output. */
const SHADER_FINALIZE = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dr_gain: f32,
  out_stride: u32,
  clip_r: f32,
  clip_g: f32,
  clip_b: f32,
  _pad1: f32,
  cc_row0: vec4f,
  cc_row1: vec4f,
  cc_row2: vec4f,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> refavg: array<f32>;
@group(0) @binding(2) var<storage, read> clip_mask: array<u32>;
@group(0) @binding(3) var<storage, read> chroma_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  // Decode chroma means from atomic accumulation buffer
  var chroma: array<f32, 3>;
  for (var c: u32 = 0u; c < 3u; c++) {
    let s = bitcast<f32>(chroma_buf[c]);
    let n = f32(chroma_buf[3u + c]);
    chroma[c] = select(0.0, s / n, n > 100.0);
  }

  let idx = y * params.width + x;
  let base3 = idx * 3u;
  let base4 = (y * params.out_stride + x) * 4u;
  let mask = clip_mask[idx];

  var r = data[base3];
  var g = data[base3 + 1u];
  var b = data[base3 + 2u];

  // Clip ratio (from WB'd values, before HL extension)
  let cr = max(r / params.clip_r, max(g / params.clip_g, b / params.clip_b));

  // Inpaint-opposed highlight extension
  if ((mask & 1u) != 0u) { r = max(r, refavg[base3] + chroma[0]); }
  if ((mask & 2u) != 0u) { g = max(g, refavg[base3 + 1u] + chroma[1]); }
  if ((mask & 4u) != 0u) { b = max(b, refavg[base3 + 2u] + chroma[2]); }

  // Color correction (camera → sRGB matrix)
  let rr = params.cc_row0.x * r + params.cc_row0.y * g + params.cc_row0.z * b;
  let gg = params.cc_row1.x * r + params.cc_row1.y * g + params.cc_row1.z * b;
  let bb = params.cc_row2.x * r + params.cc_row2.y * g + params.cc_row2.z * b;
  r = max(rr, 0.0);
  g = max(gg, 0.0);
  b = max(bb, 0.0);

  // DR exposure compensation
  r *= params.dr_gain;
  g *= params.dr_gain;
  b *= params.dr_gain;

  output[base4] = r;
  output[base4 + 1u] = g;
  output[base4 + 2u] = b;
  output[base4 + 3u] = min(cr, 1.0);
}
`;

/** Finalize without HL recovery (no clipped pixels). */
const SHADER_FINALIZE_SIMPLE = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dr_gain: f32,
  out_stride: u32,
  clip_r: f32,
  clip_g: f32,
  clip_b: f32,
  _pad1: f32,
  cc_row0: vec4f,
  cc_row1: vec4f,
  cc_row2: vec4f,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = y * params.width + x;
  let base3 = idx * 3u;
  let base4 = (y * params.out_stride + x) * 4u;

  var r = data[base3];
  var g = data[base3 + 1u];
  var b = data[base3 + 2u];

  let cr = max(r / params.clip_r, max(g / params.clip_g, b / params.clip_b));

  // Color correction
  let rr = params.cc_row0.x * r + params.cc_row0.y * g + params.cc_row0.z * b;
  let gg = params.cc_row1.x * r + params.cc_row1.y * g + params.cc_row1.z * b;
  let bb = params.cc_row2.x * r + params.cc_row2.y * g + params.cc_row2.z * b;
  r = max(rr, 0.0);
  g = max(gg, 0.0);
  b = max(bb, 0.0);

  r *= params.dr_gain;
  g *= params.dr_gain;
  b *= params.dr_gain;

  output[base4] = r;
  output[base4 + 1u] = g;
  output[base4 + 2u] = b;
  output[base4 + 3u] = min(cr, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------

interface Pipelines {
  wb: GPUComputePipeline;
  refavgClip: GPUComputePipeline;
  downsampleClip: GPUComputePipeline;
  dilate: GPUComputePipeline;
  chroma: GPUComputePipeline;
  finalize: GPUComputePipeline;
  finalizeSimple: GPUComputePipeline;
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
    wb: makePipeline(device, SHADER_WB),
    refavgClip: makePipeline(device, SHADER_REFAVG_CLIP),
    downsampleClip: makePipeline(device, SHADER_DOWNSAMPLE_CLIP),
    dilate: makePipeline(device, SHADER_DILATE),
    chroma: makePipeline(device, SHADER_CHROMA),
    finalize: makePipeline(device, SHADER_FINALIZE),
    finalizeSimple: makePipeline(device, SHADER_FINALIZE_SIMPLE),
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

/** Cast typed array to satisfy WebGPU's strict GPUAllowSharedBufferSource type. */
function asGpu(data: ArrayBufferView): GPUAllowSharedBufferSource {
  return data as unknown as GPUAllowSharedBufferSource;
}

/** Pad width to 16-pixel alignment so bytesPerRow (width × 16 bytes) is 256-aligned. */
function padWidth16(w: number): number {
  return Math.ceil(w / 16) * 16;
}

function writeFinalizeParams(
  device: GPUDevice, paramBuf: GPUBuffer,
  width: number, height: number, outStride: number,
  clips: [number, number, number],
  ccMatrix: Float32Array | null,
  drGain: number,
): void {
  const ab = new ArrayBuffer(80);
  const u32 = new Uint32Array(ab);
  const f32 = new Float32Array(ab);
  u32[0] = width;
  u32[1] = height;
  f32[2] = drGain;
  u32[3] = outStride;
  f32[4] = clips[0];
  f32[5] = clips[1];
  f32[6] = clips[2];
  f32[7] = 0;
  if (ccMatrix) {
    f32[8] = ccMatrix[0]; f32[9] = ccMatrix[1]; f32[10] = ccMatrix[2]; f32[11] = 0;
    f32[12] = ccMatrix[3]; f32[13] = ccMatrix[4]; f32[14] = ccMatrix[5]; f32[15] = 0;
    f32[16] = ccMatrix[6]; f32[17] = ccMatrix[7]; f32[18] = ccMatrix[8]; f32[19] = 0;
  } else {
    // Identity
    f32[8] = 1; f32[11] = 0;
    f32[13] = 1; f32[15] = 0;
    f32[18] = 1; f32[19] = 0;
    f32[9] = 0; f32[10] = 0; f32[12] = 0; f32[14] = 0; f32[16] = 0; f32[17] = 0;
  }
  device.queue.writeBuffer(paramBuf, 0, asGpu(new Uint8Array(ab)));
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

export interface PostprocessResult {
  /** RGBA32F GPU buffer with row-padded layout, ready for copyBufferToTexture. */
  buffer: GPUBuffer;
  /** Bytes per row (256-aligned) for copyBufferToTexture. */
  bytesPerRow: number;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * GPU post-demosaic processing: WB → highlight recovery → CC → DR → RGBA.
 *
 * Returns a GPU-resident RGBA32F buffer (row-padded for 256-byte alignment).
 * The caller takes ownership of the buffer and must destroy it after use
 * (e.g. after copyBufferToTexture into the renderer's image texture).
 *
 * @param device   WebGPU device (shared with renderer via setSharedDevice)
 * @param rawHwc   Raw demosaic output in HWC layout (no WB applied).
 *                 Can be a Float32Array (uploaded to GPU) or a GPUBuffer
 *                 already on the GPU (e.g. from GPU tile blending).
 * @param width    Image width
 * @param height   Image height
 * @param wb       White balance multipliers [R, G, B] normalized to G=1
 * @param clips    Per-channel clip levels after WB (clipNorm × wb)
 * @param ccMatrix 3×3 row-major camera→sRGB color correction (null = identity)
 * @param drGain   DR exposure compensation multiplier (1.0 = none)
 */
export async function gpuPostprocess(
  device: GPUDevice,
  rawHwc: Float32Array | GPUBuffer,
  width: number,
  height: number,
  wb: Float32Array,
  clips: [number, number, number],
  ccMatrix: Float32Array | null,
  drGain: number,
): Promise<PostprocessResult> {
  const n = width * height;
  const pipes = getPipelines(device);

  const S = GPUBufferUsage.STORAGE;
  const D = GPUBufferUsage.COPY_DST;
  const C = GPUBufferUsage.COPY_SRC;
  const U = GPUBufferUsage.UNIFORM;

  const gpuInput = rawHwc instanceof GPUBuffer;

  // Check if anything will clip after WB (saves ~1 GB VRAM when nothing clips).
  // When input is already a GPUBuffer we can't scan on CPU — conservatively
  // assume clipping is possible (the HL recovery buffers cost ~1 GB but are
  // only allocated for images that actually have clipped highlights).
  let anyClipped = gpuInput;
  if (!gpuInput) {
    const cpuData = rawHwc as Float32Array;
    for (let i = 0; i < n * 3; i += 3) {
      if (cpuData[i] * wb[0] >= clips[0] ||
          cpuData[i + 1] * wb[1] >= clips[1] ||
          cpuData[i + 2] * wb[2] >= clips[2]) {
        anyClipped = true;
        break;
      }
    }
  }

  // ── Buffers ─────────────────────────────────────────────────────────────

  let dataBuf: GPUBuffer;
  if (gpuInput) {
    dataBuf = rawHwc as GPUBuffer;
  } else {
    dataBuf = buf(device, n * 3 * 4, S | D);
    device.queue.writeBuffer(dataBuf, 0, asGpu(rawHwc as Float32Array));
  }

  // Output buffer: RGBA32F with row padding for 256-byte bytesPerRow alignment
  const paddedW = padWidth16(width);
  const bytesPerRow = paddedW * 16;  // 4 channels × 4 bytes × paddedW pixels
  const outputBuf = buf(device, bytesPerRow * height, S | C);

  // WB params
  const wbParamBuf = buf(device, 24, U | D);
  {
    const ab = new ArrayBuffer(24);
    new Uint32Array(ab, 0, 2).set([width, height]);
    new Float32Array(ab, 8, 4).set([wb[0], wb[1], wb[2], 0]);
    device.queue.writeBuffer(wbParamBuf, 0, asGpu(new Uint8Array(ab)));
  }

  // Finalize params (shared between both finalize variants)
  const finalParamBuf = buf(device, 80, U | D);
  writeFinalizeParams(device, finalParamBuf, width, height, paddedW, clips, ccMatrix, drGain);

  // HL recovery buffers (only allocated when needed)
  const hlBufs: GPUBuffer[] = [];

  const enc = device.createCommandEncoder();
  const wgX = Math.ceil(width / WG);
  const wgY = Math.ceil(height / WG);

  // ── Pass 1: White balance (in-place) ────────────────────────────────────

  let pass = enc.beginComputePass();
  pass.setPipeline(pipes.wb);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipes.wb.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dataBuf } },
      { binding: 1, resource: { buffer: wbParamBuf } },
    ],
  }));
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  if (anyClipped) {
    // ── Passes 2–5: Inpaint-opposed highlight recovery ──────────────────

    const dsW = Math.floor(width / 3);
    const dsH = Math.floor(height / 3);
    const loClips: [number, number, number] = [clips[0] * 0.2, clips[1] * 0.2, clips[2] * 0.2];
    const maxClip = Math.max(...clips);

    const refavgBuf = buf(device, n * 3 * 4, S);
    const clipMaskBuf = buf(device, n * 4, S);
    const maskDsBuf = buf(device, dsW * dsH * 4, S);
    const dilatedDsBuf = buf(device, dsW * dsH * 4, S);
    const chromaBuf = buf(device, 6 * 4, S | D);
    device.queue.writeBuffer(chromaBuf, 0, asGpu(new Uint32Array(6)));
    hlBufs.push(refavgBuf, clipMaskBuf, maskDsBuf, dilatedDsBuf, chromaBuf);

    // Refavg + clip mask params
    const rcParamBuf = buf(device, 24, U | D);
    {
      const ab = new ArrayBuffer(24);
      new Uint32Array(ab, 0, 2).set([width, height]);
      new Float32Array(ab, 8, 3).set(clips);
      device.queue.writeBuffer(rcParamBuf, 0, asGpu(new Uint8Array(ab)));
    }
    hlBufs.push(rcParamBuf);

    // Pass 2: REFAVG_CLIP
    pass = enc.beginComputePass();
    pass.setPipeline(pipes.refavgClip);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipes.refavgClip.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuf } },
        { binding: 1, resource: { buffer: refavgBuf } },
        { binding: 2, resource: { buffer: clipMaskBuf } },
        { binding: 3, resource: { buffer: rcParamBuf } },
      ],
    }));
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();

    // Pass 3: DOWNSAMPLE_CLIP
    const dsParamBuf = buf(device, 16, U | D);
    device.queue.writeBuffer(dsParamBuf, 0, asGpu(new Uint32Array([width, height, dsW, dsH])));
    hlBufs.push(dsParamBuf);

    const dsWgX = Math.ceil(dsW / WG);
    const dsWgY = Math.ceil(dsH / WG);

    pass = enc.beginComputePass();
    pass.setPipeline(pipes.downsampleClip);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipes.downsampleClip.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: clipMaskBuf } },
        { binding: 1, resource: { buffer: maskDsBuf } },
        { binding: 2, resource: { buffer: dsParamBuf } },
      ],
    }));
    pass.dispatchWorkgroups(dsWgX, dsWgY);
    pass.end();

    // Pass 4: DILATE
    const dilRadii = clips.map(c => {
      const ratio = maxClip / Math.max(c, 1e-6);
      return ((Math.min(21, Math.max(7, Math.floor(7 * ratio))) | 1) - 1) >> 1;
    });
    const dilParamBuf = buf(device, 24, U | D);
    device.queue.writeBuffer(dilParamBuf, 0, asGpu(new Uint32Array([dsW, dsH, ...dilRadii, 0])));
    hlBufs.push(dilParamBuf);

    pass = enc.beginComputePass();
    pass.setPipeline(pipes.dilate);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipes.dilate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: maskDsBuf } },
        { binding: 1, resource: { buffer: dilatedDsBuf } },
        { binding: 2, resource: { buffer: dilParamBuf } },
      ],
    }));
    pass.dispatchWorkgroups(dsWgX, dsWgY);
    pass.end();

    // Pass 5: CHROMA
    const chromaParamBuf = buf(device, 48, U | D);
    {
      const ab = new ArrayBuffer(48);
      new Uint32Array(ab, 0, 4).set([width, height, dsW, dsH]);
      new Float32Array(ab, 16, 6).set([...clips, ...loClips]);
      device.queue.writeBuffer(chromaParamBuf, 0, asGpu(new Uint8Array(ab)));
    }
    hlBufs.push(chromaParamBuf);

    pass = enc.beginComputePass();
    pass.setPipeline(pipes.chroma);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipes.chroma.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuf } },
        { binding: 1, resource: { buffer: refavgBuf } },
        { binding: 2, resource: { buffer: clipMaskBuf } },
        { binding: 3, resource: { buffer: dilatedDsBuf } },
        { binding: 4, resource: { buffer: chromaBuf } },
        { binding: 5, resource: { buffer: chromaParamBuf } },
      ],
    }));
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();

    // ── Pass 6: Finalize (HL extend + CC + DR → RGBA) ────────────────────

    pass = enc.beginComputePass();
    pass.setPipeline(pipes.finalize);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipes.finalize.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuf } },
        { binding: 1, resource: { buffer: refavgBuf } },
        { binding: 2, resource: { buffer: clipMaskBuf } },
        { binding: 3, resource: { buffer: chromaBuf } },
        { binding: 4, resource: { buffer: outputBuf } },
        { binding: 5, resource: { buffer: finalParamBuf } },
      ],
    }));
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  } else {
    // ── No clipping: simple finalize (CC + DR → RGBA) ─────────────────

    pass = enc.beginComputePass();
    pass.setPipeline(pipes.finalizeSimple);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipes.finalizeSimple.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: finalParamBuf } },
      ],
    }));
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  // ── Submit & cleanup intermediates ──────────────────────────────────────

  device.queue.submit([enc.finish()]);

  for (const b of [dataBuf, wbParamBuf, finalParamBuf, ...hlBufs]) {
    b.destroy();
  }

  // outputBuf ownership transferred to caller
  return { buffer: outputBuf, bytesPerRow };
}
