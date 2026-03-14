// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-present X-Veon contributors
/**
 * Post-demosaic highlight recovery for RGB data.
 *
 * Pass 1: Inpaint-opposed via WGSL compute shaders (GPU).
 * Pass 2: Segmentation-based refinement (CPU, reuses segmentation machinery).
 *
 * Designed for the pipeline: raw CFA → neural demosaic → WB → highlight recovery → color correction.
 * Clipping is WB-induced: channels exceed clip_level after white balance multiplication.
 */

import {
  type Segmentation,
  HL_BORDER, HL_POWERF, SEG_ID_MASK,
  createSegmentation, getSegmentId,
  segmentizePlane, segmentsCombine,
  calcPlaneCandidates, extendBorder,
} from './highlight-segments';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WG = 16; // workgroup size (16×16 = 256 threads)
const CLIP_THRESHOLD = 0.9;

// ---------------------------------------------------------------------------
// WGSL shaders for GPU Pass 1 (inpaint-opposed)
// ---------------------------------------------------------------------------

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

// Chrominance accumulation with workgroup reduction + CAS-based f32 atomics.
// chroma_buf layout: [sum_r, sum_g, sum_b, cnt_r, cnt_g, cnt_b] as atomic<u32>
// sums use bitcast f32↔u32 via CAS, counts use regular atomicAdd.
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

  // Tree reduction
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

// Extend clipped pixels. Reads chroma_buf directly (no CPU readback needed).
const SHADER_EXTEND = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
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

  // Compute chroma means from accumulated sums/counts
  var chroma: array<f32, 3>;
  for (var c: u32 = 0u; c < 3u; c++) {
    let s = bitcast<f32>(chroma_buf[c]);
    let n = f32(chroma_buf[3u + c]);
    chroma[c] = select(0.0, s / n, n > 100.0);
  }

  let idx = y * params.width + x;
  let base = idx * 3u;
  let mask = clip_mask[idx];

  for (var c: u32 = 0u; c < 3u; c++) {
    let val = input[base + c];
    let bit = 1u << c;
    if ((mask & bit) != 0u) {
      output[base + c] = max(val, refavg[base + c] + chroma[c]);
    } else {
      output[base + c] = val;
    }
  }
}
`;

// ---------------------------------------------------------------------------
// GPU Pass 1 orchestration
// ---------------------------------------------------------------------------

function createBuf(
  device: GPUDevice, size: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return device.createBuffer({ size: Math.max(size, 4), usage });
}

function createPipeline(device: GPUDevice, code: string): GPUComputePipeline {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
}

export async function reconstructOpposedGpu(
  device: GPUDevice,
  rgb: Float32Array,
  width: number,
  height: number,
  clips: [number, number, number],
): Promise<Float32Array> {
  const n = width * height;
  const dsW = Math.floor(width / 3);
  const dsH = Math.floor(height / 3);
  const loClips: [number, number, number] = [clips[0] * 0.2, clips[1] * 0.2, clips[2] * 0.2];
  const maxClip = Math.max(...clips);

  const S = GPUBufferUsage.STORAGE;
  const C = GPUBufferUsage.COPY_SRC;
  const D = GPUBufferUsage.COPY_DST;
  const U = GPUBufferUsage.UNIFORM;

  // Buffers
  const inputBuf = createBuf(device, n * 3 * 4, S | D);
  device.queue.writeBuffer(inputBuf, 0, rgb);
  const refavgBuf = createBuf(device, n * 3 * 4, S);
  const clipMaskBuf = createBuf(device, n * 4, S);
  const maskDsBuf = createBuf(device, dsW * dsH * 4, S);
  const dilatedDsBuf = createBuf(device, dsW * dsH * 4, S);
  const chromaBuf = createBuf(device, 6 * 4, S | D);
  device.queue.writeBuffer(chromaBuf, 0, new Uint32Array(6)); // zero
  const outputBuf = createBuf(device, n * 3 * 4, S | C);

  // Params
  const p1Buf = createBuf(device, 24, U | D);
  const p1 = new ArrayBuffer(24);
  new Uint32Array(p1, 0, 2).set([width, height]);
  new Float32Array(p1, 8, 3).set(clips);
  device.queue.writeBuffer(p1Buf, 0, p1);

  const p2Buf = createBuf(device, 16, U | D);
  device.queue.writeBuffer(p2Buf, 0, new Uint32Array([width, height, dsW, dsH]));

  // Dilation radii (adaptive per channel, matching Python)
  const dilRadii = clips.map(c => {
    const ratio = maxClip / Math.max(c, 1e-6);
    return ((Math.min(21, Math.max(7, Math.floor(7 * ratio))) | 1) - 1) >> 1; // radius from diameter
  });
  const p3Buf = createBuf(device, 24, U | D);
  device.queue.writeBuffer(p3Buf, 0, new Uint32Array([dsW, dsH, ...dilRadii, 0]));

  const p4Buf = createBuf(device, 48, U | D);
  const p4 = new ArrayBuffer(48);
  new Uint32Array(p4, 0, 4).set([width, height, dsW, dsH]);
  new Float32Array(p4, 16, 6).set([...clips, ...loClips]);
  device.queue.writeBuffer(p4Buf, 0, p4);

  const p5Buf = createBuf(device, 16, U | D);
  device.queue.writeBuffer(p5Buf, 0, new Uint32Array([width, height, 0, 0]));

  // Pipelines
  const pipe1 = createPipeline(device, SHADER_REFAVG_CLIP);
  const pipe2 = createPipeline(device, SHADER_DOWNSAMPLE_CLIP);
  const pipe3 = createPipeline(device, SHADER_DILATE);
  const pipe4 = createPipeline(device, SHADER_CHROMA);
  const pipe5 = createPipeline(device, SHADER_EXTEND);

  // Bind groups
  const bg1 = device.createBindGroup({
    layout: pipe1.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: refavgBuf } },
      { binding: 2, resource: { buffer: clipMaskBuf } },
      { binding: 3, resource: { buffer: p1Buf } },
    ],
  });

  const bg2 = device.createBindGroup({
    layout: pipe2.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: clipMaskBuf } },
      { binding: 1, resource: { buffer: maskDsBuf } },
      { binding: 2, resource: { buffer: p2Buf } },
    ],
  });

  const bg3 = device.createBindGroup({
    layout: pipe3.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: maskDsBuf } },
      { binding: 1, resource: { buffer: dilatedDsBuf } },
      { binding: 2, resource: { buffer: p3Buf } },
    ],
  });

  const bg4 = device.createBindGroup({
    layout: pipe4.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: refavgBuf } },
      { binding: 2, resource: { buffer: clipMaskBuf } },
      { binding: 3, resource: { buffer: dilatedDsBuf } },
      { binding: 4, resource: { buffer: chromaBuf } },
      { binding: 5, resource: { buffer: p4Buf } },
    ],
  });

  const bg5 = device.createBindGroup({
    layout: pipe5.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: refavgBuf } },
      { binding: 2, resource: { buffer: clipMaskBuf } },
      { binding: 3, resource: { buffer: chromaBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: p5Buf } },
    ],
  });

  // Dispatch all 5 passes in one command buffer
  const enc = device.createCommandEncoder();

  const wgX = Math.ceil(width / WG);
  const wgY = Math.ceil(height / WG);
  const dsWgX = Math.ceil(dsW / WG);
  const dsWgY = Math.ceil(dsH / WG);

  let pass: GPUComputePassEncoder;

  pass = enc.beginComputePass();
  pass.setPipeline(pipe1);
  pass.setBindGroup(0, bg1);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(pipe2);
  pass.setBindGroup(0, bg2);
  pass.dispatchWorkgroups(dsWgX, dsWgY);
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(pipe3);
  pass.setBindGroup(0, bg3);
  pass.dispatchWorkgroups(dsWgX, dsWgY);
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(pipe4);
  pass.setBindGroup(0, bg4);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(pipe5);
  pass.setBindGroup(0, bg5);
  pass.dispatchWorkgroups(wgX, wgY);
  pass.end();

  // Readback
  const staging = createBuf(device, n * 3 * 4, GPUBufferUsage.MAP_READ | C);
  enc.copyBufferToBuffer(outputBuf, 0, staging, 0, n * 3 * 4);

  device.queue.submit([enc.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(new Float32Array(staging.getMappedRange()).slice(0));
  staging.unmap();

  // Cleanup
  for (const b of [inputBuf, refavgBuf, clipMaskBuf, maskDsBuf, dilatedDsBuf,
                    chromaBuf, outputBuf, staging, p1Buf, p2Buf, p3Buf, p4Buf, p5Buf]) {
    b.destroy();
  }

  return result;
}

// ---------------------------------------------------------------------------
// CPU Pass 1 fallback (inpaint-opposed for RGB)
// ---------------------------------------------------------------------------

function dilateEllipseRgb(
  mask: Uint8Array, width: number, height: number, dilSize: number,
): Uint8Array {
  const r = (dilSize - 1) >> 1;
  const r2 = r * r;
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      const yMin = Math.max(0, y - r);
      const yMax = Math.min(height - 1, y + r);
      for (let ny = yMin; ny <= yMax; ny++) {
        const dyv = ny - y;
        const xRange = Math.floor(Math.sqrt(r2 - dyv * dyv));
        const xMin = Math.max(0, x - xRange);
        const xMax = Math.min(width - 1, x + xRange);
        for (let nx = xMin; nx <= xMax; nx++) {
          out[ny * width + nx] = 1;
        }
      }
    }
  }
  return out;
}

export function reconstructOpposedCpu(
  rgb: Float32Array, width: number, height: number,
  clips: [number, number, number],
): Float32Array {
  const n = width * height;
  const loClips: [number, number, number] = [clips[0] * 0.2, clips[1] * 0.2, clips[2] * 0.2];

  // Check if anything clipped
  let anyClipped = false;
  for (let i = 0; i < n * 3; i += 3) {
    if (rgb[i] >= clips[0] || rgb[i + 1] >= clips[1] || rgb[i + 2] >= clips[2]) {
      anyClipped = true; break;
    }
  }
  if (!anyClipped) return new Float32Array(rgb);

  // Clip mask at 1/3 resolution
  const mw = Math.floor(width / 3);
  const mh = Math.floor(height / 3);
  const masks: Uint8Array[] = [new Uint8Array(mw * mh), new Uint8Array(mw * mh), new Uint8Array(mw * mh)];

  for (let y = 0; y < height; y++) {
    const my = Math.floor(y / 3);
    if (my >= mh) continue;
    for (let x = 0; x < width; x++) {
      const mx = Math.floor(x / 3);
      if (mx >= mw) continue;
      const base = (y * width + x) * 3;
      for (let c = 0; c < 3; c++) {
        if (rgb[base + c] >= clips[c]) masks[c][my * mw + mx] = 1;
      }
    }
  }

  // Zero borders
  for (let c = 0; c < 3; c++) {
    const m = masks[c];
    for (let x = 0; x < mw; x++) { m[x] = 0; m[(mh - 1) * mw + x] = 0; }
    for (let y = 0; y < mh; y++) { m[y * mw] = 0; m[y * mw + mw - 1] = 0; }
  }

  // Adaptive dilation
  const maxClip = Math.max(...clips);
  const dilated: Uint8Array[] = [];
  for (let c = 0; c < 3; c++) {
    const ratio = maxClip / Math.max(clips[c], 1e-6);
    const dilSize = Math.min(21, Math.max(7, Math.floor(7 * ratio))) | 1;
    dilated.push(dilateEllipseRgb(masks[c], mw, mh, dilSize));
  }

  // Refavg (opposed reference average, cubed to linear)
  const refavgLinear = new Float32Array(n * 3);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - 1), y1 = Math.min(height - 1, y + 1);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - 1), x1 = Math.min(width - 1, x + 1);
      let sr = 0, sg = 0, sb = 0, cnt = 0;
      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          const b = (ny * width + nx) * 3;
          sr += Math.max(0, rgb[b]);
          sg += Math.max(0, rgb[b + 1]);
          sb += Math.max(0, rgb[b + 2]);
          cnt++;
        }
      }
      const cr0 = Math.cbrt(sr / cnt);
      const cr1 = Math.cbrt(sg / cnt);
      const cr2 = Math.cbrt(sb / cnt);
      const base = (y * width + x) * 3;
      const opp0 = 0.5 * (cr1 + cr2);
      const opp1 = 0.5 * (cr0 + cr2);
      const opp2 = 0.5 * (cr0 + cr1);
      refavgLinear[base] = opp0 ** 3;
      refavgLinear[base + 1] = opp1 ** 3;
      refavgLinear[base + 2] = opp2 ** 3;
    }
  }

  // Chrominance accumulation
  const chromSum = [0, 0, 0];
  const chromCnt = [0, 0, 0];
  for (let y = 3; y < height - 3; y++) {
    const my = Math.min(Math.floor(y / 3), mh - 1);
    for (let x = 3; x < width - 3; x++) {
      const mx = Math.min(Math.floor(x / 3), mw - 1);
      const base = (y * width + x) * 3;
      for (let c = 0; c < 3; c++) {
        const val = rgb[base + c];
        if (val >= clips[c] || val <= loClips[c]) continue;
        if (!dilated[c][my * mw + mx]) continue;
        chromSum[c] += val - refavgLinear[base + c];
        chromCnt[c]++;
      }
    }
  }

  const chrom = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    if (chromCnt[c] > 100) chrom[c] = chromSum[c] / chromCnt[c];
  }

  // Extend clipped pixels
  const out = new Float32Array(rgb);
  for (let i = 0; i < n; i++) {
    const base = i * 3;
    for (let c = 0; c < 3; c++) {
      if (rgb[base + c] >= clips[c]) {
        out[base + c] = Math.max(rgb[base + c], refavgLinear[base + c] + chrom[c]);
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// CPU Pass 2: Segmentation-based reconstruction for RGB
// ---------------------------------------------------------------------------

/** Cube-root opposed reference average at full resolution for a single pixel/channel. */
function refavgCrRgb(
  rgb: Float32Array, width: number, height: number,
  row: number, col: number, ch: number,
): number {
  const y0 = Math.max(0, row - 1), y1 = Math.min(height - 1, row + 1);
  const x0 = Math.max(0, col - 1), x1 = Math.min(width - 1, col + 1);
  let sr = 0, sg = 0, sb = 0, cnt = 0;
  for (let ny = y0; ny <= y1; ny++) {
    for (let nx = x0; nx <= x1; nx++) {
      const b = (ny * width + nx) * 3;
      sr += Math.max(0, rgb[b]);
      sg += Math.max(0, rgb[b + 1]);
      sb += Math.max(0, rgb[b + 2]);
      cnt++;
    }
  }
  const cr0 = Math.cbrt(sr / cnt);
  const cr1 = Math.cbrt(sg / cnt);
  const cr2 = Math.cbrt(sb / cnt);
  if (ch === 0) return 0.5 * (cr1 + cr2);
  if (ch === 1) return 0.5 * (cr0 + cr2);
  return 0.5 * (cr0 + cr1);
}

export function reconstructSegmentedCpu(
  rgb: Float32Array,
  width: number,
  height: number,
  clips: [number, number, number],
  combineRadius = 2,
  candidating = 0.5,
  originalRgb?: Float32Array,
): Float32Array {
  const original = originalRgb ?? rgb;
  const out = new Float32Array(rgb);

  const roundEven = (v: number) => (v + 1) & ~1;
  const nSpRows = Math.floor(height / 3);
  const nSpCols = Math.floor(width / 3);
  const pwidth = roundEven(nSpCols) + 2 * HL_BORDER;
  const pheight = roundEven(nSpRows) + 2 * HL_BORDER;
  const psize = pwidth * pheight;

  const cubeClips: [number, number, number] = [
    Math.cbrt(clips[0]), Math.cbrt(clips[1]), Math.cbrt(clips[2]),
  ];

  const maxSegments = Math.max(256, Math.floor((width * height) / 4000));
  const planes: Float32Array[] = [];
  const refavgs: Float32Array[] = [];
  const segs: Segmentation[] = [];
  for (let c = 0; c < 3; c++) {
    planes.push(new Float32Array(psize));
    refavgs.push(new Float32Array(psize));
    segs.push(createSegmentation(pwidth, pheight, HL_BORDER + 1, maxSegments));
  }

  // Build downsampled planes via 3×3 area averaging per channel
  let anyClipped = 0;
  for (let sr = 0; sr < nSpRows; sr++) {
    for (let sc = 0; sc < nSpCols; sc++) {
      const mean = [0, 0, 0];
      let cnt = 0;
      for (let ky = 0; ky < 3; ky++) {
        const fy = sr * 3 + ky;
        if (fy >= height) continue;
        for (let kx = 0; kx < 3; kx++) {
          const fx = sc * 3 + kx;
          if (fx >= width) continue;
          const b = (fy * width + fx) * 3;
          mean[0] += Math.max(0, rgb[b]);
          mean[1] += Math.max(0, rgb[b + 1]);
          mean[2] += Math.max(0, rgb[b + 2]);
          cnt++;
        }
      }
      if (cnt === 0) continue;
      for (let c = 0; c < 3; c++) mean[c] = Math.cbrt(mean[c] / cnt);

      const opp = [
        0.5 * (mean[1] + mean[2]),
        0.5 * (mean[0] + mean[2]),
        0.5 * (mean[0] + mean[1]),
      ];

      const o = (HL_BORDER + sr) * pwidth + HL_BORDER + sc;
      for (let c = 0; c < 3; c++) {
        planes[c][o] = mean[c];
        refavgs[c][o] = opp[c];
        if (mean[c] >= cubeClips[c]) {
          segs[c].data[o] = 1;
          anyClipped++;
        }
      }
    }
  }

  if (anyClipped < 20) return out;

  for (let c = 0; c < 3; c++) extendBorder(planes[c], pwidth, pheight, HL_BORDER);
  for (let c = 0; c < 3; c++) {
    segmentsCombine(segs[c], combineRadius);
    segmentizePlane(segs[c]);
  }
  for (let c = 0; c < 3; c++) {
    calcPlaneCandidates(planes[c], refavgs[c], segs[c], cubeClips[c], candidating);
  }

  // Reconstruct clipped pixels at full resolution
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const base = idx * 3;

      for (let c = 0; c < 3; c++) {
        const inval = Math.max(0, original[base + c]);
        if (inval < clips[c]) continue;

        const pRow = HL_BORDER + Math.floor(y / 3);
        const pCol = HL_BORDER + Math.floor(x / 3);
        if (pRow >= pheight || pCol >= pwidth) continue;

        const o = pRow * pwidth + pCol;
        const pid = getSegmentId(segs[c], o);
        if (pid <= 1 || pid >= segs[c].nr) continue;

        const candidate = segs[c].val1[pid];
        if (candidate === 0) continue;

        const candRef = segs[c].val2[pid];
        const ra = refavgCrRgb(original, width, height, y, x, c);
        const oval = (ra + candidate - candRef) ** HL_POWERF;
        out[base + c] = Math.max(inval, oval);
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Combined entry point
// ---------------------------------------------------------------------------

export async function reconstructHighlightsRgb(
  rgb: Float32Array,
  width: number,
  height: number,
  clipLevels: [number, number, number],
  options?: {
    device?: GPUDevice;
    clipThreshold?: number;
    combineRadius?: number;
    candidating?: number;
  },
): Promise<Float32Array> {
  const ct = options?.clipThreshold ?? CLIP_THRESHOLD;
  const clips: [number, number, number] = [
    clipLevels[0] * ct, clipLevels[1] * ct, clipLevels[2] * ct,
  ];

  // Quick check: any clipped pixels?
  const n = width * height;
  let anyClipped = false;
  for (let i = 0; i < n * 3; i += 3) {
    if (rgb[i] >= clips[0] || rgb[i + 1] >= clips[1] || rgb[i + 2] >= clips[2]) {
      anyClipped = true; break;
    }
  }
  if (!anyClipped) return new Float32Array(rgb);

  const original = new Float32Array(rgb);

  // Pass 1: inpaint-opposed
  let result: Float32Array;
  if (options?.device) {
    result = await reconstructOpposedGpu(options.device, rgb, width, height, clips);
  } else {
    result = reconstructOpposedCpu(rgb, width, height, clips);
  }

  // Pass 2: segmentation-based refinement
  result = reconstructSegmentedCpu(
    result, width, height, clips,
    options?.combineRadius ?? 2,
    options?.candidating ?? 0.5,
    original,
  );

  return result;
}
