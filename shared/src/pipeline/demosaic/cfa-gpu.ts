/**
 * Upload the u16 CFA (half the bytes of float32) and normalise it on the GPU through the
 * per-colour table, into the float32 CFA buffer the GPU demosaic methods read. The table holds
 * the values the CPU normalisation produced, so the result is bit-identical to it.
 */
import { NORM_LUT_SIZE } from '../preprocess/preprocessor';
import type { DemosaicInput } from './strategy';

const WG = 16;

const SHADER_NORMALIZE = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  period: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> raw: array<u32>;       // two u16 per word, little-endian
@group(0) @binding(1) var<storage, read> lut: array<f32>;       // 3 × ${NORM_LUT_SIZE}
@group(0) @binding(2) var<storage, read> pattern: array<u32>;   // period², canonical
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let i = gid.y * params.width + gid.x;
  let v = (raw[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
  let p = params.period;
  let c = pattern[(gid.y % p) * p + gid.x % p];
  out[i] = lut[c * ${NORM_LUT_SIZE}u + v];
}
`;

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();

function pipelineFor(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    p = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: SHADER_NORMALIZE }), entryPoint: 'main' },
    });
    pipelineCache.set(device, p);
  }
  return p;
}

/** Write a typed array whose byte length may not be a multiple of 4 (WebGPU requires it). */
function writePadded(device: GPUDevice, buffer: GPUBuffer, data: Uint16Array): void {
  const even = data.length & ~1;
  if (even > 0) device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, even * 2);
  if (even < data.length) {
    device.queue.writeBuffer(buffer, even * 2, new Uint16Array([data[even], 0]));
  }
}

/**
 * The normalised float32 CFA (padded size) on `device`. The caller owns the returned buffer;
 * `extraUsage` adds usages beyond STORAGE.
 */
export function uploadNormalizedCfa(
  device: GPUDevice, input: DemosaicInput, extraUsage: GPUBufferUsageFlags = 0,
): GPUBuffer {
  const { width, height } = input;
  const n = width * height;
  const S = GPUBufferUsage.STORAGE;
  const D = GPUBufferUsage.COPY_DST;
  const temporaries: GPUBuffer[] = [];
  const temp = (size: number, usage: GPUBufferUsageFlags) => {
    const b = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage });
    temporaries.push(b);
    return b;
  };
  const out = device.createBuffer({ size: Math.max(4, n * 4), usage: S | extraUsage });
  try {
    const rawBuf = temp(n * 2, S | D);
    writePadded(device, rawBuf, input.cfa);
    const lutBuf = temp(input.lut.byteLength, S | D);
    device.queue.writeBuffer(lutBuf, 0, input.lut.buffer, input.lut.byteOffset, input.lut.byteLength);
    const patternBuf = temp(input.pattern.byteLength, S | D);
    device.queue.writeBuffer(patternBuf, 0, input.pattern.buffer, input.pattern.byteOffset, input.pattern.byteLength);
    const paramBuf = temp(16, GPUBufferUsage.UNIFORM | D);
    device.queue.writeBuffer(paramBuf, 0, new Uint32Array([width, height, input.period, 0]));

    const pipeline = pipelineFor(device);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rawBuf } },
        { binding: 1, resource: { buffer: lutBuf } },
        { binding: 2, resource: { buffer: patternBuf } },
        { binding: 3, resource: { buffer: out } },
        { binding: 4, resource: { buffer: paramBuf } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(width / WG), Math.ceil(height / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
    return out;
  } catch (error) {
    out.destroy();
    throw error;
  } finally {
    // Destroying after submit is safe: submitted work keeps its resources alive.
    for (const b of temporaries) b.destroy();
  }
}
