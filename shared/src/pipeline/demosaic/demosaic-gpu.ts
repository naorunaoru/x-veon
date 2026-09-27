const BILINEAR_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dy: u32,
  dx: u32,
  period: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> cfa_pattern: array<u32>;

fn cfa_ch(y: u32, x: u32) -> u32 {
  let p = params.period;
  return cfa_pattern[((y + params.dy) % p) * p + ((x + params.dx) % p)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = y * params.width + x;
  let known = cfa_ch(y, x);
  let val = input[idx];
  let w = i32(params.width);
  let h = i32(params.height);

  // Interpolate each channel: known channel uses CFA value directly,
  // missing channels average same-color neighbors in a 5x5 window.
  var rgb = array<f32, 3>(0.0, 0.0, 0.0);
  rgb[known] = val;

  for (var ch: u32 = 0u; ch < 3u; ch++) {
    if (ch == known) { continue; }
    var sum: f32 = 0.0;
    var count: f32 = 0.0;
    for (var ky: i32 = -2; ky <= 2; ky++) {
      for (var kx: i32 = -2; kx <= 2; kx++) {
        let ny = i32(y) + ky;
        let nx = i32(x) + kx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
          if (cfa_ch(u32(ny), u32(nx)) == ch) {
            sum += input[u32(ny) * params.width + u32(nx)];
            count += 1.0;
          }
        }
      }
    }
    if (count > 0.0) {
      rgb[ch] = sum / count;
    }
  }

  // Write HWC output (read in place by the post-process, padding included)
  let o = idx * 3u;
  output[o] = rgb[0u];
  output[o + 1u] = rgb[1u];
  output[o + 2u] = rgb[2u];
}
`;

// ---------- DHT Pass 1: directional green interpolation (H and V) ----------
const DHT_GREEN_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dy: u32,
  dx: u32,
  period: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> green_hv: array<f32>; // planar: [green_h | green_v]
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> cfa_pattern: array<u32>;

fn cfa_ch(y: u32, x: u32) -> u32 {
  let p = params.period;
  return cfa_pattern[((y + params.dy) % p) * p + ((x + params.dx) % p)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = y * params.width + x;
  let plane = params.width * params.height;
  let ch = cfa_ch(y, x);
  let val = input[idx];

  if (ch == 1u) {
    // Green pixel: both directions use the known value
    green_hv[idx] = val;
    green_hv[plane + idx] = val;
    return;
  }

  let w = i32(params.width);
  let h = i32(params.height);

  // Horizontal: inverse-distance-weighted green neighbors in same row, +/-3
  var sum_h: f32 = 0.0;
  var wt_h: f32 = 0.0;
  for (var kx: i32 = -3; kx <= 3; kx++) {
    let nx = i32(x) + kx;
    if (nx >= 0 && nx < w) {
      if (cfa_ch(y, u32(nx)) == 1u) {
        let d = f32(abs(kx));
        let weight = 1.0 / (1.0 + d * d);
        sum_h += input[y * params.width + u32(nx)] * weight;
        wt_h += weight;
      }
    }
  }
  green_hv[idx] = select(val, sum_h / wt_h, wt_h > 0.0);

  // Vertical: inverse-distance-weighted green neighbors in same column, +/-3
  var sum_v: f32 = 0.0;
  var wt_v: f32 = 0.0;
  for (var ky: i32 = -3; ky <= 3; ky++) {
    let ny = i32(y) + ky;
    if (ny >= 0 && ny < h) {
      if (cfa_ch(u32(ny), x) == 1u) {
        let d = f32(abs(ky));
        let weight = 1.0 / (1.0 + d * d);
        sum_v += input[u32(ny) * params.width + x] * weight;
        wt_v += weight;
      }
    }
  }
  green_hv[plane + idx] = select(val, sum_v / wt_v, wt_v > 0.0);
}
`;

// ---------- DHT Pass 2: homogeneity selection + R/B color-difference ----------
const DHT_RESOLVE_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dy: u32,
  dx: u32,
  period: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> green_hv: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> cfa_pattern: array<u32>;

fn cfa_ch(y: u32, x: u32) -> u32 {
  let p = params.period;
  return cfa_pattern[((y + params.dy) % p) * p + ((x + params.dx) % p)];
}

// Read green estimate at a neighbor pixel (average of H and V)
fn green_avg(nidx: u32, plane: u32) -> f32 {
  return (green_hv[nidx] + green_hv[plane + nidx]) * 0.5;
}

// Interpolate a missing channel via color-difference in a 5x5 window
fn interp_cd(y: u32, x: u32, target_ch: u32, green: f32, plane: u32) -> f32 {
  let w = i32(params.width);
  let h = i32(params.height);
  var cd_sum: f32 = 0.0;
  var cd_wt: f32 = 0.0;
  for (var ky: i32 = -2; ky <= 2; ky++) {
    for (var kx: i32 = -2; kx <= 2; kx++) {
      let ny = i32(y) + ky;
      let nx = i32(x) + kx;
      if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
        if (cfa_ch(u32(ny), u32(nx)) == target_ch) {
          let nidx = u32(ny) * params.width + u32(nx);
          let cd = input[nidx] - green_avg(nidx, plane);
          let d = f32(abs(ky) + abs(kx));
          let weight = 1.0 / (1.0 + d);
          cd_sum += cd * weight;
          cd_wt += weight;
        }
      }
    }
  }
  if (cd_wt > 0.0) {
    return green + cd_sum / cd_wt;
  }
  return green;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let w = i32(params.width);
  let h = i32(params.height);
  let idx = y * params.width + x;
  let plane = params.width * params.height;
  let ch = cfa_ch(y, x);

  // Read directional greens for this pixel
  let gh = green_hv[idx];
  let gv = green_hv[plane + idx];

  // Homogeneity test: sum of absolute green differences in 3x3 window
  var hom_h: f32 = 0.0;
  var hom_v: f32 = 0.0;
  for (var ky: i32 = -1; ky <= 1; ky++) {
    for (var kx: i32 = -1; kx <= 1; kx++) {
      if (ky == 0 && kx == 0) { continue; }
      let ny = u32(clamp(i32(y) + ky, 0, h - 1));
      let nx = u32(clamp(i32(x) + kx, 0, w - 1));
      let nidx = ny * params.width + nx;
      hom_h += abs(green_hv[nidx] - gh);
      hom_v += abs(green_hv[plane + nidx] - gv);
    }
  }

  // Smooth blending: direction with lower variation gets more weight
  let alpha = hom_v / (hom_h + hom_v + 1e-6);
  let green = alpha * gh + (1.0 - alpha) * gv;

  // Build RGB
  var rgb = array<f32, 3>(0.0, 0.0, 0.0);

  if (ch == 1u) {
    // Green pixel: green is known, interpolate R and B via color-difference
    rgb[1] = input[idx];
    rgb[0] = interp_cd(y, x, 0u, input[idx], plane);
    rgb[2] = interp_cd(y, x, 2u, input[idx], plane);
  } else {
    // R or B pixel: known channel from CFA, green from DHT, other via color-diff
    rgb[ch] = input[idx];
    rgb[1] = green;
    let other = 2u - ch; // ch=0 -> other=2, ch=2 -> other=0
    rgb[other] = interp_cd(y, x, other, green, plane);
  }

  // Write HWC output (read in place by the post-process, padding included)
  let o = idx * 3u;
  output[o] = rgb[0];
  output[o + 1u] = rgb[1];
  output[o + 2u] = rgb[2];
}
`;

interface Pipelines {
  bilinear: GPUComputePipeline;
  dhtGreen: GPUComputePipeline;
  dhtResolve: GPUComputePipeline;
}

// The GPU methods run on the pipeline's shared device (ONNX Runtime's), so their output stays
// on the GPU for the post-process instead of a readback and re-upload through a second device.
const pipelineCache = new WeakMap<GPUDevice, Pipelines>();

function makePipeline(device: GPUDevice, code: string): GPUComputePipeline {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
}

function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    p = {
      bilinear: makePipeline(device, BILINEAR_WGSL),
      dhtGreen: makePipeline(device, DHT_GREEN_WGSL),
      dhtResolve: makePipeline(device, DHT_RESOLVE_WGSL),
    };
    pipelineCache.set(device, p);
  }
  return p;
}

/** Track each run allocation immediately; `keep` survives the run, everything else is released. */
function runBuffers(device: GPUDevice) {
  const owned: GPUBuffer[] = [];
  return {
    allocate(descriptor: GPUBufferDescriptor): GPUBuffer {
      const buffer = device.createBuffer(descriptor);
      owned.push(buffer);
      return buffer;
    },
    dispose(keep?: GPUBuffer): void {
      for (const buffer of owned) if (buffer !== keep) buffer.destroy();
    },
  };
}

function checkFits(device: GPUDevice, ...sizes: number[]): void {
  if (sizes.some((size) => size > device.limits.maxStorageBufferBindingSize)) {
    throw new Error('Image too large for GPU storage buffer');
  }
}

function writeCommon(
  device: GPUDevice, allocate: (d: GPUBufferDescriptor) => GPUBuffer,
  width: number, height: number, cfaPattern: Uint32Array, period: number,
): { paramsBuffer: GPUBuffer; cfaPatternBuffer: GPUBuffer } {
  const paramsBuffer = allocate({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // The CFA is canonically aligned, so the pattern shift is always 0.
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([width, height, 0, 0, period]));
  const cfaPatternBuffer = allocate({
    size: cfaPattern.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cfaPatternBuffer, 0, cfaPattern.buffer, cfaPattern.byteOffset, cfaPattern.byteLength);
  return { paramsBuffer, cfaPatternBuffer };
}

/**
 * Bilinear demosaic of the normalised float32 CFA in `cfa` (width × height). Returns an HWC
 * float32 buffer of the same size that the caller owns; `cfa` stays the caller's.
 */
export function runBilinearGpu(
  device: GPUDevice, cfa: GPUBuffer, width: number, height: number,
  cfaPattern: Uint32Array, period: number,
): GPUBuffer {
  const outputBytes = 3 * width * height * 4;
  checkFits(device, outputBytes);
  const pipes = getPipelines(device);

  const buffers = runBuffers(device);
  let output: GPUBuffer | undefined;
  try {
    output = buffers.allocate({ size: outputBytes, usage: GPUBufferUsage.STORAGE });
    const { paramsBuffer, cfaPatternBuffer } = writeCommon(device, buffers.allocate, width, height, cfaPattern, period);

    const bindGroup = device.createBindGroup({
      layout: pipes.bilinear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cfa } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: cfaPatternBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipes.bilinear);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    pass.end();
    device.queue.submit([encoder.finish()]);
    buffers.dispose(output);
    return output;
  } catch (error) {
    buffers.dispose();
    throw error;
  }
}

/**
 * DHT demosaic of the normalised float32 CFA in `cfa` (width × height). Returns an HWC float32
 * buffer of the same size that the caller owns; `cfa` stays the caller's.
 */
export function runDhtGpu(
  device: GPUDevice, cfa: GPUBuffer, width: number, height: number,
  cfaPattern: Uint32Array, period: number,
): GPUBuffer {
  const pixels = width * height;
  const greenHvBytes = 2 * pixels * 4;  // planar: [green_h | green_v]
  const outputBytes = 3 * pixels * 4;
  checkFits(device, greenHvBytes, outputBytes);
  const pipes = getPipelines(device);

  const buffers = runBuffers(device);
  let output: GPUBuffer | undefined;
  try {
    output = buffers.allocate({ size: outputBytes, usage: GPUBufferUsage.STORAGE });
    const { paramsBuffer, cfaPatternBuffer } = writeCommon(device, buffers.allocate, width, height, cfaPattern, period);
    // Intermediate green buffer (pass 1 output, pass 2 input)
    const greenHvBuffer = buffers.allocate({ size: greenHvBytes, usage: GPUBufferUsage.STORAGE });

    const wgX = Math.ceil(width / 16);
    const wgY = Math.ceil(height / 16);

    // Pass 1: directional green interpolation
    const greenBindGroup = device.createBindGroup({
      layout: pipes.dhtGreen.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cfa } },
        { binding: 1, resource: { buffer: greenHvBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
        { binding: 3, resource: { buffer: cfaPatternBuffer } },
      ],
    });

    // Pass 2: homogeneity selection + R/B color-difference
    const resolveBindGroup = device.createBindGroup({
      layout: pipes.dhtResolve.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cfa } },
        { binding: 1, resource: { buffer: greenHvBuffer } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: cfaPatternBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();

    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(pipes.dhtGreen);
    pass1.setBindGroup(0, greenBindGroup);
    pass1.dispatchWorkgroups(wgX, wgY);
    pass1.end();

    const pass2 = encoder.beginComputePass();
    pass2.setPipeline(pipes.dhtResolve);
    pass2.setBindGroup(0, resolveBindGroup);
    pass2.dispatchWorkgroups(wgX, wgY);
    pass2.end();

    device.queue.submit([encoder.finish()]);
    buffers.dispose(output);
    return output;
  } catch (error) {
    buffers.dispose();
    throw error;
  }
}
