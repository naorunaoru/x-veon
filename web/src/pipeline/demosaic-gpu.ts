const BILINEAR_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dy: u32,
  dx: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// X-Trans 6x6 CFA pattern (flattened): R=0, G=1, B=2
const XTRANS = array<u32, 36>(
  0, 2, 1, 2, 0, 1,
  1, 1, 0, 1, 1, 2,
  1, 1, 2, 1, 1, 0,
  2, 0, 1, 0, 2, 1,
  1, 1, 2, 1, 1, 0,
  1, 1, 0, 1, 1, 2
);

fn cfa_ch(y: u32, x: u32) -> u32 {
  return XTRANS[((y + params.dy) % 6u) * 6u + ((x + params.dx) % 6u)];
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

  // Write CHW planar output
  let plane = params.width * params.height;
  output[idx] = rgb[0u];
  output[plane + idx] = rgb[1u];
  output[2u * plane + idx] = rgb[2u];
}
`;

// ---------- DHT Pass 1: directional green interpolation (H and V) ----------
const DHT_GREEN_WGSL = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  dy: u32,
  dx: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> green_hv: array<f32>; // planar: [green_h | green_v]
@group(0) @binding(2) var<uniform> params: Params;

const XTRANS = array<u32, 36>(
  0, 2, 1, 2, 0, 1,
  1, 1, 0, 1, 1, 2,
  1, 1, 2, 1, 1, 0,
  2, 0, 1, 0, 2, 1,
  1, 1, 2, 1, 1, 0,
  1, 1, 0, 1, 1, 2
);

fn cfa_ch(y: u32, x: u32) -> u32 {
  return XTRANS[((y + params.dy) % 6u) * 6u + ((x + params.dx) % 6u)];
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
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> green_hv: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const XTRANS = array<u32, 36>(
  0, 2, 1, 2, 0, 1,
  1, 1, 0, 1, 1, 2,
  1, 1, 2, 1, 1, 0,
  2, 0, 1, 0, 2, 1,
  1, 1, 2, 1, 1, 0,
  1, 1, 0, 1, 1, 2
);

fn cfa_ch(y: u32, x: u32) -> u32 {
  return XTRANS[((y + params.dy) % 6u) * 6u + ((x + params.dx) % 6u)];
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

  // Write CHW planar output
  output[idx] = rgb[0];
  output[plane + idx] = rgb[1];
  output[2u * plane + idx] = rgb[2];
}
`;

let device: GPUDevice | null = null;
let pipeline: GPUComputePipeline | null = null;
let dhtGreenPipeline: GPUComputePipeline | null = null;
let dhtResolvePipeline: GPUComputePipeline | null = null;

export async function initDemosaicGpu(): Promise<boolean> {
  if (!navigator.gpu) {
    console.warn('[demosaic-gpu] WebGPU not available');
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    // Request the adapter's max buffer limits so we can handle large images.
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    const shaderModule = device.createShaderModule({ code: BILINEAR_WGSL });
    pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    const dhtGreenModule = device.createShaderModule({ code: DHT_GREEN_WGSL });
    dhtGreenPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: dhtGreenModule, entryPoint: 'main' },
    });

    const dhtResolveModule = device.createShaderModule({ code: DHT_RESOLVE_WGSL });
    dhtResolvePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: dhtResolveModule, entryPoint: 'main' },
    });

    console.log(
      `[demosaic-gpu] initialized: ${adapter.info?.device ?? 'unknown GPU'}, ` +
      `maxStorageBuffer=${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)}MB`,
    );
    return true;
  } catch (e) {
    console.warn('[demosaic-gpu] init failed:', e);
    return false;
  }
}

export function gpuAvailable(): boolean {
  return device !== null && pipeline !== null;
}

export async function runBilinearGpu(
  cfa: Float32Array,
  width: number,
  height: number,
  dy: number,
  dx: number,
): Promise<Float32Array> {
  if (!device || !pipeline) throw new Error('GPU demosaic not initialized');

  const inputBytes = width * height * 4;
  const outputBytes = 3 * width * height * 4;

  // Check buffer size fits device limits
  if (outputBytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error('Image too large for GPU storage buffer');
  }

  const inputBuffer = device.createBuffer({
    size: inputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, cfa.buffer, cfa.byteOffset, cfa.byteLength);

  const outputBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([width, height, dy, dx]));

  const stagingBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(width / 16),
    Math.ceil(height / 16),
  );
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBytes);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(new Float32Array(stagingBuffer.getMappedRange()));
  stagingBuffer.unmap();

  inputBuffer.destroy();
  outputBuffer.destroy();
  paramsBuffer.destroy();
  stagingBuffer.destroy();

  return result;
}

export async function runDhtGpu(
  cfa: Float32Array,
  width: number,
  height: number,
  dy: number,
  dx: number,
): Promise<Float32Array> {
  if (!device || !dhtGreenPipeline || !dhtResolvePipeline) {
    throw new Error('GPU DHT not initialized');
  }

  const pixels = width * height;
  const inputBytes = pixels * 4;
  const greenHvBytes = 2 * pixels * 4;  // planar: [green_h | green_v]
  const outputBytes = 3 * pixels * 4;

  const maxBinding = device.limits.maxStorageBufferBindingSize;
  if (greenHvBytes > maxBinding || outputBytes > maxBinding) {
    throw new Error('Image too large for GPU storage buffer');
  }

  // Shared buffers
  const inputBuffer = device.createBuffer({
    size: inputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, cfa.buffer, cfa.byteOffset, cfa.byteLength);

  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([width, height, dy, dx]));

  // Intermediate green buffer (pass 1 output, pass 2 input)
  const greenHvBuffer = device.createBuffer({
    size: greenHvBytes,
    usage: GPUBufferUsage.STORAGE,
  });

  // Final output
  const outputBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const stagingBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const wgX = Math.ceil(width / 16);
  const wgY = Math.ceil(height / 16);

  // Pass 1: directional green interpolation
  const greenBindGroup = device.createBindGroup({
    layout: dhtGreenPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: greenHvBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  // Pass 2: homogeneity selection + R/B color-difference
  const resolveBindGroup = device.createBindGroup({
    layout: dhtResolvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: greenHvBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();

  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(dhtGreenPipeline);
  pass1.setBindGroup(0, greenBindGroup);
  pass1.dispatchWorkgroups(wgX, wgY);
  pass1.end();

  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(dhtResolvePipeline);
  pass2.setBindGroup(0, resolveBindGroup);
  pass2.dispatchWorkgroups(wgX, wgY);
  pass2.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBytes);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(new Float32Array(stagingBuffer.getMappedRange()));
  stagingBuffer.unmap();

  inputBuffer.destroy();
  paramsBuffer.destroy();
  greenHvBuffer.destroy();
  outputBuffer.destroy();
  stagingBuffer.destroy();

  return result;
}
