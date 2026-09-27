import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createGpuNNPipeline, blendWeights1d } from './tile-blend-gpu';
import { generateTiles } from '../preprocess/preprocessor';
import { PATCH_SIZE as NN_PATCH, OVERLAP as NN_OVERLAP } from '../constants';
import type { ChannelMasks } from '../types';

// jsdom has no real WebGPU, so these tests drive createGpuNNPipeline against a small
// fake GPUDevice that just records every buffer it's asked to create. The compute
// passes themselves are irrelevant here — what's under test is the pipeline's own
// buffer-ownership bookkeeping: a partial-construction failure releases what was
// already allocated, finalize() transfers its output without destroying it, and every
// other buffer is released exactly once even if destroy() is called again afterward.

class FakeBuffer {
  mapState: GPUBufferMapState = 'unmapped';
  destroy = vi.fn();
  constructor(public size: number, public usage: number) {}
}

interface FakeDeviceState {
  failAt: number | null; // 1-based index of the createBuffer call that should throw
  failSubmit: boolean;
}

function makeFakeDevice(): { device: GPUDevice; buffers: FakeBuffer[]; state: FakeDeviceState } {
  const buffers: FakeBuffer[] = [];
  const state: FakeDeviceState = { failAt: null, failSubmit: false };
  let created = 0;

  const device = {
    createBuffer: ({ size, usage }: { size: number; usage: number }) => {
      created += 1;
      if (state.failAt !== null && created === state.failAt) {
        throw new Error(`fake device: buffer allocation ${created} failed`);
      }
      const b = new FakeBuffer(size, usage);
      buffers.push(b);
      return b;
    },
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
      }),
      finish: () => ({}),
    }),
    queue: {
      writeBuffer: () => {},
      submit: () => { if (state.failSubmit) throw new Error('device lost'); },
    },
  } as unknown as GPUDevice;

  return { device, buffers, state };
}

function makeMasks(patchSize: number): ChannelMasks {
  const n = patchSize * patchSize;
  return { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) };
}

const PATCH_SIZE = 8;

/** The CFA buffer handed to the pipeline (it takes ownership; not created through the device). */
let cfaBuf: FakeBuffer;

/** Three tiles, small enough to keep the fake buffer count easy to reason about. */
function buildPipeline(device: GPUDevice) {
  cfaBuf = new FakeBuffer(PATCH_SIZE * PATCH_SIZE * 4, 0);
  return createGpuNNPipeline(
    device,
    cfaBuf as unknown as GPUBuffer, PATCH_SIZE, PATCH_SIZE,
    makeMasks(PATCH_SIZE),
    [1, 1, 1],
    [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }],
    PATCH_SIZE, PATCH_SIZE,
    PATCH_SIZE, 2,
    1, 2,
    2,
  );
}

describe('createGpuNNPipeline buffer lifecycle', () => {
  beforeEach(() => {
    vi.stubGlobal('GPUBufferUsage', {
      STORAGE: 0x0080, COPY_DST: 0x0008, COPY_SRC: 0x0004, UNIFORM: 0x0040,
    });
  });

  it('allocates one tracked buffer per construction-time upload', () => {
    const { device, buffers } = makeFakeDevice();
    buildPipeline(device);
    expect(buffers.length).toBe(8);
    for (const b of [...buffers, cfaBuf]) expect(b.destroy).not.toHaveBeenCalled();
  });

  it('releases already-created buffers, and the CFA it was given, when construction fails', () => {
    const { device, buffers, state } = makeFakeDevice();
    state.failAt = 5; // blendOutBuf: 4 buffers already created before this one throws
    expect(() => buildPipeline(device)).toThrow(/allocation 5 failed/);
    expect(buffers.length).toBe(4);
    for (const b of [...buffers, cfaBuf]) expect(b.destroy).toHaveBeenCalledTimes(1);
  });

  it('reuses one extract buffer per batch size', () => {
    const { device, buffers } = makeFakeDevice();
    const gpu = buildPipeline(device);
    const first = gpu.extractBatch(0, 2);
    expect(gpu.extractBatch(0, 2)).toBe(first);
    const last = gpu.extractBatch(2, 1);
    expect(last).not.toBe(first);
    expect(gpu.extractBatch(2, 1)).toBe(last);
    expect(buffers.length).toBe(10);
  });

  it('finalize hands over the padded blend buffer with the visible offset and releases the rest once', async () => {
    const { device, buffers } = makeFakeDevice();
    const gpu = buildPipeline(device);
    const blendOutBuf = buffers[4];

    gpu.extractBatch(0, 2);
    gpu.accumulateBatch({} as GPUBuffer, 0, 2);
    const output = await gpu.finalize();

    expect(output).toEqual({ buffer: blendOutBuf, stride: PATCH_SIZE, offsetX: 2, offsetY: 1 });
    for (const b of [...buffers, cfaBuf]) {
      expect(b.destroy).toHaveBeenCalledTimes(b === blendOutBuf ? 0 : 1);
    }

    // A later cleanup call (e.g. defensive teardown) must not double-free anything,
    // and the transferred output must stay alive.
    gpu.destroy();
    for (const b of [...buffers, cfaBuf]) {
      expect(b.destroy).toHaveBeenCalledTimes(b === blendOutBuf ? 0 : 1);
    }
  });

  it('leaves every buffer intact when finalize fails, so destroy() releases each exactly once', async () => {
    const { device, buffers, state } = makeFakeDevice();
    const gpu = buildPipeline(device);

    gpu.extractBatch(0, 2);
    gpu.accumulateBatch({} as GPUBuffer, 0, 2);
    state.failSubmit = true;
    await expect(gpu.finalize()).rejects.toThrow('device lost');

    // 8 constructor buffers + extractOut + finParamBuf
    expect(buffers.length).toBe(10);
    for (const b of [...buffers, cfaBuf]) expect(b.destroy).not.toHaveBeenCalled();

    gpu.destroy();
    for (const b of [...buffers, cfaBuf]) expect(b.destroy).toHaveBeenCalledTimes(1);

    gpu.destroy(); // idempotent — nothing left to release
    for (const b of [...buffers, cfaBuf]) expect(b.destroy).toHaveBeenCalledTimes(1);
  });
});

describe('blend weights', () => {
  it('give every pixel of the image positive weight, including the first row and column', () => {
    const w = blendWeights1d(NN_PATCH, NN_OVERLAP);
    const { tiles, wPad } = generateTiles(1000, 1000, NN_PATCH, NN_OVERLAP);
    const sum = new Float64Array(wPad);
    for (const x0 of new Set(tiles.map((t) => t.x))) for (let i = 0; i < NN_PATCH; i++) sum[x0 + i] += w[i];
    expect(Math.min(...sum.slice(0, 1000))).toBeGreaterThan(0);
    // Where two tiles overlap, their ramps add up to exactly one.
    const stride = NN_PATCH - NN_OVERLAP;
    for (let i = 0; i < NN_OVERLAP; i++) expect(sum[stride + i]).toBeCloseTo(1, 6);
  });
});
