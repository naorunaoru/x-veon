import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createGpuNNPipeline } from './tile-blend-gpu';
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
  onSubmittedWorkDone: () => Promise<void>;
}

function makeFakeDevice(): { device: GPUDevice; buffers: FakeBuffer[]; state: FakeDeviceState } {
  const buffers: FakeBuffer[] = [];
  const state: FakeDeviceState = { failAt: null, onSubmittedWorkDone: () => Promise.resolve() };
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
      submit: () => {},
      onSubmittedWorkDone: () => state.onSubmittedWorkDone(),
    },
  } as unknown as GPUDevice;

  return { device, buffers, state };
}

function makeMasks(patchSize: number): ChannelMasks {
  const n = patchSize * patchSize;
  return { r: new Float32Array(n), g: new Float32Array(n), b: new Float32Array(n) };
}

const PATCH_SIZE = 8;

/** One tile, small enough to keep the fake buffer count easy to reason about. */
function buildPipeline(device: GPUDevice) {
  return createGpuNNPipeline(
    device,
    new Float32Array(PATCH_SIZE * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
    makeMasks(PATCH_SIZE),
    [1, 1, 1],
    [{ x: 0, y: 0 }],
    PATCH_SIZE, PATCH_SIZE,
    PATCH_SIZE, 2,
    0, 0,
    4, 4,
    1,
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
    expect(buffers.length).toBe(10);
    for (const b of buffers) expect(b.destroy).not.toHaveBeenCalled();
  });

  it('releases already-created buffers when construction fails partway through', () => {
    const { device, buffers, state } = makeFakeDevice();
    state.failAt = 6; // blendOutBuf: 5 buffers already created before this one throws
    expect(() => buildPipeline(device)).toThrow(/allocation 6 failed/);
    expect(buffers.length).toBe(5);
    for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
  });

  it('finalize transfers the output buffer and releases every other buffer exactly once', async () => {
    const { device, buffers } = makeFakeDevice();
    const gpu = buildPipeline(device);
    const cropOutBuf = buffers[9]; // last buffer allocated during construction

    gpu.extractBatch(0, 1);
    gpu.accumulateBatch({} as GPUBuffer, 0, 1);
    const output = await gpu.finalize();

    expect(output).toBe(cropOutBuf as unknown as GPUBuffer);
    for (const b of buffers) {
      expect(b.destroy).toHaveBeenCalledTimes(b === cropOutBuf ? 0 : 1);
    }

    // A later cleanup call (e.g. defensive teardown) must not double-free anything,
    // and the transferred output must stay alive.
    gpu.destroy();
    for (const b of buffers) {
      expect(b.destroy).toHaveBeenCalledTimes(b === cropOutBuf ? 0 : 1);
    }
  });

  it('leaves every buffer intact when finalize fails, so destroy() releases each exactly once', async () => {
    const { device, buffers, state } = makeFakeDevice();
    const gpu = buildPipeline(device);
    state.onSubmittedWorkDone = () => Promise.reject(new Error('device lost'));

    gpu.extractBatch(0, 1);
    gpu.accumulateBatch({} as GPUBuffer, 0, 1);
    await expect(gpu.finalize()).rejects.toThrow('device lost');

    // 10 constructor buffers + extractOut + batchPosBuf + finParamBuf + cropParamBuf
    expect(buffers.length).toBe(14);
    for (const b of buffers) expect(b.destroy).not.toHaveBeenCalled();

    gpu.destroy();
    for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);

    gpu.destroy(); // idempotent — nothing left to release
    for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
  });
});
