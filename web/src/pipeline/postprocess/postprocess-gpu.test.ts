import { afterEach, describe, expect, it, vi } from 'vitest';
import { gpuPostprocess } from './postprocess-gpu';
function fixture(fail?: string, allocation = 0) {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 });
  const buffers: { destroy: ReturnType<typeof vi.fn> }[] = [];
  const check = (stage: string) => { if (fail === stage) throw new Error(stage); };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const device = {
    createShaderModule: () => { check('setup'); return {}; },
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: () => {
      if (fail === 'allocation' && buffers.length === allocation) throw new Error('allocation');
      const buffer = { destroy: vi.fn() }; buffers.push(buffer); return buffer;
    },
    createBindGroup: () => { check('binding'); return {}; },
    createCommandEncoder: () => { check('encoding'); return { beginComputePass: () => pass, finish: () => ({}) }; },
    queue: { writeBuffer: () => check('upload'), submit: () => check('submit') },
  } as unknown as GPUDevice;
  const input = { destroy: vi.fn() } as unknown as GPUBuffer;
  const run = (cpu?: Float32Array) => gpuPostprocess(device, cpu ?? input, 6, 6,
    new Float32Array([1, 1, 1]), [1, 1, 1], null, 1);
  return { buffers, input, run };
}
afterEach(() => vi.unstubAllGlobals());
describe('gpuPostprocess ownership', () => {
  it.each(['setup', 'encoding', 'binding', 'upload', 'submit'])('releases resources after %s failure', async stage => {
    const f = fixture(stage);
    await expect(f.run()).rejects.toThrow(stage);
    expect(f.input.destroy).toHaveBeenCalledTimes(1);
    for (const b of f.buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
  });
  it.each(Array.from({ length: 14 }, (_, i) => i))('cleans partial allocation at buffer %i', async allocation => {
    const f = fixture('allocation', allocation);
    await expect(f.run()).rejects.toThrow('allocation');
    expect(f.buffers).toHaveLength(allocation);
    expect(f.input.destroy).toHaveBeenCalledTimes(1);
    for (const b of f.buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
  });
  it.each([false, true])('transfers only output on success (CPU input: %s)', async cpu => {
    const f = fixture(); const data = new Float32Array(108).fill(0.5);
    const result = await f.run(cpu ? data : undefined);
    expect(result.bytesPerRow).toBe(256);
    expect(f.input.destroy).toHaveBeenCalledTimes(cpu ? 0 : 1);
    expect(data).toEqual(new Float32Array(108).fill(0.5));
    for (const b of f.buffers) expect(b.destroy).toHaveBeenCalledTimes(b === (result.buffer as unknown) ? 0 : 1);
    result.buffer.destroy();
    for (const b of f.buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
  });
  it('releases CPU upload buffer if upload fails', async () => {
    const f = fixture('upload');
    await expect(f.run(new Float32Array(108))).rejects.toThrow('upload');
    expect(f.buffers).toHaveLength(1);
    expect(f.buffers[0].destroy).toHaveBeenCalledTimes(1);
  });
});
