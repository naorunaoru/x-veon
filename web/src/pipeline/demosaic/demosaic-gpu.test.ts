import { afterEach, describe, expect, it, vi } from 'vitest';
import { runBilinearGpu, runDhtGpu } from './demosaic-gpu';
function fixture(fail?: string, at = 0) {
  const buffers: { destroy: ReturnType<typeof vi.fn> }[] = [];
  let uploads = 0;
  const check = (stage: string) => { if (fail === stage) throw new Error(stage); };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  // A fresh device object per fixture, so the per-device pipeline cache starts empty.
  const device = {
    limits: { maxStorageBufferBindingSize: 1024 * 1024 },
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => { check('binding'); return {}; },
    createBuffer: () => {
      if (fail === 'allocation' && buffers.length === at) throw new Error('allocation');
      const buffer = { destroy: vi.fn() };
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder: () => { check('encoding'); return { beginComputePass: () => pass, finish: () => ({}) }; },
    queue: {
      writeBuffer: () => { if (fail === 'upload' && uploads++ === at) throw new Error('upload'); },
      submit: () => check('submit'),
    },
  } as unknown as GPUDevice;
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16 });
  return { device, buffers };
}
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
for (const [name, run, count] of [['bilinear', runBilinearGpu, 3], ['dht', runDhtGpu, 4]] as const) {
  describe(`${name} GPU resource lifecycle`, () => {
    const cfa = {} as GPUBuffer;
    const invoke = (device: GPUDevice) => run(device, cfa, 6, 6, new Uint32Array([0, 1, 1, 2]), 2);
    it.each(Array.from({ length: count }, (_, i) => i))('cleans partial allocation %i', at => {
      const { device, buffers } = fixture('allocation', at);
      expect(() => invoke(device)).toThrow('allocation'); expect(buffers).toHaveLength(at);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
    });
    it.each([0, 1])('cleans failed upload %i, including CFA pattern upload', at => {
      const { device, buffers } = fixture('upload', at);
      expect(() => invoke(device)).toThrow('upload'); expect(buffers.length).toBeGreaterThan(0);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
    });
    it.each(['binding', 'encoding', 'submit'])('cleans %s failure', fail => {
      const { device, buffers } = fixture(fail); expect(() => invoke(device)).toThrow(fail);
      expect(buffers).toHaveLength(count);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
    });
    it('keeps the output on the GPU for the caller and releases every other buffer', () => {
      const { device, buffers } = fixture();
      const output = invoke(device);
      expect(output).toBe(buffers[0]);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(b === buffers[0] ? 0 : 1);
    });
  });
}
