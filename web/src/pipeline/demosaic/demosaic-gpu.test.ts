import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { initDemosaicGpu, runBilinearGpu, runDhtGpu } from './demosaic-gpu';
async function fixture(fail?: string, at = 0) {
  const buffers: { destroy: ReturnType<typeof vi.fn>; unmap: ReturnType<typeof vi.fn>; mapState: string }[] = [];
  let uploads = 0;
  const check = (stage: string) => { if (fail === stage) throw new Error(stage); };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const device = {
    limits: { maxStorageBufferBindingSize: 1024 * 1024 },
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => { check('binding'); return {}; },
    createBuffer: ({ size }: { size: number }) => {
      if (fail === 'allocation' && buffers.length === at) throw new Error('allocation');
      const buffer = {
        mapState: 'unmapped', destroy: vi.fn(), unmap: vi.fn(() => { buffer.mapState = 'unmapped'; }),
        mapAsync: async () => { check('map'); buffer.mapState = 'mapped'; },
        getMappedRange: () => { check('range'); return new ArrayBuffer(size); },
      };
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder: () => { check('encoding'); return { beginComputePass: () => pass, copyBufferToBuffer() {}, finish: () => ({}) }; },
    queue: {
      writeBuffer: () => { if (fail === 'upload' && uploads++ === at) throw new Error('upload'); },
      submit: () => check('submit'),
    },
  };
  vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => ({ requestDevice: async () => device, limits: device.limits }) } });
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16 });
  vi.stubGlobal('GPUMapMode', { READ: 1 });
  expect(await initDemosaicGpu()).toBe(true);
  return buffers;
}
beforeEach(() => vi.spyOn(console, 'log').mockImplementation(() => {}));
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
for (const [name, run, count] of [['bilinear', runBilinearGpu, 5], ['dht', runDhtGpu, 6]] as const) {
  describe(`${name} GPU resource lifecycle`, () => {
    const invoke = () => run(new Float32Array(36), 6, 6, 0, 0, new Uint32Array([0, 1, 1, 2]), 2);
    it.each(Array.from({ length: count }, (_, i) => i))('cleans partial allocation %i', async at => {
      const buffers = await fixture('allocation', at);
      await expect(invoke()).rejects.toThrow('allocation'); expect(buffers).toHaveLength(at);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
    });
    it.each([0, 1, 2])('cleans failed upload %i, including CFA pattern upload', async at => {
      const buffers = await fixture('upload', at);
      await expect(invoke()).rejects.toThrow('upload'); expect(buffers.length).toBeGreaterThan(0);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
    });
    it.each(['binding', 'encoding', 'submit', 'map', 'range'])('cleans %s failure', async fail => {
      const buffers = await fixture(fail); await expect(invoke()).rejects.toThrow(fail);
      expect(buffers).toHaveLength(count);
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
      expect(buffers.at(-1)!.unmap).toHaveBeenCalledTimes(fail === 'range' ? 1 : 0);
    });
    it('copies results to CPU and releases all GPU buffers on success', async () => {
      const buffers = await fixture(); expect(await invoke()).toEqual(new Float32Array(108));
      for (const b of buffers) expect(b.destroy).toHaveBeenCalledTimes(1);
      expect(buffers.at(-1)!.unmap).toHaveBeenCalledTimes(1);
    });
  });
}
