import { afterEach, describe, expect, it, vi } from 'vitest';
const { getDevice } = vi.hoisted(() => ({ getDevice: vi.fn() }));
vi.mock('@/gpu/device', () => ({ getDevice }));
import { createRenderer } from './index';
function fixture(failAt?: number) {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, COPY_SRC: 4, STORAGE: 8 });
  vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4, COPY_DST: 8 });
  vi.stubGlobal('GPUShaderStage', { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 });
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  const resources: { destroy: ReturnType<typeof vi.fn>; createView: () => object }[] = [];
  const allocate = () => {
    if (resources.length === failAt) throw new Error('allocation');
    const resource = { destroy: vi.fn(), createView: () => ({}) }; resources.push(resource); return resource;
  };
  const copy = vi.fn();
  const device = {
    features: new Set(['float32-blendable']),
    createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createRenderPipeline: () => ({}), createComputePipeline: () => ({}), createSampler: () => ({}),
    createBindGroup: () => ({}), createBuffer: allocate, createTexture: allocate,
    createCommandEncoder: () => ({ copyBufferToTexture: copy, finish: () => ({}) }),
    queue: { submit: vi.fn() },
  };
  getDevice.mockResolvedValue(device);
  const canvas = { getContext: () => ({ configure: vi.fn() }) } as unknown as HTMLCanvasElement;
  return { resources, canvas, copy };
}
afterEach(() => vi.unstubAllGlobals());
describe('renderer ownership', () => {
  it.each([0, 1, 2, 3, 4, 5, 6])('cleans partial construction at resource %i', async failAt => {
    const f = fixture(failAt);
    await expect(createRenderer(f.canvas)).rejects.toThrow('allocation');
    expect(f.resources).toHaveLength(failAt);
    for (const r of f.resources) expect(r.destroy).toHaveBeenCalledTimes(1);
  });
  it('borrows image buffers on both successful and failed uploads', async () => {
    const f = fixture(); const renderer = await createRenderer(f.canvas);
    const buffer = { destroy: vi.fn() } as unknown as GPUBuffer;
    const image = { buffer, width: 1, height: 1, bytesPerRow: 256 };
    renderer.setImage(image);
    expect(buffer.destroy).not.toHaveBeenCalled();
    f.copy.mockImplementation(() => { throw new Error('copy'); });
    expect(() => renderer.setImage(image)).toThrow('copy');
    expect(buffer.destroy).not.toHaveBeenCalled();
    renderer.dispose();
    for (const r of f.resources) expect(r.destroy).toHaveBeenCalledTimes(1);
  });
});
