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
    features: new Set<string>(),
    createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createRenderPipeline: vi.fn(() => ({})), createComputePipeline: () => ({}), createSampler: () => ({}),
    createBindGroup: () => ({}), createBuffer: allocate, createTexture: allocate,
    createCommandEncoder: () => ({ copyBufferToTexture: copy, finish: () => ({}) }),
    queue: { submit: vi.fn() },
  };
  getDevice.mockResolvedValue(device);
  const context = { configure: vi.fn() };
  const canvas = { getContext: () => context } as unknown as HTMLCanvasElement;
  return { resources, canvas, copy, device, context };
}
afterEach(() => vi.unstubAllGlobals());
describe('renderer ownership', () => {
  it.each([0, 1, 2, 3, 4, 5, 6])('cleans partial construction at resource %i', async failAt => {
    const f = fixture(failAt);
    await expect(createRenderer(f.canvas)).rejects.toThrow('allocation');
    expect(f.resources).toHaveLength(failAt);
    for (const r of f.resources) expect(r.destroy).toHaveBeenCalledTimes(1);
  });
  it('samples the image texture it is given, without copying or ever destroying it', async () => {
    const f = fixture(); const renderer = await createRenderer(f.canvas);
    const texture = (): GPUTexture => ({ destroy: vi.fn(), createView: () => ({}) }) as unknown as GPUTexture;
    const first = texture(), second = texture();
    renderer.setImage({ texture: first, width: 1, height: 1 });
    renderer.setImage({ texture: second, width: 2, height: 2 });
    renderer.dispose();
    expect(first.destroy).not.toHaveBeenCalled();
    expect(second.destroy).not.toHaveBeenCalled();
    expect(f.copy).not.toHaveBeenCalled();
    for (const r of f.resources) expect(r.destroy).toHaveBeenCalledTimes(1);
  });
  it('switches between SDR and HDR in place, rebuilding only the display pipeline', async () => {
    const f = fixture(); const renderer = await createRenderer(f.canvas);
    const pipelines = f.device.createRenderPipeline.mock.calls.length;
    const resources = f.resources.length;
    renderer.setDisplay({ hdr: true, headroom: 4 });
    expect(renderer.display).toEqual({ hdr: true, headroom: 4 });
    expect(f.context.configure).toHaveBeenLastCalledWith(expect.objectContaining({ format: 'rgba16float', toneMapping: { mode: 'extended' } }));
    expect(f.device.createRenderPipeline.mock.calls.length).toBe(pipelines + 1);
    renderer.setDisplay({ hdr: true, headroom: 2 });  // headroom only: no reconfiguration
    expect(f.context.configure).toHaveBeenCalledTimes(2);
    renderer.setDisplay({ hdr: false, headroom: 1 });
    expect(f.context.configure).toHaveBeenLastCalledWith(expect.objectContaining({ format: 'bgra8unorm', toneMapping: { mode: 'standard' } }));
    expect(f.resources.length).toBe(resources);
  });
  it('exports through rgba32float even without float32-blendable', async () => {
    const f = fixture(); await createRenderer(f.canvas);
    const formats = f.device.createRenderPipeline.mock.calls.map((c: unknown[]) =>
      [...((c[0] as GPURenderPipelineDescriptor).fragment?.targets ?? [])][0]?.format);
    expect(formats).toContain('rgba32float');
    expect(formats).not.toContain('rgba16float');
  });
});
