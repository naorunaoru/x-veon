import { afterEach, describe, expect, it, vi } from 'vitest';
const { getDevice, readHwc } = vi.hoisted(() => ({ getDevice: vi.fn(), readHwc: vi.fn() }));
vi.mock('@/gpu/device', () => ({ getDevice }));
vi.mock('./histogram', () => ({ Histogram: class {
  isDisplayMode = false;
  setImage() {} encodeScenePass() {} renderVizToTargets() {} dispose() {}
} }));
vi.mock('./readback', () => ({ ExportTarget: class {
  ensure() { return {}; } encodeCopy() {} readHwc = readHwc; dispose() {}
} }));
import { createRenderer } from './index';
import { configFromPreset, configWithOverrides, computeTonescaleParams } from './grading/opendrt-params';
import { U_CWP_C0, U_FLAGS, U_ODRT_RS, U_PREPROCESS } from './uniforms';

function fixture() {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2, COPY_SRC: 4 });
  vi.stubGlobal('GPUShaderStage', { VERTEX: 1, FRAGMENT: 2 });
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  const resource = () => ({ destroy() {}, createView: () => ({}) });
  const writes: Float32Array[] = [];
  const frames = new Map<number, FrameRequestCallback>();
  let nextFrame = 0;
  vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
    frames.set(++nextFrame, callback);
    return nextFrame;
  }));
  vi.stubGlobal('cancelAnimationFrame', vi.fn((id: number) => frames.delete(id)));
  const flush = () => {
    const callbacks = [...frames.values()];
    frames.clear();
    for (const callback of callbacks) callback(0);
  };
  const draw = vi.fn();
  const createRenderPipeline = vi.fn(resource);
  getDevice.mockResolvedValue({
    features: new Set(), limits: { maxTextureDimension2D: 8192 }, createShaderModule: resource, createBindGroupLayout: resource,
    createPipelineLayout: resource, createRenderPipeline, createSampler: resource,
    createBuffer: resource, createTexture: resource, createBindGroup: resource,
    queue: { submit() {}, writeBuffer: (_b: unknown, _o: number, data: Float32Array) => writes.push(data.slice()) },
    createCommandEncoder: () => ({
      copyBufferToTexture() {}, finish: resource,
      beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw, end() {} }),
    }),
  });
  const canvas = { getContext: () => ({ configure() {}, getCurrentTexture: resource }) } as unknown as HTMLCanvasElement;
  readHwc.mockResolvedValue(new Float32Array(3));
  return { canvas, writes, flush, frames, draw, createRenderPipeline };
}
afterEach(() => vi.unstubAllGlobals());

describe('export gamut is independent of preview', () => {
  it.each(['rec709', 'rec2020'] as const)('packs identical %s exports and restores each preview', async gamut => {
    const exports: Float32Array[] = [];
    const previews: Float32Array[] = [];
    for (const hdr of [false, true]) {
      const { canvas, writes, flush } = fixture();
      const renderer = await createRenderer(canvas, { hdr, headroom: hdr ? 4 : 1 });
      renderer.setImage({ texture: { createView: () => ({}) } as unknown as GPUTexture, width: 1, height: 1 });
      const cfg = configWithOverrides(configFromPreset('umbra'), {});
      const ts = computeTonescaleParams(cfg);
      renderer.setGrade(cfg, ts);
      await renderer.readback(cfg, ts, gamut);
      flush();
      exports.push(writes[0]);
      previews.push(writes[1]);
      expect(writes[0][U_FLAGS + 2]).toBe(1);
      expect(writes[1][U_FLAGS + 2]).toBe(0);
      expect(writes[1][U_FLAGS + 1]).toBe(Number(hdr));
      expect(writes[1][U_ODRT_RS + 3]).toBe(0);
      renderer.dispose();
    }
    expect(exports[0]).toEqual(exports[1]);
    expect(exports[0][U_ODRT_RS + 3]).toBe(Number(gamut === 'rec2020'));
    expect(previews[0].slice(U_CWP_C0)).not.toEqual(previews[1].slice(U_CWP_C0));
  });
});

async function previewFixture() {
  const f = fixture();
  const renderer = await createRenderer(f.canvas);
  renderer.setImage({ texture: { createView: () => ({}) } as unknown as GPUTexture, width: 6000, height: 4000 });
  const cfg = configWithOverrides(configFromPreset('default'), {});
  renderer.setGrade(cfg, computeTonescaleParams(cfg));
  return { ...f, renderer, cfg };
}

it('coalesces rapid grade, viewport and scope requests into one draw with the latest state', async () => {
  const { renderer, canvas, cfg, flush, frames, writes, draw } = await previewFixture();
  for (let exposure = 0; exposure < 10; exposure++) {
    const grade = { ...cfg, exposure };
    renderer.setGrade(grade, computeTonescaleParams(grade));
    renderer.setViewport({ width: 800, height: 600, dpr: 2, scale: 0.1, offsetX: exposure, offsetY: 0, orientation: 'Normal' });
    renderer.requestRender();
  }
  expect(frames.size).toBe(1);
  expect(draw).not.toHaveBeenCalled();
  flush();
  expect(draw).toHaveBeenCalledExactlyOnceWith(6);
  expect(writes).toHaveLength(1);
  expect(writes[0][U_PREPROCESS]).toBe(9);
  expect([canvas.width, canvas.height]).toEqual([1600, 1200]);
  renderer.requestRender();
  flush();
  expect(draw).toHaveBeenCalledTimes(2);
  renderer.dispose();
});

it('cancels a pending frame on disposal and ignores later requests', async () => {
  const { renderer, flush, frames, draw } = await previewFixture();
  renderer.requestRender();
  renderer.dispose();
  expect(frames.size).toBe(0);
  renderer.requestRender();
  flush();
  expect(draw).not.toHaveBeenCalled();
});

it('uses display uniforms while export readback is pending', async () => {
  const { renderer, cfg, flush, writes } = await previewFixture();
  let finish!: (pixels: Float32Array) => void;
  readHwc.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  const exporting = renderer.readback({ ...cfg, exposure: 3 }, computeTonescaleParams(cfg), 'rec2020');
  flush();
  expect(writes[0][U_FLAGS + 2]).toBe(1);
  expect(writes[1][U_FLAGS + 2]).toBe(0);
  expect(writes[1][U_PREPROCESS]).toBe(cfg.exposure);
  finish(new Float32Array(3));
  await exporting;
  renderer.dispose();
});
