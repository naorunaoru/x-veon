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
import { U_CWP_C0, U_FLAGS, U_ODRT_RS } from './uniforms';

function fixture() {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2, COPY_SRC: 4 });
  vi.stubGlobal('GPUShaderStage', { VERTEX: 1, FRAGMENT: 2 });
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  const resource = () => ({ destroy() {}, createView: () => ({}) });
  const writes: Float32Array[] = [];
  getDevice.mockResolvedValue({
    features: new Set(), createShaderModule: resource, createBindGroupLayout: resource,
    createPipelineLayout: resource, createRenderPipeline: resource, createSampler: resource,
    createBuffer: resource, createTexture: resource, createBindGroup: resource,
    queue: { submit() {}, writeBuffer: (_b: unknown, _o: number, data: Float32Array) => writes.push(data.slice()) },
    createCommandEncoder: () => ({
      copyBufferToTexture() {}, finish: resource,
      beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }),
    }),
  });
  const canvas = { getContext: () => ({ configure() {}, getCurrentTexture: resource }) } as unknown as HTMLCanvasElement;
  readHwc.mockResolvedValue(new Float32Array(3));
  return { canvas, writes };
}
afterEach(() => vi.unstubAllGlobals());

describe('export gamut is independent of preview', () => {
  it.each(['rec709', 'rec2020'] as const)('packs identical %s exports and restores each preview', async gamut => {
    const exports: Float32Array[] = [];
    const previews: Float32Array[] = [];
    for (const hdr of [false, true]) {
      const { canvas, writes } = fixture();
      const renderer = await createRenderer(canvas, { hdr, headroom: hdr ? 4 : 1 });
      renderer.setImage({ texture: { createView: () => ({}) } as unknown as GPUTexture, width: 1, height: 1 });
      const cfg = configWithOverrides(configFromPreset('umbra'), {});
      const ts = computeTonescaleParams(cfg);
      renderer.setGrade(cfg, ts);
      await renderer.readback(cfg, ts, gamut);
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
