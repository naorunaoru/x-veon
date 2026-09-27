import { afterEach, describe, expect, it, vi } from 'vitest';
import { ExportTarget, readTextureRgba } from './readback';
function fixture(fail?: string) {
  vi.stubGlobal('GPUBufferUsage', { COPY_DST: 1, MAP_READ: 2 });
  vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 1, COPY_SRC: 2 });
  vi.stubGlobal('GPUMapMode', { READ: 1 });
  const data = new ArrayBuffer(512);
  const buffer = {
    mapState: 'unmapped', destroy: vi.fn(),
    mapAsync: vi.fn(async () => { if (fail === 'map') throw new Error('map'); buffer.mapState = 'mapped'; }),
    getMappedRange: vi.fn(() => { if (fail === 'range') throw new Error('range'); return data; }),
    unmap: vi.fn(() => { buffer.mapState = 'unmapped'; }),
  };
  const texture = { destroy: vi.fn(), createView: () => ({}) };
  const copy = vi.fn();
  const device = {
    createTexture: vi.fn(() => texture),
    createBuffer: vi.fn(() => { if (fail === 'allocate') throw new Error('allocate'); return buffer; }),
    createCommandEncoder: () => ({ copyTextureToBuffer: copy, finish: () => ({}) }),
    queue: { submit: vi.fn() },
  } as unknown as GPUDevice;
  return { device, texture, buffer, data, copy };
}
afterEach(() => vi.unstubAllGlobals());
describe('export readback', () => {
  it.each(['rgba32float', 'rgba16float'] as const)('unpacks padded %s rows without alpha', async format => {
    const f = fixture(); const target = new ExportTarget(f.device, format);
    target.ensure(1, 2);
    if (format === 'rgba32float') {
      const data = new Float32Array(f.data); data.set([1, 2, 3, 99]); data.set([4, 5, 6, 99], 64);
    } else {
      const data = new Uint16Array(f.data); data.set([0x3c00, 0x4000, 0x4200, 0]); data.set([0x4400, 0x4500, 0x4600, 0], 128);
    }
    expect(await target.readHwc()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    expect(f.buffer.unmap).toHaveBeenCalledTimes(1);
    target.dispose(); expect(f.buffer.destroy).toHaveBeenCalledTimes(1); expect(f.texture.destroy).toHaveBeenCalledTimes(1);
  });
  it('releases a texture if staging allocation fails', () => {
    const f = fixture('allocate'); const target = new ExportTarget(f.device, 'rgba32float');
    expect(() => target.ensure(1, 2)).toThrow('allocate');
    target.dispose(); expect(f.texture.destroy).toHaveBeenCalledTimes(1);
  });
  it('unmaps if reading mapped data fails', async () => {
    const f = fixture('range'); const target = new ExportTarget(f.device, 'rgba32float'); target.ensure(1, 2);
    await expect(target.readHwc()).rejects.toThrow('range'); expect(f.buffer.unmap).toHaveBeenCalledTimes(1);
  });
});
describe('scene readback', () => {
  it('removes row padding but retains alpha', async () => {
    const f = fixture(); const data = new Float32Array(f.data);
    data.set([1, 2, 3, 4]); data.set([5, 6, 7, 8], 64);
    expect(await readTextureRgba(f.device, f.texture as unknown as GPUTexture, 1, 2)).toEqual(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    expect(f.buffer.destroy).toHaveBeenCalledTimes(1); expect(f.buffer.unmap).toHaveBeenCalledTimes(1);
  });
  it.each(['map', 'range'])('cleans up on %s failure', async fail => {
    const f = fixture(fail);
    await expect(readTextureRgba(f.device, f.texture as unknown as GPUTexture, 1, 2)).rejects.toThrow(fail);
    expect(f.buffer.destroy).toHaveBeenCalledTimes(1); expect(f.buffer.unmap).toHaveBeenCalledTimes(fail === 'map' ? 0 : 1);
  });
});
