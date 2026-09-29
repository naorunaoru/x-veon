import { beforeEach, describe, expect, it, vi } from 'vitest';
const m = vi.hoisted(() => ({
  decodeRaw: vi.fn(),
  run: vi.fn(),
  gpuPostprocess: vi.fn(),
  buildColorMatrix: vi.fn(),
  estimateColorTemperature: vi.fn(),
  destroyDemosaicPool: vi.fn(),
  initWasm: vi.fn(),
  getDevice: vi.fn(),
  setSharedDevice: vi.fn(),
  models: {
    activate: vi.fn(),
    init: vi.fn(),
    device: null as GPUDevice | null,
    backend: 'webgpu',
    size: 'S',
  },
}));
vi.mock('./decode/raf-decoder', () => ({ decodeRaw: m.decodeRaw, initWasm: m.initWasm }));
vi.mock('./demosaic', () => ({
  strategyFor: () => ({ run: m.run }),
  destroyDemosaicPool: m.destroyDemosaicPool,
}));
vi.mock('./inference', () => ({ models: m.models }));
vi.mock('./postprocess/postprocess-gpu', () => ({ gpuPostprocess: m.gpuPostprocess }));
vi.mock('./postprocess/postprocessor', () => ({ buildColorMatrix: m.buildColorMatrix }));
vi.mock('./color-temperature', () => ({ estimateColorTemperature: m.estimateColorTemperature }));
vi.mock('@/gpu/device', () => ({ getDevice: m.getDevice, setSharedDevice: m.setSharedDevice }));
import { initPipeline, processRaw, type PipelineContext } from './index';
import { prepareCfa } from './preprocess/preprocessor';
import type { RawImage } from './types';
const ctx = {
  device: { queue: { onSubmittedWorkDone: async () => {} } },
  models: m.models,
} as unknown as PipelineContext;
const options = { method: 'neural-net' as const, modelSize: 'S' as const };
let input: GPUBuffer;
let output: GPUTexture;
beforeEach(() => {
  vi.resetAllMocks();
  m.models.activate.mockResolvedValue({ model: { size: 'S', sha256: 'test' }, note: null });
  input = { destroy: vi.fn() } as unknown as GPUBuffer;
  output = { destroy: vi.fn() } as unknown as GPUTexture;
  const { data, ...raw } = {
    data: new Uint16Array(36).fill(100),
    width: 6,
    height: 6,
    crops: new Uint16Array(4),
    whiteLevels: new Uint16Array([1000]),
    blackLevels: new Uint16Array(4),
    wbCoeffs: new Float32Array([2, 1, 1.5]),
    cfaStr: 'RGGB',
    cfaWidth: 2,
    xyzToCam: new Float32Array(9),
    camToXyz: new Float32Array(9),
    orientation: 'Rotate90',
    drGain: 1,
    make: 'Test',
    model: 'Camera',
  };
  m.decodeRaw.mockResolvedValue({
    raw,
    prepared: prepareCfa({ data, ...raw } as unknown as RawImage),
    prepareError: null,
  });
  m.run.mockResolvedValue({ rgb: { buffer: input, stride: 6, offsetX: 0, offsetY: 0 }, tileCount: 1 });
  m.gpuPostprocess.mockImplementation(async () => {
    input.destroy();
    return { texture: output };
  });
  m.estimateColorTemperature.mockReturnValue({ temp: 6500, tint: 0 });
});
describe('processRaw ownership', () => {
  it('labels a neural result with the size of the loaded models, not the requested size', async () => {
    const image = await processRaw(new ArrayBuffer(0), { ...options, modelSize: 'M' }, ctx);
    expect(image.meta.metadata.modelSize).toBe('S');
    image.dispose();
  });
  it('transfers the final image to its caller', async () => {
    const image = await processRaw(new ArrayBuffer(0), options, ctx);
    expect(image.gpu).toEqual({ texture: output, width: 6, height: 6 });
    // The worker-prepared u16 CFA and its table go to the strategy as they are.
    expect(m.run.mock.calls[0][0]).toMatchObject({
      width: 6,
      height: 6,
      padTop: 0,
      padLeft: 0,
      cfaType: 'bayer',
    });
    expect(m.run.mock.calls[0][0].cfa).toBeInstanceOf(Uint16Array);
    expect(image.meta.metadata).toMatchObject({
      colorTemp: 6500,
      tileCount: 1,
      backend: 'webgpu',
      modelSize: 'S',
    });
    expect(input.destroy).toHaveBeenCalledTimes(1);
    expect(output.destroy).not.toHaveBeenCalled();

    image.dispose();
    expect(output.destroy).toHaveBeenCalledTimes(1);
    expect(m.destroyDemosaicPool).toHaveBeenCalledTimes(1);
  });
  it('releases strategy output on failure before transfer', async () => {
    m.buildColorMatrix.mockImplementation(() => {
      throw new Error('matrix');
    });
    await expect(processRaw(new ArrayBuffer(0), options, ctx)).rejects.toThrow('matrix');
    expect(input.destroy).toHaveBeenCalledTimes(1);
    expect(m.gpuPostprocess).not.toHaveBeenCalled();
  });
  it('does not double release consumed input when postprocessing fails', async () => {
    m.gpuPostprocess.mockImplementation(async () => {
      input.destroy();
      throw new Error('postprocess');
    });
    await expect(processRaw(new ArrayBuffer(0), options, ctx)).rejects.toThrow('postprocess');
    expect(input.destroy).toHaveBeenCalledTimes(1);
    expect(m.destroyDemosaicPool).toHaveBeenCalledTimes(1);
  });
  it('releases final output if metadata construction fails', async () => {
    m.estimateColorTemperature.mockImplementation(() => {
      throw new Error('metadata');
    });
    await expect(processRaw(new ArrayBuffer(0), options, ctx)).rejects.toThrow('metadata');
    expect(output.destroy).toHaveBeenCalledTimes(1);
    expect(input.destroy).toHaveBeenCalledTimes(1);
  });
  it('preserves decoder errors and closes the worker pool', async () => {
    m.decodeRaw.mockRejectedValue(new Error(' unsupported '));
    await expect(processRaw(new ArrayBuffer(0), options, ctx)).rejects.toThrow(
      "Couldn't decode this RAW file. The camera or format may not be supported by this build of the decoder — unsupported.",
    );
    expect(m.run).not.toHaveBeenCalled();
    expect(m.destroyDemosaicPool).toHaveBeenCalledTimes(1);
  });
  it('reports a CFA that decoded but could not be prepared, without the decoder wording', async () => {
    const { raw } = await m.decodeRaw();
    m.decodeRaw.mockResolvedValue({ raw, prepared: null, prepareError: 'Unsupported CFA: 9 chars, width 3' });
    await expect(processRaw(new ArrayBuffer(0), options, ctx)).rejects.toThrow(
      /^Unsupported CFA: 9 chars, width 3$/,
    );
    expect(m.run).not.toHaveBeenCalled();
  });
});
it('initializes the engines and shares their device', async () => {
  m.models.device = {} as GPUDevice;
  m.getDevice.mockResolvedValue(m.models.device);
  const result = await initPipeline({ modelSize: 'S' });
  expect(m.initWasm).toHaveBeenCalledTimes(1);
  expect(m.models.init).toHaveBeenCalledWith('S');
  expect(m.setSharedDevice).toHaveBeenCalledWith(m.models.device);
  expect(result).toEqual({ device: m.models.device, models: m.models });
});

it('records demosaic time after GPU completion and before postprocessing', async () => {
  let finish!: () => void;
  const gpuDone = new Promise<void>(resolve => { finish = resolve; });
  const timingCtx = { ...ctx, device: { queue: { onSubmittedWorkDone: () => gpuDone } } } as unknown as PipelineContext;
  const timings: { demosaicMs?: number } = {};
  const pending = processRaw(new ArrayBuffer(0), { ...options, timings }, timingCtx);
  await vi.waitFor(() => expect(m.run).toHaveBeenCalled());
  expect(timings.demosaicMs).toBeUndefined();
  m.gpuPostprocess.mockImplementation(async () => {
    expect(timings.demosaicMs).toEqual(expect.any(Number));
    input.destroy();
    return { texture: output };
  });
  finish();
  const result = await pending;
  expect(timings.demosaicMs).toBeGreaterThanOrEqual(0);
  result.dispose();
});
