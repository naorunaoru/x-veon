import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { CfaType, ModelSize } from '@/lib/types';

// createGpuNNPipeline is mocked so these tests exercise neuralNetStrategy's own
// resource-lifecycle logic (dispose/destroy ordering on every failure path) without
// needing the real GPU pipeline — that's tile-blend-gpu.test.ts's job.
const { extractBatch, accumulateBatch, finalize, destroy, createGpuNNPipeline } = vi.hoisted(() => {
  const extractBatch = vi.fn();
  const accumulateBatch = vi.fn();
  const finalize = vi.fn();
  const destroy = vi.fn();
  const createGpuNNPipeline = vi.fn(() => ({ extractBatch, accumulateBatch, finalize, destroy }));
  return { extractBatch, accumulateBatch, finalize, destroy, createGpuNNPipeline };
});

vi.mock('./tile-blend-gpu', () => ({ createGpuNNPipeline }));
const cfaBuf = vi.hoisted(() => ({}) as GPUBuffer);
vi.mock('./cfa-gpu', () => ({ uploadNormalizedCfa: () => cfaBuf }));

import { neuralNetStrategy } from './nn';
import type { DemosaicInput } from './strategy';
import type { PipelineContext, ProcessOptions } from '../context';

/** width=height=1 (< PATCH_SIZE) yields exactly one tile — keeps the batch loop to one pass. */
function makeInput(): DemosaicInput {
  return {
    cfa: new Uint16Array(1),
    lut: new Float32Array(3 * 65536),
    width: 1,
    height: 1,
    padTop: 0,
    padLeft: 0,
    visibleWidth: 1,
    visibleHeight: 1,
    pattern: new Uint32Array([0, 1, 1, 2]),
    period: 2,
    cfaType: 'bayer' as CfaType,
    masks: { r: new Float32Array(1), g: new Float32Array(1), b: new Float32Array(1) },
    clipNorm: [0.987, 0.987, 0.987],
  };
}

function makeCtx(runBatchGpu: PipelineContext['models']['runBatchGpu']): PipelineContext {
  return {
    device: {} as GPUDevice,
    models: {
      init: vi.fn(),
      switchSize: vi.fn(),
      availableSizes: vi.fn(() => new Set<ModelSize>()),
      size: 'S',
      runBatchGpu,
      backend: null,
      device: null,
    },
  };
}

function makeOpts(onProgress?: ProcessOptions['onProgress']): ProcessOptions {
  return { method: 'neural-net', modelSize: 'S', onProgress };
}

describe('neuralNetStrategy resource lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    extractBatch.mockReturnValue({} as GPUBuffer);
    accumulateBatch.mockImplementation(() => {});
  });

  it('extracts, infers, accumulates, and finalizes to produce the output and tile count', async () => {
    const inputBuf = {} as GPUBuffer;
    const inferBuf = {} as GPUBuffer;
    const output = { buffer: {} as GPUBuffer, stride: 1, offsetX: 0, offsetY: 0 };
    const dispose = vi.fn();
    extractBatch.mockReturnValue(inputBuf);
    const runBatchGpu = vi.fn().mockResolvedValue({ buffer: inferBuf, dispose });
    finalize.mockResolvedValue(output);
    const onProgress = vi.fn();

    const result = await neuralNetStrategy.run(makeInput(), makeCtx(runBatchGpu), makeOpts(onProgress));

    expect(result).toEqual({ rgb: output, tileCount: 1 });
    expect(createGpuNNPipeline.mock.calls[0]).toContain(cfaBuf);
    expect(extractBatch).toHaveBeenCalledWith(0, 1);
    expect(runBatchGpu).toHaveBeenCalledWith('bayer', inputBuf, 1, 288);
    expect(accumulateBatch).toHaveBeenCalledWith(inferBuf, 0, 1);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledWith(1, 1);
    expect(destroy).not.toHaveBeenCalled();
  });

  it('destroys the pipeline and propagates the error when inference rejects', async () => {
    const err = new Error('inference failed');
    const runBatchGpu = vi.fn().mockRejectedValue(err);

    await expect(neuralNetStrategy.run(makeInput(), makeCtx(runBatchGpu), makeOpts())).rejects.toBe(err);

    expect(accumulateBatch).not.toHaveBeenCalled();
    expect(finalize).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  it('still disposes the inference output when accumulateBatch throws', async () => {
    const err = new Error('accumulate failed');
    const dispose = vi.fn();
    const runBatchGpu = vi.fn().mockResolvedValue({ buffer: {} as GPUBuffer, dispose });
    accumulateBatch.mockImplementation(() => { throw err; });

    await expect(neuralNetStrategy.run(makeInput(), makeCtx(runBatchGpu), makeOpts())).rejects.toBe(err);

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(finalize).not.toHaveBeenCalled();
  });

  it('destroys the pipeline when the progress callback throws', async () => {
    const err = new Error('progress failed');
    const dispose = vi.fn();
    const runBatchGpu = vi.fn().mockResolvedValue({ buffer: {} as GPUBuffer, dispose });
    const onProgress = vi.fn(() => { throw err; });

    await expect(neuralNetStrategy.run(makeInput(), makeCtx(runBatchGpu), makeOpts(onProgress))).rejects.toBe(err);

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(finalize).not.toHaveBeenCalled();
  });

  it('destroys the pipeline when finalize rejects', async () => {
    const err = new Error('finalize failed');
    const dispose = vi.fn();
    const runBatchGpu = vi.fn().mockResolvedValue({ buffer: {} as GPUBuffer, dispose });
    finalize.mockRejectedValue(err);

    await expect(neuralNetStrategy.run(makeInput(), makeCtx(runBatchGpu), makeOpts())).rejects.toBe(err);

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});
