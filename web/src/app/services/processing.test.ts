import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import type { ProcessedImage } from '@/pipeline';

const processRaw = vi.fn();
vi.mock('@/pipeline', () => ({ processRaw: (...args: unknown[]) => processRaw(...args) }));
const readRaw = vi.fn();
vi.mock('@/app/storage/opfs-storage', () => ({ readRaw: (...args: unknown[]) => readRaw(...args) }));
// The monolithic store still persists and matches lenses inside its actions (until Tasks 8–9).
vi.mock('@/app/storage/idb-storage', () => ({
  putFile: vi.fn().mockResolvedValue(undefined), putSetting: vi.fn().mockResolvedValue(undefined),
  debouncedPutFile: vi.fn(), deleteFile: vi.fn().mockResolvedValue(undefined),
}));
vi.mock('@/app/services/library', () => ({ matchLensFor: vi.fn() }));

import {
  processFile, setPipeline, takeResult, getResult, discardResult, isProcessing,
} from './processing';

function makeFile(id: string, file: File | null = null): QueuedFile {
  return {
    id, file, name: id, originalName: `${id}.raf`, thumbnailUrl: null,
    metadata: null, cfaType: 'xtrans', status: 'queued', error: null, progress: null,
    result: null, resultMethod: null, lensProfile: null, lookPreset: 'default',
    openDrtOverrides: {}, preProcessOverrides: {},
  };
}

function fakeImage(): ProcessedImage & { disposed: number } {
  const image = {
    disposed: 0,
    gpu: { buffer: {} as GPUBuffer, width: 4, height: 2, bytesPerRow: 256 },
    meta: {
      exportData: { width: 4, height: 2, xyzToCam: null, wbCoeffs: new Float32Array(3), camToXyz: new Float32Array(12), orientation: 'Normal' },
      metadata: { make: 'F', model: 'X', width: 4, height: 2, tileCount: 1, inferenceTime: 0, backend: 'webgpu', exposureBias: 0, lensModel: 'L', focalLength: 0, fNumber: 0, colorTemp: 0, tint: 0 },
    },
    dispose() { image.disposed += 1; },
  };
  return image;
}

const ctx = { device: {} as GPUDevice, models: { backend: 'webgpu' } } as never;

describe('processing service', () => {
  beforeEach(() => {
    discardResult('a');
    discardResult('b');
    processRaw.mockReset();
    readRaw.mockReset();
    readRaw.mockResolvedValue(new ArrayBuffer(3));
    setPipeline(ctx);
    useAppStore.setState({ files: [makeFile('a')], selectedFileId: 'a', demosaicMethod: 'dht', modelSize: 'S', processingFileId: null });
  });

  afterEach(() => { discardResult('a'); discardResult('b'); });

  it('walks queued → processing → done and hands the result over once', async () => {
    const image = fakeImage();
    processRaw.mockResolvedValue(image);
    const seen: string[] = [];
    const unsubscribe = useAppStore.subscribe((s) => { seen.push(`${s.files[0].status}/${s.processingFileId}`); });

    await processFile('a');
    unsubscribe();

    expect(seen[0]).toBe('queued/a');
    expect(seen).toContain('processing/a');
    expect(seen.at(-1)).toBe('done/null');
    expect(readRaw).toHaveBeenCalledWith('a');
    expect(processRaw).toHaveBeenCalledWith(expect.any(ArrayBuffer), { method: 'dht', modelSize: 'S' }, ctx);
    const file = useAppStore.getState().files[0];
    expect(file.result).toBe(image.meta);
    expect(file.resultMethod).toBe('dht');
    expect(getResult('a')).toBe(image);
    expect(takeResult('a')).toBe(image);
    expect(takeResult('a')).toBeNull();
    expect(image.disposed).toBe(0);
  });

  it('reads the File object for fresh drops', async () => {
    const file = new File(['raw'], 'a.raf');
    Object.defineProperty(file, 'arrayBuffer', { value: () => Promise.resolve(new ArrayBuffer(7)) });
    useAppStore.setState({ files: [makeFile('a', file)] });
    processRaw.mockResolvedValue(fakeImage());
    await processFile('a');
    expect(readRaw).not.toHaveBeenCalled();
    expect((processRaw.mock.calls[0][0] as ArrayBuffer).byteLength).toBe(7);
  });

  it('reports a RAW missing from storage as an error', async () => {
    readRaw.mockResolvedValue(null);
    await processFile('a');
    const file = useAppStore.getState().files[0];
    expect(file.status).toBe('error');
    expect(file.error).toMatch(/re-add this file/);
    expect(processRaw).not.toHaveBeenCalled();
  });

  it('propagates pipeline errors verbatim and leaves no result behind', async () => {
    processRaw.mockRejectedValue(new Error("Couldn't decode this RAW file. X"));
    await processFile('a');
    const file = useAppStore.getState().files[0];
    expect(file.status).toBe('error');
    expect(file.error).toBe("Couldn't decode this RAW file. X");
    expect(getResult('a')).toBeNull();
    expect(useAppStore.getState().processingFileId).toBeNull();
  });

  it('disposes a replaced result and a discarded one', async () => {
    const first = fakeImage();
    const second = fakeImage();
    processRaw.mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    await processFile('a');
    await processFile('a');
    expect(first.disposed).toBe(1);
    expect(second.disposed).toBe(0);
    discardResult('a');
    expect(second.disposed).toBe(1);
    expect(getResult('a')).toBeNull();
  });

  it('ignores a second call while a run is in flight', async () => {
    let release!: (image: ProcessedImage) => void;
    processRaw.mockImplementation(() => new Promise((resolve) => { release = resolve; }));
    const running = processFile('a');
    expect(isProcessing()).toBe(true);
    await processFile('a');           // returns immediately: locked
    expect(processRaw).toHaveBeenCalledTimes(1);
    release(fakeImage());
    await running;
    expect(isProcessing()).toBe(false);
  });
  it('disposes a removed in-flight result while holding the lock until completion', async () => {
    let release!: (image: ProcessedImage) => void;
    processRaw.mockImplementation(() => new Promise(resolve => { release = resolve; }));
    const running = processFile('a');
    await vi.waitFor(() => expect(processRaw).toHaveBeenCalledTimes(1));
    discardResult('a');
    useAppStore.setState({ files: [makeFile('b')] });
    await processFile('b');
    expect(isProcessing()).toBe(true);
    expect(processRaw).toHaveBeenCalledTimes(1);
    const image = fakeImage(); release(image); await running;
    expect(image.disposed).toBe(1);
    expect(getResult('a')).toBeNull();
    expect(useAppStore.getState().files[0].status).toBe('queued');
    expect(isProcessing()).toBe(false);
  });
  it('evicts an unclaimed result when a different file starts', async () => {
    useAppStore.setState({ files: [makeFile('a'), makeFile('b')] });
    const a = fakeImage(), b = fakeImage();
    processRaw.mockResolvedValueOnce(a).mockResolvedValueOnce(b);
    await processFile('a'); await processFile('b');
    expect(a.disposed).toBe(1); expect(b.disposed).toBe(0);
    expect(getResult('a')).toBeNull(); expect(getResult('b')).toBe(b);
  });
  it('clears the previous image if a retry fails', async () => {
    const image = fakeImage();
    processRaw.mockResolvedValueOnce(image).mockRejectedValueOnce(new Error('retry'));
    await processFile('a'); await processFile('a');
    expect(image.disposed).toBe(1); expect(getResult('a')).toBeNull();
    expect(useAppStore.getState().files[0].status).toBe('error');
  });

});
