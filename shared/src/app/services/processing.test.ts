import { setHost } from './host';
import { fakeHost } from '@/test/fake-host';
import { factsOf, fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto, defaultEdit } from '@/test/fake-host';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAppStore } from '@/app/store';
import type { QueuedFile } from '@/app/store';
import type { ProcessedImage } from '@/pipeline';
import { startPersistence, flushPersistence, unsavedEdits, cancelPhotoSave } from './persistence';
import { switchFolder, startLibraryWatching } from './library';

const processRaw = vi.fn();
vi.mock('@/pipeline', () => ({ processRaw: (...args: unknown[]) => processRaw(...args) }));
const readRaw = vi.fn();
vi.mock('@/app/storage/opfs-storage', () => ({ readRaw: (...args: unknown[]) => readRaw(...args) }));
// The monolithic store still persists and matches lenses inside its actions (until Tasks 8–9).
vi.mock('@/app/storage/idb-storage', () => ({
  putFile: vi.fn().mockResolvedValue(undefined),
  putSetting: vi.fn().mockResolvedValue(undefined),
  debouncedPutFile: vi.fn(),
  deleteFile: vi.fn().mockResolvedValue(undefined),
}));
vi.mock('@/app/lens/lensfun', () => ({ matchLens: vi.fn(async () => null) }));
vi.mock('@/app/services/library', async (importOriginal) => ({
  ...await importOriginal<typeof import('./library')>(),
  matchLensFor: vi.fn(),
}));

import {
  processFile,
  setPipeline,
  acquireResult,
  getResult,
  discardResult,
  isProcessing,
  RETAINED_RESULTS,
} from './processing';

function makeFile(id: string, file: File | null = null): QueuedFile {
  return {
    ...fromLibraryPhoto(fakePhoto()),
    id,
    name: id,
    originalName: `${id}.raf`,
    thumbnailUrl: null,
    metadata: null,
    cfaType: 'xtrans',
    status: 'queued',
    error: null,
    progress: null,
    result: null,
    resultMethod: null,
    lensProfile: null,
    edit: { ...defaultEdit(), lookPreset: 'default', openDrtOverrides: {}, preProcessOverrides: {} },
  };
}

function fakeImage(): ProcessedImage & { disposed: number } {
  const image = {
    disposed: 0,
    get method(): import('@/lib/types').DemosaicMethod { return image.meta.metadata.modelIdentity ? 'neural-net' : 'dht'; },
    gpu: { texture: {} as GPUTexture, width: 4, height: 2 },
    meta: {
      exportData: {
        width: 4,
        height: 2,
        xyzToCam: null,
        wbCoeffs: new Float32Array(3),
        camToXyz: new Float32Array(12),
        orientation: 'Normal',
      },
      metadata: {
        modelIdentity: undefined as import('@/lib/types').ModelIdentity | undefined,
        modelNote: undefined as string | undefined,
        make: 'F',
        model: 'X',
        width: 4,
        height: 2,
        tileCount: 1,
        inferenceTime: 0,
        backend: 'webgpu',
        exposureBias: 0,
        lensModel: 'L',
        focalLength: 0,
        fNumber: 0,
        colorTemp: 0,
        tint: 0,
      },
    },
    dispose() {
      image.disposed += 1;
    },
  };
  return image;
}

const switchSize = vi.fn();
const ctx = { device: {} as GPUDevice, models: { backend: 'webgpu', switchSize } } as never;
let host: ReturnType<typeof fakeHost>;

describe('processing service', () => {
  beforeEach(() => {
    for (const id of ['a', 'b', 'c']) discardResult(id);
    processRaw.mockReset();
    host = fakeHost();
    host.library.readRaw = readRaw;
    setHost(host);
    readRaw.mockReset();
    readRaw.mockResolvedValue(new ArrayBuffer(3));
    setPipeline(ctx);
    useAppStore.setState({
      files: [makeFile('a')],
      selectedFileId: 'a',
      demosaicMethod: 'dht',
      modelSize: 'S',
      processingFileId: null,
    });
  });

  afterEach(() => {
    for (const id of ['a', 'b', 'c']) discardResult(id);
  });

  it.each(['focus', 'own-save', 'unrelated-add', 'in-flight'] as const)('retains live work and history during an unchanged %s replacement', async cause => {
    const original = fakePhoto('a');
    useAppStore.setState({ files: [fromLibraryPhoto(original)], folder: { id: 'A', name: 'A' } });
    const image = fakeImage();
    let finish!: (image: ProcessedImage) => void;
    processRaw.mockImplementationOnce(() => cause === 'in-flight' ? new Promise(resolve => { finish = resolve; }) : Promise.resolve(image));
    const running = processFile('a');
    await vi.waitFor(() => expect(processRaw).toHaveBeenCalled());
    if (cause !== 'in-flight') await running;
    const stopPersistence = startPersistence();
    let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
    host.library.onChange = listener => { publish = listener; return () => {}; };
    const stopWatching = startLibraryWatching();
    try {
      if (cause === 'own-save') {
        useAppStore.getState().setFileLookPreset('a', 'umbra');
        await flushPersistence();
        expect(unsavedEdits()).toEqual([]);
      }
      const before = useAppStore.getState().files[0];
      const incoming = cause === 'own-save' ? { ...original, edit: before.edit, facts: factsOf(before) } : original;
      publish({ kind: 'replace', snapshot: { folder: { id: 'A', name: 'A' }, complete: true, photos: [incoming, ...(cause === 'unrelated-add' ? [fakePhoto('b')] : [])] } });
      const after = useAppStore.getState().files[0];
      expect(after.processedKey).toBe(before.processedKey);
      expect(after.lookHistory).toBe(before.lookHistory);
      expect(after.status).toBe(before.status);
      if (cause === 'in-flight') { finish(image); await running; }
      expect(getResult('a')).toBe(image);
      expect(image.disposed).toBe(0);
      expect(useAppStore.getState().files[0].status).toBe('done');
    } finally { if (finish) finish(image); await running; stopWatching(); stopPersistence(); await cancelPhotoSave('a'); }
  });

  it.each(['raw', 'method', 'look', 'removed'] as const)('reloads a genuine external %s change and invalidates only affected processing', async change => {
    const original = { ...fakePhoto('a'), sourceVersion: 'a'.repeat(64) };
    useAppStore.setState({ files: [fromLibraryPhoto(original)], folder: { id: 'A', name: 'A' } });
    const image = fakeImage(); processRaw.mockResolvedValueOnce(image); await processFile('a');
    let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
    host.library.onChange = listener => { publish = listener; return () => {}; };
    const stop = startLibraryWatching();
    const changed = { ...original, ...(change === 'raw' ? { sourceVersion: 'b'.repeat(64) } : {}), edit: { ...original.edit, ...(change === 'method' ? { demosaicMethod: 'bilinear' as const } : {}), ...(change === 'look' ? { lookPreset: 'umbra' as const } : {}) } };
    publish({ kind: 'replace', snapshot: { folder: { id: 'A', name: 'A' }, complete: true, photos: change === 'removed' ? [] : [changed] } });
    expect(image.disposed).toBe(change === 'look' ? 0 : 1);
    if (change !== 'removed') expect(useAppStore.getState().files[0].edit).toEqual(changed.edit);
    stop();
  });

  it.each(['folder', 'watch'] as const)(
    'ignores a discarded processing rejection after a same-ID %s replacement',
    async (replacement) => {
      vi.useFakeTimers();
      const stop = startPersistence();
      let stopWatching = () => {};
      let reject!: (error: Error) => void;
      readRaw.mockImplementationOnce(() => new Promise<ArrayBuffer>((_resolve, fail) => { reject = fail; }));
      useAppStore.setState({ demosaicMethod: 'neural-net', folder: { id: 'A', name: 'A' } });
      const processing = processFile('a');
      useAppStore.getState().setFileLookPreset('a', 'umbra');
      const revision = useAppStore.getState().files[0].editRevision;
      const reopened = { photos: [{ ...fakePhoto('a'), fileSize: 99 }], folder: { id: 'A', name: 'A' }, complete: true };
      try {
        if (replacement === 'folder') {
          await switchFolder(async () => ({ photos: [], folder: { id: 'B', name: 'B' }, complete: true }));
          await switchFolder(async () => reopened);
        } else {
          let publish!: Parameters<NonNullable<typeof host.library.onChange>>[0];
          host.library.onChange = listener => { publish = listener; return () => {}; };
          stopWatching = startLibraryWatching();
          publish({ kind: 'replace', snapshot: reopened });
        }
        const accepted = useAppStore.getState().files[0];
        expect(accepted).toMatchObject({ status: 'queued', modelNeedsResolution: true, editRevision: revision });
        reject(new Error('old RAW read failed'));
        await processing;
        expect(useAppStore.getState().files[0]).toBe(accepted);
        expect(unsavedEdits()[0]).toMatchObject({ revision, deferred: true, error: null, facts: { status: 'queued', error: null } });
        expect(host.library.save).not.toHaveBeenCalled();
        const image = fakeImage();
        image.meta.metadata.modelIdentity = { size: 'S', sha256: 'reopened' };
        processRaw.mockResolvedValueOnce(image);
        await processFile('a');
        await vi.advanceTimersByTimeAsync(301);
        expect(host.library.save).toHaveBeenCalledTimes(1);
        expect(host.library.save).toHaveBeenCalledWith('a', expect.objectContaining({ model: { size: 'S', sha256: 'reopened' } }), expect.anything());
        expect(unsavedEdits()).toEqual([]);
      } finally {
        reject(new Error('test cleanup'));
        await processing;
        stopWatching();
        await Promise.all(unsavedEdits().map(entry => cancelPhotoSave(entry.id)));
        stop();
        vi.useRealTimers();
      }
    },
  );

  it('passes the photo model independently of defaults and preserves an unknown recorded identity', async () => {
    const photo = makeFile('a');
    photo.edit = { ...photo.edit, demosaicMethod: 'neural-net', model: { size: 'M', sha256: 'unknown' } };
    useAppStore.setState({ files: [photo], demosaicMethod: 'dht', modelSize: 'S' });
    const image = fakeImage();
    image.meta.metadata.modelIdentity = { size: 'M', sha256: 'used' };
    image.meta.metadata.modelNote = 'different model';
    processRaw.mockResolvedValue(image);
    await processFile('a');
    expect(processRaw).toHaveBeenCalledWith(
      expect.any(ArrayBuffer),
      { method: 'neural-net', modelSize: 'S', model: { size: 'M', sha256: 'unknown' } },
      ctx,
    );
    expect(useAppStore.getState().files[0].edit.model?.sha256).toBe('unknown');
    useAppStore.getState().setFileLookPreset('a', 'umbra');
    expect(useAppStore.getState().files[0].edit.model?.sha256).toBe('used');
  });

  it('leaves an untouched edit untouched after processing', async () => {
    const image = fakeImage();
    image.meta.metadata.modelIdentity = { size: 'S', sha256: 'used' };
    processRaw.mockResolvedValue(image);
    useAppStore.setState({ demosaicMethod: 'neural-net' });
    await processFile('a');
    const file = useAppStore.getState().files[0];
    expect(file.edit).toEqual(defaultEdit());
    expect(file.resultMethod).toBe('neural-net');
    expect(file.actualModel?.size).toBe('S');
  });

  it('runs a Bayer photo through the neural net when the X-Trans default is incompatible', async () => {
    const file = makeFile('a');
    file.cfaType = 'bayer';
    useAppStore.setState({ files: [file], demosaicMethod: 'markesteijn3' });
    const image = fakeImage();
    image.meta.metadata.modelIdentity = { size: 'S', sha256: 'used' };
    processRaw.mockResolvedValue(image);
    await processFile('a');
    expect(processRaw).toHaveBeenCalledWith(
      expect.any(ArrayBuffer),
      { method: 'neural-net', modelSize: 'S', model: null, resolveDefault: true },
      ctx,
    );
    expect(useAppStore.getState().files[0].edit).toEqual(defaultEdit());
    expect(useAppStore.getState().files[0].resultMethod).toBe('neural-net');
  });

  it('saves processing facts while leaving an untouched edit null', async () => {
    vi.useFakeTimers();
    const stop = startPersistence();
    try {
      const image = fakeImage();
      image.meta.metadata.modelIdentity = { size: 'S', sha256: 'used' };
      processRaw.mockResolvedValue(image);
      useAppStore.setState({ demosaicMethod: 'neural-net' });
      await processFile('a');
      await vi.advanceTimersByTimeAsync(301);
      expect(host.library.save).not.toHaveBeenCalled();
      expect(host.library.saveFacts).toHaveBeenCalledTimes(1);
      expect(host.library.saveFacts).toHaveBeenCalledWith('a', expect.objectContaining({ resultMethod: 'neural-net' }));
    } finally {
      stop();
      vi.useRealTimers();
    }
  });

  it('saves a first edit once its deferred neural model resolves', async () => {
    vi.useFakeTimers();
    const stop = startPersistence();
    try {
      useAppStore.setState({ demosaicMethod: 'neural-net' });
      useAppStore.getState().setFileLookPreset('a', 'umbra');
      await vi.advanceTimersByTimeAsync(301);
      expect(host.library.save).not.toHaveBeenCalled();
      const image = fakeImage();
      image.meta.metadata.modelIdentity = { size: 'S', sha256: 'used' };
      processRaw.mockResolvedValue(image);
      await processFile('a');
      await vi.advanceTimersByTimeAsync(301);
      expect(host.library.save).toHaveBeenCalledTimes(1);
      expect(host.library.save).toHaveBeenCalledWith('a', expect.objectContaining({ model: { size: 'S', sha256: 'used' } }), expect.anything());
    } finally {
      stop();
      vi.useRealTimers();
    }
  });

  it('walks queued → processing → done and lends the result out without giving it away', async () => {
    const image = fakeImage();
    processRaw.mockResolvedValue(image);
    const seen: string[] = [];
    const unsubscribe = useAppStore.subscribe((s) => {
      seen.push(`${s.files[0].status}/${s.processingFileId}`);
    });

    await processFile('a');
    unsubscribe();

    expect(seen[0]).toBe('queued/a');
    expect(seen).toContain('processing/a');
    expect(seen.at(-1)).toBe('done/null');
    expect(readRaw).toHaveBeenCalledWith('a');
    expect(processRaw).toHaveBeenCalledWith(
      expect.any(ArrayBuffer),
      { method: 'dht', modelSize: 'S', model: null, resolveDefault: true },
      ctx,
    );
    const file = useAppStore.getState().files[0];
    expect(file.result).toBe(image.meta);
    expect(file.resultMethod).toBe('dht');
    expect(getResult('a')).toBe(image);
    const lease = acquireResult('a')!;
    expect(lease.image).toBe(image);
    lease.release();
    lease.release(); // idempotent
    expect(acquireResult('a')?.image).toBe(image); // still cached: showing it again needs no run
    expect(image.disposed).toBe(0);
  });

  it('always obtains RAW bytes through the host for fresh and restored photos', async () => {
    readRaw.mockResolvedValue(new ArrayBuffer(7));
    processRaw.mockResolvedValue(fakeImage());
    await processFile('a');
    expect(readRaw).toHaveBeenCalledWith('a');
    expect((processRaw.mock.calls[0][0] as ArrayBuffer).byteLength).toBe(7);
  });

  it('reports a RAW missing from storage as an error', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    readRaw.mockRejectedValue(new Error('RAW file not found in storage. Please re-add this file.'));
    await processFile('a');
    const file = useAppStore.getState().files[0];
    expect(file.status).toBe('error');
    expect(file.error).toMatch(/re-add this file/);
    expect(processRaw).not.toHaveBeenCalled();
    expect(logged).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ message: 'RAW file not found in storage. Please re-add this file.' }));
    logged.mockRestore();
  });

  it('propagates pipeline errors verbatim and leaves no result behind', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    processRaw.mockRejectedValue(new Error("Couldn't decode this RAW file. X"));
    await processFile('a');
    const file = useAppStore.getState().files[0];
    expect(file.status).toBe('error');
    expect(file.error).toBe("Couldn't decode this RAW file. X");
    expect(getResult('a')).toBeNull();
    expect(useAppStore.getState().processingFileId).toBeNull();
    expect(logged).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ message: "Couldn't decode this RAW file. X" }));
    logged.mockRestore();
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
    processRaw.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    const running = processFile('a');
    expect(isProcessing()).toBe(true);
    await processFile('a'); // returns immediately: locked
    expect(processRaw).toHaveBeenCalledTimes(1);
    release(fakeImage());
    await running;
    expect(isProcessing()).toBe(false);
  });
  it('disposes a removed in-flight result while holding the lock until completion', async () => {
    let release!: (image: ProcessedImage) => void;
    processRaw.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    const running = processFile('a');
    await vi.waitFor(() => expect(processRaw).toHaveBeenCalledTimes(1));
    discardResult('a');
    useAppStore.setState({ files: [makeFile('b')] });
    await processFile('b');
    expect(isProcessing()).toBe(true);
    expect(processRaw).toHaveBeenCalledTimes(1);
    const image = fakeImage();
    release(image);
    await running;
    expect(image.disposed).toBe(1);
    expect(getResult('a')).toBeNull();
    expect(useAppStore.getState().files[0].status).toBe('queued');
    expect(isProcessing()).toBe(false);
  });
  it('keeps RETAINED_RESULTS results besides the selected one, least recently used evicted', async () => {
    expect(RETAINED_RESULTS).toBe(1);
    useAppStore.setState({ files: [makeFile('a'), makeFile('b'), makeFile('c')], selectedFileId: 'c' });
    const a = fakeImage(),
      b = fakeImage(),
      c = fakeImage();
    processRaw.mockResolvedValueOnce(a).mockResolvedValueOnce(b).mockResolvedValueOnce(c);
    await processFile('a');
    await processFile('b');
    // Neither is selected or shown: only the most recent one stays.
    expect(a.disposed).toBe(1);
    expect(getResult('a')).toBeNull();
    expect(getResult('b')).toBe(b);
    await processFile('c');
    // The selected photo's result doesn't count against the budget.
    expect(getResult('b')).toBe(b);
    expect(getResult('c')).toBe(c);
    expect(b.disposed).toBe(0);
  });
  it('never evicts or disposes a result on screen, and disposes a replaced one when released', async () => {
    useAppStore.setState({ files: [makeFile('a'), makeFile('b'), makeFile('c')], selectedFileId: 'a' });
    const a1 = fakeImage(),
      a2 = fakeImage(),
      b = fakeImage(),
      c = fakeImage();
    processRaw
      .mockResolvedValueOnce(a1)
      .mockResolvedValueOnce(b)
      .mockResolvedValueOnce(c)
      .mockResolvedValueOnce(a2);
    await processFile('a');
    const shown = acquireResult('a')!;
    useAppStore.setState({ selectedFileId: 'z' });
    await processFile('b');
    await processFile('c');
    expect(a1.disposed).toBe(0);
    expect(getResult('a')).toBe(a1); // on screen
    expect(b.disposed).toBe(1); // evicted instead
    await processFile('a'); // a new run replaces a1…
    expect(getResult('a')).toBe(a2);
    expect(a1.disposed).toBe(0); // …but a1 is still shown
    shown.release();
    expect(a1.disposed).toBe(1);
  });
  it('disposes a discarded result only once nothing shows it', async () => {
    const image = fakeImage();
    processRaw.mockResolvedValueOnce(image);
    await processFile('a');
    const shown = acquireResult('a')!;
    discardResult('a');
    expect(getResult('a')).toBeNull();
    expect(image.disposed).toBe(0);
    shown.release();
    expect(image.disposed).toBe(1);
  });
  it('clears the previous image if a retry fails', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    const image = fakeImage();
    processRaw.mockResolvedValueOnce(image).mockRejectedValueOnce(new Error('retry'));
    await processFile('a');
    await processFile('a');
    expect(image.disposed).toBe(1);
    expect(getResult('a')).toBeNull();
    expect(useAppStore.getState().files[0].status).toBe('error');
    expect(logged).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ message: 'retry' }));
    logged.mockRestore();
  });

  it('records the fallback after a user edit made before the first processing run', async () => {
    const photo = makeFile('a');
    photo.edit = { ...photo.edit, demosaicMethod: 'neural-net', model: { size: 'S', sha256: 'old' } };
    useAppStore.setState({ files: [photo] });
    useAppStore.getState().setFileLookPreset('a', 'umbra');
    const image = fakeImage();
    image.meta.metadata.modelIdentity = { size: 'S', sha256: 'actual' };
    processRaw.mockResolvedValue(image);
    await processFile('a');
    expect(useAppStore.getState().files[0].edit.model?.sha256).toBe('actual');
  });
  it('discards a run superseded by a method change instead of overwriting the newer edit', async () => {
    let finish!: (image: ProcessedImage) => void;
    processRaw.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const running = processFile('a');
    await vi.waitFor(() => expect(processRaw).toHaveBeenCalled());
    useAppStore.getState().setFileDemosaicMethod('a', 'bilinear');
    const image = fakeImage();
    finish(image);
    await running;
    expect(image.disposed).toBe(1);
    expect(useAppStore.getState().files[0]).toMatchObject({
      status: 'queued',
      edit: { demosaicMethod: 'bilinear' },
    });
  });
});
