import { describe, it, expect, vi } from 'vitest';
import { renderHook } from '@testing-library/react';

// Stub out WASM-backed modules so the pure predicate can be imported in isolation.
vi.mock('@/app/hooks/useProcessing', () => ({ useProcessing: vi.fn() }));
import { useProcessing } from './useProcessing';
import { shouldAutoProcess, useAutoProcess } from './useAutoProcess';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { startPersistence } from '@/app/services/persistence';

const queued = { status: 'queued' as const };
const done = { status: 'done' as const };

describe('shouldAutoProcess', () => {
  it('processes a queued file when initialized and idle', () => {
    expect(shouldAutoProcess(queued, true, false)).toBe(true);
  });
  it('does not process before init', () => {
    expect(shouldAutoProcess(queued, false, false)).toBe(false);
  });
  it('does not process while another job runs', () => {
    expect(shouldAutoProcess(queued, true, true)).toBe(false);
  });
  it('does not process a done file', () => {
    expect(shouldAutoProcess(done, true, false)).toBe(false);
  });
  it('does nothing without a file', () => {
    expect(shouldAutoProcess(undefined, true, false)).toBe(false);
  });
});

it('processes an untouched Bayer photo with the neural net when its default is X-Trans only', async () => {
  const file = fromLibraryPhoto(fakePhoto());
  file.cfaType = 'bayer';
  useAppStore.setState({ files: [file], selectedFileId: file.id, demosaicMethod: 'markesteijn3', initialized: true });
  const processFile = vi.fn();
  vi.mocked(useProcessing).mockReturnValue({ processFile, isProcessing: false } as never);
  const methodChange = vi.spyOn(useAppStore.getState(), 'setFileDemosaicMethod');
  const host = fakeHost();
  setHost(host);
  vi.useFakeTimers();
  const stop = startPersistence();
  try {
    renderHook(useAutoProcess);
    expect(processFile).toHaveBeenCalledWith(file.id);
    expect(methodChange).not.toHaveBeenCalled();
    expect(useAppStore.getState().files[0].edit.demosaicMethod).toBeNull();
    await vi.advanceTimersByTimeAsync(301);
    expect(host.library.save).not.toHaveBeenCalled();
  } finally {
    stop();
    vi.useRealTimers();
  }
  useAppStore.getState().setFileLookPreset(file.id, 'umbra');
  expect(useAppStore.getState().files[0].edit.demosaicMethod).toBe('neural-net');
});

it('reprocesses an untouched result after its default method changes', () => {
  const file = fromLibraryPhoto(fakePhoto());
  file.status = 'done';
  file.processedKey = 'dht';
  useAppStore.setState({ files: [file], selectedFileId: file.id, demosaicMethod: 'neural-net', initialized: true });
  const processFile = vi.fn();
  vi.mocked(useProcessing).mockReturnValue({ processFile, isProcessing: false } as never);
  renderHook(useAutoProcess);
  expect(processFile).toHaveBeenCalledWith(file.id);
  expect(useAppStore.getState().files[0].edit.demosaicMethod).toBeNull();
});

it('processes a restored result whose previous processing key is unavailable', () => {
  const file = fromLibraryPhoto(fakePhoto());
  file.status = 'done';
  file.resultMethod = 'dht';
  useAppStore.setState({ files: [file], selectedFileId: file.id, demosaicMethod: 'neural-net', initialized: true });
  const processFile = vi.fn();
  vi.mocked(useProcessing).mockReturnValue({ processFile, isProcessing: false } as never);
  renderHook(useAutoProcess);
  expect(processFile).toHaveBeenCalledWith(file.id);
  expect(useAppStore.getState().files[0].edit.demosaicMethod).toBeNull();
});
