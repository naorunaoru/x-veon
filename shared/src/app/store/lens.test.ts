import { beforeEach, describe, expect, it } from 'vitest';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { lensPickKey } from '@/app/store/lens';
import type { LensProfile } from '@/app/lens/lensfun';
import { DEFAULT_LENS_EDIT } from '@/lib/lens/edit';
import { fakePhoto } from '@/test/fake-host';

describe('lens slice', () => {
  beforeEach(() => useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto())], lensPicks: {} }));

  it('patches the lens block through editPhoto and keeps the rest', () => {
    const { setFileLens } = useAppStore.getState();
    setFileLens('a', { distortion: false });
    const file = useAppStore.getState().files[0];
    expect(file.edit.lens).toEqual({ ...DEFAULT_LENS_EDIT, distortion: false });
    expect(file.editing).toBe('session');
    expect(file.editRevision).toBeGreaterThan(0);
  });
  it('resets the block to absent', () => {
    useAppStore.getState().setFileLens('a', { source: 'manual' });
    useAppStore.getState().resetFileLens('a');
    expect(useAppStore.getState().files[0].edit.lens).toBeUndefined();
  });
  it('ignores edits on a view-only photo', () => {
    useAppStore.setState((s) => ({ files: s.files.map((f) => ({ ...f, editing: 'view-only' as const })) }));
    useAppStore.getState().setFileLens('a', { ca: false });
    expect(useAppStore.getState().files[0].edit.lens).toBeUndefined();
  });
  it('stores picks and marks matches', () => {
    useAppStore.getState().setLensPick(lensPickKey('misc.json', 'X'), { lensModel: 'X' } as LensProfile);
    expect(useAppStore.getState().lensPicks['misc.json#X'].lensModel).toBe('X');
    useAppStore.getState().markLensMatched('a');
    expect(useAppStore.getState().files[0].lensMatched).toBe(true);
  });
});
