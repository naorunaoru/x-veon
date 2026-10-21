import { beforeEach, expect, it, vi } from 'vitest';
import { useAppStore } from '@/app/store';
import { lensPickKey } from '@/app/store/lens';
import { fromLibraryPhoto } from '@/app/store/photo';
import { loadLensProfile } from '@/app/lens/lensfun';
import type { LensProfile } from '@/lib/lensfun-types';
import { fakePhoto } from '@/test/fake-host';
import { ensureLensPick } from './lens-picks';

vi.mock('@/app/lens/lensfun', () => ({ loadLensProfile: vi.fn() }));

const profile = { lensModel: 'm' } as LensProfile;
const choice = { file: 'f', model: 'm' };
const picks = () => useAppStore.getState().lensPicks;

beforeEach(() => {
  vi.mocked(loadLensProfile).mockReset().mockResolvedValue(profile);
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  const photo = fromLibraryPhoto(fakePhoto());
  useAppStore.setState({ files: [{ ...photo, metadata: { camera: 'X-T3', lensModel: '', focalLength: 35, fNumber: 2 } }], lensPicks: {} });
});

it('loads a pick once and stores it under its key', async () => {
  ensureLensPick('a', choice); ensureLensPick('a', choice);
  await vi.waitFor(() => expect(picks()[lensPickKey('f', 'm')]).toBeDefined());
  expect(loadLensProfile).toHaveBeenCalledTimes(1);
  expect(loadLensProfile).toHaveBeenCalledWith(choice, 'X-T3');
});
it('does nothing for a pick that is already stored', () => {
  useAppStore.getState().setLensPick(lensPickKey('f', 'm'), profile);
  ensureLensPick('a', choice);
  expect(loadLensProfile).not.toHaveBeenCalled();
});
it('does nothing for a photo that is gone', () => {
  ensureLensPick('missing', choice);
  expect(loadLensProfile).not.toHaveBeenCalled();
});
it('allows a retry after a failed or empty load, without throwing', async () => {
  const settled = () => new Promise((resolve) => setTimeout(resolve));
  vi.mocked(loadLensProfile).mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(null);
  ensureLensPick('a', choice); await settled();
  expect(console.warn).toHaveBeenCalled();
  ensureLensPick('a', choice); await settled();
  expect(loadLensProfile).toHaveBeenCalledTimes(2); expect(picks()).toEqual({});
  ensureLensPick('a', choice); await settled();
  expect(loadLensProfile).toHaveBeenCalledTimes(3); expect(picks()[lensPickKey('f', 'm')]).toBeDefined();
});
