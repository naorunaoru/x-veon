import { expect, it, vi } from 'vitest';
import { loadDisplayReader } from './display';

it('loads the addon once, passes the window handle, and sanitises readings', () => {
  const displayReadings = vi.fn(() => ({ potentialEdr: 16, junk: 1 }));
  const load = vi.fn(() => ({ displayReadings }));
  const reader = loadDisplayReader(load, '/native');
  const handle = Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]);
  expect(reader.read(handle)).toEqual({ potentialEdr: 16 });
  expect(reader.read(handle)).toEqual({ potentialEdr: 16 });
  expect(displayReadings).toHaveBeenCalledWith(handle);
  expect(load).toHaveBeenCalledOnce();
});
it('warns once and returns null on a missing addon', () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    const load = vi.fn(() => { throw new Error('missing'); });
    const reader = loadDisplayReader(load);
    expect(reader.read(null)).toBeNull(); expect(reader.read(null)).toBeNull();
    expect(load).toHaveBeenCalledOnce(); expect(warn).toHaveBeenCalledOnce();
  } finally { warn.mockRestore(); }
});
it('returns null when the addon has no display function', () => {
  expect(loadDisplayReader(() => ({})).read(null)).toBeNull();
});
it('warns and returns null when the addon throws', () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    expect(loadDisplayReader(() => ({ displayReadings: () => { throw new Error('failed'); } })).read(null)).toBeNull();
    expect(warn).toHaveBeenCalledOnce();
  } finally { warn.mockRestore(); }
});
it('returns null for garbage output', () => {
  expect(loadDisplayReader(() => ({ displayReadings: () => ({ potentialEdr: 'x' }) })).read(null)).toBeNull();
});
