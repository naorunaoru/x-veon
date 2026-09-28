import { afterEach, describe, expect, it, vi } from 'vitest';
import { probeHdrDisplay } from './display';

function environment({ hdr, permission }: { hdr: boolean; permission: PermissionState | 'unsupported' }) {
  const getScreenDetails = vi.fn(async () => ({ currentScreen: { highDynamicRangeHeadroom: 4 } }));
  vi.stubGlobal('matchMedia', (q: string) => ({ matches: hdr && q === '(dynamic-range: high)' }));
  vi.stubGlobal('getScreenDetails', getScreenDetails);
  vi.stubGlobal('navigator', {
    permissions: {
      query: async () => {
        if (permission === 'unsupported') throw new TypeError('unknown permission');
        return { state: permission };
      },
    },
  });
  return getScreenDetails;
}

describe('HDR display probe', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('never calls getScreenDetails on an SDR display', async () => {
    const details = environment({ hdr: false, permission: 'granted' });
    expect(await probeHdrDisplay()).toEqual({ supported: false, headroom: 1, accurate: true });
    expect(details).not.toHaveBeenCalled();
  });

  it.each(['prompt', 'denied', 'unsupported'] as const)(
    'does not prompt at startup on an HDR display when permission is %s',
    async (permission) => {
      const details = environment({ hdr: true, permission });
      expect(await probeHdrDisplay()).toEqual({ supported: true, headroom: 2, accurate: false });
      expect(details).not.toHaveBeenCalled();
    },
  );

  it('reads the exact headroom once permission is granted', async () => {
    const details = environment({ hdr: true, permission: 'granted' });
    expect(await probeHdrDisplay()).toEqual({ supported: true, headroom: 4, accurate: true });
    expect(details).toHaveBeenCalledTimes(1);
  });
});
