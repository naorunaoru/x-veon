import { afterEach, describe, expect, it, vi } from 'vitest';
import type { DisplayReadings } from '@/host';
import type { DesktopBridge } from '../protocol/bridge';
import { createDisplayHost, probeFromReadings } from './display';

afterEach(() => vi.unstubAllGlobals());

describe('probeFromReadings', () => {
  it.each([
    [{ currentEdr: 1, potentialEdr: 16, referenceEdr: 4 }, { supported: true, headroom: 4, accurate: true }],
    [{ currentEdr: 1, potentialEdr: 16, referenceEdr: 0 }, { supported: true, headroom: 16, accurate: true }],
    [{ currentEdr: 1, potentialEdr: 1, referenceEdr: 0 }, { supported: false, headroom: 1, accurate: true }],
    [{ hdrEnabled: true, maxLuminance: 1000, sdrWhite: 200 }, { supported: true, headroom: 5, accurate: true }],
    [{ hdrEnabled: false, maxLuminance: 1000, sdrWhite: 200 }, { supported: false, headroom: 1, accurate: true }],
    [{ hdrEnabled: true, maxLuminance: 300, sdrWhite: 400 }, { supported: false, headroom: 1, accurate: true }],
  ] as const)('converts %j to %j', (readings, expected) => {
    expect(probeFromReadings(readings)).toEqual(expected);
  });

  it.each([
    { hdrEnabled: true, maxLuminance: 1000 },
    { hdrEnabled: true, maxLuminance: 0, sdrWhite: 200 },
    {},
  ] satisfies DisplayReadings[])('returns null for incomplete readings %j', readings => {
    expect(probeFromReadings(readings)).toBeNull();
  });
});

describe('createDisplayHost', () => {
  function host(readings: () => Promise<DisplayReadings | null>, hdr: boolean) {
    vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: hdr })));
    return createDisplayHost({ displayReadings: readings } as Pick<DesktopBridge, 'displayReadings'>);
  }

  it('uses the bridge reading for the probe and exposes it', async () => {
    const readings = { currentEdr: 1, potentialEdr: 16, referenceEdr: 0 };
    const displayReadings = vi.fn(async () => readings);
    const display = host(displayReadings, false);
    await expect(display.probe()).resolves.toEqual({ supported: true, headroom: 16, accurate: true });
    await expect(display.readings!()).resolves.toEqual(readings);
    expect(displayReadings).toHaveBeenCalledTimes(2);
  });

  it.each([
    ['null', async (): Promise<DisplayReadings | null> => null],
    ['rejection', async () => { throw new Error('read failed'); }],
    ['incomplete', async () => ({ hdrEnabled: true, maxLuminance: 1000 })],
  ] as const)('uses the high dynamic-range media query after %s', async (_case, readings) => {
    const display = host(readings, true);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await expect(display.probe()).resolves.toEqual({ supported: true, headroom: 2, accurate: false });
    } finally { warn.mockRestore(); }
  });

  it('returns the SDR media-query result when high dynamic range does not match', async () => {
    const display = host(async () => null, false);
    await expect(display.probe()).resolves.toEqual({ supported: false, headroom: 1, accurate: true });
  });

  it.each([
    ['null', async (): Promise<DisplayReadings | null> => null],
    ['rejection', async () => { throw new Error('read failed'); }],
  ] as const)('exposes empty readings after %s', async (_case, readings) => {
    const display = host(readings, false);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try { await expect(display.readings!()).resolves.toEqual({}); }
    finally { warn.mockRestore(); }
  });
});
