import { describe, expect, it } from 'vitest';
import {
  buildReport,
  compareToBaseline,
  exportKey,
  hashBytes,
  hashFloat32,
  methodKey,
  overallStatus,
  type Expectation,
  type GoldenBaseline,
} from './golden-compare';

const adapter = { vendor: 'test', architecture: 'none' };
const recordedAt = '2026-09-04T00:00:00Z';

describe('golden hashing', () => {
  it('hashes Float32Array bytes deterministically', async () => {
    const input = new Float32Array([0, 0.5, 1]);
    const hash = await hashFloat32(input);

    expect(hash).toBe(await hashFloat32(input));
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(hash).not.toBe(await hashFloat32(new Float32Array([0, 0.5, 1.0000001])));
  });

  it('hashes raw bytes', async () => {
    expect(await hashBytes(new Uint8Array([1, 2, 3]))).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('golden reports', () => {
  it('groups repeated runs, detects instability, and carries errors', () => {
    const report = buildReport('quick', [
      { key: 'a.raf|neural-net:S', run: 1, display: 'h1', displayDark: 'd1', scene: 's1', elapsedMs: 10 },
      { key: 'a.raf|neural-net:S', run: 2, display: 'h1', displayDark: 'd1', scene: 's1', elapsedMs: 12 },
      { key: 'a.raf|dht', run: 1, display: 'x1', displayDark: 'y1', scene: 's2', elapsedMs: 2 },
      { key: 'a.raf|dht', run: 2, display: 'x2', displayDark: 'y1', scene: 's2', elapsedMs: 3 },
      { key: 'a.raf|ppg', run: 1, display: '', displayDark: '', scene: '', elapsedMs: 0, error: 'boom' },
    ], [{ key: 'a.raf|tiff', bytes: 10, sha256: 'ee' }], adapter, 'abc1234', recordedAt);

    expect(report.entries['a.raf|neural-net:S']).toMatchObject({
      display: 'h1', displayDark: 'd1', scene: 's1', stable: true, runs: 2, elapsedMs: [10, 12],
    });
    expect(report.entries['a.raf|dht'].stable).toBe(false);
    expect(report.entries['a.raf|ppg'].error).toBe('boom');
    expect(report.exports['a.raf|tiff']).toEqual({ bytes: 10, sha256: 'ee' });
    expect(report.commit).toBe('abc1234');
  });
});

describe('compareToBaseline', () => {
  const expected: Expectation = {
    keys: ['a|neural-net:S', 'a|dht'],
    exportKeys: ['a|tiff'],
    runs: 2,
  };
  const good = buildReport('full', [
    { key: 'a|neural-net:S', run: 1, display: 'h', displayDark: 'd', scene: 's', elapsedMs: 1 },
    { key: 'a|neural-net:S', run: 2, display: 'h', displayDark: 'd', scene: 's', elapsedMs: 1 },
    { key: 'a|dht', run: 1, display: 'q', displayDark: 'r', scene: 't', elapsedMs: 1 },
    { key: 'a|dht', run: 2, display: 'q', displayDark: 'r', scene: 't', elapsedMs: 1 },
  ], [{ key: 'a|tiff', bytes: 100, sha256: 's1' }], adapter, 'x', recordedAt);
  const baseline: GoldenBaseline = {
    adapter,
    commit: 'x',
    recordedAt,
    entries: {
      'a|neural-net:S': { display: 'h', displayDark: 'd', scene: 's' },
      'a|dht': { display: 'q', displayDark: 'r', scene: 't' },
    },
    exports: { 'a|tiff': { bytes: 100, sha256: 's1' } },
  };

  it('records every expected key when there is no baseline', () => {
    const results = compareToBaseline(good, null, expected);
    expect(results.map((result) => result.status)).toEqual(['NEW', 'NEW', 'NEW']);
    expect(overallStatus(results)).toBe('RECORDED');
  });

  it('passes only when all expected method and export hashes match', () => {
    const results = compareToBaseline(good, baseline, expected);
    expect(results.every((result) => result.status === 'PASS')).toBe(true);
    expect(overallStatus(results)).toBe('PASS');
  });

  it('fails on a differing method hash', () => {
    const changed = {
      ...baseline,
      entries: { ...baseline.entries, 'a|dht': { display: 'q', displayDark: 'WRONG', scene: 't' } },
    };
    expect(compareToBaseline(good, changed, expected).find((result) => result.key === 'a|dht')).toEqual({
      key: 'a|dht', status: 'FAIL', reason: 'displayDark differs',
    });
  });

  it('fails when the baseline has no scene hash for a key', () => {
    const noScene = {
      ...baseline,
      entries: { ...baseline.entries, 'a|dht': { display: 'q', displayDark: 'r' } },
    };
    expect(compareToBaseline(good, noScene, expected).find((result) => result.key === 'a|dht')).toEqual({
      key: 'a|dht', status: 'FAIL', reason: 'scene missing from baseline',
    });
  });

  it('fails on a differing scene hash', () => {
    const changed = {
      ...baseline,
      entries: { ...baseline.entries, 'a|dht': { display: 'q', displayDark: 'r', scene: 'WRONG' } },
    };
    expect(compareToBaseline(good, changed, expected).find((result) => result.key === 'a|dht')).toEqual({
      key: 'a|dht', status: 'FAIL', reason: 'scene differs',
    });
  });

  it('treats a scene hash that changes between runs as unstable', () => {
    const drifting = buildReport('quick', [
      { key: 'a|dht', run: 1, display: 'q', displayDark: 'r', scene: 't', elapsedMs: 1 },
      { key: 'a|dht', run: 2, display: 'q', displayDark: 'r', scene: 'DRIFTED', elapsedMs: 1 },
    ], [], adapter, 'x', recordedAt);

    expect(drifting.entries['a|dht'].stable).toBe(false);
    expect(compareToBaseline(drifting, baseline, {
      keys: ['a|dht'], exportKeys: [], runs: 2,
    })).toEqual([{ key: 'a|dht', status: 'UNSTABLE' }]);
  });

  it('fails when expected report or baseline entries are missing', () => {
    const partial = buildReport('quick', [
      { key: 'a|neural-net:S', run: 1, display: 'h', displayDark: 'd', scene: 's', elapsedMs: 1 },
      { key: 'a|neural-net:S', run: 2, display: 'h', displayDark: 'd', scene: 's', elapsedMs: 1 },
    ], [], adapter, 'x', recordedAt);
    const reportResults = compareToBaseline(partial, baseline, expected);
    expect(reportResults.find((result) => result.key === 'a|dht')).toEqual({
      key: 'a|dht', status: 'FAIL', reason: 'missing from report',
    });
    expect(reportResults.find((result) => result.key === 'a|tiff')).toEqual({
      key: 'a|tiff', status: 'FAIL', reason: 'missing from report',
    });

    const incompleteBaseline = {
      ...baseline,
      entries: { 'a|neural-net:S': baseline.entries['a|neural-net:S'] },
    };
    expect(compareToBaseline(good, incompleteBaseline, expected).find((result) => result.key === 'a|dht')).toEqual({
      key: 'a|dht', status: 'FAIL', reason: 'missing from baseline',
    });
  });

  it('fails when a key has fewer runs than expected', () => {
    const oneRun = buildReport('quick', [
      { key: 'a|neural-net:S', run: 1, display: 'h', displayDark: 'd', scene: 's', elapsedMs: 1 },
    ], [], adapter, 'x', recordedAt);
    expect(compareToBaseline(oneRun, null, {
      keys: ['a|neural-net:S'], exportKeys: [], runs: 2,
    })).toEqual([{ key: 'a|neural-net:S', status: 'FAIL', reason: 'runs 1 < 2' }]);
  });

  it('reports unstable runs and processing errors', () => {
    const report = buildReport('quick', [
      { key: 'a|neural-net:S', run: 1, display: 'h', displayDark: 'd', scene: 's', elapsedMs: 1 },
      { key: 'a|neural-net:S', run: 2, display: 'h2', displayDark: 'd', scene: 's', elapsedMs: 1 },
      { key: 'a|dht', run: 1, display: '', displayDark: '', scene: '', elapsedMs: 0, error: 'decode failed' },
      { key: 'a|dht', run: 2, display: '', displayDark: '', scene: '', elapsedMs: 0, error: 'decode failed' },
    ], [], adapter, 'x', recordedAt);
    const results = compareToBaseline(report, null, {
      keys: ['a|neural-net:S', 'a|dht'], exportKeys: [], runs: 2,
    });
    expect(results).toEqual([
      { key: 'a|neural-net:S', status: 'UNSTABLE' },
      { key: 'a|dht', status: 'FAIL', reason: 'decode failed' },
    ]);
    expect(overallStatus(results)).toBe('FAIL');
    expect(overallStatus([
      { key: 'k', status: 'UNSTABLE' },
      { key: 'j', status: 'PASS' },
    ])).toBe('UNSTABLE');
  });

  it('fails when an export hash differs, even at the same byte length', () => {
    const changed = { ...baseline, exports: { 'a|tiff': { bytes: 100, sha256: 'other' } } };
    expect(compareToBaseline(good, changed, expected).find((result) => result.key === 'a|tiff')).toEqual({
      key: 'a|tiff', status: 'FAIL', reason: 'bytes differ',
    });
  });

  it('never passes an empty report', () => {
    const empty = buildReport('quick', [], [], adapter, 'x', recordedAt);
    expect(overallStatus(compareToBaseline(empty, baseline, expected))).toBe('FAIL');
    expect(overallStatus([])).toBe('FAIL');
  });

  it('builds stable method and export keys', () => {
    expect(methodKey('a.raf', 'neural-net', 'S')).toBe('a.raf|neural-net:S');
    expect(methodKey('a.raf', 'dht')).toBe('a.raf|dht');
    expect(exportKey('a.raf', 'tiff')).toBe('a.raf|tiff');
  });
});
