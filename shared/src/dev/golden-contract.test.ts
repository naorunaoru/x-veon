import { expect, it } from 'vitest';
import { expectationFor, selectBaseline } from './golden-contract';
import type { GoldenBaseline } from './golden-compare';
const baseline = (vendor: string, architecture: string) =>
  ({
    adapter: { vendor, architecture },
    entries: {},
    exports: {},
  }) as GoldenBaseline;
it('keeps exports mandatory for full mode and every method mandatory for render mode', () => {
  expect(expectationFor('full').keys).toHaveLength(10);
  expect(expectationFor('full').exportKeys).toHaveLength(6);
  expect(expectationFor('render')).toEqual({
    ...expectationFor('full'),
    exportKeys: [],
  });
  expect(expectationFor('quick').keys).toHaveLength(4);
  expect(expectationFor('quick').runs).toBe(2);
});
it('selects the recorded adapter exactly without a fallback', () => {
  const mac = baseline('apple', 'metal-3'),
    win = baseline('nvidia', 'turing');
  expect(selectBaseline(mac.adapter, [mac, win])).toBe(mac);
  expect(selectBaseline(win.adapter, [mac, win])).toBe(win);
  expect(
    selectBaseline({ vendor: 'amd', architecture: 'rdna' }, [mac, win]),
  ).toBeNull();
  expect(() => selectBaseline(mac.adapter, [mac, mac])).toThrow('Ambiguous');
  expect(() =>
    selectBaseline({ vendor: 'unknown', architecture: 'unknown' }, []),
  ).toThrow('Unknown');
});

it('fails a rendering report that omits any required method', async () => {
  const { buildReport, compareToBaseline, overallStatus } = await import(
    './golden-compare'
  );
  const expected = expectationFor('render');
  const report = buildReport(
    'render',
    expected.keys.slice(1).map((key) => ({
      key,
      run: 1,
      scene: 's',
      display: 'd',
      displayDark: 'k',
      elapsedMs: 1,
    })),
    [],
    { vendor: 'apple', architecture: 'metal-3' },
    'test',
    '2026-09-29',
  );
  expect(overallStatus(compareToBaseline(report, null, expected))).toBe('FAIL');
});
