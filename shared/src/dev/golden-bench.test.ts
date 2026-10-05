import { expect, it } from 'vitest';
import { median } from './golden-bench';
it('finds the middle of unsorted odd and even samples', () => {
  expect(median([3, 1, 2])).toBe(2);
  expect(median([4, 1, 3, 2])).toBe(2.5);
});
it('rejects empty and nonfinite samples', () => {
  for (const values of [[], [NaN], [Infinity]]) expect(() => median(values)).toThrow();
});
