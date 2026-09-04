import { describe, it, expect } from 'vitest';
import { resolveValue, isModified, formatDelta } from './param-model';

describe('resolveValue', () => {
  it('returns the override when present', () => {
    expect(resolveValue(0.5, 1.4)).toBe(0.5);
  });
  it('returns the base when override is undefined', () => {
    expect(resolveValue(undefined, 1.4)).toBe(1.4);
  });
  it('treats 0 as a real override, not absent', () => {
    expect(resolveValue(0, 1.4)).toBe(0);
  });
  it('works for boolean params', () => {
    expect(resolveValue(true, false)).toBe(true);
    expect(resolveValue(undefined, false)).toBe(false);
  });
});

describe('isModified', () => {
  it('is false at the default', () => {
    expect(isModified(1.4, 1.4)).toBe(false);
  });
  it('is true away from the default', () => {
    expect(isModified(1.0, 1.4)).toBe(true);
  });
  it('ignores sub-epsilon float noise', () => {
    expect(isModified(0.5 + 1e-9, 0.5)).toBe(false);
  });
});

describe('formatDelta', () => {
  const f = (n: number) => n.toFixed(2);
  it('prefixes a + for positive deltas', () => {
    expect(formatDelta(1.0, 0.0, f)).toBe('+1.00');
  });
  it('keeps the - for negative deltas', () => {
    expect(formatDelta(-0.33, 0.0, f)).toBe('-0.33');
  });
  it('formats the magnitude of the difference, not the value', () => {
    expect(formatDelta(1.73, 1.4, f)).toBe('+0.33');
  });
});
