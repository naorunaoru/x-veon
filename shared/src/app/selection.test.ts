// shared/src/app/selection.test.ts
import { describe, expect, it } from 'vitest';
import { clickSelect, moveCurrent, selectAllShown, toggleSelectAll, clearSelection, prune, type SelectionState } from './selection';
const order = ['a', 'b', 'c', 'd', 'e'];
const all = { order, shown: new Set(order) };
const base: SelectionState = { currentId: 'b', selectedIds: ['b'], anchorId: 'b' };

describe('clickSelect', () => {
  it('plain click makes the photo current and the only selected one', () => {
    expect(clickSelect(base, all, 'd', { meta: false, shift: false })).toEqual({ currentId: 'd', selectedIds: ['d'], anchorId: 'd' });
  });
  it('meta-click toggles without moving current', () => {
    const s = clickSelect(base, all, 'd', { meta: true, shift: false });
    expect(s).toEqual({ currentId: 'b', selectedIds: ['b', 'd'], anchorId: 'd' });
    expect(clickSelect(s, all, 'd', { meta: true, shift: false }).selectedIds).toEqual(['b']);
  });
  it('shift-click selects the range from the anchor within shown order and moves current', () => {
    expect(clickSelect(base, all, 'e', { meta: false, shift: true })).toEqual({ currentId: 'e', selectedIds: ['b', 'c', 'd', 'e'], anchorId: 'b' });
    expect(clickSelect(base, { order, shown: new Set(['a', 'b', 'd', 'e']) }, 'e', { meta: false, shift: true }).selectedIds).toEqual(['b', 'd', 'e']);
  });
});

describe('moveCurrent', () => {
  it('steps through shown photos and collapses the selection to current', () => {
    expect(moveCurrent(base, all, 1, false)).toEqual({ currentId: 'c', selectedIds: ['c'], anchorId: 'c' });
    expect(moveCurrent(base, all, -1, false).currentId).toBe('a');
  });
  it('stops at the ends without changing state', () => {
    const atEnd = { currentId: 'e', selectedIds: ['e'], anchorId: 'e' };
    expect(moveCurrent(atEnd, all, 1, false)).toBe(atEnd);
    expect(moveCurrent(atEnd, all, 2, false)).toBe(atEnd);
  });
  it('extends the selection from the anchor with shift', () => {
    const s = moveCurrent(base, all, 2, true);
    expect(s).toEqual({ currentId: 'd', selectedIds: ['b', 'c', 'd'], anchorId: 'b' });
    expect(moveCurrent(s, all, -3, true).selectedIds).toEqual(['a', 'b']);
  });
  it('goes to the nearest shown photo when current is hidden', () => {
    const o = { order, shown: new Set(['a', 'd', 'e']) };
    expect(moveCurrent(base, o, 1, false).currentId).toBe('d');
    expect(moveCurrent(base, o, -1, false).currentId).toBe('a');
  });
  it('is a no-op with no current photo or no shown photos', () => {
    const none = { currentId: null, selectedIds: [], anchorId: null };
    expect(moveCurrent(none, all, 1, false)).toBe(none);
    expect(moveCurrent(base, { order, shown: new Set() }, 1, false)).toBe(base);
  });
});

describe('select all / clear / prune', () => {
  it('selects everything shown, in order', () => {
    expect(selectAllShown(base, { order, shown: new Set(['e', 'a']) }).selectedIds).toEqual(['a', 'e']);
  });
  it('toggle clears when everything shown is already selected', () => {
    const full = selectAllShown(base, all);
    expect(toggleSelectAll(full, all).selectedIds).toEqual([]);
    expect(toggleSelectAll(base, all).selectedIds).toEqual(order);
  });
  it('clear returns the same object when already empty', () => {
    const empty = { ...base, selectedIds: [] };
    expect(clearSelection(empty)).toBe(empty);
    expect(clearSelection(base).selectedIds).toEqual([]);
  });
  it('prune drops missing ids and the anchor, keeps current', () => {
    const s = { currentId: 'b', selectedIds: ['b', 'c'], anchorId: 'c' };
    expect(prune(s, new Set(['a', 'b']))).toEqual({ currentId: 'b', selectedIds: ['b'], anchorId: null });
    expect(prune(s, new Set(order))).toBe(s);
  });
});
