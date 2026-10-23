import { beforeEach, expect, it } from 'vitest';
import { useAppStore } from '@/app/store';
import { maxColumns, shownIds } from './selection';
import { fromLibraryPhoto } from './photo';
import { fakePhoto } from '@/test/fake-host';
const files = ['a', 'b', 'c'].map(id => fromLibraryPhoto(fakePhoto(id)));
beforeEach(() => useAppStore.setState({ files, filter: 'all', currentId: 'a', selectedIds: [], selectionAnchor: null, columnWidth: 1, lastColumnWidth: 1 }));

it('shownIds lists every file in order', () => {
  expect(shownIds(useAppStore.getState())).toEqual(['a', 'b', 'c']);
});
it('clickCell, moveCurrent and toggleSelectAll go through the reducer', () => {
  const s = useAppStore.getState();
  s.clickCell('c', { meta: false, shift: true });
  expect(useAppStore.getState()).toMatchObject({ currentId: 'c', selectedIds: ['a', 'b', 'c'] });
  s.moveCurrent(-1, false);
  expect(useAppStore.getState()).toMatchObject({ currentId: 'b', selectedIds: ['b'] });
  s.toggleSelectAll();
  expect(useAppStore.getState().selectedIds).toEqual(['a', 'b', 'c']);
  s.toggleSelectAll();
  expect(useAppStore.getState().selectedIds).toEqual([]);
});
it('does not churn state when nothing changes', () => {
  const before = useAppStore.getState();
  before.clearSelection();
  before.moveCurrent(-1, false);
  before.setColumnWidth(1);
  expect(useAppStore.getState()).toBe(before);
});
it('clamps columnWidth to the window and remembers the last open width', () => {
  const innerWidth = Object.getOwnPropertyDescriptor(window, 'innerWidth')!;
  Object.defineProperty(window, 'innerWidth', { value: 1000, configurable: true });
  try {
    expect(maxColumns(1000)).toBe(4);
    const s = useAppStore.getState();
    s.setColumnWidth(99);
    expect(useAppStore.getState().columnWidth).toBe(4);
    s.toggleColumn();
    expect(useAppStore.getState()).toMatchObject({ columnWidth: 0, lastColumnWidth: 4 });
    s.toggleColumn();
    expect(useAppStore.getState().columnWidth).toBe(4);
    s.setColumnWidth(-3);
    expect(useAppStore.getState().columnWidth).toBe(0);
  } finally {
    Object.defineProperty(window, 'innerWidth', innerWidth);
  }
});
it('ignores a non-finite width', () => {
  expect(maxColumns(undefined as unknown as number)).toBe(0);
  const before = useAppStore.getState();
  before.setColumnWidth(NaN);
  expect(useAppStore.getState()).toBe(before);
  before.restoreFromDb([fromLibraryPhoto(fakePhoto('z'))], { columnWidth: 'x' as unknown as number });
  expect(useAppStore.getState()).toMatchObject({ columnWidth: 1, lastColumnWidth: 1 });
});
it('prunes the selection when files change', () => {
  const s = useAppStore.getState();
  s.clickCell('a', { meta: false, shift: false });
  s.clickCell('b', { meta: true, shift: false });
  s.removeFile('b');
  expect(useAppStore.getState()).toMatchObject({ selectedIds: ['a'], selectionAnchor: null });
  s.restoreFromDb([fromLibraryPhoto(fakePhoto('z'))], { columnWidth: 3 });
  // restoreFromDb keeps live files, so 'a' stays selected; only the width setting is applied
  expect(useAppStore.getState()).toMatchObject({ selectedIds: ['a'], columnWidth: 3, currentId: 'a' });
});
it('setFilter prunes the selection and moves a hidden current to the first shown', () => {
  useAppStore.setState({ files: [files[0], { ...files[1], rating: 2 }, { ...files[2], rating: 3 }], selectedIds: ['a', 'c'], selectionAnchor: 'a' });
  useAppStore.getState().setFilter('star2');
  const state = useAppStore.getState();
  expect(shownIds(state)).toEqual(['b', 'c']);
  expect(state).toMatchObject({ filter: 'star2', selectedIds: ['c'], selectionAnchor: null, currentId: 'b' });
});
it('keeps a hidden current photo until the user moves', () => {
  useAppStore.setState({ files: [{ ...files[0], rating: 1 }, { ...files[1], rating: 2 }, { ...files[2], rating: 3 }], filter: 'star1', currentId: 'b', selectedIds: ['b'], selectionAnchor: 'b' });
  const s = useAppStore.getState();
  s.setFileRating('b', -1);
  expect(useAppStore.getState()).toMatchObject({ currentId: 'b', selectedIds: ['b'] });
  expect(shownIds(useAppStore.getState())).toEqual(['a', 'c']);
  s.moveCurrent(1, false);
  expect(useAppStore.getState().currentId).toBe('c');
});
