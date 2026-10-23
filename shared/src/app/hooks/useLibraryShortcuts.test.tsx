import { beforeEach as beforeHostTest } from 'vitest';
beforeHostTest(() => {
  getHost().library.remove = vi.fn(async () => {});
});
import { getHost } from '@/app/services/host';
import { renderHook, fireEvent } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
const remove = vi.hoisted(() => vi.fn());
vi.mock('@/app/services/library', () => ({ removeFile: remove }));
const startExport = vi.hoisted(() => vi.fn());
vi.mock('@/app/services/export-batch', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/app/services/export-batch')>(), startExport,
}));
import { useLibraryShortcuts } from './useLibraryShortcuts';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
beforeEach(() => {
  remove.mockClear();
  useAppStore.setState({ currentId: 'b' });
});
it.each(['Delete', 'Backspace'])('removes the selected photo with %s', (key) => {
  renderHook(useLibraryShortcuts);
  fireEvent.keyDown(document.body, { key });
  expect(remove).toHaveBeenCalledWith('b');
});
it('protects editing, dialogs, modifier shortcuts, and held keys', () => {
  renderHook(useLibraryShortcuts);
  for (const tag of ['input', 'textarea', 'select']) {
    const field = document.createElement(tag);
    document.body.append(field);
    fireEvent.keyDown(field, { key: 'Backspace' });
    field.remove();
  }
  const editor = document.createElement('div');
  editor.setAttribute('contenteditable', 'true');
  document.body.append(editor);
  fireEvent.keyDown(editor, { key: 'Delete' });
  editor.remove();
  fireEvent.keyDown(document.body, { key: 'Delete', repeat: true });
  fireEvent.keyDown(document.body, { key: 'Backspace', metaKey: true });
  const dialog = document.createElement('div');
  dialog.setAttribute('role', 'dialog');
  document.body.append(dialog);
  fireEvent.keyDown(document.body, { key: 'Delete' });
  dialog.remove();
  expect(remove).not.toHaveBeenCalled();
});
describe('navigation keys', () => {
  beforeEach(() => useAppStore.setState({
    files: [fromLibraryPhoto(fakePhoto('a')), fromLibraryPhoto(fakePhoto('b'))],
    columnWidth: 3, selectedIds: [], selectionAnchor: null,
  }));
  // The moveCurrent spies would otherwise ride along into every later store state.
  afterEach(() => vi.restoreAllMocks());
  it.each([
    ['ArrowRight', false, [1, false]], ['ArrowLeft', false, [-1, false]],
    ['ArrowDown', false, [3, false]], ['ArrowUp', true, [-3, true]],
  ])('%s moves current (shift=%s)', (key, shiftKey, args) => {
    const move = vi.spyOn(useAppStore.getState(), 'moveCurrent').mockImplementation(() => {});
    renderHook(useLibraryShortcuts);
    fireEvent.keyDown(document.body, { key, shiftKey });
    expect(move).toHaveBeenCalledWith(...args);
  });
  it('steps by one row of one when the column is a strip or collapsed', () => {
    const move = vi.spyOn(useAppStore.getState(), 'moveCurrent').mockImplementation(() => {});
    renderHook(useLibraryShortcuts);
    useAppStore.setState({ columnWidth: 1 });
    fireEvent.keyDown(document.body, { key: 'ArrowDown' });
    useAppStore.setState({ columnWidth: 0 });
    fireEvent.keyDown(document.body, { key: 'ArrowDown' });
    expect(move.mock.calls).toEqual([[1, false], [1, false]]);
  });
  it('⌘A selects all shown, Escape clears, Tab toggles the column', () => {
    useAppStore.setState({ files: [fromLibraryPhoto(fakePhoto('a')), fromLibraryPhoto(fakePhoto('b'))], currentId: 'a' });
    renderHook(useLibraryShortcuts);
    fireEvent.keyDown(document.body, { key: 'a', metaKey: true });
    expect(useAppStore.getState().selectedIds).toEqual(['a', 'b']);
    fireEvent.keyDown(document.body, { key: 'a', metaKey: true });
    expect(useAppStore.getState().selectedIds).toEqual(['a', 'b']);   // never clears
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(useAppStore.getState().selectedIds).toEqual([]);
    const tab = fireEvent.keyDown(document.body, { key: 'Tab' });
    expect(useAppStore.getState().columnWidth).toBe(0);
    expect(tab).toBe(false);                                            // defaultPrevented
  });
  it('⌘E exports the selection or the current photo', () => {
    startExport.mockClear();
    useAppStore.setState({ currentId: 'a', selectedIds: ['b', 'a'], initialized: true });
    renderHook(useLibraryShortcuts);
    expect(fireEvent.keyDown(document.body, { key: 'e', metaKey: true })).toBe(false);   // defaultPrevented
    useAppStore.setState({ selectedIds: [] });
    fireEvent.keyDown(document.body, { key: 'E', ctrlKey: true });
    fireEvent.keyDown(document.body, { key: 'у', code: 'KeyE', metaKey: true });
    expect(startExport.mock.calls).toEqual([[['b', 'a']], [['a']], [['a']]]);
    startExport.mockClear();
    fireEvent.keyDown(document.body, { key: 'e' });
    fireEvent.keyDown(document.body, { key: 'E', metaKey: true, shiftKey: true });
    fireEvent.keyDown(document.body, { key: 'e', metaKey: true, altKey: true });
    useAppStore.setState({ initialized: false });
    fireEvent.keyDown(document.body, { key: 'e', metaKey: true });              // the pipeline is not up yet
    expect(startExport).not.toHaveBeenCalled();
  });
  it('rating keys set, clear and reject the current photo', () => {
    const rate = vi.spyOn(useAppStore.getState(), 'setFileRating').mockImplementation(() => {});
    renderHook(useLibraryShortcuts);
    for (const key of ['1', '2', '3', '4', '5', '0']) {
      expect(fireEvent.keyDown(document.body, { key })).toBe(false);     // defaultPrevented
    }
    fireEvent.keyDown(document.body, { key: 'x' });
    fireEvent.keyDown(document.body, { key: 'X' });
    fireEvent.keyDown(document.body, { key: 'ч', code: 'KeyX' });
    expect(rate.mock.calls).toEqual([
      ['b', 1], ['b', 2], ['b', 3], ['b', 4], ['b', 5], ['b', 0], ['b', -1], ['b', -1], ['b', -1],
    ]);
    rate.mockClear();
    fireEvent.keyDown(document.body, { key: '3', metaKey: true });
    fireEvent.keyDown(document.body, { key: '3', shiftKey: true });
    fireEvent.keyDown(document.body, { key: 'x', metaKey: true });
    fireEvent.keyDown(document.body, { key: 'X', shiftKey: true });
    expect(rate).not.toHaveBeenCalled();
  });
  it('protects editing, dialogs and menus', () => {
    const move = vi.spyOn(useAppStore.getState(), 'moveCurrent').mockImplementation(() => {});
    const rate = vi.spyOn(useAppStore.getState(), 'setFileRating').mockImplementation(() => {});
    renderHook(useLibraryShortcuts);
    const menu = document.createElement('div');
    menu.setAttribute('role', 'menu');
    document.body.append(menu);
    fireEvent.keyDown(document.body, { key: 'ArrowRight' });
    menu.remove();
    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog');
    document.body.append(dialog);
    const tab = fireEvent.keyDown(dialog, { key: 'Tab' });
    dialog.remove();
    const input = document.createElement('input');
    document.body.append(input);
    fireEvent.keyDown(input, { key: 'ArrowLeft' });
    fireEvent.keyDown(input, { key: '3' });
    input.remove();
    expect(move).not.toHaveBeenCalled();
    expect(rate).not.toHaveBeenCalled();
    expect(tab).toBe(true);                                             // Tab in a dialog keeps its default
    expect(useAppStore.getState().columnWidth).toBe(3);
  });
  it('Tab from a control keeps its traversal', () => {
    renderHook(useLibraryShortcuts);
    const button = document.createElement('button');
    document.body.append(button);
    button.focus();
    const tab = fireEvent.keyDown(button, { key: 'Tab' });
    button.remove();
    expect(tab).toBe(true);                                             // not defaultPrevented
    expect(useAppStore.getState().columnWidth).toBe(3);
  });
  it('Tab from the column\'s cells toggles the column', () => {
    renderHook(useLibraryShortcuts);
    const listbox = document.createElement('div');
    listbox.setAttribute('role', 'listbox');
    const cell = document.createElement('button');
    listbox.append(cell);
    document.body.append(listbox);
    cell.focus();
    const tab = fireEvent.keyDown(cell, { key: 'Tab' });
    listbox.remove();
    expect(tab).toBe(false);                                            // defaultPrevented
    expect(useAppStore.getState().columnWidth).toBe(0);
  });
  it('navigates on a host without remove (desktop)', () => {
    delete getHost().library.remove;
    useAppStore.setState({ currentId: 'a' });
    renderHook(useLibraryShortcuts);
    fireEvent.keyDown(document.body, { key: 'ArrowRight' });
    expect(useAppStore.getState().currentId).toBe('b');
  });
  it('a held Tab toggles once and swallows its repeats', () => {
    renderHook(useLibraryShortcuts);
    const held = fireEvent.keyDown(document.body, { key: 'Tab', repeat: true });
    expect(held).toBe(false);                                           // defaultPrevented
    expect(useAppStore.getState().columnWidth).toBe(3);
  });
  it('leaves the keys to the browser when there are no photos', () => {
    useAppStore.setState({ files: [], currentId: null });
    renderHook(useLibraryShortcuts);
    const tab = fireEvent.keyDown(document.body, { key: 'Tab' });
    const right = fireEvent.keyDown(document.body, { key: 'ArrowRight' });
    expect(tab).toBe(true);
    expect(right).toBe(true);
    expect(useAppStore.getState().columnWidth).toBe(3);
  });
  it.each([
    { key: 'ф', code: 'KeyA', ctrlKey: true, selected: ['a', 'b'] },
    { key: 'A', metaKey: true, selected: ['a', 'b'] },
    { key: 'q', code: 'KeyA', ctrlKey: true, selected: [] },        // AZERTY Ctrl+Q
  ])('⌘A selects all shown on any layout (key $key)', ({ selected, ...init }) => {
    renderHook(useLibraryShortcuts);
    fireEvent.keyDown(document.body, init);
    expect(useAppStore.getState().selectedIds).toEqual(selected);
  });
});
