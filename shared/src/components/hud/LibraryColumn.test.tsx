// shared/src/components/hud/LibraryColumn.test.tsx
import { beforeEach, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
import { getHost } from '@/app/services/host';
vi.mock('@/app/services/export-jobs', () => ({ startExport: vi.fn() }));
// Pointer events are fired the way Minimap.test.tsx fires them; jsdom has no PointerEvent class.
import { startExport } from '@/app/services/export-jobs';
import { LibraryColumn } from './LibraryColumn';
const files = ['a', 'b', 'c'].map(id => ({ ...fromLibraryPhoto(fakePhoto(id)), status: 'done' as const, thumbnailUrl: `blob:${id}` }));
beforeEach(() => useAppStore.setState({ files, currentId: 'a', selectedIds: ['a'], selectionAnchor: 'a', columnWidth: 1, lastColumnWidth: 1, initialized: true }));

it('renders nothing without photos', () => {
  useAppStore.setState({ files: [], currentId: null });
  const { container } = render(<LibraryColumn />);
  expect(container).toBeEmptyDOMElement();
});
it('renders one option per photo with current and selected state', () => {
  files[1].edit = { ...files[1].edit, lookPreset: 'colorful' };
  render(<LibraryColumn />);
  const cells = screen.getAllByTestId('column-cell');
  expect(cells).toHaveLength(3);
  expect(screen.getByRole('listbox', { name: 'Photos' })).toHaveAttribute('aria-multiselectable', 'true');
  expect(cells[0]).toHaveAttribute('data-current', 'true');
  expect(cells[0]).toHaveAttribute('aria-selected', 'true');
  expect(cells[1]).toHaveAttribute('aria-selected', 'false');
  expect(cells[0]).toHaveAttribute('tabindex', '0');
  expect(cells[1]).toHaveAttribute('tabindex', '-1');
  expect(cells[1].querySelector('img')).toHaveAttribute('src', 'blob:b');
  expect(cells[1].querySelector('.xv-cell__edited')).not.toBeNull();
  expect(cells[2].querySelector('.xv-cell__edited')).toBeNull();
  expect(screen.getByTestId('library-column').style.getPropertyValue('--xv-columns')).toBe('1');
});
it('dispatches clicks with modifiers', () => {
  const clickCell = vi.spyOn(useAppStore.getState(), 'clickCell').mockImplementation(() => {});
  render(<LibraryColumn />);
  const cells = screen.getAllByTestId('column-cell');
  fireEvent.click(cells[2]);
  fireEvent.click(cells[1], { metaKey: true });
  fireEvent.click(cells[1], { shiftKey: true });
  expect(clickCell.mock.calls).toEqual([
    ['c', { meta: false, shift: false }], ['b', { meta: true, shift: false }], ['b', { meta: false, shift: true }],
  ]);
});
it('select-all toggles and relabels', () => {
  render(<LibraryColumn />);
  fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
  expect(useAppStore.getState().selectedIds).toEqual(['a', 'b', 'c']);
  fireEvent.click(screen.getByRole('button', { name: 'Select none' }));
  expect(useAppStore.getState().selectedIds).toEqual([]);
});
it('exports the current photo through the dialog', async () => {
  render(<LibraryColumn />);
  const button = screen.getByRole('button', { name: 'Export' });
  await waitFor(() => expect(button).toBeEnabled());        // enabled once exporter.status() resolves
  fireEvent.click(button);
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Export' }));
  expect(startExport).toHaveBeenCalledWith('a');
});
it('disables Export while the current photo is not done', () => {
  useAppStore.setState({ files: [{ ...files[0], status: 'queued' }] });
  render(<LibraryColumn />);
  expect(screen.getByRole('button', { name: 'Export' })).toBeDisabled();
});
it('gives the disabled Export the unavailability reason as its tooltip', async () => {
  vi.mocked(getHost().exporter.status).mockResolvedValue({ available: false, reason: 'The native encoder could not be loaded: test' });
  render(<LibraryColumn />);
  const button = screen.getByRole('button', { name: 'Export' });
  await waitFor(() => expect(button).toHaveAttribute('title', 'The native encoder could not be loaded: test'));
  expect(button).toBeDisabled();
});
it('collapsed shows only the edge tab, which restores the width', () => {
  useAppStore.setState({ columnWidth: 0, lastColumnWidth: 2 });
  render(<LibraryColumn />);
  expect(screen.queryByTestId('library-column')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Show photos' }));
  expect(useAppStore.getState().columnWidth).toBe(2);
});
it('drag on the edge snaps to whole columns on release', () => {
  Object.defineProperty(window, 'innerWidth', { value: 1400, configurable: true });
  render(<LibraryColumn />);
  const edge = screen.getByRole('separator', { name: 'Resize photos' });
  const pointer = (type: string, clientX: number) =>
    fireEvent(edge, Object.assign(new MouseEvent(type, { bubbles: true, clientX, button: 0 }), { pointerId: 1 }));
  pointer('pointerdown', 124);
  pointer('pointermove', 360);
  expect(useAppStore.getState().columnWidth).toBe(1);        // nothing applied until release
  expect(screen.getByTestId('library-column')).toHaveAttribute('data-dragging', 'true');
  pointer('pointerup', 360);
  expect(useAppStore.getState().columnWidth).toBe(3);        // (360 - 6) / 118 = 3.0
});
it('re-clamps the width when the window shrinks', () => {
  const innerWidth = Object.getOwnPropertyDescriptor(window, 'innerWidth')!;
  Object.defineProperty(window, 'innerWidth', { value: 1400, configurable: true });
  try {
    useAppStore.setState({ columnWidth: 5, lastColumnWidth: 5 });
    render(<LibraryColumn />);
    Object.defineProperty(window, 'innerWidth', { value: 800, configurable: true });
    fireEvent(window, new Event('resize'));
    expect(useAppStore.getState().columnWidth).toBe(2);       // maxColumns(800)
  } finally {
    Object.defineProperty(window, 'innerWidth', innerWidth);
  }
});
it('scrolls the current cell into view when current changes', () => {
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  render(<LibraryColumn />);
  scroll.mockClear();                                        // the mount scrolls 'a' into view
  act(() => useAppStore.getState().setCurrent('c'));
  expect(scroll).toHaveBeenCalledWith({ block: 'nearest' });
});
it('moves keyboard focus with the current cell', () => {
  render(<LibraryColumn />);
  const cells = screen.getAllByTestId('column-cell');
  cells[0].focus();
  act(() => useAppStore.getState().setCurrent('c'));
  expect(document.activeElement).toBe(cells[2]);
});
it('leaves focus outside the column where it is', () => {
  render(<LibraryColumn />);
  expect(document.activeElement).toBe(document.body);
  act(() => useAppStore.getState().setCurrent('c'));
  expect(document.activeElement).toBe(document.body);
});
it('scrolls the current cell into view when the column opens', () => {
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  useAppStore.setState({ columnWidth: 0 });
  render(<LibraryColumn />);
  scroll.mockClear();
  act(() => useAppStore.getState().setColumnWidth(1));
  expect(scroll).toHaveBeenCalledWith({ block: 'nearest' });
});
