// shared/src/components/hud/LibraryColumn.test.tsx
import { beforeEach, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { filterCounts } from '@/app/rating';
import { fakePhoto } from '@/test/fake-host';
import { getHost } from '@/app/services/host';
vi.mock('@/app/services/export-batch', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/app/services/export-batch')>(),
  startExport: vi.fn(), cancelBatch: vi.fn(), dismissBatch: vi.fn(), retryJob: vi.fn(),
}));
// Pointer events are fired the way Minimap.test.tsx fires them; jsdom has no PointerEvent class.
import { cancelBatch, dismissBatch, retryJob, startExport, useExportBatch } from '@/app/services/export-batch';
import type { BatchJobState, ExportBatch } from '@/app/services/export-batch';
import { LibraryColumn } from './LibraryColumn';
const files = ['a', 'b', 'c'].map(id => ({ ...fromLibraryPhoto(fakePhoto(id)), status: 'done' as const, thumbnailUrl: `blob:${id}` }));
beforeEach(() => {
  for (const mock of [startExport, cancelBatch, dismissBatch, retryJob]) vi.mocked(mock).mockClear();
  useExportBatch.setState({ batch: null });
  useAppStore.setState({ files, filter: 'all', currentId: 'a', selectedIds: ['a'], selectionAnchor: 'a', columnWidth: 1, lastColumnWidth: 1, initialized: true });
});
const batch = (status: ExportBatch['status'], jobs: [fileId: string, state: BatchJobState, error?: string][]): ExportBatch => ({
  id: 'batch', target: { token: 'folder' }, status, closed: false,
  settings: { format: 'avif', quality: 80, saveTo: { mode: 'beside' }, onExists: 'skip' },
  jobs: jobs.map(([fileId, state, error], i) => ({ id: `job-${i}`, fileId, name: `${fileId}.avif`, state, error: error ?? null })),
});

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
it('marks rated and rejected cells', () => {
  useAppStore.setState({ files: [{ ...files[0], rating: 3 }, { ...files[1], rating: -1 }, { ...files[2], rating: 0 }] });
  render(<LibraryColumn />);
  const [rated, rejected, unrated] = screen.getAllByTestId('column-cell');
  expect(rated.querySelector('.xv-cell__stars')).toHaveTextContent('★★★');
  expect(rated).toHaveAttribute('data-rating', '3');
  expect(rated).toHaveAccessibleName('a.raf, rated 3');
  expect(rated.querySelector('.xv-cell__reject')).toBeNull();
  expect(rejected.querySelector('.xv-cell__reject')).not.toBeNull();
  expect(rejected.querySelector('.xv-cell__stars')).toBeNull();
  expect(rejected).toHaveAttribute('data-rating', '-1');
  expect(rejected).toHaveAccessibleName('b.raf, rejected');
  expect(unrated.querySelector('.xv-cell__stars, .xv-cell__reject')).toBeNull();
  expect(unrated).toHaveAttribute('data-rating', '0');
  expect(unrated).toHaveAccessibleName('c.raf');
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
it('the Export button counts the selection and starts a batch for it', async () => {
  useAppStore.setState({ selectedIds: ['c', 'a'] });
  render(<LibraryColumn />);
  const button = screen.getByRole('button', { name: 'Export 2' });
  await waitFor(() => expect(button).toBeEnabled());        // enabled once exporter.status() resolves
  expect(button).toHaveClass('xv-column__export');
  expect(screen.getByRole('button', { name: 'Export settings' })).toBeDisabled();
  fireEvent.click(button);
  expect(startExport).toHaveBeenCalledWith(['c', 'a']);
});
it('Export exports the current photo when nothing is selected', async () => {
  useAppStore.setState({ selectedIds: [], currentId: 'b' });
  render(<LibraryColumn />);
  const button = screen.getByRole('button', { name: 'Export' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(startExport).toHaveBeenCalledWith(['b']);
});
it('disables Export without a target or availability', async () => {
  useAppStore.setState({ selectedIds: [], currentId: null });
  const first = render(<LibraryColumn />);
  await act(async () => {});                                  // exporter.status() resolves available
  expect(screen.getByRole('button', { name: 'Export' })).toBeDisabled();
  first.unmount();
  useAppStore.setState({ selectedIds: ['a'], currentId: 'a', initialized: false });
  const second = render(<LibraryColumn />);
  await act(async () => {});
  expect(screen.getByRole('button', { name: 'Export' })).toBeDisabled();   // the pipeline is not up yet
  second.unmount();
  useAppStore.setState({ initialized: true });
  vi.mocked(getHost().exporter.status).mockResolvedValue({ available: false, reason: 'no encoder' });
  render(<LibraryColumn />);
  await act(async () => {});
  expect(screen.getByRole('button', { name: 'Export' })).toBeDisabled();
});
it('gives the disabled Export the unavailability reason as its tooltip', async () => {
  vi.mocked(getHost().exporter.status).mockResolvedValue({ available: false, reason: 'The native encoder could not be loaded: test' });
  render(<LibraryColumn />);
  const button = screen.getByRole('button', { name: 'Export' });
  await waitFor(() => expect(button).toHaveAttribute('title', 'The native encoder could not be loaded: test'));
  expect(button).toBeDisabled();
});
it('shows n of N, a progress bar and Cancel while running', () => {
  useExportBatch.setState({ batch: batch('running', [['a', 'written'], ['b', 'encoding'], ['c', 'queued']]) });
  render(<LibraryColumn />);
  expect(screen.queryByRole('button', { name: /^Export/ })).toBeNull();
  expect(screen.getByRole('button', { name: 'Select all' })).toBeInTheDocument();
  const status = screen.getByRole('status');
  expect(status).toHaveClass('xv-column__status');
  expect(status).toHaveTextContent('2 of 3');
  const bar = within(status).getByRole('progressbar');
  expect(bar).toHaveAttribute('aria-valuenow', '1');
  expect(bar).toHaveAttribute('aria-valuemax', '3');
  expect(bar.style.getPropertyValue('--xv-progress')).toBe(String(1 / 3));
  fireEvent.click(within(status).getByRole('button', { name: 'Cancel export' }));
  expect(cancelBatch).toHaveBeenCalled();
  act(() => useExportBatch.setState({ batch: batch('running', [['a', 'written'], ['b', 'failed', 'x'], ['c', 'skipped'], ['a', 'cancelled']]) }));
  expect(screen.getByRole('status')).toHaveTextContent('3 of 3');   // clamped to the total; cancelled jobs don't count
});
it('shows N done and Reveal for one file by name', () => {
  const open = vi.fn(async () => {});
  getHost().exporter.reveal = { label: 'Show in Finder', open };
  useExportBatch.setState({ batch: batch('done', [['a', 'written'], ['b', 'cancelled']]) });
  const first = render(<LibraryColumn />);
  expect(screen.getByRole('status')).toHaveTextContent('1 done');
  fireEvent.click(screen.getByRole('button', { name: 'Show in Finder' }));
  expect(open).toHaveBeenCalledWith({ token: 'folder' }, 'a.avif');
  first.unmount();
  // Several files: the first written one is selected in its folder; all skipped: just the folder.
  useExportBatch.setState({ batch: batch('done', [['a', 'skipped'], ['b', 'written'], ['c', 'written']]) });
  const second = render(<LibraryColumn />);
  expect(screen.getByRole('status')).toHaveTextContent('3 done');
  fireEvent.click(screen.getByRole('button', { name: 'Show in Finder' }));
  expect(open).toHaveBeenLastCalledWith({ token: 'folder' }, 'b.avif');
  second.unmount();
  useExportBatch.setState({ batch: batch('done', [['a', 'skipped'], ['b', 'skipped']]) });
  const third = render(<LibraryColumn />);
  fireEvent.click(screen.getByRole('button', { name: 'Show in Finder' }));
  expect(open).toHaveBeenLastCalledWith({ token: 'folder' });
  third.unmount();
  // No target (web): no button.
  useExportBatch.setState({ batch: { ...batch('done', [['a', 'written']]), target: null } });
  render(<LibraryColumn />);
  expect(screen.getByRole('status')).toHaveTextContent('1 done');
  expect(screen.queryByRole('button', { name: 'Show in Finder' })).toBeNull();
});
it('shows failed with Why listing reasons and Retry', () => {
  useExportBatch.setState({ batch: batch('failed', [['a', 'written'], ['b', 'failed', 'disk full'], ['c', 'skipped']]) });
  render(<LibraryColumn />);
  const status = screen.getByRole('status');
  expect(status).toHaveTextContent('2 done · 1 failed');
  expect(screen.queryByRole('dialog')).toBeNull();
  fireEvent.click(within(status).getByRole('button', { name: 'Why' }));
  const dialog = screen.getByRole('dialog', { name: 'Export failures' });
  const rows = Array.from(dialog.querySelectorAll('.name'), (name) => [name.textContent, name.parentElement!.querySelector('.reason')!.textContent]);
  expect(rows).toEqual([['b.avif', 'disk full'], ['c.avif', 'skipped, exists']]);
  const retries = within(dialog).getAllByRole('button', { name: 'Retry' });
  expect(retries).toHaveLength(2);
  fireEvent.click(retries[0]);
  expect(retryJob).toHaveBeenCalledWith('job-1');
  fireEvent.keyDown(document.body, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).toBeNull();
  fireEvent.click(within(status).getByRole('button', { name: 'Why' }));
  fireEvent.pointerDown(document.body);
  expect(screen.queryByRole('dialog')).toBeNull();               // outside pointerdown closes
  fireEvent.click(within(status).getByRole('button', { name: 'Why' }));
  fireEvent.click(within(status).getByRole('button', { name: 'Why' }));
  expect(screen.queryByRole('dialog')).toBeNull();               // Why toggles
  fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
  expect(dismissBatch).not.toHaveBeenCalled();                    // a failed batch stays until ✕
  fireEvent.click(within(status).getByRole('button', { name: 'Dismiss export result' }));
  expect(dismissBatch).toHaveBeenCalledTimes(1);
  act(() => useExportBatch.setState({ batch: batch('failed', [['a', 'failed', 'gone']]) }));
  expect(screen.getByRole('status')).toHaveTextContent(/^1 failed/);
});
it('any click in the foot dismisses a done batch', () => {
  useExportBatch.setState({ batch: batch('done', [['a', 'written']]) });
  render(<LibraryColumn />);
  fireEvent.click(screen.getByRole('status'));
  expect(dismissBatch).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
  expect(dismissBatch).toHaveBeenCalledTimes(2);
  fireEvent.click(screen.getAllByTestId('column-cell')[1]);
  expect(dismissBatch).toHaveBeenCalledTimes(2);                  // outside the foot
  act(() => useExportBatch.setState({ batch: batch('running', [['a', 'queued']]) }));
  fireEvent.click(screen.getByRole('status'));
  expect(dismissBatch).toHaveBeenCalledTimes(2);                  // only a done batch
});
it('Cancel that settles a batch to done shows the result; the next click in the foot dismisses it', async () => {
  // The real store functions, not the mocks.
  const actual = await vi.importActual<typeof import('@/app/services/export-batch')>('@/app/services/export-batch');
  vi.mocked(cancelBatch).mockImplementationOnce(actual.cancelBatch);
  vi.mocked(dismissBatch).mockImplementationOnce(actual.dismissBatch);
  useExportBatch.setState({ batch: batch('running', [['a', 'written'], ['b', 'queued']]) });
  render(<LibraryColumn />);
  fireEvent.click(screen.getByRole('button', { name: 'Cancel export' }));
  expect(useExportBatch.getState().batch?.status).toBe('done');
  expect(screen.getByRole('status')).toHaveTextContent('1 done');
  const line = () => screen.getAllByTestId('column-cell')[0].querySelector('.xv-cell__line');
  expect(line()).toHaveAttribute('data-export', 'done');
  expect(dismissBatch).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('status'));
  expect(dismissBatch).toHaveBeenCalledTimes(1);
  expect(useExportBatch.getState().batch).toBeNull();
  expect(screen.getByRole('button', { name: 'Export' })).toBeInTheDocument();
  expect(line()).toBeNull();
});
it('cells carry export lines', () => {
  useExportBatch.setState({ batch: batch('running', [['a', 'written'], ['b', 'rendering'], ['b', 'failed', 'x']]) });
  render(<LibraryColumn />);
  const lines = () => screen.getAllByTestId('column-cell').map((cell) => cell.querySelector('.xv-cell__line')?.getAttribute('data-export') ?? null);
  expect(lines()).toEqual(['done', 'failed', null]);           // the latest job for a file wins
  act(() => useExportBatch.setState({ batch: batch('running', [['a', 'queued'], ['b', 'processing'], ['c', 'skipped']]) }));
  expect(lines()).toEqual(['queued', 'running', 'done']);
  act(() => useExportBatch.setState({ batch: null }));
  expect(lines()).toEqual([null, null, null]);
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
it('the filter dropdown lists the five filters with counts and applies one', () => {
  useAppStore.setState({ files: [{ ...files[0], rating: 1 }, { ...files[1], rating: 0 }, { ...files[2], rating: 2 }] });
  render(<LibraryColumn />);
  const trigger = screen.getByRole('button', { name: 'Filter photos' });
  expect(trigger).toHaveTextContent('All · 3');
  fireEvent.click(trigger);
  const items = within(screen.getByRole('menu')).getAllByRole('menuitemradio');
  expect(items.map((item) => item.textContent)).toEqual(['All · 3', '★1+ · 2', '★2+ · 1', '★3+ · 0', 'Rejected · 0']);
  expect(items.map((item) => item.getAttribute('aria-checked'))).toEqual(['true', 'false', 'false', 'false', 'false']);
  // Counts come from every file, not the selection (only 'a' is selected).
  expect(filterCounts(useAppStore.getState().files.map((f) => f.rating))).toMatchObject({ all: 3, star1: 2, star2: 1 });
  fireEvent.click(screen.getByRole('menuitemradio', { name: '★2+ · 1' }));
  expect(useAppStore.getState().filter).toBe('star2');
  expect(screen.queryByRole('menu')).toBeNull();
  expect(screen.getAllByTestId('column-cell')).toHaveLength(1);
  expect(trigger).toHaveTextContent('★2+ · 1');
});
it('Escape closes the filter menu and returns focus to the trigger', () => {
  render(<LibraryColumn />);
  const trigger = screen.getByRole('button', { name: 'Filter photos' });
  fireEvent.click(trigger);
  const first = screen.getByRole('menuitemradio', { name: 'All · 3' });
  expect(document.activeElement).toBe(first);
  fireEvent.keyDown(first, { key: 'Escape' });
  expect(screen.queryByRole('menu')).toBeNull();
  expect(document.activeElement).toBe(trigger);
});
