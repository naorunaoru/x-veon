// shared/src/components/hud/ExportPopover.test.tsx
import { beforeEach, expect, it, vi } from 'vitest';
import { act, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
import { getHost } from '@/app/services/host';
import { useLibraryShortcuts } from '@/app/hooks/useLibraryShortcuts';
vi.mock('@/app/services/export-batch', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/app/services/export-batch')>(),
  startExport: vi.fn(),
}));
import { startExport, useExportBatch } from '@/app/services/export-batch';
import { LibraryColumn } from './LibraryColumn';
const files = ['a', 'b', 'c'].map(id => ({ ...fromLibraryPhoto(fakePhoto(id)), status: 'done' as const, thumbnailUrl: `blob:${id}` }));
beforeEach(() => {
  vi.mocked(startExport).mockClear();
  useExportBatch.setState({ batch: null });
  useAppStore.setState({
    files, filter: 'all', currentId: 'a', selectedIds: ['a'], selectionAnchor: 'a', columnWidth: 1, lastColumnWidth: 1, initialized: true,
    exportFormat: 'jpeg-hdr', exportQuality: 95, exportSaveTo: { mode: 'beside' }, exportOnExists: 'skip',
  });
});
const chevron = () => screen.getByRole('button', { name: 'Export settings' });
const openPopover = () => {
  render(<LibraryColumn />);
  fireEvent.click(chevron());
  return screen.getByRole('dialog', { name: 'Export settings' });
};

it('opens from the chevron and persists format and quality to the store', () => {
  render(<LibraryColumn />);
  expect(chevron()).toBeEnabled();
  expect(screen.queryByRole('dialog', { name: 'Export settings' })).toBeNull();
  fireEvent.click(chevron());
  const popover = screen.getByRole('dialog', { name: 'Export settings' });
  expect(popover).toHaveClass('xv-export-popover');
  expect(chevron()).toHaveAttribute('aria-expanded', 'true');
  const format = within(popover).getByRole('group', { name: 'Format' });
  expect(within(format).getAllByRole('radio').map((radio) => radio.getAttribute('name'))).toEqual(['export-format', 'export-format', 'export-format']);
  expect(within(format).getByLabelText('Ultra HDR JPEG')).toBeChecked();
  fireEvent.click(within(format).getByLabelText('AVIF (BT.2020 / HLG)'));
  expect(useAppStore.getState().exportFormat).toBe('avif');
  expect(within(format).getByLabelText('AVIF (BT.2020 / HLG)')).toBeChecked();
  const quality = within(popover).getByRole('slider', { name: 'Quality' });
  expect(quality).toHaveAttribute('aria-valuenow', '95');
  fireEvent.keyDown(quality, { key: 'ArrowLeft' });
  expect(useAppStore.getState().exportQuality).toBe(94);
});
it('disables Quality for TIFF', () => {
  const popover = openPopover();
  fireEvent.click(within(popover).getByLabelText('TIFF (Linear sRGB)'));
  expect(useAppStore.getState().exportFormat).toBe('tiff');
  expect(popover.querySelector('.xv-slider')).toHaveClass('is-disabled');
  fireEvent.keyDown(within(popover).getByRole('slider', { name: 'Quality' }), { key: 'ArrowRight' });
  expect(useAppStore.getState().exportQuality).toBe(95);
});
it('hides Save to and If exists when the host has no chooseFolder', () => {
  const popover = openPopover();
  expect(within(popover).queryByRole('group', { name: 'Save to' })).toBeNull();
  expect(within(popover).queryByRole('group', { name: 'If the file exists' })).toBeNull();
});
it('shows them with a desktop-like host', async () => {
  const chooseFolder = vi.fn<() => Promise<string | null>>(async () => '/tmp/out');
  getHost().exporter.chooseFolder = chooseFolder;
  const popover = openPopover();
  const saveTo = within(popover).getByRole('group', { name: 'Save to' });
  expect(within(saveTo).getByLabelText('Next to the RAWs')).toBeChecked();
  expect(popover.querySelector('.xv-export-popover__path')).toBeNull();

  chooseFolder.mockResolvedValueOnce(null);                 // a cancelled pick keeps beside
  await act(async () => fireEvent.click(within(saveTo).getByLabelText('Folder…')));
  expect(chooseFolder).toHaveBeenCalledTimes(1);
  expect(useAppStore.getState().exportSaveTo).toEqual({ mode: 'beside' });
  expect(within(saveTo).getByLabelText('Next to the RAWs')).toBeChecked();

  await act(async () => fireEvent.click(within(saveTo).getByLabelText('Folder…')));
  expect(chooseFolder).toHaveBeenCalledTimes(2);
  expect(useAppStore.getState().exportSaveTo).toEqual({ mode: 'folder', path: '/tmp/out' });
  expect(within(saveTo).getByLabelText('Folder…')).toBeChecked();
  const path = popover.querySelector('.xv-export-popover__path') as HTMLElement;
  expect(path).toHaveTextContent('/tmp/out');

  chooseFolder.mockResolvedValueOnce('/tmp/other');
  await act(async () => fireEvent.click(within(path).getByRole('button', { name: 'Change…' })));
  expect(chooseFolder).toHaveBeenCalledTimes(3);
  expect(useAppStore.getState().exportSaveTo).toEqual({ mode: 'folder', path: '/tmp/other' });

  fireEvent.click(within(saveTo).getByLabelText('Next to the RAWs'));
  expect(useAppStore.getState().exportSaveTo).toEqual({ mode: 'beside' });

  const onExists = within(popover).getByRole('group', { name: 'If the file exists' });
  expect(within(onExists).getByLabelText('Skip')).toBeChecked();
  fireEvent.click(within(onExists).getByLabelText('Replace'));
  expect(useAppStore.getState().exportOnExists).toBe('replace');
  expect(within(onExists).getByLabelText('Replace')).toBeChecked();
});
it('its Export button starts the batch for the targets and closes', async () => {
  useAppStore.setState({ selectedIds: ['c', 'a'] });
  const popover = openPopover();
  const button = within(popover).getByRole('button', { name: 'Export 2' });
  expect(button).toHaveClass('xv-btn', 'xv-btn--primary');
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(startExport).toHaveBeenCalledWith(['c', 'a']);
  expect(screen.queryByRole('dialog', { name: 'Export settings' })).toBeNull();
  expect(chevron()).toHaveFocus();
});
it('Escape closes and returns focus to the chevron', () => {
  const popover = openPopover();
  fireEvent.keyDown(popover, { key: 'Escape' });
  expect(screen.queryByRole('dialog', { name: 'Export settings' })).toBeNull();
  expect(chevron()).toHaveFocus();
});
it('an outside click closes it; the chevron toggles it', () => {
  openPopover();
  fireEvent.pointerDown(screen.getByRole('dialog', { name: 'Export settings' }));
  expect(screen.getByRole('dialog', { name: 'Export settings' })).toBeInTheDocument();
  fireEvent.pointerDown(document.body);
  expect(screen.queryByRole('dialog', { name: 'Export settings' })).toBeNull();
  fireEvent.click(chevron());
  fireEvent.pointerDown(chevron());
  fireEvent.click(chevron());
  expect(screen.queryByRole('dialog', { name: 'Export settings' })).toBeNull();
});
it('the library keys are off while it is open', () => {
  renderHook(useLibraryShortcuts);
  render(<LibraryColumn />);
  fireEvent.keyDown(window, { key: 'ArrowRight' });
  expect(useAppStore.getState().currentId).toBe('b');      // the keys work while it is closed
  fireEvent.click(chevron());
  fireEvent.keyDown(window, { key: 'ArrowRight' });
  expect(useAppStore.getState().currentId).toBe('b');
});
