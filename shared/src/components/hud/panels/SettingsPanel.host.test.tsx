import { beforeEach, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '@/app/store';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { setHost } from '@/app/services/host';
vi.mock('@/app/hooks/useModelSizes', () => ({
  useModelSizes: () => ({
    available: new Set(['S', 'M']),
    modelFor: (size: 'S' | 'M') => ({ size, sha256: size }),
  }),
}));
vi.mock('@/app/services/library', () => ({ clearLibrary: vi.fn(async () => {}) }));
import { clearLibrary } from '@/app/services/library';
let host: ReturnType<typeof fakeHost>;
beforeEach(() => {
  vi.clearAllMocks();
  host = fakeHost();
  setHost(host);
  useAppStore.setState({
    files: [fromLibraryPhoto(fakePhoto())],
    selectedFileId: 'a',
    modelSize: 'S',
    demosaicMethod: 'neural-net',
  });
});
it('separates selected-photo model/method from defaults', () => {
  render(<SettingsPanel />);
  fireEvent.change(screen.getByLabelText('Photo demosaic method'), { target: { value: 'dht' } });
  expect(useAppStore.getState().files[0].edit.demosaicMethod).toBe('dht');
  expect(useAppStore.getState().demosaicMethod).toBe('neural-net');
  fireEvent.change(screen.getByLabelText('Default model'), { target: { value: 'M' } });
  expect(useAppStore.getState().modelSize).toBe('M');
  expect(useAppStore.getState().files[0].edit.model).toBeNull();
});
it('shows clear only as a capability and confirms removal of settings', async () => {
  host.library.clear = vi.fn();
  render(<SettingsPanel />);
  fireEvent.click(screen.getByRole('button', { name: 'Clear library' }));
  expect(
    screen.getByText('Removes every photo, edit and setting this build has stored in this browser.'),
  ).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(clearLibrary).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Clear library' }));
  fireEvent.click(screen.getByRole('button', { name: 'Remove photos, edits and settings' }));
  await waitFor(() => expect(clearLibrary).toHaveBeenCalledTimes(1));
});
it('omits absent capabilities and disables selected edits for view-only photos', () => {
  useAppStore.setState({
    files: [{ ...fromLibraryPhoto(fakePhoto()), editing: 'view-only', editingNote: 'newer schema' }],
  });
  render(<SettingsPanel />);
  expect(screen.queryByRole('button', { name: 'Clear library' })).toBeNull();
  expect(screen.getByLabelText('Photo demosaic method')).toBeDisabled();
  expect(screen.getByLabelText('Default demosaic method')).not.toBeDisabled();
});

it('shows the host supplied export-unavailable reason in Settings', async () => {
  vi.mocked(host.exporter.status).mockResolvedValue({ available: false, reason: 'Desktop export arrives in M3.' });
  render(<SettingsPanel />);
  expect(await screen.findByText('Desktop export arrives in M3.')).toBeVisible();
  expect(screen.getByText('Export')).toBeVisible();
});
it('omits the unavailable readout when the web exporter is available', async () => {
  await act(async () => { render(<SettingsPanel />); });
  expect(screen.queryByText('Export')).toBeNull();
  expect(screen.queryByText('Desktop export arrives in M3.')).toBeNull();
});
