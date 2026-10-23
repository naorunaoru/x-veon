import { beforeEach, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
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
let host: ReturnType<typeof fakeHost>;
beforeEach(() => {
  vi.clearAllMocks();
  host = fakeHost();
  setHost(host);
  useAppStore.setState({
    files: [fromLibraryPhoto(fakePhoto())],
    currentId: 'a',
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
it('disables selected edits for view-only photos', () => {
  useAppStore.setState({
    files: [{ ...fromLibraryPhoto(fakePhoto()), editing: 'view-only', editingNote: 'newer schema' }],
  });
  render(<SettingsPanel />);
  expect(screen.getByLabelText('Photo demosaic method')).toBeDisabled();
  expect(screen.getByLabelText('Default demosaic method')).not.toBeDisabled();
});

it('shows the host supplied export-unavailable reason in Settings', async () => {
  vi.mocked(host.exporter.status).mockResolvedValue({ available: false, reason: 'The native encoder could not be loaded: test' });
  render(<SettingsPanel />);
  expect(await screen.findByText('The native encoder could not be loaded: test')).toBeVisible();
  expect(screen.getByText('Export')).toBeVisible();
});
it('omits the unavailable readout when the web exporter is available', async () => {
  await act(async () => { render(<SettingsPanel />); });
  expect(screen.queryByText('Export')).toBeNull();
  expect(screen.queryByText('The native encoder could not be loaded: test')).toBeNull();
});
