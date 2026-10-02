import { readFileSync } from 'node:fs';
import { URL as NodeURL } from 'node:url';
const hudCss = readFileSync(new NodeURL('./HudRoot.css', import.meta.url), 'utf8');
const dropCss = readFileSync(new NodeURL('./DropSurface.css', import.meta.url), 'utf8');
import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { HudRoot } from './HudRoot';
vi.mock('@/components/OutputCanvas', () => ({ OutputCanvas: () => null }));
vi.mock('@/app/hooks/useModelSizes', () => ({
  useModelSizes: () => ({ available: new Set(['S']), modelFor: vi.fn() }),
}));
beforeEach(() => {
  const host = fakeHost();
  host.library.clear = vi.fn();
  setHost(host);
  useAppStore.setState({
    files: [],
    selectedFileId: null,
    initialized: true,
    initError: null,
    openPanel: null,
  });
});
it.each(['empty', 'decode-error', 'startup-error'] as const)(
  'keeps Settings and Clear reachable in %s state',
  (state) => {
    if (state === 'decode-error') {
      const file = fromLibraryPhoto(fakePhoto());
      file.status = 'error';
      file.error = 'bad RAW';
      useAppStore.setState({ files: [file], selectedFileId: file.id });
    }
    if (state === 'startup-error') useAppStore.setState({ initialized: false, initError: 'No GPU' });
    render(<HudRoot />);
    const settings = screen.getByRole('button', { name: 'Settings' });
    expect(settings.closest('[data-chrome-hidden="true"], [inert]')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Exposure' })).toBeNull();
    fireEvent.click(settings);
    const clear = screen.getByRole('button', { name: 'Clear library' });
    expect(clear.closest('[data-chrome-hidden="true"], [inert]')).toBeNull();
    fireEvent.click(clear);
    expect(
      screen.getByText('Removes every photo, edit and setting this build has stored in this browser.'),
    ).toBeVisible();
  },
);
it('keeps the Clear error and retry controls mounted when a partial failure empties the library', async () => {
  const host = fakeHost();
  host.library.clear = vi.fn().mockRejectedValueOnce(new Error('OPFS denied')).mockResolvedValue(undefined);
  setHost(host);
  const file = fromLibraryPhoto(fakePhoto());
  file.status = 'error';
  useAppStore.setState({ files: [file], selectedFileId: file.id });
  render(<HudRoot />);
  fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear library' }));
  fireEvent.click(screen.getByRole('button', { name: 'Remove photos, edits and settings' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('OPFS denied');
  fireEvent.click(screen.getByRole('button', { name: 'Remove photos, edits and settings' }));
  await vi.waitFor(() => expect(host.library.clear).toHaveBeenCalledTimes(2));
});

it('places only the recovery controls above the empty drop surface', () => {
  const style = document.createElement('style');
  style.textContent = hudCss + dropCss;
  document.head.append(style);
  try {
    const { container } = render(<HudRoot />);
    const settings = screen.getByRole('button', { name: 'Settings' });
    const overlay = settings.closest('.xv-hud-overlay')!;
    const drop = container.querySelector('.xv-drop')!;
    const photoOverlay = container.querySelector('.xv-topbar')!.closest('.xv-hud-overlay')!;
    expect(Number(getComputedStyle(photoOverlay).zIndex)).toBeLessThan(Number(getComputedStyle(drop).zIndex));
    expect(Number(getComputedStyle(overlay).zIndex)).toBeGreaterThan(Number(getComputedStyle(drop).zIndex));
  } finally {
    style.remove();
  }
});

it.each([false, true])('shows the empty web status pill while initialized=%s', initialized => {
 useAppStore.setState({ initialized, backend: initialized ? 'webgpu' : null }); render(<HudRoot />);
 expect(screen.getByText(initialized ? 'webgpu' : /Loading models and WASM/)).toBeVisible();
});
