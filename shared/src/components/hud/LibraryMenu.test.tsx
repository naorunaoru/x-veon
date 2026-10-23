import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { setHost } from '@/app/services/host';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { useAppStore } from '@/app/store';
vi.mock('@/app/services/library', () => ({ importFiles: vi.fn(), clearLibrary: vi.fn(async () => {}) }));
import { importFiles, clearLibrary } from '@/app/services/library';
import { LibraryMenu } from './LibraryMenu';
import { HudRoot } from './HudRoot';
vi.mock('@/components/OutputCanvas', () => ({ OutputCanvas: () => null }));
beforeEach(() => { vi.clearAllMocks(); setHost(fakeHost()); useAppStore.setState({ files: [], currentId: null, initialized: true, initError: null }); });

it('renders nothing on a folder host', () => {
  setHost(fakeHost({ openFolder: vi.fn(async () => null) }));
  const { container } = render(<LibraryMenu />);
  expect(container).toBeEmptyDOMElement();
});
it('adds picked files through importFiles', () => {
  const click = vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(() => {});
  render(<LibraryMenu />);
  fireEvent.click(screen.getByRole('button', { name: 'Library' }));
  fireEvent.click(screen.getByRole('menuitem', { name: 'Add files…' }));
  const input = document.querySelector('input[type=file]') as HTMLInputElement;
  expect(click).toHaveBeenCalledTimes(1);
  expect(click.mock.contexts[0]).toBe(input);
  click.mockRestore();
  expect(input.accept).toContain('.raf');
  const file = new File([''], 'x.RAF');
  // jsdom's value ignores fireEvent's files; a browser shows the picked path until it is reset.
  Object.defineProperty(input, 'value', { configurable: true, writable: true, value: 'C:\\fakepath\\x.RAF' });
  fireEvent.change(input, { target: { files: [file] } });
  expect(importFiles).toHaveBeenCalledWith([file]);
  expect(input.value).toBe('');
});
it('offers Clear library only when the host can clear, and confirms', async () => {
  render(<LibraryMenu />);
  fireEvent.click(screen.getByRole('button', { name: 'Library' }));
  expect(screen.queryByRole('menuitem', { name: 'Clear library' })).toBeNull();
  const host = fakeHost(); host.library.clear = vi.fn(); setHost(host);
  render(<LibraryMenu />);
  fireEvent.click(screen.getAllByRole('button', { name: 'Library' })[1]);
  fireEvent.click(screen.getByRole('menuitem', { name: 'Clear library' }));
  expect(screen.getByText('Removes every photo, edit and setting this build has stored in this browser.')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Remove photos, edits and settings' }));
  await waitFor(() => expect(clearLibrary).toHaveBeenCalledTimes(1));
});
it('keeps the failed-Clear notice after the dialog is cancelled', async () => {
  vi.mocked(clearLibrary).mockRejectedValueOnce(new Error('IDB blocked'));
  const host = fakeHost(); host.library.clear = vi.fn(); setHost(host);
  render(<LibraryMenu />);
  fireEvent.click(screen.getByRole('button', { name: 'Library' }));
  fireEvent.click(screen.getByRole('menuitem', { name: 'Clear library' }));
  fireEvent.click(screen.getByRole('button', { name: 'Remove photos, edits and settings' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('IDB blocked');
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(screen.getByRole('alert')).toHaveTextContent('IDB blocked');
});
it.each(['empty', 'decode-error', 'startup-error'] as const)('keeps the Library menu reachable in the %s state', (state) => {
  const host = fakeHost(); host.library.clear = vi.fn(); setHost(host);
  if (state === 'decode-error') {
    const file = { ...fromLibraryPhoto(fakePhoto()), status: 'error' as const, error: 'bad RAW' };
    useAppStore.setState({ files: [file], currentId: 'a' });
  }
  if (state === 'startup-error') useAppStore.setState({ initialized: false, initError: 'No GPU' });
  render(<HudRoot />);
  expect(screen.getByRole('button', { name: 'Library' })).toBeVisible();
});
