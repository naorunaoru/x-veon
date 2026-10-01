import { beforeEach, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { URL as NodeURL } from 'node:url';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useAppStore } from '@/app/store';
import { setHost } from '@/app/services/host';
import { fakeHost, fakePhoto } from '@/test/fake-host';
import { fromLibraryPhoto } from '@/app/store/photo';
import { TopBar } from './TopBar';
import { HudRoot } from './HudRoot';

const hudCss = readFileSync(new NodeURL('./HudRoot.css', import.meta.url), 'utf8');
const dropCss = readFileSync(new NodeURL('./DropSurface.css', import.meta.url), 'utf8');

const recent = [
  { id: '/photos/2026', name: '2026' },
  { id: '/photos/archive', name: 'Archive' },
];

beforeEach(() => {
  useAppStore.setState({ folder: null, files: [], selectedFileId: null, initialized: true, initError: null });
});

it('keeps the folder menu above the empty drop surface', () => {
  setHost(fakeHost({ openFolder: vi.fn(async () => null), recentFolders: vi.fn(async () => []) }));
  const style = document.createElement('style');
  style.textContent = hudCss + dropCss;
  document.head.append(style);
  try {
    const { container } = render(<HudRoot />);
    const trigger = screen.getByRole('button', { name: 'Open folder…' });
    const overlay = trigger.closest('.xv-hud-overlay')!;
    const drop = container.querySelector('.xv-drop')!;
    expect(trigger.closest('[data-chrome-hidden="true"]')).toBeNull();
    expect(Number(getComputedStyle(overlay).zIndex)).toBeGreaterThan(Number(getComputedStyle(drop).zIndex));
  } finally {
    style.remove();
  }
});

it('keeps folder switching reachable while the selected RAW has an error', () => {
  setHost(fakeHost({ openFolder: vi.fn(async () => null) }));
  const file = fromLibraryPhoto(fakePhoto());
  file.status = 'error';
  useAppStore.setState({ files: [file], selectedFileId: file.id });
  const style = document.createElement('style');
  style.textContent = hudCss;
  document.head.append(style);
  try {
    render(<HudRoot />);
    const trigger = screen.getByRole('button', { name: 'Open folder…' });
    expect(getComputedStyle(trigger.closest('.xv-topbar')!).opacity).not.toBe('0');
  } finally {
    style.remove();
  }
});

it('shows the current folder name and lists recent folders in host order', async () => {
  const host = fakeHost({
    openFolder: vi.fn(async () => ({ photos: [], complete: true, folder: recent[0] })),
    recentFolders: vi.fn(async () => recent),
  });
  setHost(host);
  useAppStore.setState({ folder: recent[0] });
  const user = userEvent.setup();
  render(<TopBar />);

  await user.click(screen.getByRole('button', { name: '2026' }));
  expect(await screen.findAllByRole('menuitem')).toHaveLength(3);
  expect(screen.getAllByRole('menuitem').map(item => item.textContent)).toEqual(['2026', 'Archive', 'Open folder…']);
});

it('opens a recent folder through the shared folder service', async () => {
  const openFolder = vi.fn(async () => ({ photos: [], complete: true, folder: recent[1] }));
  setHost(fakeHost({ openFolder, recentFolders: vi.fn(async () => recent) }));
  const user = userEvent.setup();
  render(<TopBar />);

  await user.click(screen.getByRole('button', { name: 'Open folder…' }));
  await user.click(await screen.findByRole('menuitem', { name: 'Archive' }));

  expect(openFolder).toHaveBeenCalledWith(recent[1]);
  expect(useAppStore.getState().folder).toEqual(recent[1]);
});

it('lets keyboard users choose Open folder… from the menu', async () => {
  const openFolder = vi.fn(async () => ({ photos: [], complete: true, folder: recent[0] }));
  setHost(fakeHost({ openFolder, recentFolders: vi.fn(async () => recent) }));
  const user = userEvent.setup();
  render(<TopBar />);

  const trigger = screen.getByRole('button', { name: 'Open folder…' });
  trigger.focus();
  await user.keyboard('{Enter}');
  await screen.findByRole('menuitem', { name: 'Open folder…' });
  await user.keyboard('{End}{Enter}');

  expect(openFolder).toHaveBeenCalledWith(undefined);
  expect(useAppStore.getState().folder).toEqual(recent[0]);
});

it('renders no folder control when the host cannot open folders', () => {
  setHost(fakeHost());
  render(<TopBar />);
  expect(screen.queryByRole('button', { name: 'Open folder…' })).toBeNull();
  expect(screen.queryByRole('menu')).toBeNull();
});

it('keeps the keyboard choice focused when delayed recent folders arrive', async () => {
  let resolve!: (folders: typeof recent) => void;
  setHost(fakeHost({ openFolder: vi.fn(async () => null), recentFolders: () => new Promise(done => { resolve = done; }) }));
  const user = userEvent.setup(); render(<TopBar />);
  screen.getByRole('button', { name: 'Open folder…' }).focus();
  await user.keyboard('{Enter}{End}');
  const choice = screen.getByRole('menuitem', { name: 'Open folder…' });
  expect(document.activeElement).toBe(choice);
  resolve(recent);
  await screen.findByRole('menuitem', { name: 'Archive' });
  expect(document.activeElement).toBe(choice);
});
