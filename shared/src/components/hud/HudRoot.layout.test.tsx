import { beforeEach, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useAppStore } from '@/app/store';
import { fromLibraryPhoto } from '@/app/store/photo';
import { fakePhoto } from '@/test/fake-host';
import { HudRoot } from './HudRoot';
vi.mock('@/components/OutputCanvas', () => ({ OutputCanvas: () => null }));
beforeEach(() => useAppStore.setState({ files: [], currentId: null, initialized: true, initError: null, backend: 'webgpu', columnWidth: 1, lastColumnWidth: 1 }));

it('shows the loading text in the empty state before initialisation', () => {
  useAppStore.setState({ initialized: false });
  render(<HudRoot />);
  expect(screen.getByText('Loading models and WASM…')).toBeInTheDocument();
  expect(screen.queryByText('Drop RAW files to start')).toBeNull();
});
it('lays the column beside the stage once photos exist, with no filmstrip or action HUD', () => {
  const file = { ...fromLibraryPhoto(fakePhoto()), status: 'done' as const };
  useAppStore.setState({ files: [file], currentId: 'a' });
  const { container } = render(<HudRoot />);
  const root = container.querySelector('.xv-hud-root')!;
  expect(root.children[0]).toHaveAttribute('data-testid', 'library-column');
  expect(root.children[1]).toHaveAttribute('data-testid', 'hud-stage');
  expect(container.querySelector('.xv-filmstrip')).toBeNull();
  expect(container.querySelector('.xv-actionhud')).toBeNull();
  expect(container.querySelector('.xv-status-pill')).toBeNull();
  expect(screen.getByTestId('hud-stage').querySelector('.xv-hud-overlay')).not.toBeNull();
});
it('shows no column and no backend string with an empty library', () => {
  const { container } = render(<HudRoot />);
  expect(screen.queryByTestId('library-column')).toBeNull();
  expect(container.textContent).not.toMatch(/webgpu/);
});
