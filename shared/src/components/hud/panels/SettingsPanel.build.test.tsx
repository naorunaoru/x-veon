import { setHost } from '@/app/services/host';
import { fakeHost } from '@/test/fake-host';
beforeEach(() =>
  setHost({
    ...fakeHost(),
    build: { channel: 'beta', sha: 'abc1234', date: '2026-09-04' },
    channelLink: { label: 'stable', href: '/x-veon/' },
  }),
);
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '@/app/store';
import { BUILD } from '@/lib/channel';

vi.mock('@/app/hooks/useModelSizes', () => ({
  useModelSizes: () => ({
    available: new Set(['S', 'M', 'L']),
    switchTo: vi.fn().mockResolvedValue(undefined),
  }),
}));
vi.mock('@/app/hooks/useProcessing', () => ({
  useProcessing: () => ({ processFile: vi.fn(), isProcessing: false }),
}));
// Pretend this is a beta build; keep the real helpers.
vi.mock('@/lib/channel', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/channel')>()),
  BUILD: { channel: 'beta', sha: 'abc1234', date: '2026-09-04' },
}));

describe('SettingsPanel build row (beta build)', () => {
  beforeEach(() => {
    delete BUILD.version;
    useAppStore.setState({ openPanel: 'settings', files: [], selectedFileId: null });
  });

  it('shows the channel, sha and date', () => {
    render(<SettingsPanel />);
    const row = screen.getByTestId('xv-build');
    expect(row).toHaveTextContent('Beta');
    expect(row).toHaveTextContent('abc1234');
    expect(row).toHaveTextContent('2026-09-04');
    expect(row).not.toHaveTextContent('2026.10.6-beta.1');
  });

  it('shows the release version between channel and sha', () => {
    BUILD.version = '2026.10.6-beta.1';
    setHost({ ...fakeHost(), build: { ...BUILD }, channelLink: { label: 'stable', href: '/x-veon/' } });
    render(<SettingsPanel />);
    expect(screen.getByTestId('xv-build')).toHaveTextContent('Beta · 2026.10.6-beta.1 · abc1234 · 2026-09-04');
  });

  it('links to the stable channel and says libraries are separate', () => {
    render(<SettingsPanel />);
    const link = screen.getByRole('link', { name: 'Open stable' });
    expect(link).toHaveAttribute('href', '/x-veon/');
    expect(screen.getByTestId('xv-build')).toHaveTextContent('separate library');
  });
});
