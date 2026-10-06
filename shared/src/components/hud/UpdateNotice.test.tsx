import { beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { fakeHost } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { UpdateNotice } from './UpdateNotice';

const notice = { name: 'X-veon Beta', version: '2026.10.7-beta.1', url: 'https://github.com/naorunaoru/x-veon/releases/tag/beta/2026-10-07' };
beforeEach(() => setHost(fakeHost()));
it('announces, links to and dismisses a newer release', async () => {
  setHost({ ...fakeHost(), updates: { check: vi.fn(async () => notice) } });
  render(<UpdateNotice />);
  expect(await screen.findByRole('status')).toHaveTextContent('X-veon Beta 2026.10.7-beta.1 is available.');
  expect(screen.getByRole('link', { name: 'Download' })).toHaveAttribute('href', notice.url);
  expect(screen.getByRole('link', { name: 'Download' })).toHaveAttribute('target', '_blank');
  expect(screen.getByRole('link', { name: 'Download' })).toHaveAttribute('rel', 'noreferrer');
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});
it.each([async () => null, async () => { throw new Error('offline'); }])('stays empty after no release or failure', async check => {
  setHost({ ...fakeHost(), updates: { check } });
  render(<UpdateNotice />);
  await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument());
});
it('stays empty without the update capability', () => {
  render(<UpdateNotice />);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});
