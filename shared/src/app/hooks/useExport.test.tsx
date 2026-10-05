import { expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { fakeHost } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { useExport } from './useExport';
it('reports an unavailable host exporter so the UI can disable export and explain why', async () => {
  const host = fakeHost();
  vi.mocked(host.exporter.status).mockResolvedValue({ available: false, reason: 'Encoder unavailable' });
  setHost(host);
  const { result } = renderHook(() => useExport());
  await waitFor(() => expect(result.current.exportAvailable).toBe(false));
  await waitFor(() => expect(result.current.unavailableReason).toBe('Encoder unavailable'));
});

it.each(['web', 'desktop'])('retries a transient %s capability failure while mounted and clears the reason', async (platform) => {
  const host = fakeHost();
  // Exercise the optional capability difference without inspecting host identity.
  if (platform === 'desktop') host.exporter.reveal = { label: 'Show in Finder', open: async () => {} };
  else delete host.exporter.reveal;
  vi.mocked(host.exporter.status).mockRejectedValueOnce(new Error('Temporarily unavailable')).mockResolvedValue({ available: true });
  setHost(host);
  const { result } = renderHook(() => useExport());
  await waitFor(() => expect(result.current.unavailableReason).toBe('Temporarily unavailable'));
  act(() => result.current.retry());
  await waitFor(() => expect(result.current.exportAvailable).toBe(true));
  expect(result.current.unavailableReason).toBeNull();
});
it('ignores obsolete status responses after retry and unmount', async () => {
  const host = fakeHost();
  let old!: (value: { available: false; reason: string }) => void;
  vi.mocked(host.exporter.status).mockImplementationOnce(() => new Promise(resolve => { old = resolve; })).mockResolvedValue({ available: true });
  setHost(host);
  const { result, unmount } = renderHook(() => useExport());
  act(() => result.current.retry());
  await waitFor(() => expect(result.current.exportAvailable).toBe(true));
  await act(async () => old({ available: false, reason: 'Stale failure' }));
  expect(result.current.unavailableReason).toBeNull();
  unmount();
});

it('ignores completion after unmount during a retry', async () => {
 const host = fakeHost(); let finish!: (value: { available: true }) => void;
 vi.mocked(host.exporter.status).mockResolvedValueOnce({ available: false, reason: 'Retry me' }).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
 setHost(host); const { result, unmount } = renderHook(() => useExport());
 await waitFor(() => expect(result.current.unavailableReason).toBe('Retry me'));
 act(() => result.current.retry()); unmount(); await act(async () => finish({ available: true }));
 expect(result.current.exportAvailable).toBe(false);
});
