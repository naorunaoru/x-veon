import { expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { fakeHost } from '@/test/fake-host';
import { setHost } from '@/app/services/host';
import { useExport } from './useExport';
it('reports an unavailable host exporter so the UI can disable export and explain why', async () => {
  const host = fakeHost();
  vi.mocked(host.exporter.status).mockResolvedValue({ available: false, reason: 'Encoder unavailable' });
  setHost(host);
  const { result } = renderHook(() => useExport());
  await waitFor(() => expect(result.current.exportAvailable).toBe(false));
  await waitFor(() => expect(result.current.exportError).toBe('Encoder unavailable'));
});
