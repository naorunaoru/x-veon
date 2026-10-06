import { expect, it, vi } from 'vitest';
import { createUpdateHost } from './updates';

it('returns a bridge notice and contains bridge failures', async () => {
  const notice = { name: 'X-veon Beta', version: '2026.10.7-beta.1', url: 'https://github.com/naorunaoru/x-veon/releases/tag/beta/2026-10-07' };
  const checkForUpdate = vi.fn().mockResolvedValueOnce(notice).mockRejectedValueOnce(new Error('offline'));
  const host = createUpdateHost({ checkForUpdate });
  await expect(host.check()).resolves.toEqual(notice);
  await expect(host.check()).resolves.toBeNull();
});
