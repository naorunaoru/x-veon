import { expect, it, vi } from 'vitest';
import { createOpfsStorage } from './opfs-storage';
it.each(['stable', 'beta', 'dev'])('removes the exact OPFS targets for %s', async (channel) => {
  const removeEntry = vi.fn(async (_name: string, _options: { recursive: boolean }) => {});
  Object.defineProperty(navigator, 'storage', {
    configurable: true,
    value: { getDirectory: async () => ({ removeEntry }) },
  });
  await createOpfsStorage(channel).clear(channel === 'stable');
  expect(removeEntry.mock.calls.map((call) => call[0])).toEqual(
    channel === 'stable' ? ['stable', 'raw', 'thumbnails', 'hwc-cache'] : [channel],
  );
});
it('ignores missing folders but rejects storage errors', async () => {
  const removeEntry = vi
    .fn()
    .mockRejectedValueOnce(new DOMException('absent', 'NotFoundError'))
    .mockRejectedValueOnce(new Error('permission denied'));
  Object.defineProperty(navigator, 'storage', {
    configurable: true,
    value: { getDirectory: async () => ({ removeEntry }) },
  });
  const storage = createOpfsStorage('beta');
  await expect(storage.clear()).resolves.toBeUndefined();
  await expect(storage.clear()).rejects.toThrow('permission denied');
});
