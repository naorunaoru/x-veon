import { expect, it } from 'vitest';
import type { Host } from '@/host';
import { createWebHost } from './index';
it('builds a typed web host with browser capabilities and an explicit settings namespace', async () => {
  const host: Host = createWebHost({ deliver: () => {}, reload: () => {} });
  expect(host.settingsDbName).toBe('xveon-dev');
  expect(host.library.remove).toBeTypeOf('function');
  expect(host.library.clear).toBeTypeOf('function');
  expect(host.channelLink).toBeUndefined();
  expect(await host.exporter.status()).toEqual({ available: true });
  const { library: _library, ...incomplete } = host;
  // @ts-expect-error LibraryHost is a required capability group.
  const invalid: Host = incomplete;
  expect(invalid).not.toHaveProperty('library');
});
