import { expect, it } from 'vitest';
import { storageNames } from './channel';
it('keeps the per-channel database and OPFS names', () => {
  for (const channel of ['stable', 'beta', 'dev'] as const)
    expect(storageNames(channel)).toEqual({ dbName: `xveon-${channel}`, opfsRoot: channel });
});
