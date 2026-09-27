import { describe, it, expect } from 'vitest';
import { DB_NAME } from './idb-storage';
import { OPFS_ROOT } from './opfs-storage';
import { storageNames, BUILD } from '@/lib/channel';

// Vitest runs without the __XV_BUILD__ define, i.e. as the `dev` channel.
describe('storage namespace follows the build channel', () => {
  it('IndexedDB database name', () => {
    expect(DB_NAME).toBe(storageNames(BUILD.channel).dbName);
    expect(DB_NAME).toBe('xveon-dev');
  });

  it('OPFS root folder', () => {
    expect(OPFS_ROOT).toBe(storageNames(BUILD.channel).opfsRoot);
    expect(OPFS_ROOT).toBe('dev');
  });
});
