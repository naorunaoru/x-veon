import 'fake-indexeddb/auto';
import { expect, it, vi } from 'vitest';
import { closeDatabase, openDatabase } from './database';
import { getSetting, putSetting } from './settings-storage';
import { setHost } from '@/app/services/host';
import { fakeHost } from '@/test/fake-host';
it('shares both version-1 stores and resets the cached connection on close', async () => {
  const name = 'connection-test';
  const first = await openDatabase(name);
  expect([...first.objectStoreNames]).toEqual(['files', 'settings']);
  expect(first.version).toBe(1);
  expect(await openDatabase(name)).toBe(first);
  await closeDatabase(name);
  expect(await openDatabase(name)).not.toBe(first);
  await closeDatabase(name);
});
it('uses the supplied settings database and does not cross namespaces', async () => {
  setHost({ ...fakeHost(), settingsDbName: 'settings-a' });
  await putSetting('modelSize', 'M');
  setHost({ ...fakeHost(), settingsDbName: 'settings-b' });
  expect(await getSetting('modelSize')).toBeUndefined();
  setHost({ ...fakeHost(), settingsDbName: 'settings-a' });
  expect(await getSetting('modelSize')).toBe('M');
  await closeDatabase('settings-a');
  await closeDatabase('settings-b');
});
it('retries an open after failure and closes on versionchange', async () => {
  const original = indexedDB.open.bind(indexedDB);
  const spy = vi.spyOn(indexedDB, 'open').mockImplementationOnce(() => {
    throw new Error('open failed');
  });
  await expect(openDatabase('retry-test')).rejects.toThrow('open failed');
  spy.mockRestore();
  const db = await openDatabase('retry-test');
  await new Promise<void>((resolve, reject) => {
    const req = original('retry-test', 2);
    req.onsuccess = () => {
      req.result.close();
      resolve();
    };
    req.onerror = () => reject(req.error);
  });
  await expect(openDatabase('retry-test')).rejects.toBeDefined();
});
