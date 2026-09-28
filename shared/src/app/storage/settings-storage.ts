import { getHost } from '@/app/services/host';
import { transact } from './database';
let paused = false;
const writes = new Set<Promise<void>>();
export async function getSetting<T>(key: string): Promise<T | undefined> {
  const row = await transact(getHost().settingsDbName, 'settings', 'readonly', (store) => store.get(key));
  return row?.value as T | undefined;
}
export function putSetting(key: string, value: unknown): Promise<void> {
  if (paused) return Promise.resolve();
  const saving = transact(getHost().settingsDbName, 'settings', 'readwrite', (store) =>
    store.put({ key, value }),
  ).then(() => {});
  writes.add(saving);
  void saving.then(
    () => writes.delete(saving),
    () => writes.delete(saving),
  );
  return saving;
}
export async function pauseSettings(): Promise<void> {
  paused = true;
  await Promise.allSettled([...writes]);
}
export function resumeSettings(): void {
  paused = false;
}
