import { readFileSync, renameSync, realpathSync, statSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import type { FolderRef } from '@/host';
import { writeFileAtomic } from '../worker/atomic-write';
type Entry = FolderRef & { path: string; openedAt: string };
export function folderId(realPath: string): string {
  return createHash('sha256').update(process.platform === 'win32' ? realPath.toLowerCase() : realPath).digest('base64url').slice(0, 22);
}
export function createFolderStore(file: string) {
  let entries: Entry[] = [], lastId: string | null = null;
  let pending = Promise.resolve();
  try {
    const data = JSON.parse(readFileSync(file, 'utf8'));
    if (data?.version !== 1 || !Array.isArray(data.recent) || !(data.last === null || typeof data.last === 'string') || !data.recent.every((e: Entry) => e && ['id', 'name', 'path', 'openedAt'].every(key => typeof e[key as keyof Entry] === 'string'))) throw new Error('Invalid folder store');
    const seen = new Set<string>();
    entries = data.recent.filter((entry: Entry) => {
      try { if (!statSync(entry.path).isDirectory() || seen.has(entry.path)) return false; seen.add(entry.path); return true; } catch { return false; }
    }).slice(0, 10);
    lastId = entries.some(e => e.id === data.last) ? data.last : null;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') renameSync(file, file + '.bad');
  }
  const ref = ({ id, name }: Entry): FolderRef => ({ id, name });
  return {
    recent: () => entries.map(ref),
    resolve: (id: string) => entries.find(e => e.id === id)?.path ?? null,
    last: () => { const entry = entries.find(e => e.id === lastId); return entry ? ref(entry) : null; },
    remember(folder: string): FolderRef {
      const realPath = realpathSync(folder);
      const entry: Entry = { id: folderId(realPath), path: realPath, name: path.basename(realPath) || realPath, openedAt: new Date().toISOString() };
      entries = [entry, ...entries.filter(e => e.id !== entry.id)].slice(0, 10); lastId = entry.id;
      const snapshot = JSON.stringify({ version: 1, recent: entries, last: lastId });
      pending = pending.catch(() => {}).then(() => writeFileAtomic(file, snapshot));
      // Keep the error observable through flush without an unhandled rejection.
      void pending.catch(() => {});
      return ref(entry);
    },
    flush: () => pending,
  };
}
