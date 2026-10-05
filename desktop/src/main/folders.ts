import { readFileSync, renameSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import type { FolderRef } from '@/host';
import { writeFileAtomic } from '../worker/atomic-write';
type Entry = FolderRef & { path: string; openedAt: string };
export function isMissingFolder(error: unknown): boolean {
  return ['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException)?.code ?? '');
}
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
      if (seen.has(entry.path)) return false; seen.add(entry.path); return true;
    }).slice(0, 10);
    lastId = entries.some(e => e.id === data.last) ? data.last : null;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') renameSync(file, file + '.bad');
  }
  function persist() {
    const snapshot = JSON.stringify({ version: 1, recent: entries, last: lastId });
    pending = pending.catch(() => {}).then(() => writeFileAtomic(file, snapshot));
    // Keep the error observable through flush without an unhandled rejection.
    void pending.catch(() => {});
  }
  function forget(entry: Entry): boolean {
    if (!entries.includes(entry)) return false;
    entries = entries.filter(e => e !== entry);
    if (lastId === entry.id) lastId = null;
    persist(); return true;
  }
  const ref = ({ id, name }: Entry): FolderRef => ({ id, name });
  return {
    recent: () => entries.map(ref),
    resolve: (id: string) => entries.find(e => e.id === id)?.path ?? null,
    last: () => { const entry = entries.find(e => e.id === lastId); return entry ? ref(entry) : null; },
    forgetPath(folder: string): boolean {
      const entry = entries.find(e => e.path === folder);
      return entry ? forget(entry) : false;
    },
    remember(folder: string): FolderRef {
      const realPath = folder;
      const entry: Entry = { id: folderId(realPath), path: realPath, name: path.basename(realPath) || realPath, openedAt: new Date().toISOString() };
      entries = [entry, ...entries.filter(e => e.id !== entry.id)].slice(0, 10); lastId = entry.id;
      persist();
      return ref(entry);
    },
    flush: () => pending,
  };
}
