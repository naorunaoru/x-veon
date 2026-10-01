import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import type { PhotoFacts } from '@/host';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { FolderEntry } from './folder';

// One worker owns the cache; share ordering across cache instances for the same file.
const factsWrites = new Map<string, Promise<void>>();

export function queuedFacts(): PhotoFacts {
  return { cfaType: null, metadata: null, resultMeta: null, resultMethod: null, lensProfile: null, status: 'queued', error: null };
}

export function createCache(opts: { dir: string; limitBytes?: number }) {
  const dir = path.resolve(opts.dir);
  const limit = opts.limitBytes ?? 2 * 1024 ** 3;
  const file = (key: string, extension: string) => {
    if (!/^[a-f0-9]{64}$/.test(key)) throw new Error('Invalid cache key');
    return path.join(dir, `${key}.${extension}`);
  };
  async function touch(target: string) {
    const now = new Date();
    try { await fs.utimes(target, now, now); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  }
  async function put(target: string, data: string | Uint8Array) {
    await fs.mkdir(dir, { recursive: true });
    const temp = path.join(dir, `.${randomUUID()}.tmp`);
    try { await fs.writeFile(temp, data); await fs.rename(temp, target); }
    finally { await fs.rm(temp, { force: true }); }
  }
  async function getFacts(key: string): Promise<PhotoFacts | null> {
    const target = file(key, 'json');
    try {
      const facts = JSON.parse(await fs.readFile(target, 'utf8')) as PhotoFacts;
      await touch(target);
      return facts;
    } catch (error) {
      if (error instanceof SyntaxError || (error as NodeJS.ErrnoException).code === 'ENOENT') return null;
      throw error;
    }
  }
  function updateFacts(key: string, merge: (current: PhotoFacts | null) => PhotoFacts): Promise<PhotoFacts> {
    const target = file(key, 'json');
    const update = (factsWrites.get(target) ?? Promise.resolve()).then(async () => {
      const facts = merge(await getFacts(key));
      await put(target, JSON.stringify(facts));
      return facts;
    });
    // A failed write rejects its caller but does not block later updates.
    const settled = update.then(() => {}, () => {});
    factsWrites.set(target, settled);
    void settled.then(() => { if (factsWrites.get(target) === settled) factsWrites.delete(target); });
    return update;
  }
  return {
    key(entry: FolderEntry): string {
      return createHash('sha256').update(`${entry.path}\0${entry.size}\0${entry.mtimeMs}`).digest('hex');
    },
    getFacts,
    async putFacts(key: string, facts: PhotoFacts): Promise<void> {
      await updateFacts(key, current => ({ ...facts, metadata: facts.metadata ?? current?.metadata ?? null }));
    },
    async putHeadMetadata(key: string, metadata: QuickMetadata | null): Promise<PhotoFacts> {
      return updateFacts(key, current => ({ ...(current ?? queuedFacts()), metadata: current?.metadata ?? metadata }));
    },
    thumbPath(key: string): string { return file(key, 'jpg'); },
    async putThumb(key: string, jpeg: Uint8Array): Promise<void> { await put(file(key, 'jpg'), jpeg); },
    async hasThumb(key: string): Promise<boolean> {
      const target = file(key, 'jpg');
      try { const stat = await fs.stat(target); if (!stat.isFile()) return false; await touch(target); return true; }
      catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error; }
    },
    // Kept outside PhotoFacts: null metadata can also mean extraction completed.
    async hasHead(key: string): Promise<boolean> {
      const target = file(key, 'head');
      try { await fs.access(target); await touch(target); return true; }
      catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false; throw error; }
    },
    async markHead(key: string): Promise<void> { await put(file(key, 'head'), ''); },
    async evict(): Promise<void> {
      let names: string[];
      try { names = await fs.readdir(dir); }
      catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return; throw error; }
      const entries = new Map<string, { files: string[]; size: number; used: number }>();
      for (const name of names) {
        const match = /^([a-f0-9]{64})\.(json|jpg|head)$/.exec(name);
        if (!match) continue;
        const target = path.join(dir, name);
        try {
          const stat = await fs.stat(target);
          const entry = entries.get(match[1]) ?? { files: [], size: 0, used: 0 };
          entry.files.push(target); entry.size += stat.size; entry.used = Math.max(entry.used, stat.mtimeMs);
          entries.set(match[1], entry);
        } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
      }
      let total = [...entries.values()].reduce((sum, entry) => sum + entry.size, 0);
      for (const entry of [...entries.values()].sort((a, b) => a.used - b.used)) {
        if (total <= limit) break;
        await Promise.all(entry.files.map(target => fs.rm(target, { force: true })));
        total -= entry.size;
      }
    },
  };
}
