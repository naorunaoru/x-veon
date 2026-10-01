import fs from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import type { LibraryPhoto, PhotoEdit, PhotoFacts, PhotoId } from '@/host';
import { defaultPhotoEdit } from '@/app/photo-edit';
import { photoId } from './ids';
import { listFolder } from './folder';
import type { FolderEntry } from './folder';
import { createRoots } from './roots';
import { writeFileAtomic, removeIfExists } from './atomic-write';
import { createCache, queuedFacts } from './cache';
import { ensureHead } from './thumbnails';
import { readSidecar, mergeSidecar } from './xmp';

export interface WorkerLibrary {
 list(folderPath: string, folderId: string): AsyncIterable<{ photos: LibraryPhoto[]; registry: [PhotoId, string][] }>;
 register(entries: [PhotoId, string][]): void;
 setRoots(realRoots: string[]): void;
 saveEdit(id: PhotoId, edit: PhotoEdit): Promise<void>;
 saveFacts(id: PhotoId, facts: PhotoFacts): Promise<void>;
 thumbnail(id: PhotoId): Promise<{ path: string | null; facts: PhotoFacts | null }>;
}

function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }

async function sidecarText(target: string): Promise<string | null> {
  try {
    const stat = await fs.lstat(target);
    if (stat.isSymbolicLink()) throw new Error('The sidecar is a symlink');
    if (!stat.isFile()) throw new Error('The sidecar is not a regular file');
    return await fs.readFile(target, 'utf8');
  } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null; throw error; }
}

export function createWorkerLibrary(opts: { sessionKey: Buffer; cacheDir: string }): WorkerLibrary {
  const sessionKey = Buffer.from(opts.sessionKey);
  const registry = new Map<PhotoId, string>();
  const roots = createRoots();
  const cache = createCache({ dir: opts.cacheDir });
  function registered(id: PhotoId): string {
    const file = registry.get(id);
    if (!file) throw new Error('The photo is not registered');
    return file;
  }
  async function currentEntry(id: PhotoId): Promise<FolderEntry> {
    const file = registered(id);
    const stat = await fs.stat(file);
    return { path: file, name: path.basename(file), size: stat.size, mtimeMs: stat.mtimeMs };
  }
  return {
    async *list(folderPath, _folderId) {
      const entries = await listFolder(folderPath);
      let folderNote: string | null = null;
      try { await fs.access(folderPath, constants.W_OK); }
      catch (error) { folderNote = `This folder can't be written: ${message(error)}`; }
      for (let offset = 0; offset < entries.length; offset += 250) {
        const photos: LibraryPhoto[] = [];
        const batchRegistry: [PhotoId, string][] = [];
        for (const entry of entries.slice(offset, offset + 250)) {
          const id = photoId(sessionKey, entry.path);
          let edit = defaultPhotoEdit();
          let editing: LibraryPhoto['editing'] = folderNote ? 'session' : 'saved';
          let editingNote = folderNote;
          try {
            const state = readSidecar(await sidecarText(entry.path + '.xmp'));
            if (state.kind === 'ok' || state.kind === 'newer') edit = state.edit;
            if (state.kind === 'unreadable' || state.kind === 'newer') {
              editing = 'view-only';
              editingNote = state.kind === 'unreadable' ? `Sidecar is unreadable: ${state.reason}` : `Sidecar uses newer schema ${state.schemaVersion}`;
            }
          } catch (error) { editing = 'view-only'; editingNote = `Sidecar is unreadable: ${message(error)}`; }
          const facts = await cache.getFacts(cache.key(entry)) ?? queuedFacts();
          photos.push({ id, name: entry.name.replace(/\.[^.]+$/, ''), originalName: entry.name, fileSize: entry.size, sourceVersion: cache.key(entry), thumbnailUrl: `xveon-photo://thumb/${id}`, edit, facts, editing, editingNote });
          registry.set(id, entry.path);
          batchRegistry.push([id, entry.path]);
        }
        yield { photos, registry: batchRegistry };
      }
    },
    register(entries) {
      for (const [id, file] of entries) {
        if (photoId(sessionKey, file) !== id) throw new Error('Photo ID does not match its registered path');
      }
      for (const [id, file] of entries) registry.set(id, path.resolve(file));
    },
    setRoots: roots.set,
    async saveEdit(id, edit) {
      const raw = registered(id);
      const checked = await roots.checkWrite(raw);
      await fs.access(checked.realDir, constants.W_OK);
      const existing = await sidecarText(checked.sidecarPath);
      const merged = mergeSidecar(existing, edit);
      if (merged === existing) return;
      const current = await roots.checkWrite(raw);
      if (current.sidecarPath !== checked.sidecarPath || current.dev !== checked.dev || current.ino !== checked.ino) throw new Error('The photo is no longer in its folder');
      if (merged === null) await removeIfExists(current.sidecarPath, checked);
      else await writeFileAtomic(current.sidecarPath, merged, { directory: checked });
    },
    async saveFacts(id, facts) {
      await cache.putFacts(cache.key(await currentEntry(id)), facts);
      await cache.evict();
    },
    async thumbnail(id) {
      const entry = await currentEntry(id);
      const head = await ensureHead(entry, cache);
      const facts = await cache.getFacts(cache.key(entry));
      await cache.evict();
      return { path: head.thumbnail, facts };
    },
  };
}
