import { afterEach, expect, it, vi } from 'vitest';
import { realpathSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import fs from 'node:fs/promises';
import os from 'node:os';
import { createFolderStore } from './folders';
const dirs: string[] = [];
afterEach(() => { for (const dir of dirs) rmSync(dir, { recursive: true, force: true }); });
function setup() { const dir = realpathSync(mkdtempSync(path.join(os.tmpdir(), 'folders-'))); dirs.push(dir); return { dir, file: path.join(dir, 'folders.json') }; }
it('persists ordered deduplicated recent folders capped at ten and last', async () => {
  const { dir, file } = setup(); const store = createFolderStore(file);
  for (let i = 0; i < 12; i++) { mkdirSync(path.join(dir, `folder${i}`)); store.remember(path.join(dir, `folder${i}`)); }
  const chosen = store.remember(path.join(dir, 'folder3')); await store.flush();
  expect(store.recent().map(f => f.name)).toEqual(['folder3', 'folder11', 'folder10', 'folder9', 'folder8', 'folder7', 'folder6', 'folder5', 'folder4', 'folder2']);
  expect(store.last()).toEqual(chosen); expect(store.resolve(chosen.id)).toBe(path.join(dir, 'folder3'));
  expect(createFolderStore(file).recent()).toEqual(store.recent());
  const saved = JSON.parse(readFileSync(file, 'utf8')); expect(saved.version).toBe(1); expect(saved.last).toBe(chosen.id); expect(saved.recent[0].openedAt).toEqual(expect.any(String));
});
it('retains recent paths for lazy validation without touching missing/offline folders', async () => {
  const { dir, file } = setup(); const folder = path.join(dir, 'gone'); mkdirSync(folder); const store = createFolderStore(file); store.remember(folder); await store.flush(); rmSync(folder, { recursive: true });
  const restored = createFolderStore(file); expect(restored.recent()).toEqual(store.recent()); expect(restored.last()).toEqual(store.last());
});
it('quarantines corrupt files and starts empty', () => {
  const { file } = setup(); writeFileSync(file, '{invalid'); const store = createFolderStore(file);
  expect(store.recent()).toEqual([]); expect(readFileSync(file + '.bad', 'utf8')).toBe('{invalid');
});
it('surfaces an atomic-write failure and lets a later accepted snapshot recover', async () => {
  const fs = (await import('node:fs/promises')).default;
  const { dir, file } = setup(); mkdirSync(path.join(dir, 'A')); mkdirSync(path.join(dir, 'B'));
  const store = createFolderStore(file); const rename = vi.spyOn(fs, 'rename').mockRejectedValueOnce(Object.assign(new Error('ENOSPC'), { code: 'ENOSPC' }));
  try {
    store.remember(path.join(dir, 'A')); await expect(store.flush()).rejects.toThrow('ENOSPC');
    const last = store.remember(path.join(dir, 'B')); await store.flush(); expect(JSON.parse(readFileSync(file, 'utf8')).last).toBe(last.id);
  } finally { rename.mockRestore(); }
});

it('prunes missing recents independently of an unresolved network entry and refreshes the menu', async () => {
 const { dir, file } = setup(), store = createFolderStore(file);
 const gone = path.join(dir, 'gone'), offline = path.join(dir, 'offline'), current = path.join(dir, 'current');
 mkdirSync(current); store.remember(gone); store.remember(offline); const last = store.remember(current); await store.flush();
 let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
 const original = fs.stat.bind(fs), menu = vi.fn();
 const probe = vi.spyOn(fs, 'stat').mockImplementation(async target => { if (target === offline) { await gate; throw Object.assign(new Error('EHOSTUNREACH'), { code: 'EHOSTUNREACH' }); } return original(target); });
 try {
   store.pruneMissing(menu);
   expect(store.last()).toEqual(last); expect(store.recent()).toHaveLength(3);
   await vi.waitFor(() => expect(menu).toHaveBeenCalledOnce()); await store.flush();
   expect(store.recent().map(f => f.name)).toEqual(['current', 'offline']);
   expect(createFolderStore(file).recent()).toEqual(store.recent()); expect(store.last()).toEqual(last);
 } finally { release(); await gate; probe.mockRestore(); }
});
it('a late missing check cannot delete a re-remembered folder', async () => {
 const { dir, file } = setup(), store = createFolderStore(file), target = path.join(dir, 'folder');
 store.remember(target); await store.flush();
 let reject!: (reason: Error) => void; const gate = new Promise<never>((_, r) => { reject = r; });
 const probe = vi.spyOn(fs, 'stat').mockImplementationOnce(() => gate);
 try {
   const menu = vi.fn(); store.pruneMissing(menu); mkdirSync(target); const latest = store.remember(target);
   reject(Object.assign(new Error('ENOENT'), { code: 'ENOENT' })); await Promise.resolve(); await Promise.resolve(); await store.flush();
   expect(store.last()).toEqual(latest); expect(store.recent()).toEqual([latest]); expect(menu).not.toHaveBeenCalled();
 } finally { probe.mockRestore(); }
});
