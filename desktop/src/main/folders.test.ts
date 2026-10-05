import { afterEach, expect, it, vi } from 'vitest';
import { realpathSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
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
