import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { beforeEach, afterEach, it, expect, vi } from 'vitest';
import { writeFileAtomic, removeIfExists, directoryIdentity } from './atomic-write';
let dir: string; let target: string;
beforeEach(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-atomic-')); target = path.join(dir, 'a.xmp'); await fs.writeFile(target, 'original'); });
afterEach(async () => { vi.restoreAllMocks(); await fs.rm(dir, { recursive: true, force: true }); });
it('replaces atomically in the same directory and removes idempotently', async () => { await writeFileAtomic(target, 'new'); expect(await fs.readFile(target, 'utf8')).toBe('new'); expect(await fs.readdir(dir)).toEqual(['a.xmp']); await removeIfExists(target); await removeIfExists(target); expect(await fs.readdir(dir)).toEqual([]); });
it('preserves the original and cleans temporary files after rename failure', async () => { vi.spyOn(fs, 'rename').mockRejectedValue(Object.assign(new Error('EIO: rename failed'), { code: 'EIO' })); await expect(writeFileAtomic(target, 'new')).rejects.toThrow('EIO: rename failed'); expect(await fs.readFile(target, 'utf8')).toBe('original'); expect(await fs.readdir(dir)).toEqual(['a.xmp']); });
it('retries EBUSY twice and eventually replaces the original', async () => { vi.spyOn(fs, 'rename').mockRejectedValueOnce(Object.assign(new Error('busy'), { code: 'EBUSY' })).mockRejectedValueOnce(Object.assign(new Error('busy'), { code: 'EBUSY' })); await writeFileAtomic(target, 'new', { renameRetries: [1, 1] }); expect(await fs.readFile(target, 'utf8')).toBe('new'); expect(await fs.readdir(dir)).toEqual(['a.xmp']); });
it('propagates ENOSPC from a partial write and cleans up', async () => { const write = fs.writeFile.bind(fs); vi.spyOn(fs, 'writeFile').mockImplementationOnce(async (...args) => { await write(args[0], 'partial'); throw Object.assign(new Error('ENOSPC: no space left on device'), { code: 'ENOSPC' }); }); await expect(writeFileAtomic(target, 'new')).rejects.toThrow('ENOSPC: no space left on device'); expect(await fs.readFile(target, 'utf8')).toBe('original'); expect(await fs.readdir(dir)).toEqual(['a.xmp']); });

it('does not reject a committed write if its parent moves after rename', async () => {
 const rename = fs.rename.bind(fs); const moved = dir + '-moved';
 vi.spyOn(fs, 'rename').mockImplementationOnce(async (...args) => { await rename(...args); await rename(dir, moved); });
 try { await expect(writeFileAtomic(target, 'committed')).resolves.toBeUndefined(); expect(await fs.readFile(path.join(moved, 'a.xmp'), 'utf8')).toBe('committed'); }
 finally { await rename(moved, dir); }
});
it('syncs the owned temp file before publishing it', async () => {
 const open = fs.open.bind(fs); const steps: string[] = [];
 vi.spyOn(fs, 'open').mockImplementation(async (...args) => { const handle = await open(...args); const sync = handle.sync.bind(handle); vi.spyOn(handle, 'sync').mockImplementation(async () => { steps.push('sync'); await sync(); }); return handle; });
 const rename = fs.rename.bind(fs); vi.spyOn(fs, 'rename').mockImplementation(async (...args) => { steps.push('rename'); await rename(...args); });
 await writeFileAtomic(target, 'durable'); expect(steps).toEqual(['sync', 'rename']);
});

it.each(['EBUSY', 'EPERM', 'EACCES'])('retries locked deletion after %s', async code => {
 vi.spyOn(fs, 'unlink').mockRejectedValueOnce(Object.assign(new Error('locked'), { code }));
 await removeIfExists(target); await expect(fs.stat(target)).rejects.toMatchObject({ code: 'ENOENT' }); expect(fs.unlink).toHaveBeenCalledTimes(2);
});

it('rechecks parent identity before retrying a locked deletion', async () => {
 const identity = await directoryIdentity(dir); const moved = dir + '-old';
 vi.spyOn(fs, 'unlink').mockImplementationOnce(async () => {
   await fs.rename(dir, moved); await fs.mkdir(dir); await fs.writeFile(target, 'replacement directory');
   throw Object.assign(new Error('locked'), { code: 'EPERM' });
 });
 try { await expect(removeIfExists(target, identity)).rejects.toThrow('The folder changed while saving'); expect(await fs.readFile(target, 'utf8')).toBe('replacement directory'); expect(await fs.readFile(path.join(moved, 'a.xmp'), 'utf8')).toBe('original'); }
 finally { await fs.rm(moved, { recursive: true, force: true }); }
});

it('preserves a replacement temp leaf across a cleanup retry and retains the save error', async () => {
 const rename = fs.rename.bind(fs);
 const saveError = Object.assign(new Error('EIO: original rename failure'), { code: 'EIO' });
 vi.spyOn(fs, 'rename').mockRejectedValue(saveError);
 const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
 let temp = '', ownedInode = 0, replacementInode = 0;
 const unlink = vi.spyOn(fs, 'unlink').mockImplementationOnce(async file => {
   temp = String(file); ownedInode = (await fs.lstat(temp)).ino;
   await rename(temp, temp + '.retained');
   await fs.writeFile(temp, 'replacement must survive', { flag: 'wx' });
   replacementInode = (await fs.lstat(temp)).ino;
   throw Object.assign(new Error('busy owned temp'), { code: 'EBUSY' });
 });
 await expect(writeFileAtomic(target, 'owned edit')).rejects.toBe(saveError);
 expect(replacementInode).not.toBe(ownedInode);
 expect(await fs.readFile(temp, 'utf8')).toBe('replacement must survive');
 expect(await fs.readFile(temp + '.retained', 'utf8')).toBe('owned edit');
 expect(await fs.readFile(target, 'utf8')).toBe('original');
 expect(unlink).toHaveBeenCalledTimes(1);
 expect(warn).toHaveBeenCalledExactlyOnceWith('Could not safely clean sidecar temp:', temp, expect.objectContaining({ message: 'The sidecar temp changed' }));
});

it('writes exact binary bytes', async () => { await writeFileAtomic(target, new Uint8Array([0, 128, 255])); expect(await fs.readFile(target)).toEqual(Buffer.from([0, 128, 255])); });
it('reports a moved folder while preserving ENOENT and cause', async () => {
 const directory = await directoryIdentity(dir), moved = dir + '-moved'; await fs.rename(dir, moved);
 try { await expect(writeFileAtomic(target, 'new', { directory })).rejects.toMatchObject({ message: expect.stringMatching(/The folder changed while saving.*ENOENT/), code: 'ENOENT', cause: expect.objectContaining({ code: 'ENOENT' }) }); }
 finally { await fs.rename(moved, dir); }
});
it('cancels a partial write before rename and cleans its own temp', async () => {
 const abort = new AbortController(); const write = fs.writeFile.bind(fs);
 vi.spyOn(fs, 'writeFile').mockImplementationOnce(async (...args) => { await write(...args); abort.abort(new Error('Export cancelled')); });
 await expect(writeFileAtomic(target, 'new', { signal: abort.signal })).rejects.toThrow('Export cancelled');
 expect(await fs.readFile(target, 'utf8')).toBe('original'); expect(await fs.readdir(dir)).toEqual(['a.xmp']);
});
it('checks cancellation after directory validation and before rename', async () => {
 const abort = new AbortController(); const stat = fs.stat.bind(fs); let calls = 0;
 vi.spyOn(fs, 'stat').mockImplementation(async (...args) => { const result = await stat(...args); if (++calls === 3) abort.abort(new Error('Export cancelled')); return result; });
 await expect(writeFileAtomic(target, 'new', { signal: abort.signal })).rejects.toThrow('Export cancelled');
 expect(await fs.readFile(target, 'utf8')).toBe('original'); expect(await fs.readdir(dir)).toEqual(['a.xmp']);
});
it('retains a successful commit when cancelled after rename', async () => {
 const abort = new AbortController(); const rename = fs.rename.bind(fs);
 vi.spyOn(fs, 'rename').mockImplementationOnce(async (...args) => { await rename(...args); abort.abort(new Error('Export cancelled')); });
 await expect(writeFileAtomic(target, 'committed', { signal: abort.signal })).resolves.toBeUndefined();
 expect(await fs.readFile(target, 'utf8')).toBe('committed');
});

it('replaces a legal near-limit filename on the real filesystem', async () => {
 const name = 'a'.repeat(235) + '.avif', longTarget = path.join(dir, name);
 await fs.writeFile(longTarget, 'original');
 await writeFileAtomic(longTarget, new Uint8Array([1, 2, 3]));
 expect(await fs.readFile(longTarget)).toEqual(Buffer.from([1, 2, 3]));
 expect((await fs.readdir(dir)).sort()).toEqual(['a.xmp', name].sort());
});
