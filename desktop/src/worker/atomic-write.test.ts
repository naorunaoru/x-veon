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
 try { await expect(removeIfExists(target, identity)).rejects.toThrow('directory changed'); expect(await fs.readFile(target, 'utf8')).toBe('replacement directory'); expect(await fs.readFile(path.join(moved, 'a.xmp'), 'utf8')).toBe('original'); }
 finally { await fs.rm(moved, { recursive: true, force: true }); }
});
