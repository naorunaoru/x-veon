import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { setTimeout } from 'node:timers/promises';

export interface DirectoryIdentity { realDir: string; dev: number; ino: number }
export async function directoryIdentity(directory: string): Promise<DirectoryIdentity> {
  const realDir = await fs.realpath(directory);
  const stat = await fs.stat(realDir);
  if (!stat.isDirectory()) throw new Error('The photo directory changed');
  return { realDir, dev: stat.dev, ino: stat.ino };
}
export async function checkDirectory(identity: DirectoryIdentity): Promise<void> {
  const current = await directoryIdentity(identity.realDir);
  if (current.realDir !== identity.realDir || current.dev !== identity.dev || current.ino !== identity.ino)
    throw new Error('The photo directory changed');
}

// Portable Node checks reject persistent directory swaps. They cannot atomically exclude
// repeated malicious ancestor renames between these checks and filesystem syscalls.
export async function writeFileAtomic(target: string, data: string, opts?: { renameRetries?: number[]; directory?: DirectoryIdentity }): Promise<void> {
  const directory = opts?.directory ?? await directoryIdentity(path.dirname(target));
  const dir = directory.realDir;
  await checkDirectory(directory);
  const destination = path.join(dir, path.basename(target));
  const temp = path.join(dir, `.${path.basename(target)}.${randomUUID()}.tmp`);
  let owned: { dev: number; ino: number } | undefined;
  let committed = false;
  try {
    const handle = await fs.open(temp, 'wx');
    try {
      owned = await handle.stat();
      await fs.writeFile(handle, data);
      await handle.sync();
    } finally { await handle.close(); }
    const delays = opts?.renameRetries ?? [50, 100, 200, 400, 800];
    for (let attempt = 0; ; attempt++) {
      try {
        await checkDirectory(directory);
        await fs.rename(temp, destination);
        committed = true;
        break;
      } catch (error) {
        if (!['EBUSY', 'EPERM', 'EACCES'].includes((error as NodeJS.ErrnoException).code ?? '') || attempt >= delays.length) throw error;
        await setTimeout(delays[attempt]);
      }
    }
  } finally {
    // A successful rename is the commit point. Later directory changes cannot undo it.
    if (!committed && owned) {
      try {
        await removeIfExists(temp, directory, owned);
      } catch (error) { console.warn('Could not safely clean sidecar temp:', temp, error); }
    }
  }
}

export async function removeIfExists(target: string, directory?: DirectoryIdentity, owned?: { dev: number; ino: number }): Promise<void> {
  const delays = [50, 100, 200, 400, 800];
  for (let attempt = 0; ; attempt++) {
    if (directory) await checkDirectory(directory);
    try {
      // A retry delay can replace the leaf without changing its parent.
      // Cleanup may unlink only the same regular file this task created.
      if (owned) {
        const current = await fs.lstat(target);
        if (!current.isFile() || current.dev !== owned.dev || current.ino !== owned.ino)
          throw new Error('The sidecar temp changed');
      }
      await fs.unlink(target); return;
    }
    catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') return;
      if (!['EBUSY', 'EPERM', 'EACCES'].includes(code ?? '') || attempt >= delays.length) throw error;
      await setTimeout(delays[attempt]);
    }
  }
}
