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
  try {
    await fs.writeFile(temp, data, { flag: 'wx' });
    const delays = opts?.renameRetries ?? [50, 100, 200, 400, 800];
    for (let attempt = 0; ; attempt++) {
      try {
        await checkDirectory(directory);
        await fs.rename(temp, destination);
        break;
      } catch (error) {
        if (!['EBUSY', 'EPERM', 'EACCES'].includes((error as NodeJS.ErrnoException).code ?? '') || attempt >= delays.length) throw error;
        await setTimeout(delays[attempt]);
      }
    }
  } finally {
    // Never follow a changed parent even while cleaning a failed write.
    await checkDirectory(directory);
    await removeIfExists(temp);
  }
}

export async function removeIfExists(target: string, directory?: DirectoryIdentity): Promise<void> {
  if (directory) await checkDirectory(directory);
  try { await fs.unlink(target); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
}
