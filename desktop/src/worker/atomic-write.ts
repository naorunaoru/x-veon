import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { setTimeout } from 'node:timers/promises';

export async function writeFileAtomic(target: string, data: string, opts?: { renameRetries?: number[] }): Promise<void> {
  const dir = await fs.realpath(path.dirname(target));
  const destination = path.join(dir, path.basename(target));
  const temp = path.join(dir, `.${path.basename(target)}.${randomUUID()}.tmp`);
  try {
    await fs.writeFile(temp, data, { flag: 'wx' });
    const delays = opts?.renameRetries ?? [50, 100, 200, 400, 800];
    for (let attempt = 0; ; attempt++) {
      try {
        await fs.rename(temp, destination);
        break;
      } catch (error) {
        if (!['EBUSY', 'EPERM', 'EACCES'].includes((error as NodeJS.ErrnoException).code ?? '') || attempt >= delays.length) throw error;
        await setTimeout(delays[attempt]);
      }
    }
  } finally {
    await removeIfExists(temp);
  }
}

export async function removeIfExists(target: string): Promise<void> {
  try { await fs.unlink(target); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
}
