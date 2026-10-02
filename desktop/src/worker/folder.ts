import fs from 'node:fs/promises';
import path from 'node:path';
import { RAW_EXTENSIONS } from '@/lib/catalog';

export interface FolderEntry { path: string; name: string; size: number; mtimeMs: number }
const natural = new Intl.Collator(undefined, { numeric: true });

export async function listFolder(folderPath: string): Promise<FolderEntry[]> {
  const folder = await fs.realpath(folderPath);
  const entries: FolderEntry[] = [];
  for (const name of await fs.readdir(folder)) {
    if (name.startsWith('._') || !RAW_EXTENSIONS.includes(path.extname(name).toLowerCase())) continue;
    const file = path.join(folder, name);
    try {
      const real = await fs.realpath(file);
      const relative = path.relative(folder, real);
      if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) continue;
      const stat = await fs.stat(file);
      if (stat.isFile()) entries.push({ path: file, name, size: stat.size, mtimeMs: stat.mtimeMs });
    } catch (error) {
      if (!['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code ?? '')) throw error;
    }
  }
  return entries.sort((a, b) => natural.compare(a.name, b.name));
}
