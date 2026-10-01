import fs from 'node:fs/promises';
import path from 'node:path';
import type { DirectoryIdentity } from './atomic-write';

function inside(root: string, file: string): boolean {
  const relative = path.relative(root, file);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export function createRoots() {
  let roots: string[] = [];
  return {
    set(realRoots: string[]): void { roots = realRoots.map(root => path.resolve(root)); },
    async checkWrite(rawPath: string): Promise<DirectoryIdentity & { sidecarPath: string }> {
      let realDir: string;
      let realRaw: string;
      try {
        realDir = await fs.realpath(path.dirname(rawPath));
        realRaw = await fs.realpath(rawPath);
        if (!(await fs.stat(realRaw)).isFile()) throw new Error('The photo is no longer in its folder');
      } catch (error) {
        if (['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code ?? '')) {
          throw new Error('The photo is no longer in its folder');
        }
        throw error;
      }
      if (!roots.some(root => inside(root, realDir) && inside(root, realRaw))) {
        throw new Error('The photo is outside the opened folders');
      }
      const sidecarPath = path.join(realDir, path.basename(rawPath) + '.xmp');
      try {
        const sidecar = await fs.lstat(sidecarPath);
        if (sidecar.isSymbolicLink()) throw new Error('The sidecar is a symlink');
        if (!sidecar.isFile()) throw new Error('The sidecar is not a regular file');
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
      const directory = await fs.stat(realDir);
      return { realDir, sidecarPath, dev: directory.dev, ino: directory.ino };
    },
  };
}
