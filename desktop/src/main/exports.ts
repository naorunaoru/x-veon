import path from 'node:path';
import { randomUUID } from 'node:crypto';
import type { PhotoId, SaveTo } from '@/host';
import { isExportTargetRequest } from '../protocol/security';
import { isLeafName } from '../protocol/rpc';

export function createExportTargets(deps: {
  rawPath(photoId: PhotoId): string | undefined;
  register(token: string, directory: string): Promise<void>;
  showItem(file: string): void;
  chooseFolder(): Promise<string | null>;
  stat(path: string): Promise<{ isDirectory(): boolean }>;
  fixedDir?: string;
}) {
  const issued = new Map<string, string>();
  async function directory(saveTo: SaveTo, photoIds: PhotoId[]): Promise<string> {
    if (deps.fixedDir) return deps.fixedDir;
    if (saveTo.mode === 'beside') {
      const dirs = photoIds.map(id => { const raw = deps.rawPath(id); if (!raw) throw new Error('Unknown photo'); return path.dirname(raw); });
      if (dirs.some(dir => dir !== dirs[0])) throw new Error('Photos must be in one folder to export beside them');
      return dirs[0];
    }
    const dir = saveTo.path;
    try {
      if (!path.isAbsolute(dir)) throw new Error('not an absolute path');
      if (!(await deps.stat(dir)).isDirectory()) throw new Error('not a folder');
    } catch (error) { throw new Error(`Couldn't export to ${dir}: ${error instanceof Error ? error.message : String(error)}`, { cause: error }); }
    return dir;
  }
  return {
    async resolve(saveTo: SaveTo, photoIds: PhotoId[]): Promise<{ token: string } | null> {
      // Also protect direct callers: the bridge validated the same shape.
      if (!isExportTargetRequest(saveTo, photoIds)) throw new Error('Invalid export target');
      const dir = await directory(saveTo, photoIds);
      const token = randomUUID();
      await deps.register(token, dir);
      issued.set(token, dir);
      if (issued.size > 50) issued.delete(issued.keys().next().value!);
      return { token };
    },
    reveal(token: string, name?: string) {
      const dir = issued.get(token);
      if (!dir) return;
      // Without a name the folder itself is shown, never opened.
      if (name === undefined) deps.showItem(dir);
      else if (isLeafName(name)) deps.showItem(path.join(dir, name));
    },
    chooseFolder: () => deps.chooseFolder(),
  };
}
