import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { exportFormatInfo } from '@/lib/catalog';
import type { ExportFormat } from '@/lib/types';
import type { PhotoId } from '@/host';

export function createExportDestinations(deps: {
  showSaveDialog(options: Electron.SaveDialogOptions): Promise<Electron.SaveDialogReturnValue>;
  rawPath(photoId: PhotoId): string | undefined;
  register(token: string, target: string): Promise<void>;
  reveal(target: string): void;
  fallbackDir(): string;
  fixedDir?: string;
}) {
  const issued = new Map<string, string>();
  return {
    async choose(photoId: PhotoId, format: ExportFormat): Promise<{ token: string; name: string } | null> {
      // Also protect direct callers: fixture destinations must always be single leaves.
      if (!/^[A-Za-z0-9_-]{22}$/.test(photoId) || !['jpeg-hdr', 'avif', 'tiff'].includes(format)) throw new Error('Invalid export destination');
      const info = exportFormatInfo(format);
      const raw = deps.rawPath(photoId);
      const suggested = `${raw ? path.parse(raw).name : 'export'}.${info.ext}`;
      let target: string;
      if (deps.fixedDir) target = path.join(deps.fixedDir, `${photoId}-${format}.${info.ext}`);
      else {
        const result = await deps.showSaveDialog({
          defaultPath: path.join(raw ? path.dirname(raw) : deps.fallbackDir(), suggested),
          filters: [{ name: info.label, extensions: [info.ext] }],
          properties: ['createDirectory', 'showOverwriteConfirmation'],
        });
        if (result.canceled || !result.filePath) return null;
        target = result.filePath;
      }
      const token = randomUUID();
      await deps.register(token, target);
      issued.set(token, target);
      if (issued.size > 50) issued.delete(issued.keys().next().value!);
      return { token, name: path.basename(target) };
    },
    reveal(token: string) { const target = issued.get(token); if (target) deps.reveal(target); },
  };
}
