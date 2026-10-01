import path from 'node:path';
import fs, { type FileHandle } from 'node:fs/promises';
import { constants } from 'node:fs';
import { Readable } from 'node:stream';
import { parsePhotoUrl } from '../protocol/photo-url';
import type { PhotoId } from '@/host';
type Deps = { protocol: { handle(scheme: string, handler: (request: { url: string; method: string }) => Promise<Response>): unknown }; registry: ReadonlyMap<PhotoId, string>; roots: readonly string[]; cacheDir: string; thumbnail(id: PhotoId): Promise<string | null> };
export function isInside(folderReal: string, fileReal: string, platform: NodeJS.Platform = process.platform): boolean {
  const paths = platform === 'win32' ? path.win32 : path.posix;
  if (!paths.isAbsolute(folderReal) || !paths.isAbsolute(fileReal)) return false;
  const normalize = (p: string) => platform === 'win32' ? paths.normalize(p).toLowerCase() : paths.normalize(p);
  const relative = paths.relative(normalize(folderReal), normalize(fileReal));
  return relative !== '' && relative !== '..' && !relative.startsWith('..' + paths.sep) && !paths.isAbsolute(relative);
}
export function registerPhotoProtocol(deps: Deps) {
  deps.protocol.handle('xveon-photo', async request => {
    const missing = () => new Response(null, { status: 404 });
    const parsed = parsePhotoUrl(request.url);
    if (request.method !== 'GET' || !parsed || !deps.registry.has(parsed.id)) return missing();
    let handle: FileHandle | undefined;
    try {
      const raw = await fs.realpath(deps.registry.get(parsed.id)!);
      if (!deps.roots.some(root => isInside(root, raw))) return missing();
      const requested = parsed.kind === 'raw' ? raw : await deps.thumbnail(parsed.id);
      if (!requested) return missing();
      const file = await fs.realpath(requested);
      const allowedRoots = parsed.kind === 'raw' ? deps.roots : [await fs.realpath(deps.cacheDir)];
      const contained = (resolved: string) => allowedRoots.some(root => isInside(root, resolved));
      if (!contained(file)) return missing();
      const before = await fs.lstat(file, { bigint: true });
      if (!before.isFile() || before.ino === 0n) return missing();
      // NOFOLLOW protects the final component on POSIX. Identity and path checks
      // also cover ordinary ancestor replacement, including Windows reparse paths.
      const noFollow = process.platform === 'win32' ? 0 : constants.O_NOFOLLOW;
      handle = await fs.open(file, constants.O_RDONLY | noFollow);
      const opened = await handle.stat({ bigint: true });
      if (!opened.isFile() || opened.dev !== before.dev || opened.ino !== before.ino) return missing();
      const finalPath = await fs.realpath(file);
      if (finalPath !== file || !contained(finalPath)) return missing();
      const after = await fs.lstat(file, { bigint: true });
      if (!after.isFile() || after.dev !== opened.dev || after.ino !== opened.ino) return missing();
      // Node has no portable atomic beneath-root open. Repeated adversarial
      // ancestor renames between checks need native APIs; never reopen for streaming.
      const response = new Response(Readable.toWeb(handle.createReadStream()) as ReadableStream, { headers: { 'Access-Control-Allow-Origin': 'app://bundle', 'Content-Type': parsed.kind === 'raw' ? 'application/octet-stream' : 'image/jpeg', 'Content-Length': String(opened.size) } });
      handle = undefined; // The stream now owns and closes this checked descriptor.
      return response;
    } catch { return missing(); }
    finally { await handle?.close().catch(() => {}); }
  });
}
