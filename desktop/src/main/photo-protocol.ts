import path from 'node:path';
import { realpath, open } from 'node:fs/promises';
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
    try {
      const raw = await realpath(deps.registry.get(parsed.id)!);
      if (!deps.roots.some(root => isInside(root, raw))) return missing();
      const requested = parsed.kind === 'raw' ? raw : await deps.thumbnail(parsed.id);
      if (!requested) return missing();
      const file = await realpath(requested);
      if (parsed.kind === 'thumb' && !isInside(await realpath(deps.cacheDir), file)) return missing();
      const handle = await open(file, 'r');
      const stat = await handle.stat();
      if (!stat.isFile()) { await handle.close(); return missing(); }
      return new Response(Readable.toWeb(handle.createReadStream()) as ReadableStream, { headers: { 'Content-Type': parsed.kind === 'raw' ? 'application/octet-stream' : 'image/jpeg', 'Content-Length': String(stat.size) } });
    } catch { return missing(); }
  });
}
