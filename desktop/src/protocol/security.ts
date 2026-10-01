import { isListingFrame, isWorkerIdentity } from './rpc';
export function assetName(
  raw: string,
  method: string,
  files: ReadonlySet<string>,
): string | null {
  if (method !== 'GET' || /(?:%2e|%2f|%5c|\\|\/\.\.?\/)/i.test(raw))
    return null;
  try {
    const url = new URL(raw);
    if (
      url.protocol !== 'app:' ||
      url.host !== 'bundle' ||
      url.username ||
      url.password
    )
      return null;
    const name = decodeURIComponent(url.pathname.slice(1)) || 'index.html';
    return files.has(name) ? name : null;
  } catch {
    return null;
  }
}
export function acceptsSender(url: string, mainFrame: boolean): boolean {
  try {
    const u = new URL(url);
    return (
      mainFrame &&
      u.protocol === 'app:' &&
      u.host === 'bundle' &&
      u.pathname === '/'
    );
  } catch {
    return false;
  }
}
export type DesktopRequest = { version: 2; kind: 'loadLast' | 'recentFolders' } | { version: 2; kind: 'requestWorkerPort'; requestId: string } | { version: 2; kind: 'openFolder'; folderId?: string } | { version: 2; kind: 'openDropped'; paths: string[] };
export const CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; img-src 'self' blob: data: xveon-photo:; style-src 'self' 'unsafe-inline'; connect-src 'self' xveon-photo:; object-src 'none'; base-uri 'none'; frame-src 'none'";
function object(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function envelope(value: unknown): value is Record<string, unknown> {
  if (!object(value) || value.version !== 2) return false;
  try { return new TextEncoder().encode(JSON.stringify(value)).byteLength <= 1_000_000; } catch { return false; }
}
function dense(value: unknown, check: (item: unknown) => boolean): boolean {
  if (!Array.isArray(value)) return false;
  for (let i = 0; i < value.length; i++) if (!Object.hasOwn(value, i) || !check(value[i])) return false;
  return true;
}
function unsaved(value: unknown): boolean {
  return dense(value, item => object(item) && typeof item.id === 'string' && /^[A-Za-z0-9_-]{22}$/.test(item.id)
    && typeof item.name === 'string' && (item.error === null || typeof item.error === 'string')
    && (item.folder === null || (object(item.folder) && typeof item.folder.id === 'string' && typeof item.folder.name === 'string')));
}
export function isDesktopRequest(value: unknown): value is DesktopRequest {
  if (!envelope(value)) return false;
  switch (value.kind) {
    case 'loadLast': case 'recentFolders': return true;
    case 'requestWorkerPort': return correlationId(value.requestId);
    case 'openFolder': return value.folderId === undefined || typeof value.folderId === 'string';
    case 'openDropped': return dense(value.paths, p => typeof p === 'string');
    default: return false;
  }
}
export function isUnsavedUpdate(value: unknown): value is { version: 2; edits: import('@/host').UnsavedSummary[] } { return envelope(value) && unsaved(value.edits); }
export function isFlushResponse(value: unknown): value is { version: 2; requestId: number; unsaved: import('@/host').UnsavedSummary[] } { return envelope(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 && unsaved(value.unsaved); }

export function isBridgeEvent(value: unknown): value is import('./bridge').BridgeEvent & { version: 2 } {
  if (!envelope(value)) return false;
  switch (value.kind) {
    case 'listing': return isListingFrame(value.frame) && !('registry' in value.frame);
    case 'folder-request': return value.folderId === undefined || typeof value.folderId === 'string';
    case 'flush-request': return Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0;
    case 'worker-restarted': return value.worker === undefined || isWorkerIdentity(value.worker);
    case 'worker-stopped': return typeof value.reason === 'string';
    default: return false;
  }
}

function correlationId(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
}
export function isWorkerPortDelivery(value: unknown): value is { version: 2; requestId: string } {
  return envelope(value) && correlationId(value.requestId);
}
