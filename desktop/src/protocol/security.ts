import type { ExportFormat } from '@/lib/types';
import type { DisplayReadings, NewerRelease } from '@/host';
import { parseReleaseTag, type ReleaseTag } from '../release/tags';
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
export type DesktopRequest = { version: 2; kind: 'chooseExportDestination'; photoId: string; format: ExportFormat } | { version: 2; kind: 'revealExport'; token: string } | { version: 2; kind: 'loadLast' | 'recentFolders' | 'displayReadings' | 'checkForUpdate' } | { version: 2; kind: 'requestWorkerPort'; requestId: string } | { version: 2; kind: 'openFolder'; folderId?: string } | { version: 2; kind: 'openDropped'; paths: string[] };
export const CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; img-src 'self' blob: data: xveon-photo:; style-src 'self' 'unsafe-inline'; connect-src 'self' xveon-photo:; object-src 'none'; base-uri 'none'; frame-src 'none'";
function object(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value); }
const RELEASE_PAGE = 'https://github.com/naorunaoru/x-veon/releases/tag/';
/** The release page of a validated tag: the only link the update notice opens. */
export function releasePage(tag: ReleaseTag): string { return RELEASE_PAGE + tag.tag; }
export function isReleasePage(url: unknown): url is string {
  return typeof url === 'string' && url.startsWith(RELEASE_PAGE) && parseReleaseTag(url.slice(RELEASE_PAGE.length)) !== null;
}
export function newerReleaseFrom(value: unknown): NewerRelease | null {
  if (!object(value)) return null;
  const short = (s: unknown): s is string => typeof s === 'string' && s.length > 0 && s.length <= 100;
  return short(value.name) && short(value.version) && isReleasePage(value.url) ? { name: value.name, version: value.version, url: value.url } : null;
}
const READING_NUMBERS = ['currentEdr', 'potentialEdr', 'referenceEdr', 'maxLuminance', 'maxFullFrameLuminance', 'minLuminance', 'sdrWhite'] as const;
/** The known display fields with valid values; null when none remain. Main applies it to the
 * addon's output and preload to the bridge response. */
export function displayReadingsFrom(value: unknown): DisplayReadings | null {
  if (!object(value)) return null;
  const readings: DisplayReadings = {};
  for (const key of READING_NUMBERS) {
    const n = value[key];
    if (typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1_000_000) readings[key] = n;
  }
  if (typeof value.hdrEnabled === 'boolean') readings.hdrEnabled = value.hdrEnabled;
  return Object.keys(readings).length ? readings : null;
}
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
    case 'loadLast': case 'recentFolders': case 'displayReadings': case 'checkForUpdate': return true;
    case 'chooseExportDestination': return typeof value.photoId === 'string' && /^[A-Za-z0-9_-]{22}$/.test(value.photoId) && ['jpeg-hdr', 'avif', 'tiff'].includes(value.format as string);
    case 'revealExport': return correlationId(value.token);
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
