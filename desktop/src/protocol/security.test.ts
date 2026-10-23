import { expect, it } from 'vitest';
import { assetName, acceptsSender, displayReadingsFrom, isBridgeEvent, isDesktopRequest, isReleasePage, newerReleaseFrom, releasePage } from './security';
import { parseReleaseTag } from '../release/tags';
it('accepts only canonical release-page links and bounded notices', () => {
  expect(isDesktopRequest({ version: 2, kind: 'checkForUpdate' })).toBe(true);
  const url = 'https://github.com/naorunaoru/x-veon/releases/tag/beta/2026-10-07';
  expect(releasePage(parseReleaseTag('beta/2026-10-07')!)).toBe(url);
  expect(isReleasePage(url)).toBe(true);
  for (const value of [
    'https://github.com/naorunaoru/x-veon/releases/tag/../../../../other/repo',
    'https://github.com/naorunaoru/x-veon/releases/tag/beta/2026-02-30',
    `${url}?x=1`, `${url}#x`, 'https://github.com/naorunaoru/x-veon/releases/tag/',
    'https://github.com/other/x-veon/releases/tag/beta/2026-10-07',
    'http://github.com/naorunaoru/x-veon/releases/tag/beta/2026-10-07',
    'javascript:alert(1)', null, 1,
  ]) expect(isReleasePage(value)).toBe(false);
  const notice = { name: 'X-veon Beta', version: '2026.10.7-beta.1', url };
  expect(newerReleaseFrom(notice)).toEqual(notice);
  for (const bad of [{ url: 'https://example.com' }, { name: '' }, { name: 'x'.repeat(101) }, { version: '' }, { version: 'x'.repeat(101) }])
    expect(newerReleaseFrom({ ...notice, ...bad })).toBeNull();
});
it('accepts display requests and keeps only valid known readings', () => {
  expect(isDesktopRequest({ version: 2, kind: 'displayReadings' })).toBe(true);
  expect(displayReadingsFrom({ potentialEdr: 16, referenceEdr: 0, hdrEnabled: false })).toEqual({ potentialEdr: 16, referenceEdr: 0, hdrEnabled: false });
  expect(displayReadingsFrom({ currentEdr: NaN, potentialEdr: -1, referenceEdr: Infinity, sdrWhite: '80', maxLuminance: 16, junk: 1 })).toEqual({ maxLuminance: 16 });
  for (const value of [{}, null, [], { potentialEdr: 'x' }]) expect(displayReadingsFrom(value)).toBeNull();
});
it('serves only exact bundle assets via GET', () => {
  const files = new Set(['index.html', 'assets/decoder.wasm']);
  expect(assetName('app://bundle/', 'GET', files)).toBe('index.html');
  expect(assetName('app://bundle/assets/decoder.wasm', 'GET', files)).toBe(
    'assets/decoder.wasm',
  );
  for (const url of [
    'app://other/index.html',
    'app://bundle/missing',
    'app://bundle/%2e%2e/secret',
    'app://bundle/assets/%2fsecret',
    'file:///index.html',
    'app://bundle/assets/../index.html',
  ])
    expect(assetName(url, 'GET', files)).toBeNull();
  expect(assetName('app://bundle/', 'POST', files)).toBeNull();
});
it('requires the trusted top frame', () => {
  expect(acceptsSender('app://bundle/?golden=render', true)).toBe(true);
  expect(acceptsSender('app://bundle/', false)).toBe(false);
  expect(acceptsSender('https://example.com/', true)).toBe(false);

});
it('validates bounded version-2 desktop requests and dense nested unsaved summaries', async () => {
  const { isDesktopRequest, isUnsavedUpdate, isFlushResponse, CONTENT_SECURITY_POLICY } = await import('./security');
  expect(isDesktopRequest({ version: 2, kind: 'openFolder', folderId: 'a' })).toBe(true);
  expect(isDesktopRequest({ version: 2, kind: 'openDropped', paths: ['/a'] })).toBe(true);
  for (const request of [{ version: 1, kind: 'loadLast' }, { version: 2, kind: 'openFolder', folderId: 1 }, { version: 2, kind: 'openDropped', paths: Array(2) }, { version: 2, kind: 'openDropped', paths: ['a'.repeat(1_000_001)] }, { version: 2, kind: 'readFile', path: '/secret' }]) expect(isDesktopRequest(request)).toBe(false);
  const edits = [{ id: 'a'.repeat(22), name: 'a', folder: { id: 'f', name: 'Photos' }, error: null }];
  expect(isUnsavedUpdate({ version: 2, edits })).toBe(true); expect(isFlushResponse({ version: 2, requestId: 1, unsaved: edits })).toBe(true);
  expect(isUnsavedUpdate({ version: 2, edits: Array(1) })).toBe(false); expect(isUnsavedUpdate({ version: 2, edits: [{ ...edits[0], folder: { id: 4 } }] })).toBe(false);
  expect(isFlushResponse({ version: 2, requestId: NaN, unsaved: edits })).toBe(false);
  expect(CONTENT_SECURITY_POLICY).toMatch(/img-src[^;]*xveon-photo:/); expect(CONTENT_SECURITY_POLICY).toMatch(/connect-src[^;]*xveon-photo:/);
});

it('validates the complete outgoing bridge event and rejects oversized wrapped listings', async () => {
  const { isBridgeEvent } = await import('./security');
  const frame = { v: 1, kind: 'listing-begin', token: 't', activation: 'a', folder: { id: 'f', name: 'Photos' }, total: 0, purpose: 'open' };
  expect(isBridgeEvent({ version: 2, kind: 'listing', frame })).toBe(true);
  const large = { ...frame, folder: { id: 'f', name: '' } };
  large.folder.name = 'x'.repeat(999_990 - Buffer.byteLength(JSON.stringify(large)));
  expect(isBridgeEvent({ version: 2, kind: 'listing', frame: large })).toBe(false);
  expect(isBridgeEvent({ version: 1, kind: 'worker-restarted' })).toBe(false);
  expect(isBridgeEvent({ version: 2, kind: 'flush-request', requestId: 1 })).toBe(true);
  expect(isBridgeEvent({ version: 2, kind: 'folder-request', folderId: 42 })).toBe(false);
});

it('requires a bounded correlation UUID for worker-port handshakes', async () => {
  const { isDesktopRequest } = await import('./security');
  expect(isDesktopRequest({ version: 2, kind: 'requestWorkerPort', requestId: '00000000-0000-4000-8000-000000000001' })).toBe(true);
  for (const requestId of [undefined, 1, '', 'old', 'x'.repeat(1_000_001)])
    expect(isDesktopRequest({ version: 2, kind: 'requestWorkerPort', requestId })).toBe(false);
});

it('validates worker identities on native restart events', () => {
  expect(isBridgeEvent({ version: 2, kind: 'worker-restarted', worker: '11111111-1111-4111-8111-111111111111' })).toBe(true);
  for (const worker of [null, 1, 'invalid', 'a'.repeat(100)]) expect(isBridgeEvent({ version: 2, kind: 'worker-restarted', worker })).toBe(false);
});

it('validates export target, reveal and folder requests', () => {
 const id = 'a'.repeat(22), token = '00000000-0000-4000-8000-000000000001';
 const resolve = { version: 2, kind: 'resolveExportTarget', saveTo: { mode: 'beside' }, photoIds: [id] };
 expect(isDesktopRequest(resolve)).toBe(true);
 expect(isDesktopRequest({ ...resolve, saveTo: { mode: 'folder', path: '/Users/me/Exports' } })).toBe(true);
 expect(isDesktopRequest({ ...resolve, photoIds: Array(10_000).fill(id) })).toBe(true);
 expect(isDesktopRequest({ ...resolve, saveTo: { mode: 'folder', path: '/Users/x/out' } })).toBe(true);
 for (const unc of ['\\\\host\\share', '//host/share', '\\/host/share', '/\\host\\share']) expect(isDesktopRequest({ ...resolve, saveTo: { mode: 'folder', path: unc } })).toBe(false);
 for (const patch of [{ photoIds: [] }, { photoIds: Array(10_001).fill(id) }, { photoIds: ['../raw'] }, { photoIds: [id, , id] }, { saveTo: { mode: 'folder', path: '' } }, { saveTo: { mode: 'folder', path: 'a\0b' } }, { saveTo: { mode: 'folder', path: 'x'.repeat(4097) } }, { saveTo: { mode: 'folder' } }, { saveTo: { mode: 'desktop' } }, { saveTo: null }])
  expect(isDesktopRequest({ ...resolve, ...patch })).toBe(false);
 expect(isDesktopRequest({ version: 2, kind: 'revealExport', token })).toBe(true);
 expect(isDesktopRequest({ version: 2, kind: 'revealExport', token, name: 'a.avif' })).toBe(true);
 for (const name of ['../a.avif', 'a/b', '', '..', 'x'.repeat(256)]) expect(isDesktopRequest({ version: 2, kind: 'revealExport', token, name })).toBe(false);
 expect(isDesktopRequest({ version: 2, kind: 'revealExport', token: 'bad' })).toBe(false);
 expect(isDesktopRequest({ version: 2, kind: 'chooseExportFolder' })).toBe(true);
 expect(isDesktopRequest({ version: 2, kind: 'chooseExportDestination', photoId: id, format: 'avif' })).toBe(false);
});
