import { expect, it } from 'vitest';
import { assetName, acceptsSender, isRequest } from './security';
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
it('requires the trusted top frame and versioned requests', () => {
  expect(acceptsSender('app://bundle/?golden=render', true)).toBe(true);
  expect(acceptsSender('app://bundle/', false)).toBe(false);
  expect(acceptsSender('https://example.com/', true)).toBe(false);
  expect(isRequest({ version: 1, kind: 'connect' })).toBe(true);
  for (const value of [
    null,
    {},
    { version: 2, kind: 'connect' },
    { version: 1, kind: 'read-file' },
  ])
    expect(isRequest(value)).toBe(false);
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
