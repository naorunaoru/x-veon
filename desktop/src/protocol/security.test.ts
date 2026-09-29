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
