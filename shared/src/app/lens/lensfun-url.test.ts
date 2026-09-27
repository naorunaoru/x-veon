import { describe, it, expect } from 'vitest';
import { lensfunUrl } from './lensfun';

describe('lensfunUrl', () => {
  it('prefixes lens data files with the given base path', () => {
    expect(lensfunUrl('index.json', '/x-veon/beta/')).toBe('/x-veon/beta/lensfun/index.json');
    expect(lensfunUrl('fujifilm.json', '/')).toBe('/lensfun/fujifilm.json');
  });

  it('tolerates a base without a trailing slash', () => {
    expect(lensfunUrl('index.json', '/x-veon')).toBe('/x-veon/lensfun/index.json');
  });

  it('defaults to the Vite base URL', () => {
    expect(lensfunUrl('index.json')).toBe(`${import.meta.env.BASE_URL}lensfun/index.json`);
  });
});
