import { describe, expect, it } from 'vitest';
import { parsePhotoUrl, photoUrl } from './photo-url';
const id = 'abcdefghijklmnopqrstuv';
it('round trips RAW and thumbnail URLs', () => {
  for (const kind of ['raw', 'thumb'] as const) expect(parsePhotoUrl(photoUrl(kind, id))).toEqual({ kind, id });
});
describe('rejects unsafe or ambiguous URLs', () => {
  it.each([`xveon-photo://other/${id}`, `https://raw/${id}`, `xveon-photo://raw/../${id}`, `xveon-photo://raw/%2e%2e/${id}`, `xveon-photo://raw/${id}%2f`, `xveon-photo://raw/${id}%5c`, `xveon-photo://raw/${id}?a`, `xveon-photo://raw/${id}#a`, `xveon-photo://user@raw/${id}`, `xveon-photo://raw:80/${id}`, 'xveon-photo://raw/short', `xveon-photo://raw/${id}a`])('%s', raw => expect(parsePhotoUrl(raw)).toBeNull());
});
it('rejects trailing line breaks without normalizing the input', () => {
  expect(parsePhotoUrl(`xveon-photo://raw/${id}\n`)).toBeNull();
  expect(() => photoUrl('raw', id + '\n')).toThrow('Invalid photo URL');
});
