import { describe, it, expect } from 'vitest';
import { photoId } from './ids';
describe('photo IDs', () => {
  it('uses a session-keyed, normalized, 22-character URL-safe identity', () => {
    const key = Buffer.from('session');
    expect(photoId(key, '/photos/a/../b.RAF')).toBe(photoId(key, '/photos/b.RAF'));
    expect(photoId(key, '/photos/b.RAF')).toMatch(/^[A-Za-z0-9_-]{22}$/);
    expect(photoId(key, '/photos/b.RAF')).not.toBe(photoId(Buffer.from('other'), '/photos/b.RAF'));
  });
});
