import { afterEach, expect, it, vi } from 'vitest';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { isInside, registerPhotoProtocol } from './photo-protocol';
const dirs: string[] = [];
afterEach(async () => { for (const dir of dirs) await fs.rm(dir, { recursive: true, force: true }); });
it('handles path boundaries, Windows drive casing and UNC shares', () => {
  expect(isInside('/photos', '/photos/a.RAF')).toBe(true); expect(isInside('/photos', '/photos-other/a.RAF')).toBe(false);
  expect(isInside('c:\\photos', 'C:\\Photos\\A.RAF', 'win32')).toBe(true);
  expect(isInside('\\\\server\\share\\photos', '\\\\SERVER\\SHARE\\photos\\a.RAF', 'win32')).toBe(true);
  expect(isInside('\\\\server\\share', '\\\\server\\other\\a.RAF', 'win32')).toBe(false);
});
it('streams only registered files in opened real roots and thumbnails in the real cache', async () => {
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'protocol-'))); dirs.push(dir);
  const root = path.join(dir, 'photos'), cache = path.join(dir, 'cache'); await fs.mkdir(root); await fs.mkdir(cache);
  const raw = path.join(root, 'a.RAF'), outside = path.join(dir, 'secret'), thumb = path.join(cache, 'a.jpg');
  await fs.writeFile(raw, 'raw bytes'); await fs.writeFile(outside, 'secret'); await fs.writeFile(thumb, 'thumbnail'); await fs.symlink(outside, path.join(root, 'escape.RAF'));
  let handler!: (r: { url: string; method: string }) => Promise<Response>;
  const registry = new Map([['a'.repeat(22), raw], ['b'.repeat(22), outside], ['c'.repeat(22), path.join(root, 'escape.RAF')]]);
  let thumbPath: string | null = thumb;
  registerPhotoProtocol({ protocol: { handle: (_scheme, h) => { handler = h; } }, registry, roots: [root], cacheDir: cache, thumbnail: async () => thumbPath });
  const request = (kind: string, id: string) => handler({ url: `xveon-photo://${kind}/${id.repeat(22)}`, method: 'GET' });
  expect(await (await request('raw', 'a')).text()).toBe('raw bytes');
  for (const id of ['b', 'c', 'z']) expect((await request('raw', id)).status).toBe(404);
  expect(await (await request('thumb', 'a')).text()).toBe('thumbnail'); thumbPath = outside; expect((await request('thumb', 'a')).status).toBe(404);
  expect((await handler({ url: `xveon-photo://raw/${'a'.repeat(22)}`, method: 'POST' })).status).toBe(404);
});
