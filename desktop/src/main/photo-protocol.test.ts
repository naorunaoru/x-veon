import { requireFileSymlinks, directoryLinkType } from '../test/symlinks';
import { afterEach, expect, it, vi } from 'vitest';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { isInside, registerPhotoProtocol } from './photo-protocol';
const dirs: string[] = [];
afterEach(async () => { vi.restoreAllMocks(); for (const dir of dirs) await fs.rm(dir, { recursive: true, force: true }); });
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
  await fs.writeFile(raw, 'raw bytes'); await fs.writeFile(outside, 'secret'); await fs.writeFile(thumb, 'thumbnail');
  let handler!: (r: { url: string; method: string }) => Promise<Response>;
  const registry = new Map([['a'.repeat(22), raw], ['b'.repeat(22), outside]]);
  let thumbPath: string | null = thumb;
  registerPhotoProtocol({ protocol: { handle: (_scheme, h) => { handler = h; } }, registry, roots: [root], cacheDir: cache, thumbnail: async () => thumbPath });
  const request = (kind: string, id: string) => handler({ url: `xveon-photo://${kind}/${id.repeat(22)}`, method: 'GET' });
  const allowed = await request('raw', 'a');
  expect(allowed.headers.get('Access-Control-Allow-Origin')).toBe('app://bundle');
  expect(await allowed.text()).toBe('raw bytes');
  for (const id of ['b', 'z']) expect((await request('raw', id)).status).toBe(404);
  expect(await (await request('thumb', 'a')).text()).toBe('thumbnail'); thumbPath = outside; expect((await request('thumb', 'a')).status).toBe(404);
  expect((await handler({ url: `xveon-photo://raw/${'a'.repeat(22)}`, method: 'POST' })).status).toBe(404);
});

it.for(['second-resolution', 'leaf-before-open', 'parent-before-open'])('rejects outside bytes after a controlled %s symlink swap', async (phase, context) => {
  if (phase !== 'parent-before-open') requireFileSymlinks(context);
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'protocol-swap-'))); dirs.push(dir);
  const root = path.join(dir, 'photos'), outside = path.join(dir, 'outside'); await fs.mkdir(root); await fs.mkdir(outside);
  const raw = path.join(root, 'a.RAF'); await fs.writeFile(raw, 'safe raw'); await fs.writeFile(path.join(outside, 'a.RAF'), 'outside secret');
  let handler!: (r: { url: string; method: string }) => Promise<Response>;
  registerPhotoProtocol({ protocol: { handle: (_scheme, h) => { handler = h; } }, registry: new Map([['a'.repeat(22), raw]]), roots: [root], cacheDir: dir, thumbnail: async () => null });
  const originalRealpath = fs.realpath.bind(fs), originalOpen = fs.open.bind(fs);
  let swapped = false;
  const swap = async () => {
    if (swapped) return; swapped = true;
    if (phase === 'parent-before-open') { await fs.rename(root, root + '-old'); await fs.symlink(outside, root, directoryLinkType); }
    else { await fs.unlink(raw); await fs.symlink(path.join(outside, 'a.RAF'), raw); }
  };
  if (phase === 'second-resolution') vi.spyOn(fs, 'realpath').mockImplementation(async (...args: Parameters<typeof fs.realpath>) => { const result = await originalRealpath(...args); if (args[0] === raw) await swap(); return result; });
  else vi.spyOn(fs, 'open').mockImplementation(async (...args) => { if (args[0] === raw) await swap(); return originalOpen(...args); });
  const response = await handler({ url: `xveon-photo://raw/${'a'.repeat(22)}`, method: 'GET' });
  const bytes = await response.text();
  expect(swapped).toBe(true); expect(response.status).toBe(404); expect(bytes).not.toContain('outside secret');
});

it('rejects an already registered RAW symlink escaping its opened root', async context => {
  requireFileSymlinks(context);
  const dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'protocol-escape-'))); dirs.push(dir);
  const root = path.join(dir, 'photos'); await fs.mkdir(root);
  const outside = path.join(dir, 'secret'), escape = path.join(root, 'escape.RAF');
  await fs.writeFile(outside, 'outside secret'); await fs.symlink(outside, escape, 'file');
  let handler!: (r: { url: string; method: string }) => Promise<Response>;
  registerPhotoProtocol({ protocol: { handle: (_scheme, h) => { handler = h; } }, registry: new Map([['c'.repeat(22), escape]]), roots: [root], cacheDir: dir, thumbnail: async () => null });
  const response = await handler({ url: `xveon-photo://raw/${'c'.repeat(22)}`, method: 'GET' });
  expect(response.status).toBe(404); expect(await response.text()).not.toContain('outside secret');
});
