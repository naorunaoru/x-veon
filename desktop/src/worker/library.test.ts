import { requireFileSymlinks, directoryLinkType } from '../test/symlinks';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { beforeEach, afterEach, it, expect, vi } from 'vitest';
import { defaultPhotoEdit } from '@/app/photo-edit';
import type { PhotoFacts } from '@/host';
import { createWorkerLibrary } from './library';
import { photoId } from './ids';
import { readSidecar, XVEON_NS } from './xmp';
let dir: string; let folder: string; let raw: string; let lib: ReturnType<typeof createWorkerLibrary>;
const sessionKey = Buffer.from('session-key');
const edit = { ...defaultPhotoEdit(), preProcessOverrides: { exposure: 1.25 } };
const facts: PhotoFacts = { cfaType: 'xtrans', metadata: null, resultMeta: null, resultMethod: null, lensProfile: null, status: 'done', error: null };
const id = () => photoId(sessionKey, raw);
async function collect(library = lib) { const batches = []; for await (const batch of library.list(folder, 'folder-id')) batches.push(batch); return batches; }
beforeEach(async () => {
 dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-library-'))); folder = path.join(dir, 'photos'); raw = path.join(folder, 'a.RAF'); await fs.mkdir(folder); await fs.writeFile(raw, 'raw');
 lib = createWorkerLibrary({ sessionKey, cacheDir: path.join(dir, 'cache') }); lib.setRoots([folder]); lib.register([[id(), raw]]);
});
afterEach(async () => { vi.restoreAllMocks(); await fs.chmod(folder, 0o755).catch(() => {}); await fs.chmod(raw + '.xmp', 0o644).catch(() => {}); await fs.rm(dir, { recursive: true, force: true }); });
it('lists defaults, protocol URLs and queued facts without RAW content reads', async () => {
 const read = vi.spyOn(fs, 'readFile'); const open = vi.spyOn(fs, 'open'); const photo = (await collect())[0].photos[0];
 expect(photo).toMatchObject({ id: id(), name: 'a', originalName: 'a.RAF', fileSize: 3, thumbnailUrl: `xveon-photo://thumb/${id()}`, edit: defaultPhotoEdit(), editing: 'saved', editingNote: null, facts: { status: 'queued' } });
 expect(open).not.toHaveBeenCalled(); expect(read.mock.calls.every(args => String(args[0]).endsWith('.xmp') || String(args[0]).startsWith(path.join(dir, 'cache')))).toBe(true);
});
it('yields 2,000 RAWs in eight naturally ordered batches of at most 250', async () => {
 await fs.unlink(raw); await Promise.all(Array.from({ length: 2000 }, (_, i) => fs.writeFile(path.join(folder, `DSCF${i}.RAF`), '')));
 const batches = await collect(); expect(batches.map(b => b.photos.length)).toEqual(Array(8).fill(250));
 expect(batches.flatMap(b => b.photos.map(p => p.originalName))).toEqual(Array.from({ length: 2000 }, (_, i) => `DSCF${i}.RAF`));
 expect(batches.flatMap(b => b.registry).length).toBe(2000);
});
it('uses stable IDs with a shared key and rejects mismatched restart registrations', async () => {
 const other = createWorkerLibrary({ sessionKey, cacheDir: path.join(dir, 'cache2') }); expect((await collect(other))[0].photos[0].id).toBe(id());
 const different = createWorkerLibrary({ sessionKey: Buffer.from('other'), cacheDir: path.join(dir, 'cache3') }); expect((await collect(different))[0].photos[0].id).not.toBe(id());
 expect(() => lib.register([['wrong', raw]])).toThrow(/id|identity/i); await expect(lib.saveEdit('unregistered', edit)).rejects.toThrow(/registered/i);
});
it('writes the full RAW name plus xmp and preserves other namespace content', async () => {
 await fs.writeFile(raw + '.xmp', '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" dc:title="kept"/></rdf:RDF></x:xmpmeta>');
 await lib.saveEdit(id(), edit); const text = await fs.readFile(raw + '.xmp', 'utf8'); expect(text).toContain('dc:title="kept"'); expect(readSidecar(text)).toEqual({ kind: 'ok', edit }); expect((await collect())[0].photos[0].edit).toEqual(edit);
});
it('creates nothing for default edits and deletes an only-ours sidecar on reset', async () => {
 await lib.saveEdit(id(), defaultPhotoEdit()); expect(await fs.readdir(folder)).toEqual(['a.RAF']); await lib.saveEdit(id(), edit); expect(await fs.readdir(folder)).toEqual(['a.RAF', 'a.RAF.xmp']); await lib.saveEdit(id(), defaultPhotoEdit()); expect(await fs.readdir(folder)).toEqual(['a.RAF']);
});
it('rejects a RAW deleted after registration', async () => { await fs.unlink(raw); await expect(lib.saveEdit(id(), edit)).rejects.toThrow('The photo is no longer in its folder'); expect(await fs.readdir(folder)).toEqual([]); });
it('rejects a parent replaced by an outside symlink without writing outside', async () => {
 const outside = path.join(dir, 'outside'); await fs.mkdir(outside); await fs.writeFile(path.join(outside, 'a.RAF'), 'outside'); await fs.rename(folder, folder + '-old'); await fs.symlink(outside, folder, directoryLinkType);
 await expect(lib.saveEdit(id(), edit)).rejects.toThrow(/outside|opened/i); expect(await fs.readdir(outside)).toEqual(['a.RAF']);
});
it('rejects a sidecar that became a symlink, retaining its target bytes', async context => { requireFileSymlinks(context);
 const target = path.join(dir, 'target'); await fs.writeFile(target, 'unchanged'); await fs.symlink(target, raw + '.xmp'); await expect(lib.saveEdit(id(), edit)).rejects.toThrow(/sidecar.*symlink/i); expect(await fs.readFile(target, 'utf8')).toBe('unchanged');
});
it('refuses malformed and newer-schema sidecars and lists them as view-only', async () => {
 for (const text of ['<broken', `<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:xveon="${XVEON_NS}" xveon:SchemaVersion="2" xveon:Exposure="2"/></rdf:RDF></x:xmpmeta>`]) {
 await fs.writeFile(raw + '.xmp', text); const photo = (await collect())[0].photos[0]; expect(photo.editing).toBe('view-only'); expect(photo.editingNote).toBeTruthy(); await expect(lib.saveEdit(id(), edit)).rejects.toThrow(/sidecar/i); expect(await fs.readFile(raw + '.xmp', 'utf8')).toBe(text);
 }
});
it.skipIf(process.platform === 'win32' || process.getuid?.() === 0)('reports POSIX read-only folders as session and rejects with the OS message', async () => {
 await fs.chmod(folder, 0o555); const photo = (await collect())[0].photos[0]; expect(photo.editing).toBe('session'); expect(photo.editingNote).toMatch(/This folder can't be written:.*EACCES/);
 await expect(lib.saveEdit(id(), edit)).rejects.toThrow(/EACCES/); expect(await fs.readdir(folder)).toEqual(['a.RAF']);
});
it.skipIf(process.platform === 'win32' || process.getuid?.() === 0)('refuses unreadable sidecars without changing bytes', async () => {
 await fs.writeFile(raw + '.xmp', 'unchanged'); await fs.chmod(raw + '.xmp', 0o000); expect((await collect())[0].photos[0].editing).toBe('view-only'); await expect(lib.saveEdit(id(), edit)).rejects.toThrow(/EACCES/); await fs.chmod(raw + '.xmp', 0o644); expect(await fs.readFile(raw + '.xmp', 'utf8')).toBe('unchanged');
});
it('saves facts only in cache and handles deletion between listings', async () => {
 await lib.saveFacts(id(), facts); expect((await collect())[0].photos[0].facts).toEqual(facts); expect(await fs.readdir(folder)).toEqual(['a.RAF']); await fs.rm(path.join(dir, 'cache'), { recursive: true }); expect((await collect())[0].photos[0].facts.status).toBe('queued');
});
it('resolves lazy thumbnails through the registry and returns cached facts', async () => {
 const bytes = Buffer.alloc(0x80); bytes.write('FUJIFILMCCD-RAW'); bytes.write('X-T5', 0x1c); bytes.writeUInt32BE(0x70, 0x54); bytes.writeUInt32BE(4, 0x58); bytes.set([255, 216, 255, 217], 0x70); await fs.writeFile(raw, bytes);
 const result = await lib.thumbnail(id()); expect(result.path).not.toBeNull(); expect(result.facts?.metadata?.camera).toBe('Fujifilm X-T5'); expect(await fs.readFile(result.path!)).toEqual(Buffer.from([255, 216, 255, 217]));
});
it('propagates failed rename and ENOSPC without losing the original sidecar', async () => {
 await lib.saveEdit(id(), edit); const original = await fs.readFile(raw + '.xmp', 'utf8');
 vi.spyOn(fs, 'rename').mockRejectedValueOnce(Object.assign(new Error('ENOSPC: no space left'), { code: 'ENOSPC' })); await expect(lib.saveEdit(id(), { ...edit, preProcessOverrides: { exposure: 2 } })).rejects.toThrow('ENOSPC: no space left'); expect(await fs.readFile(raw + '.xmp', 'utf8')).toBe(original); expect(await fs.readdir(folder)).toEqual(['a.RAF', 'a.RAF.xmp']);
});

it('retains extracted metadata after incomplete facts saves, including a fresh library instance', async () => {
  const bytes = Buffer.alloc(0x80);
  bytes.write('FUJIFILMCCD-RAW'); bytes.write('X-T5', 0x1c);
  bytes.writeUInt32BE(0x70, 0x54); bytes.writeUInt32BE(4, 0x58);
  bytes.set([255, 216, 255, 217], 0x70);
  await fs.writeFile(raw, bytes);
  const extracted = await lib.thumbnail(id());
  expect(extracted.facts?.metadata?.camera).toBe('Fujifilm X-T5');

  await lib.saveFacts(id(), { ...facts, metadata: null });
  const open = vi.spyOn(fs, 'open');
  const restored = createWorkerLibrary({ sessionKey, cacheDir: path.join(dir, 'cache') });
  restored.register([[id(), raw]]);
  for (const library of [lib, restored]) {
    const result = await library.thumbnail(id());
    expect(result.path).toBe(extracted.path);
    expect(result.facts).toEqual({ ...facts, metadata: { camera: 'Fujifilm X-T5', lensModel: '', focalLength: 0, fNumber: 0 } });
    expect((await collect(library))[0].photos[0].facts).toEqual(result.facts);
  }
  expect(open).not.toHaveBeenCalled();
});

it.each(['write', 'reset'] as const)('rejects a persistent parent swap after validation during %s', async operation => {
  await lib.saveEdit(id(), edit);
  const outside = path.join(dir, 'outside');
  await fs.mkdir(outside);
  await fs.writeFile(path.join(outside, 'a.RAF'), 'outside raw');
  await fs.writeFile(path.join(outside, 'a.RAF.xmp'), 'outside sidecar');
  const realpath = fs.realpath.bind(fs);
  let parents = 0;
  vi.spyOn(fs, 'realpath').mockImplementation(async target => {
    if (String(target) === folder && ++parents === 3) {
      await fs.rename(folder, folder + '-old');
      await fs.symlink(outside, folder, directoryLinkType);
    }
    return realpath(target);
  });
  await expect(lib.saveEdit(id(), operation === 'reset' ? defaultPhotoEdit() : { ...edit, preProcessOverrides: { exposure: 2 } })).rejects.toThrow(/directory|folder/i);
  expect(await fs.readFile(path.join(outside, 'a.RAF.xmp'), 'utf8')).toBe('outside sidecar');
  expect(await fs.readdir(outside)).toEqual(['a.RAF', 'a.RAF.xmp']);
});

it('rejects a parent swap between a locked rename and its retry', async () => {
  const outside = path.join(dir, 'outside'); await fs.mkdir(outside);
  await fs.writeFile(path.join(outside, 'a.RAF.xmp'), 'outside sidecar');
  vi.spyOn(fs, 'rename').mockImplementationOnce(async () => {
    await fs.rename(folder, folder + '-old'); await fs.symlink(outside, folder, directoryLinkType);
    throw Object.assign(new Error('busy'), { code: 'EBUSY' });
  });
  await expect(lib.saveEdit(id(), edit)).rejects.toThrow(/directory|folder/i);
  expect(await fs.readFile(path.join(outside, 'a.RAF.xmp'), 'utf8')).toBe('outside sidecar');
});

it('changes the opaque RAW revision for same-size content changed at the same registered path', async () => {
  const before = (await collect())[0].photos[0];
  await fs.writeFile(raw, 'new'); await fs.utimes(raw, 1000, 1000);
  const after = (await collect())[0].photos[0];
  expect(before.sourceVersion).toMatch(/^[a-f0-9]{64}$/);
  expect(after).toMatchObject({ id: before.id, fileSize: before.fileSize });
  expect(after.sourceVersion).not.toBe(before.sourceVersion);
});
