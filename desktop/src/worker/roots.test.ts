import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { beforeEach, afterEach, it, expect } from 'vitest';
import { createRoots } from './roots';
let dir: string; let raw: string;
beforeEach(async () => { dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-roots-'))); raw = path.join(dir, 'photos', 'a.RAF'); await fs.mkdir(path.dirname(raw)); await fs.writeFile(raw, 'raw'); });
afterEach(async () => { await fs.rm(dir, { recursive: true, force: true }); });
it('allows an existing regular RAW inside an opened root', async () => {
 const roots = createRoots(); roots.set([dir]); expect(await roots.checkWrite(raw)).toEqual({ realDir: path.dirname(raw), sidecarPath: raw + '.xmp', dev: expect.any(Number), ino: expect.any(Number) });
});
it('rejects deleted RAWs', async () => { const roots = createRoots(); roots.set([dir]); await fs.unlink(raw); await expect(roots.checkWrite(raw)).rejects.toThrow('The photo is no longer in its folder'); });
it('rejects an existing sidecar symlink without following it', async () => { const roots = createRoots(); roots.set([dir]); await fs.symlink(raw, raw + '.xmp'); await expect(roots.checkWrite(raw)).rejects.toThrow(/sidecar.*symlink/i); });
it('rejects RAW targets outside the root and nonregular sidecars', async () => {
 const roots = createRoots(); roots.set([path.dirname(raw)]); await fs.writeFile(path.join(dir, 'outside.RAF'), 'raw'); await fs.unlink(raw); await fs.symlink('../outside.RAF', raw);
 await expect(roots.checkWrite(raw)).rejects.toThrow(/outside|opened/i);
 await fs.unlink(raw); await fs.writeFile(raw, 'raw'); await fs.mkdir(raw + '.xmp'); await expect(roots.checkWrite(raw)).rejects.toThrow(/sidecar.*regular/i);
});
