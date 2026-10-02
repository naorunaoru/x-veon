import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createPackage } from '@electron/asar';
import { expect, it } from 'vitest';
// @ts-expect-error Node packaging script intentionally has no TS runtime dependency.
import { checkArchive, ARCHIVE_BUDGET_BYTES } from '../../scripts/check-dist.mjs';
it('checks forbidden paths everywhere, all three JS bundles, and the archive budget', async () => {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-package-'));
 try {
   const source = path.join(dir, 'source'), archive = path.join(dir, 'app.asar');
   const files = ['out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json', 'out/main/index.js', 'out/preload/index.js', 'out/renderer/assets/index.js'];
   for (const file of files) { await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}'); }
   await createPackage(source, archive); expect(checkArchive(archive).failures).toEqual([]);
   for (const file of ['elsewhere/samples/secret.txt', 'nested/photo.NEF', 'node_modules/@xveon/shared/source.ts', 'other/golden-fixture.json']) { await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}'); }
   for (const area of ['main', 'preload', 'renderer/assets']) await fs.writeFile(path.join(source, `out/${area}/index.js`), 'createGoldenHost');
   await createPackage(source, archive); const failures = checkArchive(archive).failures.join('\n');
   for (const text of ['samples path', 'RAW file', 'workspace package', 'golden path', 'out/main/index.js contains', 'out/preload/index.js contains', 'out/renderer/assets/index.js contains']) expect(failures).toContain(text);
   await fs.truncate(archive, ARCHIVE_BUDGET_BYTES + 1); expect(checkArchive(archive).failures.join('\n')).toContain('over');
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
});
