import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createPackageWithOptions } from '@electron/asar';
import { expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { requireFileSymlinks } from '../test/symlinks';
// @ts-expect-error Node packaging script intentionally has no TS runtime dependency.
import { checkArchive, ARCHIVE_BUDGET_BYTES } from '../../scripts/check-dist.mjs';
const fixtureAddon = 'xveon-native.darwin-arm64.node';

it('checks forbidden paths everywhere, all three JS bundles, and the archive budget', async () => {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-package-'));
 try {
   const source = path.join(dir, 'source'), archive = path.join(dir, 'app.asar');
   const files = [`native/${fixtureAddon}`, 'out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json', 'out/main/index.js', 'out/preload/index.js', 'out/renderer/assets/index.js'];
   for (const file of files) { await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}'); }
   await createPackageWithOptions(source, archive, { unpack: '*.node' }); expect(checkArchive(archive, { addon: fixtureAddon }).failures).toEqual([]);
   for (const file of ['elsewhere/samples/secret.txt', 'nested/photo.NEF', 'node_modules/@xveon/shared/source.ts', 'other/golden-fixture.json']) { await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}'); }
   for (const area of ['main', 'preload', 'renderer/assets']) await fs.writeFile(path.join(source, `out/${area}/index.js`), 'createGoldenHost');
   await createPackageWithOptions(source, archive, { unpack: '*.node' }); const failures = checkArchive(archive, { addon: fixtureAddon }).failures.join('\n');
   for (const text of ['samples path', 'RAW file', 'workspace package', 'golden path', 'out/main/index.js contains', 'out/preload/index.js contains', 'out/renderer/assets/index.js contains']) expect(failures).toContain(text);
   await fs.truncate(archive, ARCHIVE_BUDGET_BYTES + 1); expect(checkArchive(archive, { addon: fixtureAddon }).failures.join('\n')).toContain('over');
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
});

it.for(['ordinary', 'symlink', 'linux'])('runs the actual CLI through a %s path for valid and invalid archives', async (route, context) => {
 if (route === 'symlink') requireFileSymlinks(context);
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-package-cli-'));
 try {
   const source = path.join(dir, 'source'), archive = path.join(dir, 'app.asar');
   for (const file of [`native/${fixtureAddon}`, 'out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json', 'out/main/index.js', 'out/preload/index.js', 'out/renderer/index.js']) {
     await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}');
   }
   const script = fileURLToPath(new URL('../../scripts/check-dist.mjs', import.meta.url)), link = path.join(dir, 'check-dist.mjs');
   if (route === 'symlink') await fs.symlink(script, link, 'file');
   const preload = path.join(dir, 'linux.mjs');
   if (route === 'linux') await fs.writeFile(preload, "Object.defineProperty(process, 'platform', { value: 'linux' }); Object.defineProperty(process, 'arch', { value: 'x64' });");
   const runtimeArgs = route === 'linux' ? ['--import', pathToFileURL(preload).href] : [];
   for (const invalid of [false, true]) {
     if (invalid) await fs.writeFile(path.join(source, 'secret.RAF'), 'raw');
     await createPackageWithOptions(source, archive, { unpack: '*.node' });
     {
       const entry = route === 'symlink' ? link : script;
       const result = spawnSync(process.execPath, [...runtimeArgs, entry, archive, '--addon', fixtureAddon], { encoding: 'utf8' });
       expect(result.error).toBeUndefined(); expect(result.status, result.stderr).toBe(invalid ? 1 : 0);
       expect(result.stdout + result.stderr).toContain(invalid ? 'RAW file: secret.RAF' : 'Package check passed');
       if (route === 'linux' && !invalid) {
         const defaultResult = spawnSync(process.execPath, [...runtimeArgs, entry, archive], { encoding: 'utf8' });
         expect(defaultResult.status).toBe(1);
         expect(defaultResult.stderr).toContain('unsupported native addon platform: linux-x64');
         const apiResult = spawnSync(process.execPath, [...runtimeArgs, '--input-type=module', '--eval', `const { checkArchive } = await import(${JSON.stringify(new URL('../../scripts/check-dist.mjs', import.meta.url).href)}); if (checkArchive(${JSON.stringify(archive)}, { addon: ${JSON.stringify(fixtureAddon)} }).failures.length) throw new Error('fixture failed');`], { encoding: 'utf8' });
         expect(apiResult.status, apiResult.stderr).toBe(0);
       }
     }
   }
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
});

it.each(['stdin', 'eval-missing', 'eval-checker', 'file-missing'])('allows programmatic import from %s without running the CLI', async route => {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-package-import-'));
 try {
   const script = new URL('../../scripts/check-dist.mjs', import.meta.url);
   const missing = path.join(dir, 'missing.asar');
   const source = `${route === 'file-missing' ? `process.argv[1] = ${JSON.stringify(missing)};` : ''}
     const { checkArchive } = await import(${JSON.stringify(script.href)});
     if (typeof checkArchive !== 'function') throw new Error('missing export');
     let failed = false;
     try { checkArchive(${JSON.stringify(missing)}, { addon: ${JSON.stringify(fixtureAddon)} }); } catch { failed = true; }
     if (!failed) throw new Error('archive failure was swallowed');
     console.log('imported; archive errors remain observable');`;
   const importer = path.join(dir, 'import.mjs');
   if (route === 'file-missing') await fs.writeFile(importer, source);
   const args = route === 'stdin' ? ['--input-type=module', '-']
     : route === 'file-missing' ? [importer]
     : ['--input-type=module', '--eval', source, route === 'eval-checker' ? fileURLToPath(script) : missing];
   const result = spawnSync(process.execPath, args, { encoding: 'utf8', ...(route === 'stdin' ? { input: source } : {}) });
   expect(result.error).toBeUndefined(); expect(result.status, result.stderr).toBe(0);
   expect(result.stdout.trim()).toBe('imported; archive errors remain observable'); expect(result.stderr).toBe('');
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
});

it('rejects golden mode flags in normal bundles', async () => {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-golden-mode-'));
 try {
  const source = path.join(dir, 'source'), archive = path.join(dir, 'app.asar');
  for (const file of [`native/${fixtureAddon}`, 'out/main/index.js', 'out/preload/index.js', 'out/renderer/index.js', 'out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json']) {
   await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}');
  }
  await fs.writeFile(path.join(source, 'out/main/index.js'), 'process.argv.find(arg => arg.startsWith("--golden-mode="))');
  await createPackageWithOptions(source, archive, { unpack: '*.node' });
  expect(checkArchive(archive, { addon: fixtureAddon }).failures).toContain('out/main/index.js contains --golden-mode');
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
});

it.each(['unpacked', 'missing', 'packed', 'extra', 'wrong-platform', 'missing-physical'])('checks native addon integrity: %s', async scenario => {
 const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'xveon-native-package-'));
 const addon = 'xveon-native.darwin-arm64.node';
 try {
  const source = path.join(dir, 'source'), archive = path.join(dir, 'app.asar');
  const files = ['out/main/index.js', 'out/preload/index.js', 'out/renderer/index.js', 'out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json'];
  if (scenario !== 'missing') files.push(`native/${scenario === 'wrong-platform' ? 'xveon-native.win32-x64-msvc.node' : addon}`);
  if (scenario === 'extra') files.push('node_modules/x/y.node');
  for (const file of files) { await fs.mkdir(path.dirname(path.join(source, file)), { recursive: true }); await fs.writeFile(path.join(source, file), '{}'); }
  await createPackageWithOptions(source, archive, scenario === 'packed' ? {} : { unpack: '*.node' });
  if (scenario === 'missing-physical') await fs.rm(`${archive}.unpacked/native/${addon}`);
  const failures = checkArchive(archive, { addon }).failures;
  if (scenario === 'unpacked') expect(failures).toEqual([]);
  else expect(failures.join('\n')).toContain({ missing: 'missing native addon', packed: 'native addon is not unpacked', extra: 'unexpected native module', 'wrong-platform': 'missing native addon', 'missing-physical': 'missing unpacked native addon' }[scenario]);
 } finally { await fs.rm(dir, { recursive: true, force: true }); }
});

it.for([['--addon'], ['--addon', '../escape.node'], ['--addon', 'xveon-native.linux-x64.node'], ['--unknown'], ['--addon', fixtureAddon, '--addon', fixtureAddon]])('rejects invalid CLI arguments: %j', args => {
 const script = fileURLToPath(new URL('../../scripts/check-dist.mjs', import.meta.url));
 const result = spawnSync(process.execPath, [script, '/missing.asar', ...args], { encoding: 'utf8' });
 expect(result.status).toBe(1);
 expect(result.stderr).toContain('Usage:');
 expect(result.stderr).not.toContain('ENOENT');
});
