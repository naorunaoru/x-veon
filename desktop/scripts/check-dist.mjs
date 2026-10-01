import { listPackage, extractFile } from '@electron/asar';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const archive = process.argv[2]
  ? resolve(process.argv[2])
  : fileURLToPath(new URL('../dist/mac-arm64/X-veon Beta.app/Contents/Resources/app.asar', import.meta.url));
const entries = listPackage(archive).map(entry => entry.replace(/^\//, ''));
const required = ['out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json'];
const failures = required.filter(file => !entries.includes(file)).map(file => `missing ${file}`);

if (entries.some(file => file === 'out/renderer/samples' || file.startsWith('out/renderer/samples/'))) {
  failures.push('samples/ is packaged');
}

const rendererScripts = entries.filter(file => /^out\/renderer\/assets\/.*\.js$/.test(file));
if (rendererScripts.length === 0) failures.push('missing renderer JavaScript');
for (const file of rendererScripts) {
  const source = extractFile(archive, file).toString('utf8');
  for (const forbidden of ['createGoldenHost', 'runSpike', 'spike-timing']) {
    if (source.includes(forbidden)) failures.push(`${file} contains ${forbidden}`);
  }
}

if (failures.length) {
  console.error(`Package check failed (${archive}):\n${failures.map(message => `- ${message}`).join('\n')}`);
  process.exitCode = 1;
} else {
  console.log(`Package check passed (${archive}): required data present; samples and development code absent`);
}
