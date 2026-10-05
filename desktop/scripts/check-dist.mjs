import { listPackage, extractFile, uncache } from '@electron/asar';
import { realpathSync, statSync } from 'node:fs';
import { normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Bundled output is about 47 MB. 80 MiB leaves room for model/runtime growth,
// while rejecting accidentally shipped workspace sources and redundant runtimes.
export const ARCHIVE_BUDGET_BYTES = 80 * 1024 ** 2;
export function checkArchive(archive) {
 uncache(archive);
 const entries = listPackage(archive).map(entry => entry.replace(/\\/g, '/').replace(/^\//, ''));
 const required = ['out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json'];
 const failures = required.filter(file => !entries.includes(file)).map(file => `missing ${file}`);
 const size = statSync(archive).size;
 if (size > ARCHIVE_BUDGET_BYTES) failures.push(`archive is ${size} bytes, over ${ARCHIVE_BUDGET_BYTES}-byte budget`);
 for (const file of entries) {
   if (/(^|\/)samples(\/|$)/i.test(file)) failures.push(`samples path: ${file}`);
   if (/\.(raf|cr2|cr3|nef|nrw|arw|dng|rw2|orf|pef|srw|erf|kdc|dcr|mef)$/i.test(file)) failures.push(`RAW file: ${file}`);
   if (/(^|\/)node_modules\/@xveon(\/|$)/i.test(file)) failures.push(`workspace package: ${file}`);
   if (/golden/i.test(file)) failures.push(`golden path: ${file}`);
 }
 for (const area of ['main', 'preload', 'renderer']) {
   const scripts = entries.filter(file => file.startsWith(`out/${area}/`) && /\.[cm]?js$/.test(file));
   if (!scripts.length) failures.push(`missing ${area} JavaScript`);
   for (const file of scripts) {
     const source = extractFile(archive, normalize(file)).toString('utf8');
     for (const forbidden of ['createGoldenHost', 'runSpike', 'spike-timing', '__golden', 'golden-report'])
       if (source.includes(forbidden)) failures.push(`${file} contains ${forbidden}`);
   }
 }
 return { size, entries: entries.length, failures };
}
if (process.argv[1] && realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url))) {
 const archive = process.argv[2] ? resolve(process.argv[2])
   : fileURLToPath(new URL('../dist/mac-arm64/X-veon Beta.app/Contents/Resources/app.asar', import.meta.url));
 const { size, failures } = checkArchive(archive);
 if (failures.length) {
   console.error(`Package check failed (${archive}):\n${failures.map(message => `- ${message}`).join('\n')}`); process.exitCode = 1;
 } else console.log(`Package check passed (${archive}): ${size} bytes; required data present; samples, RAWs, workspace sources and development code absent`);
}
