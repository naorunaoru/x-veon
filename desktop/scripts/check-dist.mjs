import { listPackage, extractFile, statFile, uncache } from '@electron/asar';
import { existsSync, realpathSync, statSync } from 'node:fs';
import { normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Bundled output is about 47 MB. 80 MiB leaves room for model/runtime growth,
// while rejecting accidentally shipped workspace sources and redundant runtimes.
export const ARCHIVE_BUDGET_BYTES = 80 * 1024 ** 2;
function platformAddon() {
 if (process.platform === 'darwin' && process.arch === 'arm64') return 'xveon-native.darwin-arm64.node';
 if (process.platform === 'win32' && process.arch === 'x64') return 'xveon-native.win32-x64-msvc.node';
 throw new Error(`unsupported native addon platform: ${process.platform}-${process.arch}`);
}
export function checkArchive(archive, { addon = platformAddon() } = {}) {
 uncache(archive);
 const entries = listPackage(archive).map(entry => entry.replace(/\\/g, '/').replace(/^\//, ''));
 const required = ['out/renderer/index.html', 'out/renderer/checkpoints/models.json', 'out/renderer/lensfun/index.json'];
 const failures = required.filter(file => !entries.includes(file)).map(file => `missing ${file}`);
 const nativePath = `native/${addon}`;
 if (!entries.includes(nativePath)) failures.push(`missing native addon: ${nativePath}`);
 else if (!statFile(archive, nativePath).unpacked) failures.push(`native addon is not unpacked: ${nativePath}`);
 if (!existsSync(`${archive}.unpacked/${nativePath}`)) failures.push(`missing unpacked native addon: ${nativePath}`);
 const size = statSync(archive).size;
 if (size > ARCHIVE_BUDGET_BYTES) failures.push(`archive is ${size} bytes, over ${ARCHIVE_BUDGET_BYTES}-byte budget`);
 for (const file of entries) {
   if (/\.node$/i.test(file) && file !== nativePath) failures.push(`unexpected native module: ${file}`);
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
     for (const forbidden of ['createGoldenHost', 'runSpike', 'spike-timing', '__golden', 'golden-report', '--golden-mode'])
       if (source.includes(forbidden)) failures.push(`${file} contains ${forbidden}`);
   }
 }
 return { size, entries: entries.length, failures };
}
function isDirectInvocation() {
 // stdin/eval arguments belong to the caller, even if one names this script.
 if (!process.argv[1] || process.argv[1] === '-' || process.execArgv.some(arg => /^-[ep]|^--(?:eval|print)(?:=|$)/.test(arg))) return false;
 try { return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url)); }
 catch { return false; } // An importing program may use a non-file argv[1].
}
function cliArguments(args) {
 let archive, addon;
 const usage = 'Usage: check-dist.mjs [archive.asar] [--addon xveon-native.<darwin-arm64|win32-x64-msvc>.node]';
 for (let index = 0; index < args.length; index++) {
   const argument = args[index];
   if (argument === '--addon') {
     if (addon !== undefined || !/^xveon-native\.(darwin-arm64|win32-x64-msvc)\.node$/.test(args[index + 1] ?? '')) throw new Error(usage);
     addon = args[++index];
   } else if (argument.startsWith('-') || archive !== undefined) throw new Error(usage);
   else archive = resolve(argument);
 }
 return { archive: archive ?? fileURLToPath(new URL('../dist/mac-arm64/X-veon Beta.app/Contents/Resources/app.asar', import.meta.url)), addon };
}
if (isDirectInvocation()) {
 try {
   const { archive, addon } = cliArguments(process.argv.slice(2));
   const { size, failures } = checkArchive(archive, { addon });
   if (failures.length) {
     console.error(`Package check failed (${archive}):\n${failures.map(message => `- ${message}`).join('\n')}`); process.exitCode = 1;
   } else console.log(`Package check passed (${archive}): ${size} bytes; required data and unpacked native addon present; samples, RAWs, workspace sources and development code absent`);
 } catch (error) {
   console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;
 }
}
