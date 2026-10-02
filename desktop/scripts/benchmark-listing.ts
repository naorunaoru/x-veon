import { performance } from 'node:perf_hooks';
import { photo } from '../src/protocol/test-fixtures';
import { listingFrames, bridgeListingFrames, createListingAssembler } from '../src/protocol/listing';
import { createWorkerController } from '../src/worker/controller';
import { isListingFrame } from '../src/protocol/rpc';
const photos = Array.from({ length: 2000 }, (_, i) => photo(i));
const registry: [string, string][] = photos.map(p => [p.id, `/photos/写真/${p.originalName}`]);
const header = { token: 'benchmark', activation: 'benchmark', folder: { id: 'photos', name: 'Photos' }, purpose: 'open' as const };
const measure = (run: () => unknown) => { const start = performance.now(); run(); return +(performance.now() - start).toFixed(3); };
const workerMs = measure(() => listingFrames(header, photos, registry));
const mainMs = measure(() => bridgeListingFrames(header, photos));
let complete = 0, maxEnvelope = 0;
const window = createListingAssembler((_folder, items) => { complete = items.length; }, (_token, reason) => { throw new Error(reason); });
const main = createListingAssembler((folder, items, purpose) => {
 for (const frame of bridgeListingFrames({ ...header, folder, purpose }, items)) {
   const envelope = structuredClone({ version: 2, kind: 'listing', frame });
   maxEnvelope = Math.max(maxEnvelope, Buffer.byteLength(JSON.stringify(envelope)));
   window.push(envelope.frame);
 }
}, (_token, reason) => { throw new Error(reason); });
const controller = createWorkerController({
 postToMain(message) { if (isListingFrame(message)) main.push(structuredClone(message)); },
 createLibrary: () => ({ async *list() { for (let offset = 0; offset < photos.length; offset += 250) yield { photos: photos.slice(offset, offset + 250), registry: registry.slice(offset, offset + 250) }; }, register() {}, setRoots() {}, async saveEdit() {}, async saveFacts() {}, async thumbnail() { return { path: null, facts: null }; } }),
});
async function run() {
 await controller.handleMain({ v: 1, kind: 'session', key: 'a2V5', cacheDir: '/unused' });
 const start = performance.now();
 await controller.handleMain({ v: 1, rid: 1, kind: 'list', path: '/photos', folderId: 'photos', token: 'benchmark', activation: 'benchmark', purpose: 'open' });
 console.log(JSON.stringify({ count: photos.length, averagePhotoBytes: Buffer.byteLength(JSON.stringify(photos)) / photos.length, workerMs, mainMs, controllerToWindowMs: +(performance.now() - start).toFixed(3), complete, maxEnvelope, node: process.version, platform: process.platform, arch: process.arch }, null, 2));
 if (complete !== 2000 || maxEnvelope > 1_000_000) throw new Error('Incomplete or oversized benchmark flow');
}
void run();
