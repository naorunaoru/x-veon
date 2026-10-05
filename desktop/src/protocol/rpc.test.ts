import { describe, expect, it } from 'vitest';
import { isPortRequest, isPortReply, isPortEvent, isMainToWorker, isWorkerToMain, isListingFrame } from './rpc';
import { photo } from './test-fixtures';
const p = photo();
const folder = { id: 'folder', name: 'Photos' };
const examples = [
  [isPortRequest, { v: 1, rid: 1, op: 'saveEdit', id: p.id, edit: p.edit }],
  [isPortRequest, { v: 1, rid: 2, op: 'saveFacts', id: p.id, facts: p.facts }],
  [isPortRequest, { v: 1, rid: 3, op: 'rescan' }],
  [isPortReply, { v: 1, rid: 1, ok: true }],
  [isPortReply, { v: 1, rid: 1, ok: false, error: 'failed' }],
  [isPortEvent, { v: 1, event: 'facts', activation: 'a', folder, photos: [p] }],
  [isMainToWorker, { v: 1, kind: 'session', key: Buffer.from('session').toString('base64'), cacheDir: '/cache' }],
  [isMainToWorker, { v: 1, kind: 'connect' }],
  [isMainToWorker, { v: 1, kind: 'register', entries: [[p.id, '/a.RAF']] }],
  [isMainToWorker, { v: 1, kind: 'roots', realRoots: ['/photos'] }],
  [isMainToWorker, { v: 1, rid: 1, kind: 'list', path: '/photos', folderId: 'folder', token: 't', activation: 't', purpose: 'open' }],
  [isMainToWorker, { v: 1, kind: 'cancel-list', token: 't' }],
  [isMainToWorker, { v: 1, rid: 1, kind: 'thumbnail', id: p.id }],
  [isMainToWorker, { v: 1, kind: 'watch', path: '/photos', folderId: 'folder', activation: 't' }],
  [isMainToWorker, { v: 1, kind: 'watch', path: null, folderId: null, activation: null }],
  [isWorkerToMain, { v: 1, rid: 1, kind: 'thumbnail', path: null }],
  [isWorkerToMain, { v: 1, rid: 1, kind: 'error', error: 'failed' }],
  [isListingFrame, { v: 1, kind: 'listing-begin', token: 't', activation: 't', folder, total: 1, purpose: 'open' }],
  [isListingFrame, { v: 1, kind: 'listing-batch', token: 't', seq: 0, photos: [p], registry: [[p.id, '/a.RAF']] }],
  [isListingFrame, { v: 1, kind: 'listing-end', token: 't', total: 1 }],
] as const;
describe('versioned process protocol', () => {
  it.each(examples)('accepts a complete valid message %#', (validate, message) => { expect(validate(message)).toBe(true); });
  it.each(examples)('rejects null, version, and serialized size violations %#', (validate, message) => {
    expect(validate(null)).toBe(false); expect(validate({ ...message, v: 2 })).toBe(false);
    expect(validate({ ...message, padding: '🦊'.repeat(300_000) })).toBe(false);
  });
  it('rejects unknown operations, unsafe rids, invalid IDs and wrong field types', () => {
    for (const patch of [{ op: 'delete' }, { rid: 1.5 }, { rid: Number.MAX_SAFE_INTEGER + 1 }, { id: 'wrong' }, { edit: [] }]) expect(isPortRequest({ ...examples[0][1], ...patch })).toBe(false);
    expect(isMainToWorker({ v: 1, kind: 'roots', realRoots: [5] })).toBe(false);
    expect(isMainToWorker({ v: 1, kind: 'session', key: '!!', cacheDir: 1 })).toBe(false);
    expect(isMainToWorker({ v: 1, kind: 'watch', path: null, folderId: 'folder', activation: 'a' })).toBe(false);
    expect(isMainToWorker({ v: 1, kind: 'register', entries: Array(1001).fill([p.id, '/a']) })).toBe(false);
    expect(isPortReply({ v: 1, rid: 1, ok: false, error: 42 })).toBe(false);
    expect(isWorkerToMain({ v: 1, rid: 1, kind: 'thumbnail', path: false })).toBe(false);
    expect(isPortEvent({ ...examples[5][1], photos: Array(251).fill(p) })).toBe(false);
    expect(isListingFrame({ ...examples[18][1], seq: -1 })).toBe(false);
  });
  it('validates nested edits, metadata, result arrays, and lens coefficients', () => {
    const edits: unknown[] = [ { ...p.edit, lookPreset: 'unknown' }, { ...p.edit, preProcessOverrides: { exposure: '1' } }, { ...p.edit, openDrtOverrides: { tn_lcon_enable: 1 } }, { ...p.edit, openDrtOverrides: { constructor: 1 } }, { ...p.edit, model: { size: 'XL', sha256: 'a' } } ];
    for (const edit of edits) expect(isPortRequest({ ...examples[0][1], edit })).toBe(false);
    const facts = [ { ...p.facts, metadata: { ...p.facts.metadata, focalLength: '35' } }, { ...p.facts, resultMeta: { ...p.facts.resultMeta, exportData: { ...p.facts.resultMeta!.exportData, wbCoeffs: [1, '2'] } } }, { ...p.facts, resultMeta: { ...p.facts.resultMeta, metadata: { ...p.facts.resultMeta!.metadata, width: '7728' } } }, { ...p.facts, lensProfile: { ...p.facts.lensProfile, distortion: [{ model: 'ptlens', focal: 35, a: 'bad' }] } }, { ...p.facts, error: 42 } ];
    for (const value of facts) {
      expect(isPortRequest({ ...examples[1][1], facts: value })).toBe(false);
      expect(isListingFrame({ ...examples[18][1], photos: [{ ...p, facts: value }] })).toBe(false);
    }
    expect(isListingFrame({ ...examples[18][1], photos: [{ ...p, fileSize: NaN }] })).toBe(false);
  });
});

describe('structured-cloned sparse arrays', () => {
  const sparse = (value: unknown) => [value, , value];
  const sparseFacts = (field: 'xyzToCam' | 'wbCoeffs' | 'camToXyz') => ({ ...p.facts, resultMeta: { ...p.facts.resultMeta, exportData: { ...p.facts.resultMeta!.exportData, [field]: sparse(1) } } });
  const sparseLens = (field: 'distortion' | 'tca' | 'vignetting') => ({ ...p.facts, lensProfile: { ...p.facts.lensProfile, [field]: sparse(p.facts.lensProfile![field][0]) } });
  const cases = [
    ['listing photos', isListingFrame, { ...examples[18][1], photos: sparse(p) }],
    ['event photos', isPortEvent, { ...examples[5][1], photos: sparse(p) }],
    ['listing registry', isListingFrame, { ...examples[18][1], registry: sparse([p.id, '/a.RAF']) }],
    ['registered entries', isMainToWorker, { v: 1, kind: 'register', entries: sparse([p.id, '/a.RAF']) }],
    ['registry tuple', isMainToWorker, { v: 1, kind: 'register', entries: [[p.id, ,]] }],
    ['roots', isMainToWorker, { v: 1, kind: 'roots', realRoots: sparse('/photos') }],
    ...(['xyzToCam', 'wbCoeffs', 'camToXyz'] as const).map(field => [field, isPortRequest, { ...examples[1][1], facts: sparseFacts(field) }] as const),
    ...(['distortion', 'tca', 'vignetting'] as const).map(field => [field, isPortRequest, { ...examples[1][1], facts: sparseLens(field) }] as const),
  ] as const;
  it.each(cases)('rejects holes in %s', (_name, validate, message) => {
    expect(validate(structuredClone(message))).toBe(false);
  });
});

it('bounds and validates state stamps, listing order and opaque source revisions', () => {
  const stamp = { worker: '11111111-1111-4111-8111-111111111111', revision: 2 };
  expect(isPortReply({ v: 1, rid: 1, ok: true, stamp })).toBe(true);
  expect(isListingFrame({ ...examples[17][1], stamp: { ...stamp, scan: 1 } })).toBe(true);
  for (const invalid of [{ ...stamp, worker: 'x'.repeat(100) }, { ...stamp, revision: -1 }, { ...stamp, revision: 1.5 }, { ...stamp, revision: Number.MAX_SAFE_INTEGER + 1 }]) {
    expect(isPortReply({ v: 1, rid: 1, ok: true, stamp: invalid })).toBe(false);
    expect(isListingFrame({ ...examples[17][1], stamp: { ...invalid, scan: 1 } })).toBe(false);
  }
  for (const scan of [-1, 1.5, NaN]) expect(isListingFrame({ ...examples[17][1], stamp: { ...stamp, scan } })).toBe(false);
  for (const sourceVersion of ['a'.repeat(64), 'x'.repeat(64), 'a'.repeat(65), null, 1]) {
    const expected = sourceVersion === 'a'.repeat(64);
    expect(isListingFrame({ ...examples[18][1], photos: [{ ...p, sourceVersion }] })).toBe(expected);
    expect(isPortEvent({ ...examples[5][1], photos: [{ ...p, sourceVersion }] })).toBe(expected);
  }
});

describe('export protocol', () => {
  const begin = { v: 1, rid: 1, op: 'exportBegin', job: 'job-0001', destination: 'dest-001', format: 'avif', width: 2, height: 2, orientation: '', quality: 95, peakLuminance: 1000, planes: 1 };
  const chunk = { v: 1, rid: 2, op: 'exportChunk', job: 'job-0001', plane: 0, offset: 0, data: new ArrayBuffer(4) };
  it('accepts each format and all export operations', () => {
    for (const format of ['avif', 'jpeg-hdr', 'tiff']) expect(isPortRequest({ ...begin, format, planes: format === 'jpeg-hdr' ? 2 : 1 })).toBe(true);
    for (const op of ['exportStatus', 'exportCommit', 'exportCancel']) expect(isPortRequest({ v: 1, rid: 1, op, job: 'job-0001' })).toBe(true);
    expect(isPortRequest({ ...chunk, data: new ArrayBuffer(64_000_000) })).toBe(true);
  });
  it('rejects invalid begin fields', () => {
    for (const patch of [{ width: 0 }, { width: 65_536 }, { width: 20_000, height: 20_000 }, { quality: 0 }, { quality: 101 }, { quality: 95.5 }, { peakLuminance: NaN }, { format: 'png' }, { format: 'jpeg-hdr', planes: 1 }, { planes: 2 }, { job: 'has spaces' }, { destination: 'short' }]) expect(isPortRequest({ ...begin, ...patch })).toBe(false);
  });
  it('rejects invalid chunks', () => {
    for (const patch of [{ data: new Float32Array(1) }, { data: new ArrayBuffer(64_000_004) }, { data: new ArrayBuffer(6) }, { data: new ArrayBuffer(0) }, { offset: -4 }, { offset: 2 }, { plane: 2 }]) expect(isPortRequest({ ...chunk, ...patch })).toBe(false);
  });
  it('validates receipts, availability and destination registration', () => {
    const receipt = { name: 'test.avif', bytes: 3, sha256: 'a'.repeat(64), encodeMs: 2 };
    expect(isPortReply({ v: 1, rid: 1, ok: true, receipt })).toBe(true);
    for (const patch of [{ bytes: -1 }, { sha256: 'aaa' }, { encodeMs: NaN }]) expect(isPortReply({ v: 1, rid: 1, ok: true, receipt: { ...receipt, ...patch } })).toBe(false);
    for (const availability of [{ available: true }, { available: false, reason: 'no addon' }]) expect(isPortReply({ v: 1, rid: 1, ok: true, availability })).toBe(true);
    expect(isPortReply({ v: 1, rid: 1, ok: true, availability: { available: false } })).toBe(false);
    expect(isMainToWorker({ v: 1, kind: 'export-destination', token: 'dest-001', path: '/a.avif' })).toBe(true);
    expect(isMainToWorker({ v: 1, kind: 'export-destination', token: 'dest-001', path: '' })).toBe(false);
  });
});
it('validates destination acknowledgments and optional request IDs', () => {
 const request = { v: 1, rid: 8, kind: 'export-destination', token: 'dest-001', path: '/test.avif' };
 const reply = { v: 1, rid: 8, kind: 'export-destination', token: 'dest-001' };
 expect(isMainToWorker(request)).toBe(true); expect(isWorkerToMain(reply)).toBe(true);
 expect(isMainToWorker({ ...request, rid: 1.5 })).toBe(false); expect(isWorkerToMain({ ...reply, token: 'bad' })).toBe(false); expect(isWorkerToMain({ ...reply, rid: undefined })).toBe(false);
});
import { runInNewContext } from 'node:vm';
it('accepts an ArrayBuffer from another realm while retaining the envelope limit', () => {
 const chunk = { v: 1, rid: 1, op: 'exportChunk', job: 'job-0001', plane: 0, offset: 0, data: runInNewContext('new ArrayBuffer(4)') };
 expect(isPortRequest(chunk)).toBe(true); expect(isPortRequest({ ...chunk, padding: 'x'.repeat(1_000_000) })).toBe(false);
});
