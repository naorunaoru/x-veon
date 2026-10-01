import { expect, it, vi } from 'vitest';
import { createListingAssembler, listingFrames } from './listing';
import type { ListingFrame } from './listing';
import { isListingFrame } from './rpc';
import { photo } from './test-fixtures';
const folder = { id: 'f', name: 'Photos' };
const begin: ListingFrame = { v: 1, kind: 'listing-begin', token: 't', activation: 'a', folder, total: 2, purpose: 'open' };
it('assembles 2,000 realistic photos from eight batches below 1 MB', () => {
  const photos = Array.from({ length: 2000 }, (_, i) => photo(i));
  const frames = listingFrames({ token: 't', activation: 'a', folder, purpose: 'open' }, photos);
  expect(frames.filter(f => f.kind === 'listing-batch').length).toBe(8);
  const done = vi.fn(); const error = vi.fn(); const assembler = createListingAssembler(done, error);
  for (const frame of frames) { expect(isListingFrame(frame)).toBe(true); expect(Buffer.byteLength(JSON.stringify(frame))).toBeLessThanOrEqual(1_000_000); assembler.push(frame); }
  expect(error).not.toHaveBeenCalled(); expect(done).toHaveBeenCalledExactlyOnceWith(folder, photos, 'open');
});
it('ignores unknown and cancelled tokens, including a late begin', () => {
  const done = vi.fn(); const error = vi.fn(); const a = createListingAssembler(done, error);
  a.push({ v: 1, kind: 'listing-end', token: 'unknown', total: 0 }); a.push(begin); a.cancel('t');
  for (const frame of listingFrames({ token: 't', activation: 'a', folder, purpose: 'replace' }, [photo(), photo(1)])) a.push(frame);
  expect(done).not.toHaveBeenCalled(); expect(error).not.toHaveBeenCalled();
});
it.each(['gap', 'begin-total', 'end-total', 'duplicate', 'oversized'])('drops the whole listing on %s with no partial application', kind => {
  const done = vi.fn(); const error = vi.fn(); const a = createListingAssembler(done, error); a.push(begin);
  a.push({ v: 1, kind: 'listing-batch', token: 't', seq: kind === 'gap' ? 1 : 0, photos: kind === 'oversized' ? Array(251).fill(photo()) : [photo()] });
  if (kind === 'duplicate') a.push({ v: 1, kind: 'listing-batch', token: 't', seq: 0, photos: [photo(1)] });
  a.push({ v: 1, kind: 'listing-end', token: 't', total: kind === 'end-total' ? 3 : kind === 'begin-total' ? 1 : 2 });
  expect(done).not.toHaveBeenCalled(); expect(error).toHaveBeenCalledOnce();
});
it('splits below the count limit when large metadata would exceed the byte limit', () => {
  const photos = Array.from({ length: 250 }, (_, i) => ({ ...photo(i), editingNote: 'x'.repeat(10_000) }));
  const frames = listingFrames({ token: 'large', activation: 'a', folder, purpose: 'replace' }, photos);
  const batches = frames.filter(f => f.kind === 'listing-batch');
  expect(batches.length).toBeGreaterThan(1); expect(batches.flatMap(f => f.photos)).toEqual(photos);
  expect(frames.every(f => Buffer.byteLength(JSON.stringify(f)) <= 1_000_000)).toBe(true);
});
it('drops a structured-cloned sparse photo batch without applying undefined photos', () => {
  const done = vi.fn(), error = vi.fn(); const assembler = createListingAssembler(done, error);
  assembler.push({ ...begin, total: 3 });
  assembler.push(structuredClone({ v: 1, kind: 'listing-batch', token: 't', seq: 0, photos: [photo(), , photo(2)] }) as ListingFrame);
  assembler.push({ v: 1, kind: 'listing-end', token: 't', total: 3 });
  expect(done).not.toHaveBeenCalled(); expect(error).toHaveBeenCalledExactlyOnceWith('t', 'Invalid listing frame');
});
