import type { FolderRef, LibraryPhoto, PhotoId } from '@/host';
import { isListingFrame, MAX_MESSAGE_BYTES, type ListingStamp } from './rpc';
export type ListingFrame =
  | { v: 1; kind: 'listing-begin'; token: string; activation: string; folder: FolderRef; total: number; purpose: 'open' | 'replace'; stamp?: ListingStamp }
  | { v: 1; kind: 'listing-batch'; token: string; seq: number; photos: LibraryPhoto[]; registry?: [PhotoId, string][] }
  | { v: 1; kind: 'listing-end'; token: string; total: number };
type Header = Pick<Extract<ListingFrame, { kind: 'listing-begin' }>, 'token' | 'activation' | 'folder' | 'purpose' | 'stamp'>;

/** Splits by both count and serialized size, including long paths/large cached facts. */
export function listingFrames(header: Header, photos: LibraryPhoto[], registry?: [PhotoId, string][], byteLimit = MAX_MESSAGE_BYTES): ListingFrame[] {
  const encoder = new TextEncoder();
  const bytes = (value: unknown) => encoder.encode(JSON.stringify(value)).byteLength;
  const valid = (frame: ListingFrame) => isListingFrame(frame) && (byteLimit === MAX_MESSAGE_BYTES || bytes(frame) <= byteLimit);
  const frames: ListingFrame[] = [{ v: 1, kind: 'listing-begin', ...header, total: photos.length }];
  const paths = registry && new Map(registry);
  const empty = (seq: number): Extract<ListingFrame, { kind: 'listing-batch' }> => ({ v: 1, kind: 'listing-batch', token: header.token, seq, photos: [], ...(paths ? { registry: [] } : {}) });
  let batch = empty(0), size = bytes(batch);
  function flush() {
    if (!batch.photos.length) return;
    if (!valid(batch)) throw new Error('A photo exceeds the listing message limit or has invalid data');
    frames.push(batch);
    batch = empty(batch.seq + 1); size = bytes(batch);
  }
  for (const photo of photos) {
    const entry = paths?.get(photo.id);
    if (paths && entry === undefined) throw new Error('Missing registry entry');
    const pair: [PhotoId, string] | undefined = paths ? [photo.id, entry!] : undefined;
    const payload = bytes(photo) + (pair ? bytes(pair) : 0);
    const commas = () => batch.photos.length ? (pair ? 2 : 1) : 0;
    if (batch.photos.length >= 250 || size + payload + commas() > byteLimit) flush();
    size += payload + commas();
    if (size > byteLimit) throw new Error('A photo exceeds the listing message limit or has invalid data');
    batch.photos.push(photo);
    if (pair) batch.registry!.push(pair);
  }
  flush();
  frames.push({ v: 1, kind: 'listing-end', token: header.token, total: photos.length });
  if (!valid(frames[0]) || !valid(frames[frames.length - 1])) throw new Error('Invalid listing');
  return frames;
}

export function createListingAssembler(
  onComplete: (folder: FolderRef, photos: LibraryPhoto[], purpose: 'open' | 'replace', stamp?: ListingStamp) => void,
  onError: (token: string, reason: string) => void,
) {
  const active = new Map<string, { begin: Extract<ListingFrame, { kind: 'listing-begin' }>; photos: LibraryPhoto[]; seq: number }>();
  const cancelled = new Set<string>();
  function cancel(token?: string) {
    if (token === undefined) { for (const key of active.keys()) cancelled.add(key); active.clear(); }
    else { active.delete(token); cancelled.add(token); }
  }
  function fail(token: string, reason: string) { cancel(token); onError(token, reason); }
  return {
    cancel,
    push(frame: ListingFrame) {
      if (!frame || typeof frame.token !== 'string' || cancelled.has(frame.token)) return;
      const token = frame.token;
      const state = active.get(token);
      if (frame.kind !== 'listing-begin' && !state) return;
      if (!isListingFrame(frame)) { fail(token, 'Invalid listing frame'); return; }
      if (frame.kind === 'listing-begin') {
        if (state) { fail(frame.token, 'Duplicate listing begin'); return; }
        active.set(frame.token, { begin: frame, photos: [], seq: 0 }); return;
      }
      if (!state) return;
      if (frame.kind === 'listing-batch') {
        if (frame.seq !== state.seq || state.photos.length + frame.photos.length > state.begin.total) { fail(frame.token, 'Listing batch sequence or total mismatch'); return; }
        state.photos.push(...frame.photos); state.seq++; return;
      }
      if (frame.total !== state.begin.total || frame.total !== state.photos.length) { fail(frame.token, 'Listing total mismatch'); return; }
      active.delete(frame.token); cancelled.add(frame.token);
      onComplete(state.begin.folder, state.photos, state.begin.purpose, state.begin.stamp);
    },
  };
}

/** Account for the exact main→preload event envelope, not just the worker frame. */
export function bridgeListingFrames(header: Header, photos: LibraryPhoto[]): ListingFrame[] {
  const overhead = new TextEncoder().encode(JSON.stringify({ version: 2, kind: 'listing', frame: null })).byteLength - 4;
  return listingFrames(header, photos, undefined, MAX_MESSAGE_BYTES - overhead);
}
