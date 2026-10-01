import type { FolderRef, LibraryPhoto, PhotoId } from '@/host';
import { isListingFrame } from './rpc';
export type ListingFrame =
  | { v: 1; kind: 'listing-begin'; token: string; activation: string; folder: FolderRef; total: number; purpose: 'open' | 'replace' }
  | { v: 1; kind: 'listing-batch'; token: string; seq: number; photos: LibraryPhoto[]; registry?: [PhotoId, string][] }
  | { v: 1; kind: 'listing-end'; token: string; total: number };
type Header = Pick<Extract<ListingFrame, { kind: 'listing-begin' }>, 'token' | 'activation' | 'folder' | 'purpose'>;

/** Splits by both count and serialized size, including long paths/large cached facts. */
export function listingFrames(header: Header, photos: LibraryPhoto[], registry?: [PhotoId, string][]): ListingFrame[] {
  const frames: ListingFrame[] = [{ v: 1, kind: 'listing-begin', ...header, total: photos.length }];
  const paths = registry && new Map(registry);
  let batch: Extract<ListingFrame, { kind: 'listing-batch' }> = { v: 1, kind: 'listing-batch', token: header.token, seq: 0, photos: [], ...(paths ? { registry: [] } : {}) };
  function flush() {
    if (!batch.photos.length) return;
    frames.push(batch);
    batch = { v: 1, kind: 'listing-batch', token: header.token, seq: batch.seq + 1, photos: [], ...(paths ? { registry: [] } : {}) };
  }
  for (const photo of photos) {
    const entry = paths?.get(photo.id);
    if (paths && entry === undefined) throw new Error('Missing registry entry');
    const next = { ...batch, photos: [...batch.photos, photo], ...(paths ? { registry: [...batch.registry!, [photo.id, entry!] as [PhotoId, string]] } : {}) };
    if (!isListingFrame(next)) flush();
    batch.photos.push(photo);
    if (paths) batch.registry!.push([photo.id, entry!]);
    if (!isListingFrame(batch)) throw new Error('A photo exceeds the listing message limit or has invalid data');
  }
  flush();
  frames.push({ v: 1, kind: 'listing-end', token: header.token, total: photos.length });
  if (!frames.every(isListingFrame)) throw new Error('Invalid listing');
  return frames;
}

export function createListingAssembler(
  onComplete: (folder: FolderRef, photos: LibraryPhoto[], purpose: 'open' | 'replace') => void,
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
      onComplete(state.begin.folder, state.photos, state.begin.purpose);
    },
  };
}
