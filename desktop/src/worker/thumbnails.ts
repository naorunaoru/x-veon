import fs from 'node:fs/promises';
import { extractRafThumbnail, extractRafQuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { FolderEntry } from './folder';
import type { createCache } from './cache';

type Cache = ReturnType<typeof createCache>;
type Head = { thumbnail: string | null; metadata: QuickMetadata | null };
// Slots cover the entire miss operation, including buffers retained during cache writes.
let active = 0;
const waiting: (() => void)[] = [];
async function withHeadSlot<T>(run: () => Promise<T>): Promise<T> {
  if (active >= 4) await new Promise<void>(resolve => waiting.push(resolve));
  else active++;
  try { return await run(); }
  finally { const next = waiting.shift(); if (next) next(); else active--; }
}
const pending = new WeakMap<Cache, Map<string, Promise<Head>>>();

export function ensureHead(entry: FolderEntry, cache: Cache): Promise<Head> {
  const key = cache.key(entry);
  let requests = pending.get(cache);
  if (!requests) { requests = new Map(); pending.set(cache, requests); }
  const existing = requests.get(key);
  if (existing) return existing;
  const request = readHead(entry, cache, key).finally(() => { requests.delete(key); });
  requests.set(key, request);
  return request;
}

async function readHead(entry: FolderEntry, cache: Cache, key: string): Promise<Head> {
  if (await cache.hasHead(key)) {
    return { thumbnail: await cache.hasThumb(key) ? cache.thumbPath(key) : null, metadata: (await cache.getFacts(key))?.metadata ?? null };
  }
  return withHeadSlot(async () => {
    const handle = await fs.open(entry.path, 'r');
    let head: ArrayBuffer;
    try {
      const buffer = Buffer.alloc(Math.min(entry.size, 16 * 1024 ** 2));
      let length = 0;
      while (length < buffer.length) {
        const { bytesRead } = await handle.read(buffer, length, buffer.length - length, length);
        if (!bytesRead) break;
        length += bytesRead;
      }
      // alloc owns its backing store. Only a concurrently truncated RAW needs a smaller copy.
      head = length === buffer.length ? buffer.buffer as ArrayBuffer
        : buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + length) as ArrayBuffer;
    } finally { await handle.close(); }
    const jpeg = extractRafThumbnail(head);
    const metadata = extractRafQuickMetadata(head);
    if (jpeg) await cache.putThumb(key, new Uint8Array(await jpeg.arrayBuffer()));
    const facts = await cache.putHeadMetadata(key, metadata);
    await cache.markHead(key);
    return { thumbnail: jpeg ? cache.thumbPath(key) : null, metadata: facts.metadata };
  });
}
