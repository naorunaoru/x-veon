import fs from 'node:fs/promises';
import { extractRafThumbnail, extractRafQuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { QuickMetadata } from '@/pipeline/decode/raf-thumbnail';
import type { FolderEntry } from './folder';
import type { createCache } from './cache';

type Cache = ReturnType<typeof createCache>;
type Head = { thumbnail: string | null; metadata: QuickMetadata | null };
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
  const handle = await fs.open(entry.path, 'r');
  let head: ArrayBuffer;
  try {
    const buffer = Buffer.alloc(Math.min(entry.size, 16 * 1024 ** 2));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    head = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + bytesRead) as ArrayBuffer;
  } finally { await handle.close(); }
  const jpeg = extractRafThumbnail(head);
  const metadata = extractRafQuickMetadata(head);
  if (jpeg) await cache.putThumb(key, new Uint8Array(await jpeg.arrayBuffer()));
  const facts = await cache.putHeadMetadata(key, metadata);
  await cache.markHead(key);
  return { thumbnail: jpeg ? cache.thumbPath(key) : null, metadata: facts.metadata };
}
