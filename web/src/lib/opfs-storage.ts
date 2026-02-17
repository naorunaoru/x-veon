// ── Directory handles (lazy-init singletons) ───────────────────────────────

let hwcDir: FileSystemDirectoryHandle | null = null;
let rawDir: FileSystemDirectoryHandle | null = null;
let thumbDir: FileSystemDirectoryHandle | null = null;

async function getHwcDir(): Promise<FileSystemDirectoryHandle> {
  if (hwcDir) return hwcDir;
  const root = await navigator.storage.getDirectory();
  hwcDir = await root.getDirectoryHandle('hwc-cache', { create: true });
  return hwcDir;
}

async function getRawDir(): Promise<FileSystemDirectoryHandle> {
  if (rawDir) return rawDir;
  const root = await navigator.storage.getDirectory();
  rawDir = await root.getDirectoryHandle('raw', { create: true });
  return rawDir;
}

async function getThumbDir(): Promise<FileSystemDirectoryHandle> {
  if (thumbDir) return thumbDir;
  const root = await navigator.storage.getDirectory();
  thumbDir = await root.getDirectoryHandle('thumbnails', { create: true });
  return thumbDir;
}

// ── Compression helpers ─────────────────────────────────────────────────────

/** Magic bytes at the start of compressed HWC files ("HWC1" LE). */
const HWC_MAGIC = 0x31435748;

/**
 * Byte-shuffle: reorder [b0,b1,b2,b3, b0,b1,b2,b3, ...] into
 * [all b0s, all b1s, all b2s, all b3s]. Groups similar bytes together
 * (e.g. exponent bytes) for dramatically better gzip compression on float data.
 */
function byteShuffle4(input: Uint8Array): Uint8Array {
  const n = input.length;
  const stride = n >>> 2;
  const out = new Uint8Array(n);
  for (let i = 0; i < stride; i++) {
    const src = i * 4;
    out[i] = input[src];
    out[stride + i] = input[src + 1];
    out[stride * 2 + i] = input[src + 2];
    out[stride * 3 + i] = input[src + 3];
  }
  return out;
}

/** Reverse byte-shuffle: restore interleaved byte order. */
function byteUnshuffle4(input: Uint8Array): Uint8Array {
  const n = input.length;
  const stride = n >>> 2;
  const out = new Uint8Array(n);
  for (let i = 0; i < stride; i++) {
    const dst = i * 4;
    out[dst] = input[i];
    out[dst + 1] = input[stride + i];
    out[dst + 2] = input[stride * 2 + i];
    out[dst + 3] = input[stride * 3 + i];
  }
  return out;
}

async function gzipCompress(data: Uint8Array): Promise<ArrayBuffer> {
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

async function gzipDecompress(data: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

// ── HWC (demosaiced result) storage ─────────────────────────────────────────

/**
 * Single-slot write-through handoff: holds the most recent writeHwc buffer
 * so the immediate readHwc (from OutputCanvas) skips the decompress round-trip.
 * Not consumed on read — React strict mode double-fires effects.
 * Overwritten on next writeHwc, so at most one buffer (~300 MB) in memory.
 */
let hwcHandoff: { key: string; data: Float32Array } | null = null;

/** Build the OPFS key for a file+method pair. */
export function hwcKey(fileId: string, method: string): string {
  return `${fileId}::${method}`;
}

/** Write a Float32Array to OPFS (byte-shuffled + gzip). */
export async function writeHwc(key: string, hwc: Float32Array): Promise<void> {
  hwcHandoff = { key, data: hwc };

  const dir = await getHwcDir();
  const fh = await dir.getFileHandle(key, { create: true });

  const raw = new Uint8Array(hwc.buffer, hwc.byteOffset, hwc.byteLength);
  const t0 = performance.now();
  const compressed = await gzipCompress(byteShuffle4(raw));
  const dt = performance.now() - t0;

  const origMB = (raw.byteLength / (1024 * 1024)).toFixed(1);
  const compMB = (compressed.byteLength / (1024 * 1024)).toFixed(1);
  const ratio = (raw.byteLength / compressed.byteLength).toFixed(1);
  console.log(`[opfs] compress ${origMB} MB → ${compMB} MB (${ratio}x) in ${dt.toFixed(0)} ms`);

  const out = new Uint8Array(4 + compressed.byteLength);
  new DataView(out.buffer).setUint32(0, HWC_MAGIC, true);
  out.set(new Uint8Array(compressed), 4);

  const writable = await fh.createWritable();
  await writable.write(out.buffer);
  await writable.close();
}

/** Read a Float32Array from OPFS. Returns null if missing. */
export async function readHwc(key: string): Promise<Float32Array | null> {
  // Fast path: buffer still in memory from writeHwc (avoids decompress round-trip)
  if (hwcHandoff?.key === key) {
    console.log(`[opfs] handoff hit (skipped decompress)`);
    return hwcHandoff.data;
  }

  try {
    const dir = await getHwcDir();
    const fh = await dir.getFileHandle(key);
    const file = await fh.getFile();
    const buffer = await file.arrayBuffer();

    // Compressed format: 4-byte magic + gzip(shuffled)
    if (buffer.byteLength > 4 && new DataView(buffer).getUint32(0, true) === HWC_MAGIC) {
      const t0 = performance.now();
      const decompressed = await gzipDecompress(buffer.slice(4));
      const unshuffled = byteUnshuffle4(new Uint8Array(decompressed));
      const dt = performance.now() - t0;
      const compMB = ((buffer.byteLength - 4) / (1024 * 1024)).toFixed(1);
      const origMB = (decompressed.byteLength / (1024 * 1024)).toFixed(1);
      console.log(`[opfs] decompress ${compMB} MB → ${origMB} MB in ${dt.toFixed(0)} ms`);
      return new Float32Array(unshuffled.buffer);
    }

    // Legacy: uncompressed Float32Array
    return new Float32Array(buffer);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return null;
    throw e;
  }
}

/** Delete all HWC entries for a given file ID (all method variants). */
export async function deleteHwcForFile(fileId: string): Promise<void> {
  try {
    const dir = await getHwcDir();
    const prefix = `${fileId}::`;
    // @ts-expect-error keys() exists at runtime but missing from TS lib types
    for await (const key of dir.keys() as AsyncIterableIterator<string>) {
      if (key.startsWith(prefix)) {
        await dir.removeEntry(key);
      }
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return;
    throw e;
  }
}

// ── RAW file storage ────────────────────────────────────────────────────────

/** Write original RAW file bytes to OPFS. */
export async function writeRaw(fileId: string, buffer: ArrayBuffer): Promise<void> {
  const dir = await getRawDir();
  const fh = await dir.getFileHandle(fileId, { create: true });
  const writable = await fh.createWritable();
  await writable.write(buffer);
  await writable.close();
}

/** Read original RAW file bytes from OPFS. Returns null if missing. */
export async function readRaw(fileId: string): Promise<ArrayBuffer | null> {
  try {
    const dir = await getRawDir();
    const fh = await dir.getFileHandle(fileId);
    const file = await fh.getFile();
    return await file.arrayBuffer();
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return null;
    throw e;
  }
}

/** Delete the RAW file for a given file ID. */
export async function deleteRawForFile(fileId: string): Promise<void> {
  try {
    const dir = await getRawDir();
    await dir.removeEntry(fileId);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return;
    throw e;
  }
}

// ── Thumbnail storage ───────────────────────────────────────────────────────

/** Write a thumbnail Blob to OPFS. */
export async function writeThumbnail(fileId: string, blob: Blob): Promise<void> {
  const dir = await getThumbDir();
  const fh = await dir.getFileHandle(fileId, { create: true });
  const writable = await fh.createWritable();
  await writable.write(blob);
  await writable.close();
}

/** Read a thumbnail Blob from OPFS. Returns null if missing. */
export async function readThumbnail(fileId: string): Promise<Blob | null> {
  try {
    const dir = await getThumbDir();
    const fh = await dir.getFileHandle(fileId);
    return await fh.getFile();
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return null;
    throw e;
  }
}

/** Delete the thumbnail for a given file ID. */
async function deleteThumbnailForFile(fileId: string): Promise<void> {
  try {
    const dir = await getThumbDir();
    await dir.removeEntry(fileId);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return;
    throw e;
  }
}

// ── Combined cleanup ────────────────────────────────────────────────────────

/** Delete all OPFS data (raw + hwc + thumbnail) for a file. */
export async function deleteAllForFile(fileId: string): Promise<void> {
  await Promise.all([
    deleteRawForFile(fileId),
    deleteHwcForFile(fileId),
    deleteThumbnailForFile(fileId),
  ]);
}

/** Check if HWC data exists for a file+method without reading the full buffer. */
export async function hasHwc(key: string): Promise<boolean> {
  try {
    const dir = await getHwcDir();
    await dir.getFileHandle(key);
    return true;
  } catch {
    return false;
  }
}

/** List all file IDs that have entries in the raw/ directory. */
export async function listRawFileIds(): Promise<Set<string>> {
  const ids = new Set<string>();
  try {
    const dir = await getRawDir();
    // @ts-expect-error keys() exists at runtime but missing from TS lib types
    for await (const key of dir.keys() as AsyncIterableIterator<string>) {
      ids.add(key);
    }
  } catch {
    // Directory may not exist yet
  }
  return ids;
}

/** List all file IDs that have entries in the hwc-cache/ directory. */
export async function listHwcFileIds(): Promise<Set<string>> {
  const ids = new Set<string>();
  try {
    const dir = await getHwcDir();
    // @ts-expect-error keys() exists at runtime but missing from TS lib types
    for await (const key of dir.keys() as AsyncIterableIterator<string>) {
      // Keys are "fileId::method" — extract fileId
      const sep = key.indexOf('::');
      if (sep > 0) ids.add(key.substring(0, sep));
    }
  } catch {
    // Directory may not exist yet
  }
  return ids;
}
