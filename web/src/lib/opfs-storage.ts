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

// ── HWC (demosaiced result) storage ─────────────────────────────────────────

/** Build the OPFS key for a file+method pair. */
export function hwcKey(fileId: string, method: string): string {
  return `${fileId}::${method}`;
}

// ── Off-thread HWC worker (compress/decompress + OPFS I/O) ──────────────────

let hwcWorker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, {
  resolve: (v: Float32Array | null) => void;
}>();

function getHwcWorker(): Worker {
  if (hwcWorker) return hwcWorker;
  hwcWorker = new Worker(new URL('./hwc-worker.ts', import.meta.url), { type: 'module' });
  hwcWorker.onmessage = (e: MessageEvent) => {
    const { id, buffer, error } = e.data;
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    if (error || !buffer) {
      p.resolve(null);
    } else {
      p.resolve(new Float32Array(buffer));
    }
  };
  return hwcWorker;
}

/**
 * Write a Float32Array to OPFS (byte-shuffled + gzip) on a background worker.
 * Returns immediately — compression and I/O happen off the main thread.
 */
export function writeHwc(key: string, hwc: Float32Array): void {
  const worker = getHwcWorker();
  const copy = hwc.buffer.slice(hwc.byteOffset, hwc.byteOffset + hwc.byteLength);
  worker.postMessage({ type: 'write', key, buffer: copy }, [copy]);
}

/** Read a Float32Array from OPFS on a background worker. Returns null if missing. */
export async function readHwc(key: string): Promise<Float32Array | null> {
  const worker = getHwcWorker();
  const id = nextId++;
  return new Promise((resolve) => {
    pending.set(id, { resolve });
    worker.postMessage({ type: 'read', id, key });
  });
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
