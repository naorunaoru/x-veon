// ── Directory handles (lazy-init singletons) ───────────────────────────────

let hwcDir: FileSystemDirectoryHandle | null = null;
let rawDir: FileSystemDirectoryHandle | null = null;

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

// ── HWC (demosaiced result) storage ─────────────────────────────────────────

/** Build the OPFS key for a file+method pair. */
export function hwcKey(fileId: string, method: string): string {
  return `${fileId}::${method}`;
}

/** Write a Float32Array to OPFS. */
export async function writeHwc(key: string, hwc: Float32Array): Promise<void> {
  const dir = await getHwcDir();
  const fh = await dir.getFileHandle(key, { create: true });
  const writable = await fh.createWritable();
  await writable.write(hwc.buffer as ArrayBuffer);
  await writable.close();
}

/** Read a Float32Array from OPFS. Returns null if missing. */
export async function readHwc(key: string): Promise<Float32Array | null> {
  try {
    const dir = await getHwcDir();
    const fh = await dir.getFileHandle(key);
    const file = await fh.getFile();
    const buffer = await file.arrayBuffer();
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

// ── Combined cleanup ────────────────────────────────────────────────────────

/** Delete all OPFS data (raw + hwc) for a file. */
export async function deleteAllForFile(fileId: string): Promise<void> {
  await Promise.all([
    deleteRawForFile(fileId),
    deleteHwcForFile(fileId),
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
