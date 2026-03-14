// ── Directory handles (lazy-init singletons) ───────────────────────────────

let rawDir: FileSystemDirectoryHandle | null = null;
let thumbDir: FileSystemDirectoryHandle | null = null;

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

/** Delete all OPFS data (raw + thumbnail) for a file. */
export async function deleteAllForFile(fileId: string): Promise<void> {
  await Promise.all([
    deleteRawForFile(fileId),
    deleteThumbnailForFile(fileId),
  ]);
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
