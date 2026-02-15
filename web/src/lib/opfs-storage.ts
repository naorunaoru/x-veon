let hwcDir: FileSystemDirectoryHandle | null = null;

async function getHwcDir(): Promise<FileSystemDirectoryHandle> {
  if (hwcDir) return hwcDir;
  const root = await navigator.storage.getDirectory();
  hwcDir = await root.getDirectoryHandle('hwc-cache', { create: true });
  return hwcDir;
}

/** Write a Float32Array to OPFS, keyed by file ID. */
export async function writeHwc(fileId: string, hwc: Float32Array): Promise<void> {
  const dir = await getHwcDir();
  const fh = await dir.getFileHandle(fileId, { create: true });
  const writable = await fh.createWritable();
  await writable.write(hwc.buffer as ArrayBuffer);
  await writable.close();
}

/** Read a Float32Array from OPFS by file ID. Returns null if missing. */
export async function readHwc(fileId: string): Promise<Float32Array | null> {
  try {
    const dir = await getHwcDir();
    const fh = await dir.getFileHandle(fileId);
    const file = await fh.getFile();
    const buffer = await file.arrayBuffer();
    return new Float32Array(buffer);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return null;
    throw e;
  }
}

/** Delete the OPFS file for a given file ID. No-op if missing. */
export async function deleteHwc(fileId: string): Promise<void> {
  try {
    const dir = await getHwcDir();
    await dir.removeEntry(fileId);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'NotFoundError') return;
    throw e;
  }
}

/** Clear all OPFS hwc files. Called on startup to avoid stale data. */
export async function clearAllHwc(): Promise<void> {
  try {
    const dir = await getHwcDir();
    // @ts-expect-error keys() exists at runtime but missing from TS lib types
    for await (const key of dir.keys() as AsyncIterableIterator<string>) {
      await dir.removeEntry(key);
    }
  } catch {
    // OPFS unavailable or empty
  }
}
