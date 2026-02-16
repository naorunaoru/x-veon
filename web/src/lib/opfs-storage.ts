let hwcDir: FileSystemDirectoryHandle | null = null;

async function getHwcDir(): Promise<FileSystemDirectoryHandle> {
  if (hwcDir) return hwcDir;
  const root = await navigator.storage.getDirectory();
  hwcDir = await root.getDirectoryHandle('hwc-cache', { create: true });
  return hwcDir;
}

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

/** Delete all OPFS entries for a given file ID (all method variants). */
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
