/**
 * Off-thread HWC worker: handles byte-shuffle/unshuffle, gzip compress/decompress,
 * and OPFS I/O so the main thread stays responsive during large buffer operations.
 */

const HWC_MAGIC = 0x31435748;

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

let hwcDir: FileSystemDirectoryHandle | null = null;
async function getHwcDir(): Promise<FileSystemDirectoryHandle> {
  if (hwcDir) return hwcDir;
  const root = await navigator.storage.getDirectory();
  hwcDir = await root.getDirectoryHandle('hwc-cache', { create: true });
  return hwcDir;
}

self.onmessage = async (e: MessageEvent) => {
  const { type, id, key, buffer } = e.data;

  if (type === 'write') {
    const raw = new Uint8Array(buffer);
    const t0 = performance.now();
    const compressed = await gzipCompress(byteShuffle4(raw));
    const dt = performance.now() - t0;

    const origMB = (raw.byteLength / (1024 * 1024)).toFixed(1);
    const compMB = (compressed.byteLength / (1024 * 1024)).toFixed(1);
    const ratio = (raw.byteLength / compressed.byteLength).toFixed(1);
    console.log(`[opfs] compress ${origMB} MB \u2192 ${compMB} MB (${ratio}x) in ${dt.toFixed(0)} ms`);

    const out = new Uint8Array(4 + compressed.byteLength);
    new DataView(out.buffer).setUint32(0, HWC_MAGIC, true);
    out.set(new Uint8Array(compressed), 4);

    const dir = await getHwcDir();
    const fh = await dir.getFileHandle(key, { create: true });
    const writable = await fh.createWritable();
    await writable.write(out.buffer);
    await writable.close();
    return;
  }

  if (type === 'read') {
    try {
      const dir = await getHwcDir();
      const fh = await dir.getFileHandle(key);
      const file = await fh.getFile();
      const buf = await file.arrayBuffer();

      if (buf.byteLength > 4 && new DataView(buf).getUint32(0, true) === HWC_MAGIC) {
        const t0 = performance.now();
        const decompressed = await gzipDecompress(buf.slice(4));
        const result = byteUnshuffle4(new Uint8Array(decompressed));
        const dt = performance.now() - t0;
        const compMB = ((buf.byteLength - 4) / (1024 * 1024)).toFixed(1);
        const origMB = (decompressed.byteLength / (1024 * 1024)).toFixed(1);
        console.log(`[opfs] decompress ${compMB} MB \u2192 ${origMB} MB in ${dt.toFixed(0)} ms`);
        self.postMessage({ id, buffer: result.buffer }, [result.buffer] as any);
      } else {
        // Legacy: uncompressed
        self.postMessage({ id, buffer: buf }, [buf] as any);
      }
    } catch (e) {
      self.postMessage({ id, error: (e as DOMException).name });
    }
    return;
  }
};
