/**
 * Extract embedded JPEG thumbnail and quick metadata from RAW files.
 *
 * Supports:
 * - Fujifilm RAF: custom header with known JPEG offset at 0x54
 * - TIFF-based formats (ARW, CR2, NEF, DNG, RW2, ORF, PEF, etc.):
 *   parses IFD chain to find JPEG thumbnail and Make/Model tags
 */

// ── RAF extraction ──────────────────────────────────────────────────

function extractRafJpeg(buffer: ArrayBuffer): Blob | null {
  if (buffer.byteLength < 0x5C) return null;
  const view = new DataView(buffer);

  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 16));
  if (!magic.startsWith('FUJIFILMCCD-RAW')) return null;

  const jpegOffset = view.getUint32(0x54, false);
  const jpegLength = view.getUint32(0x58, false);

  if (jpegOffset === 0 || jpegLength === 0) return null;
  if (jpegOffset + jpegLength > buffer.byteLength) return null;

  const soi = view.getUint16(jpegOffset, false);
  if (soi !== 0xFFD8) return null;

  return new Blob([new Uint8Array(buffer, jpegOffset, jpegLength)], { type: 'image/jpeg' });
}

function extractRafMeta(buffer: ArrayBuffer): { camera: string } | null {
  if (buffer.byteLength < 0x70) return null;
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 16));
  if (!magic.startsWith('FUJIFILMCCD-RAW')) return null;

  const camera = new TextDecoder()
    .decode(new Uint8Array(buffer, 0x1C, 0x20))
    .replace(/\0+$/, '')
    .trim();

  return { camera };
}

// ── TIFF/EXIF extraction ────────────────────────────────────────────

const TAG_MAKE = 0x010F;
const TAG_MODEL = 0x0110;
const TAG_JPEG_OFFSET = 0x0201;   // JPEGInterchangeFormat
const TAG_JPEG_LENGTH = 0x0202;   // JPEGInterchangeFormatLength
const TAG_STRIP_OFFSETS = 0x0111;
const TAG_STRIP_LENGTHS = 0x0117;
const TAG_SUB_IFD = 0x014A;
const TAG_NEW_SUBFILE_TYPE = 0x00FE;

const TYPE_SIZES: Record<number, number> = {
  1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8,
};

function isTiff(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 8) return false;
  const b = new Uint8Array(buffer, 0, 4);
  return (b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2A && b[3] === 0x00) || // II (LE)
         (b[0] === 0x4D && b[1] === 0x4D && b[2] === 0x00 && b[3] === 0x2A);   // MM (BE)
}

function getU16(view: DataView, offset: number, le: boolean): number {
  return view.getUint16(offset, le);
}

function getU32(view: DataView, offset: number, le: boolean): number {
  return view.getUint32(offset, le);
}

function readString(buffer: ArrayBuffer, offset: number, length: number): string {
  return new TextDecoder()
    .decode(new Uint8Array(buffer, offset, length))
    .replace(/\0+$/, '')
    .trim();
}

interface IfdEntry {
  tag: number;
  type: number;
  count: number;
  valueOffset: number; // raw 4 bytes from the entry (value or offset)
}

function parseIfd(
  view: DataView,
  buffer: ArrayBuffer,
  ifdOffset: number,
  le: boolean,
): { entries: IfdEntry[]; nextIfd: number } | null {
  if (ifdOffset + 2 > buffer.byteLength) return null;
  const count = getU16(view, ifdOffset, le);
  if (ifdOffset + 2 + count * 12 + 4 > buffer.byteLength) return null;

  const entries: IfdEntry[] = [];
  for (let i = 0; i < count; i++) {
    const base = ifdOffset + 2 + i * 12;
    const tag = getU16(view, base, le);
    const type = getU16(view, base + 2, le);
    const cnt = getU32(view, base + 4, le);
    const valueOffset = getU32(view, base + 8, le);
    entries.push({ tag, type, count: cnt, valueOffset });
  }

  const nextIfd = getU32(view, ifdOffset + 2 + count * 12, le);
  return { entries, nextIfd };
}

function getEntryString(entry: IfdEntry, buffer: ArrayBuffer): string {
  const size = entry.count;
  if (size <= 4) {
    // Inline ASCII (rare for make/model but handle it)
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setUint32(0, entry.valueOffset, false);
    return new TextDecoder().decode(bytes).replace(/\0+$/, '').trim();
  }
  if (entry.valueOffset + size > buffer.byteLength) return '';
  return readString(buffer, entry.valueOffset, size);
}

interface TiffResult {
  jpeg: Blob | null;
  make: string;
  model: string;
}

function parseTiff(buffer: ArrayBuffer): TiffResult {
  const result: TiffResult = { jpeg: null, make: '', model: '' };
  if (!isTiff(buffer)) return result;

  const view = new DataView(buffer);
  const le = new Uint8Array(buffer, 0, 1)[0] === 0x49;
  const firstIfd = getU32(view, 4, le);

  let bestJpeg: { offset: number; length: number } | null = null;

  // Walk the IFD chain (IFD0, IFD1, ...)
  let ifdOffset = firstIfd;
  let ifdIndex = 0;
  const maxIfds = 10;

  while (ifdOffset > 0 && ifdIndex < maxIfds) {
    const ifd = parseIfd(view, buffer, ifdOffset, le);
    if (!ifd) break;

    let jpegOff = 0;
    let jpegLen = 0;

    for (const entry of ifd.entries) {
      switch (entry.tag) {
        case TAG_MAKE:
          if (!result.make) result.make = getEntryString(entry, buffer);
          break;
        case TAG_MODEL:
          if (!result.model) result.model = getEntryString(entry, buffer);
          break;
        case TAG_JPEG_OFFSET:
          jpegOff = entry.type === 3
            ? getU16(view, entry.count > 2 ? entry.valueOffset : ifdOffset + 2 + ifd.entries.indexOf(entry) * 12 + 8, le)
            : entry.valueOffset;
          break;
        case TAG_JPEG_LENGTH:
          jpegLen = entry.type === 3
            ? getU16(view, entry.count > 2 ? entry.valueOffset : ifdOffset + 2 + ifd.entries.indexOf(entry) * 12 + 8, le)
            : entry.valueOffset;
          break;
        case TAG_SUB_IFD: {
          // Parse SubIFDs for larger previews
          const subCount = Math.min(entry.count, 4);
          for (let s = 0; s < subCount; s++) {
            const subOffset = subCount === 1 && (TYPE_SIZES[entry.type] ?? 4) * entry.count <= 4
              ? entry.valueOffset
              : entry.valueOffset + s * 4 <= buffer.byteLength - 4
                ? getU32(view, entry.valueOffset + s * 4, le)
                : 0;
            if (subOffset > 0 && subOffset < buffer.byteLength) {
              const sub = parseIfd(view, buffer, subOffset, le);
              if (sub) {
                let subJpegOff = 0, subJpegLen = 0;
                let isReduced = false;
                for (const se of sub.entries) {
                  if (se.tag === TAG_NEW_SUBFILE_TYPE) {
                    const val = se.valueOffset;
                    if (val & 1) isReduced = true; // reduced resolution image
                  }
                  if (se.tag === TAG_JPEG_OFFSET) subJpegOff = se.valueOffset;
                  if (se.tag === TAG_JPEG_LENGTH) subJpegLen = se.valueOffset;
                  if (se.tag === TAG_STRIP_OFFSETS) subJpegOff = se.valueOffset;
                  if (se.tag === TAG_STRIP_LENGTHS) subJpegLen = se.valueOffset;
                }
                if (isReduced && subJpegOff > 0 && subJpegLen > 0) {
                  if (!bestJpeg || subJpegLen > bestJpeg.length) {
                    bestJpeg = { offset: subJpegOff, length: subJpegLen };
                  }
                }
              }
            }
          }
          break;
        }
      }
    }

    // IFD1 typically contains the EXIF thumbnail
    if (jpegOff > 0 && jpegLen > 0) {
      if (!bestJpeg || jpegLen > bestJpeg.length) {
        bestJpeg = { offset: jpegOff, length: jpegLen };
      }
    }

    ifdOffset = ifd.nextIfd;
    ifdIndex++;
  }

  // Validate and extract best JPEG
  if (bestJpeg && bestJpeg.offset + bestJpeg.length <= buffer.byteLength) {
    const soi = getU16(view, bestJpeg.offset, false); // JPEG is always big-endian
    if (soi === 0xFFD8) {
      result.jpeg = new Blob(
        [new Uint8Array(buffer, bestJpeg.offset, bestJpeg.length)],
        { type: 'image/jpeg' },
      );
    }
  }

  return result;
}

// ── Public API (unchanged signatures) ───────────────────────────────

export function extractRafThumbnail(buffer: ArrayBuffer): Blob | null {
  // Try RAF first
  const raf = extractRafJpeg(buffer);
  if (raf) return raf;

  // Fall back to TIFF/EXIF
  return parseTiff(buffer).jpeg;
}

export function extractRafQuickMetadata(buffer: ArrayBuffer): { camera: string } | null {
  // Try RAF first
  const raf = extractRafMeta(buffer);
  if (raf) return raf;

  // Fall back to TIFF/EXIF
  const tiff = parseTiff(buffer);
  if (tiff.make || tiff.model) {
    const camera = [tiff.make, tiff.model].filter(Boolean).join(' ');
    return { camera };
  }

  return null;
}
