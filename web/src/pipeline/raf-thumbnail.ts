/**
 * Extract the embedded JPEG thumbnail from a RAF file.
 *
 * RAF header layout (big-endian):
 *   0x00: "FUJIFILMCCD-RAW " (16 bytes magic)
 *   0x54: uint32 jpeg_offset
 *   0x58: uint32 jpeg_length
 */
export function extractRafThumbnail(buffer: ArrayBuffer): Blob | null {
  if (buffer.byteLength < 0x5C) return null;

  const view = new DataView(buffer);

  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 16));
  if (!magic.startsWith('FUJIFILMCCD-RAW')) return null;

  const jpegOffset = view.getUint32(0x54, false);
  const jpegLength = view.getUint32(0x58, false);

  if (jpegOffset === 0 || jpegLength === 0) return null;
  if (jpegOffset + jpegLength > buffer.byteLength) return null;

  // Validate JPEG SOI marker
  const soi = view.getUint16(jpegOffset, false);
  if (soi !== 0xFFD8) return null;

  const jpegBytes = new Uint8Array(buffer, jpegOffset, jpegLength);
  return new Blob([jpegBytes], { type: 'image/jpeg' });
}

/**
 * Extract camera model from RAF header (bytes 0x1C-0x3C).
 */
export function extractRafQuickMetadata(buffer: ArrayBuffer): { camera: string } | null {
  if (buffer.byteLength < 0x70) return null;

  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 16));
  if (!magic.startsWith('FUJIFILMCCD-RAW')) return null;

  const camera = new TextDecoder()
    .decode(new Uint8Array(buffer, 0x1C, 0x20))
    .replace(/\0+$/, '')
    .trim();

  return { camera };
}
