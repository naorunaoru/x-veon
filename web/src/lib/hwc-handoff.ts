/**
 * Single-slot transient HWC handoff between processing (producer) and
 * OutputCanvas (consumer). Avoids the OPFS compress/decompress round-trip
 * for immediate display after processing completes.
 *
 * At most one Float32Array lives here at a time. Consumed on first read.
 */

let slot: { key: string; hwc: Float32Array } | null = null;
let clipSlot: { key: string; mask: Float32Array } | null = null;

export function setHwc(key: string, hwc: Float32Array): void {
  slot = { key, hwc };
}

export function takeHwc(key: string): Float32Array | null {
  if (slot?.key === key) {
    const hwc = slot.hwc;
    slot = null;
    return hwc;
  }
  return null;
}

export function setClipMask(key: string, mask: Float32Array): void {
  clipSlot = { key, mask };
}

export function takeClipMask(key: string): Float32Array | null {
  if (clipSlot?.key === key) {
    const mask = clipSlot.mask;
    clipSlot = null;
    return mask;
  }
  return null;
}
