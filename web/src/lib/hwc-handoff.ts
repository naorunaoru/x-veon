/**
 * Single-slot transient handoff between processing (producer) and
 * OutputCanvas (consumer). Transfers a GPU-resident RGBA32F buffer
 * directly — no CPU readback involved.
 *
 * At most one buffer lives here at a time. Consumed on first read.
 */

export interface GpuHandoff {
  buffer: GPUBuffer;
  bytesPerRow: number;
}

let slot: { key: string; handoff: GpuHandoff } | null = null;

export function setGpuResult(key: string, handoff: GpuHandoff): void {
  // If there's an unclaimed buffer from a previous run, destroy it
  if (slot && slot.key !== key) {
    slot.handoff.buffer.destroy();
  }
  slot = { key, handoff };
}

export function takeGpuResult(key: string): GpuHandoff | null {
  if (slot?.key === key) {
    const handoff = slot.handoff;
    slot = null;
    return handoff;
  }
  return null;
}
