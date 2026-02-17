// Probe HDR display capabilities.
// With WebGPU, HDR canvas (rgba16float + toneMapping: extended) is always available.
// The only question is whether the physical display supports extended range.

export interface HdrDisplayInfo {
  /** Whether the display supports HDR (headroom > 1.0). */
  supported: boolean;
  /** Peak-to-SDR luminance ratio (e.g. 4.0 = 400 nit peak). 1.0 if unknown/SDR. */
  headroom: number;
}

/** Read peak/SDR luminance ratio from screen API or media query. */
export function getHdrHeadroom(): number {
  if (typeof screen !== 'undefined' && screen.highDynamicRangeHeadroom != null) {
    const hr = screen.highDynamicRangeHeadroom;
    console.log(hr);
    if (typeof hr === 'number' && hr > 1.0) return hr;
  }
  if (typeof matchMedia !== 'undefined' && matchMedia('(dynamic-range: high)').matches) {
    console.log('wrong')
    return 2.0; // conservative default
  }
  return 1.0;
}

/**
 * Probe display HDR capability.
 * WebGPU always supports extended tone mapping — this just checks whether
 * the physical display has HDR headroom.
 */
export function probeHdrDisplay(): HdrDisplayInfo {
  const headroom = getHdrHeadroom();
  return { supported: headroom > 1.0, headroom };
}
