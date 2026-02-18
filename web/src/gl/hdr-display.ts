// Probe HDR display capabilities.
// With WebGPU, HDR canvas (rgba16float + toneMapping: extended) is always available.
// The only question is whether the physical display supports extended range.

export interface HdrDisplayInfo {
  /** Whether the display supports HDR (headroom > 1.0). */
  supported: boolean;
  /** Peak-to-SDR luminance ratio (e.g. 4.0 = 400 nit peak). 1.0 if unknown/SDR. */
  headroom: number;
}

/** Read peak/SDR luminance ratio from the Window Management API, screen API, or media query. */
export async function getHdrHeadroom(): Promise<number> {
  // 1. Window Management API — most accurate (gives real nit-based headroom)
  try {
    if ('getScreenDetails' in window) {
      const details = await (window as any).getScreenDetails();
      const hr = details?.currentScreen?.highDynamicRangeHeadroom;
      if (typeof hr === 'number' && hr > 1.0) return hr;
    }
  } catch {
    // Permission denied or API unavailable — fall through
  }

  // 2. screen.highDynamicRangeHeadroom (not yet available in most browsers)
  if (typeof screen !== 'undefined' && screen.highDynamicRangeHeadroom != null) {
    const hr = screen.highDynamicRangeHeadroom;
    if (typeof hr === 'number' && hr > 1.0) return hr;
  }

  // 3. Media query — knows HDR is supported but not the headroom value
  if (typeof matchMedia !== 'undefined' && matchMedia('(dynamic-range: high)').matches) {
    return 2.0; // conservative default
  }

  return 1.0;
}

/**
 * Probe display HDR capability.
 * WebGPU always supports extended tone mapping — this just checks whether
 * the physical display has HDR headroom.
 */
export async function probeHdrDisplay(): Promise<HdrDisplayInfo> {
  const headroom = await getHdrHeadroom();
  return { supported: headroom > 1.0, headroom };
}
