// Probe HDR display capabilities.
// With WebGPU, HDR canvas (rgba16float + toneMapping: extended) is always available.
// The only question is whether the physical display supports extended range.

export interface HdrDisplayInfo {
  /** Whether the display supports HDR (headroom > 1.0). */
  supported: boolean;
  /** Peak-to-SDR luminance ratio (e.g. 4.0 = 400 nit peak). 1.0 if unknown/SDR. */
  headroom: number;
  /** Whether headroom is precisely known (vs. conservative media-query fallback). */
  accurate: boolean;
}

/** Whether the Window Management API is available (may still need permission). */
export function hasWindowManagementApi(): boolean {
  return 'getScreenDetails' in window;
}

interface HeadroomResult {
  headroom: number;
  accurate: boolean;
}

/** Read peak/SDR luminance ratio from the Window Management API, screen API, or media query. */
async function getHdrHeadroom(): Promise<HeadroomResult> {
  // 1. Window Management API — most accurate (gives real nit-based headroom)
  try {
    if ('getScreenDetails' in window) {
      const details = await (window as any).getScreenDetails();
      const hr = details?.currentScreen?.highDynamicRangeHeadroom;
      if (typeof hr === 'number' && hr > 1.0) return { headroom: hr, accurate: true };
    }
  } catch {
    // Permission denied or no user gesture — fall through
  }

  // 2. screen.highDynamicRangeHeadroom (not yet available in most browsers)
  if (typeof screen !== 'undefined' && screen.highDynamicRangeHeadroom != null) {
    const hr = screen.highDynamicRangeHeadroom;
    if (typeof hr === 'number' && hr > 1.0) return { headroom: hr, accurate: true };
  }

  // 3. Media query — knows HDR is supported but not the headroom value
  if (typeof matchMedia !== 'undefined' && matchMedia('(dynamic-range: high)').matches) {
    return { headroom: 2.0, accurate: false };
  }

  return { headroom: 1.0, accurate: true };
}

/**
 * Request accurate headroom via Window Management API.
 * Must be called from a user gesture context (click handler) so the browser
 * can show the permission prompt.
 */
export async function requestWindowManagementHeadroom(): Promise<number | null> {
  try {
    if ('getScreenDetails' in window) {
      const details = await (window as any).getScreenDetails();
      const hr = details?.currentScreen?.highDynamicRangeHeadroom;
      if (typeof hr === 'number' && hr > 1.0) return hr;
    }
  } catch {
    // Permission denied
  }
  return null;
}

/**
 * Probe display HDR capability.
 * WebGPU always supports extended tone mapping — this just checks whether
 * the physical display has HDR headroom.
 */
export async function probeHdrDisplay(): Promise<HdrDisplayInfo> {
  const { headroom, accurate } = await getHdrHeadroom();
  return { supported: headroom > 1.0, headroom, accurate };
}
