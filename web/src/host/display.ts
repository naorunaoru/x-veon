import { mediaQueryHeadroom } from '@/lib/display';
import type { DisplayHost, HdrDisplayInfo } from '@/host';
// Probe HDR display capabilities.
// With WebGPU, HDR canvas (rgba16float + toneMapping: extended) is always available.
// The only question is whether the physical display supports extended range.

/** Whether the Window Management API is available (may still need permission). */
export function hasWindowManagementApi(): boolean {
  return 'getScreenDetails' in window;
}

interface HeadroomResult {
  headroom: number;
  accurate: boolean;
}

/** Whether the Window Management permission is already granted (so asking can't prompt). */
async function windowManagementGranted(): Promise<boolean> {
  try {
    const status = await navigator.permissions.query({ name: 'window-management' as PermissionName });
    return status.state === 'granted';
  } catch {
    return false;
  }
}

/** Read peak/SDR luminance ratio from the Window Management API, screen API, or media query. */
async function getHdrHeadroom(): Promise<HeadroomResult> {
  const hdrMedia = typeof matchMedia !== 'undefined' && matchMedia('(dynamic-range: high)').matches;

  // 1. Window Management API — most accurate (gives real nit-based headroom). Only on displays
  //    the browser already treats as HDR, and only once permission is granted: calling
  //    getScreenDetails() in the "prompt" state shows the browser's permission prompt and
  //    stalls initialisation until it's answered. HdrPermissionDialog asks explicitly instead.
  if (hdrMedia && 'getScreenDetails' in window && (await windowManagementGranted())) {
    try {
      const details = await (window as any).getScreenDetails();
      const hr = details?.currentScreen?.highDynamicRangeHeadroom;
      if (typeof hr === 'number' && hr > 1.0) return { headroom: hr, accurate: true };
    } catch {
      // Revoked or unavailable — fall through
    }
  }

  // 2. screen.highDynamicRangeHeadroom (not yet available in most browsers)
  if (typeof screen !== 'undefined' && screen.highDynamicRangeHeadroom != null) {
    const hr = screen.highDynamicRangeHeadroom;
    if (typeof hr === 'number' && hr > 1.0) return { headroom: hr, accurate: true };
  }

  // 3. Media query — knows HDR is supported but not the headroom value
  return mediaQueryHeadroom(hdrMedia);
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

export function createDisplayHost(): DisplayHost {
  return {
    probe: probeHdrDisplay,
    ...('getScreenDetails' in window ? { requestAccurateHeadroom: requestWindowManagementHeadroom } : {}),
  };
}
