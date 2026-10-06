import type { DisplayHost, DisplayReadings, HdrDisplayInfo } from '@/host';
import { mediaQueryHeadroom } from '@/lib/display';
import type { DesktopBridge } from '../protocol/bridge';

const positive = (n: number | undefined): n is number =>
  typeof n === 'number' && Number.isFinite(n) && n > 0;

/** Spec §8: macOS uses reference EDR above 0, otherwise potential EDR.
 * Windows uses maximum luminance divided by SDR white while HDR is enabled.
 * Incomplete readings return null so the caller can use the media query. */
export function probeFromReadings(r: DisplayReadings): HdrDisplayInfo | null {
  let headroom: number | null = null;
  if (r.referenceEdr !== undefined || r.potentialEdr !== undefined) {
    headroom = positive(r.referenceEdr) ? r.referenceEdr : positive(r.potentialEdr) ? r.potentialEdr : null;
  } else if (r.hdrEnabled === false) headroom = 1;
  else if (r.hdrEnabled === true && positive(r.maxLuminance) && positive(r.sdrWhite))
    headroom = r.maxLuminance / r.sdrWhite;
  if (headroom === null) return null;
  headroom = Math.max(1, headroom);
  return { supported: headroom > 1, headroom, accurate: true };
}

function fallback(): HdrDisplayInfo {
  const info = mediaQueryHeadroom(matchMedia('(dynamic-range: high)').matches);
  return { ...info, supported: info.headroom > 1 };
}

export function createDisplayHost(bridge: Pick<DesktopBridge, 'displayReadings'>): DisplayHost {
  const readings = () => bridge.displayReadings().catch((error: unknown) => {
    console.warn('Display readings failed:', error);
    return null;
  });
  return {
    async probe() { const r = await readings(); return (r && probeFromReadings(r)) ?? fallback(); },
    async readings() { return (await readings()) ?? {}; },
  };
}
