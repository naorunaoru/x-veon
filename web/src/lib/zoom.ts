/** Zoom = scale where 1 == 100% (1:1 pixels). The pill slider covers a log range. */
export const ZOOM_MIN = 0.1; // 10%
export const ZOOM_MAX = 4.0; // 400%

const LMIN = Math.log(ZOOM_MIN);
const LMAX = Math.log(ZOOM_MAX);

/** Slider position [0,1] → zoom (log-scaled). */
export function sliderToZoom(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return Math.exp(LMIN + c * (LMAX - LMIN));
}

/** Zoom → slider position [0,1], clamped. */
export function zoomToSlider(zoom: number): number {
  const t = (Math.log(zoom) - LMIN) / (LMAX - LMIN);
  return Math.min(1, Math.max(0, t));
}

/** Display string: "Fit" when at fit scale, else "NNN%". */
export function formatZoom(scale: number, fitScale: number): string {
  if (Math.abs(scale - fitScale) < fitScale * 0.01) return 'Fit';
  return `${Math.round(scale * 100)}%`;
}
