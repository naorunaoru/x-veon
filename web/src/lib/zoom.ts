/** Zoom = scale where 1 == 100% (1:1 pixels). The pill slider covers a log range. */
export const ZOOM_MIN = 0.1; // 10%
export const ZOOM_MAX = 4.0; // 400%

export const minimumZoom = (fitScale: number): number => fitScale * 0.75;

/** Slider position [0,1] → zoom (log-scaled). */
export function sliderToZoom(t: number, min = ZOOM_MIN, max = ZOOM_MAX): number {
  const c = Math.min(1, Math.max(0, t));
  const low = Math.log(min);
  return Math.exp(low + c * (Math.log(max) - low));
}

/** Zoom → slider position [0,1], clamped. */
export function zoomToSlider(zoom: number, min = ZOOM_MIN, max = ZOOM_MAX): number {
  const t = (Math.log(zoom) - Math.log(min)) / (Math.log(max) - Math.log(min));
  return Math.min(1, Math.max(0, t));
}

/** Display string: "Fit" when at fit scale, else "NNN%". */
export function formatZoom(scale: number, fitScale: number): string {
  if (Math.abs(scale - fitScale) < fitScale * 0.01) return 'Fit';
  return `${Math.round(scale * 100)}%`;
}
