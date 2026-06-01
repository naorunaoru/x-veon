import type { OpenDrtConfig, PreProcessConfig } from '@/gl/opendrt-params';

/** Every tool-rail panel id. Phase 2 wires exposure/whiteBalance/brilliance/looks/settings;
 *  scopes/toneCurve/advanced/crop are reserved for later phases. */
export type PanelId =
  | 'scopes' | 'exposure' | 'whiteBalance' | 'toneCurve'
  | 'brilliance' | 'looks' | 'advanced' | 'settings' | 'crop';

export interface SectionKeys {
  drt: (keyof OpenDrtConfig)[];
  pre: (keyof PreProcessConfig)[];
}

/** Override keys owned by each grading section — drives the modified dot and per-section reset. */
export const SECTION_KEYS: Partial<Record<PanelId, SectionKeys>> = {
  exposure: { drt: ['tn_con', 'tn_lcon', 'tn_lcon_enable'], pre: ['exposure'] },
  whiteBalance: { drt: [], pre: ['wb_temp', 'wb_tint'] },
  brilliance: { drt: ['brl_r', 'brl_g', 'brl_b', 'brl_enable'], pre: [] },
};

/** A section is modified when any of its override keys is present on the file. */
export function isSectionModified(
  section: PanelId,
  drtOverrides: Partial<OpenDrtConfig>,
  preOverrides: Partial<PreProcessConfig>,
): boolean {
  const keys = SECTION_KEYS[section];
  if (!keys) return false;
  return keys.drt.some((k) => k in drtOverrides) || keys.pre.some((k) => k in preOverrides);
}
