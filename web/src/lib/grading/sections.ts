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
  toneCurve: { drt: ['tn_sh', 'tn_toe'], pre: [] },
  brilliance: { drt: ['brl_r', 'brl_g', 'brl_b', 'brl_enable'], pre: [] },
  advanced: {
    drt: [
      'tn_lg', 'tn_off', 'tn_lcon_enable', 'tn_lcon_w', 'tn_lcon_pc',
      'tn_hcon_enable', 'tn_hcon', 'tn_hcon_pv', 'tn_hcon_st',
      'cwp', 'cwp_rng',
      'rs_sa', 'rs_rw', 'rs_bw',
      'pt_r', 'pt_g', 'pt_b', 'pt_rng_low', 'pt_rng_high',
      'ptl_enable', 'ptm_enable', 'ptm_low', 'ptm_low_st', 'ptm_high', 'ptm_high_st',
      'brl_c', 'brl_m', 'brl_y', 'brl_rng',
      'hs_rgb_enable', 'hs_r', 'hs_g', 'hs_b', 'hs_rgb_rng',
      'hs_cmy_enable', 'hs_c', 'hs_m', 'hs_y',
      'hc_enable', 'hc_r',
      'peak_luminance',
    ],
    pre: ['sharpen_amount'],
  },
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
