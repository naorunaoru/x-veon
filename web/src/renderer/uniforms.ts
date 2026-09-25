import { sanitizeOpenDrtConfig, type GradingConfig, type TonescaleParams } from './grading/opendrt-params';
import { IDENTITY_3X3, P3D65_TO_REC709, P3D65_TO_REC2020, computeCwpAdaptMatrix } from './color-matrices';

// Float offsets into the uniform buffer shadow array.
export const U_TS           = 0;   // vec4: ts_s, ts_s1, ts_m2, ts_dsc
export const U_FLAGS        = 4;   // vec4: ts_x0, hdrDisplay, exportMode, ptl_enable
export const U_ODRT_TONE    = 8;   // vec4: tn_con, tn_sh, tn_toe, tn_off
export const U_ODRT_RS      = 12;  // vec4: rs_sa, rs_rw, rs_bw, rec2020_output
export const U_ODRT_PT      = 16;  // vec4: pt_r, pt_g, pt_b, pt_rng_low
export const U_ODRT_PT2     = 20;  // vec4: pt_rng_high, ptm_high_st, 0, 0
export const U_ODRT_LCON    = 24;  // vec4: enable, tn_lcon, tn_lcon_w, tn_lcon_pc
export const U_ODRT_HCON    = 28;  // vec4: enable, tn_hcon, tn_hcon_pv, tn_hcon_st
export const U_ODRT_BRL_RGB = 32;  // vec4: brl_r, brl_g, brl_b, brl_rng
export const U_ODRT_BRL_CMY = 36;  // vec4: brl_c, brl_m, brl_y, enable
export const U_ODRT_PTM     = 40;  // vec4: enable, ptm_low, ptm_low_st, ptm_high
export const U_PREPROCESS   = 44;  // vec4: exposure, wb_temp, wb_tint, sharpen_amount
export const U_TEXEL        = 48;  // vec4: texel_w, texel_h, 0, 0
export const U_SRGB_P3_C0   = 52;  // 3 × vec4: columns 0–2 (offsets 52, 56, 60)
export const U_P3_DSP_C0    = 64;  // 3 × vec4: columns 0–2 (offsets 64, 68, 72)
export const U_ODRT_HS_RGB  = 76;  // vec4: enable, hs_r, hs_g, hs_b
export const U_ODRT_HS_ETC  = 80;  // vec4: hs_rgb_rng, hs_cmy_enable, hc_enable, hc_r
export const U_ODRT_HS_CMY  = 84;  // vec4: hs_c, hs_m, hs_y, cwp_rng
export const U_CWP_C0       = 88;  // 3 × vec4: cwp adaptation matrix columns (offsets 88, 92, 96)
export const U_VIEW_X       = 100; // source UV → clip X; w = display filtering
export const U_VIEW_Y       = 104; // source UV → clip Y
export const UNIFORM_FLOATS  = 108;
export const UNIFORM_BYTES   = UNIFORM_FLOATS * 4; // 432

export function setMat3(d: Float32Array, offset: number, m: Float32Array): void {
  // Column 0: [r0c0, r1c0, r2c0, 0]
  d[offset]     = m[0]; d[offset + 1] = m[3]; d[offset + 2] = m[6]; d[offset + 3] = 0;
  // Column 1: [r0c1, r1c1, r2c1, 0]
  d[offset + 4] = m[1]; d[offset + 5] = m[4]; d[offset + 6] = m[7]; d[offset + 7] = 0;
  // Column 2: [r0c2, r1c2, r2c2, 0]
  d[offset + 8] = m[2]; d[offset + 9] = m[5]; d[offset + 10] = m[8]; d[offset + 11] = 0;
}

export function applyOpenDrtUniforms(d: Float32Array, ts: TonescaleParams, cfg: GradingConfig, outputGamut: 'rec709' | 'p3' | 'rec2020'): void {

  cfg = sanitizeOpenDrtConfig(cfg);
  setMat3(d, U_P3_DSP_C0, outputGamut === 'rec709' ? P3D65_TO_REC709 : outputGamut === 'rec2020' ? P3D65_TO_REC2020 : IDENTITY_3X3);

  // Tonescale params
  d[U_TS]     = ts.ts_s;
  d[U_TS + 1] = ts.ts_s1;
  d[U_TS + 2] = ts.ts_m2;
  d[U_TS + 3] = ts.ts_dsc;
  d[U_FLAGS]  = ts.ts_x0;
  // FLAGS[1] = hdrDisplay — set by caller
  // FLAGS[2] = exportMode — set by caller
  d[U_FLAGS + 3] = cfg.ptl_enable ? 1.0 : 0.0;

  // OpenDRT packed vec4s
  d[U_ODRT_TONE]     = cfg.tn_con;
  d[U_ODRT_TONE + 1] = cfg.tn_sh;
  d[U_ODRT_TONE + 2] = cfg.tn_toe;
  d[U_ODRT_TONE + 3] = cfg.tn_off;

  d[U_ODRT_RS]     = cfg.rs_sa;
  d[U_ODRT_RS + 1] = cfg.rs_rw;
  d[U_ODRT_RS + 2] = cfg.rs_bw;
  d[U_ODRT_RS + 3] = outputGamut === 'rec2020' ? 1 : 0;

  d[U_ODRT_PT]     = cfg.pt_r;
  d[U_ODRT_PT + 1] = cfg.pt_g;
  d[U_ODRT_PT + 2] = cfg.pt_b;
  d[U_ODRT_PT + 3] = cfg.pt_rng_low;

  d[U_ODRT_PT2]     = cfg.pt_rng_high;
  d[U_ODRT_PT2 + 1] = cfg.ptm_high_st;
  d[U_ODRT_PT2 + 2] = 0;
  d[U_ODRT_PT2 + 3] = 0;

  d[U_ODRT_LCON]     = cfg.tn_lcon_enable ? 1.0 : 0.0;
  d[U_ODRT_LCON + 1] = cfg.tn_lcon;
  d[U_ODRT_LCON + 2] = cfg.tn_lcon_w;
  d[U_ODRT_LCON + 3] = cfg.tn_lcon_pc;

  d[U_ODRT_HCON]     = cfg.tn_hcon_enable ? 1.0 : 0.0;
  d[U_ODRT_HCON + 1] = cfg.tn_hcon;
  d[U_ODRT_HCON + 2] = cfg.tn_hcon_pv;
  d[U_ODRT_HCON + 3] = cfg.tn_hcon_st;

  d[U_ODRT_BRL_RGB]     = cfg.brl_r;
  d[U_ODRT_BRL_RGB + 1] = cfg.brl_g;
  d[U_ODRT_BRL_RGB + 2] = cfg.brl_b;
  d[U_ODRT_BRL_RGB + 3] = cfg.brl_rng;

  d[U_ODRT_BRL_CMY]     = cfg.brl_c;
  d[U_ODRT_BRL_CMY + 1] = cfg.brl_m;
  d[U_ODRT_BRL_CMY + 2] = cfg.brl_y;
  d[U_ODRT_BRL_CMY + 3] = cfg.brl_enable ? 1.0 : 0.0;

  d[U_ODRT_PTM]     = cfg.ptm_enable ? 1.0 : 0.0;
  d[U_ODRT_PTM + 1] = cfg.ptm_low;
  d[U_ODRT_PTM + 2] = cfg.ptm_low_st;
  d[U_ODRT_PTM + 3] = cfg.ptm_high;

  // Hue shift / hue contrast / creative white
  d[U_ODRT_HS_RGB]     = cfg.hs_rgb_enable ? 1.0 : 0.0;
  d[U_ODRT_HS_RGB + 1] = cfg.hs_r;
  d[U_ODRT_HS_RGB + 2] = cfg.hs_g;
  d[U_ODRT_HS_RGB + 3] = cfg.hs_b;

  d[U_ODRT_HS_ETC]     = cfg.hs_rgb_rng;
  d[U_ODRT_HS_ETC + 1] = cfg.hs_cmy_enable ? 1.0 : 0.0;
  d[U_ODRT_HS_ETC + 2] = cfg.hc_enable ? 1.0 : 0.0;
  d[U_ODRT_HS_ETC + 3] = cfg.hc_r;

  d[U_ODRT_HS_CMY]     = cfg.hs_c;
  d[U_ODRT_HS_CMY + 1] = cfg.hs_m;
  d[U_ODRT_HS_CMY + 2] = cfg.hs_y;
  d[U_ODRT_HS_CMY + 3] = cfg.cwp_rng;

  // Creative white adaptation matrix (identity when cwp=0)
  if (cfg.cwp > 0) {
    const cwpAdapt = computeCwpAdaptMatrix(outputGamut !== 'rec709', cfg.cwp);
    setMat3(d, U_CWP_C0, cwpAdapt);
  } else {
    setMat3(d, U_CWP_C0, IDENTITY_3X3);
  }

  // Pre-processing
  d[U_PREPROCESS]     = cfg.exposure;
  d[U_PREPROCESS + 1] = cfg.wb_temp;
  d[U_PREPROCESS + 2] = cfg.wb_tint;
  d[U_PREPROCESS + 3] = cfg.sharpen_amount;
}
