#!/usr/bin/env python3
"""OpenDRT reference implementation — Base + Default presets.

Line-for-line port of opendrt_art.ctl transform() for Base and Default presets.
Used to generate golden test vectors for validating the Rust implementation.

Input:  scene-linear P3-D65 RGB
Output: display-linear RGB in target gamut (Rec.709 or Rec.2020), before EOTF
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

SQRT3 = 1.73205080756887729353
PI = math.pi

# ── Matrices (from ART CTL lines 131-143) ──────────────────────────────────

P3D65_TO_REC709 = np.array([
    [ 1.224940181, -0.2249402404,  0.0],
    [-0.04205697775, 1.042057037, -1.4901e-08],
    [-0.01963755488,-0.07863604277, 1.098273635],
], dtype=np.float64)

P3D65_TO_REC2020 = np.array([
    [0.7538330344, 0.1985973691, 0.04756959659],
    [0.04574384897, 0.9417772198, 0.01247893122],
    [-0.001210340355, 0.0176017173, 0.9836086231],
], dtype=np.float64)

REC2020_TO_XYZ = np.array([
    [0.636958122253, 0.144616916776, 0.168880969286],
    [0.262700229883, 0.677998125553, 0.059301715344],
    [0.0,            0.028072696179, 1.060985088348],
], dtype=np.float64)

XYZ_TO_P3D65 = np.array([
    [ 2.49349691194, -0.931383617919, -0.402710784451],
    [-0.829488969562, 1.76266406032,   0.023624685842],
    [ 0.035845830244,-0.076172389268,  0.956884524008],
], dtype=np.float64)


# ── Math helpers (CTL lines 157-423) ────────────────────────────────────────

def spowf(a: float, b: float) -> float:
    """Safe power: returns a unchanged if a <= 0."""
    if a <= 0.0:
        return a
    return a ** b


def sdivf(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0
    return a / b


def compress_hyperbolic_power(x: float, s: float, p: float) -> float:
    """CTL line 338-342: x/(x+s)^p"""
    return spowf(x / (x + s), p)


def compress_toe_quadratic(x: float, toe: float, inv: bool) -> float:
    """CTL line 344-353: Quadratic toe."""
    if toe == 0.0:
        return x
    if not inv:
        return spowf(x, 2.0) / (x + toe)
    else:
        return (x + math.sqrt(x * (4.0 * toe + x))) / 2.0


def compress_toe_cubic(x: float, m: float, w: float, inv: bool) -> float:
    """CTL line 355-368: Cubic toe."""
    if m == 1.0:
        return x
    x2 = x * x
    if not inv:
        return x * (x2 + m * w) / (x2 + w)
    else:
        p0 = x2 - 3.0 * m * w
        p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w
        p2 = (max(0.0, x2 * p1 * p1 - 4 * p0**3) ** 0.5 / 2.0 + x * p1 / 2.0) ** (1.0 / 3.0)
        return p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0


def contrast_high(x: float, p: float, pv: float, pv_lx: float) -> float:
    """CTL line 382-399: High contrast with linear extension."""
    x0 = 0.18 * 2.0**pv
    if x < x0 or p == 1.0:
        return x
    o = x0 - x0 / p
    s0 = x0**(1.0 - p) / p
    x1 = x0 * 2.0**pv_lx
    k1 = p * s0 * x1**p / x1
    y1 = s0 * x1**p + o
    if x > x1:
        return k1 * (x - x1) + y1
    else:
        return s0 * x**p + o


def complement_power(x: float, p: float) -> float:
    """CTL line 370-373"""
    return 1.0 - spowf(1.0 - x, 1.0 / p)


def sigmoid_cubic(x: float, s: float) -> float:
    """CTL line 375-380"""
    if x < 0.0 or x > 1.0:
        return 1.0
    return 1.0 + s * (1.0 - 3.0 * x * x + 2.0 * x * x * x)


def gauss_window(x: float, w: float) -> float:
    """CTL line 411-416"""
    y = x / w
    return math.exp(-y * y)


def hue_offset(h: float, o: float) -> float:
    """CTL line 419-423"""
    return math.fmod(h - o + PI, 2.0 * PI) - PI


def softplus(x: float, s: float, x0: float, y0: float) -> float:
    """CTL line 401-409: Smooth clamp."""
    if x > 10.0 * s + y0 or s < 1e-3:
        return x
    m = 1.0
    if abs(y0) > 1e-6:
        m = math.exp(y0 / s)
    m = m - math.exp(x0 / s)
    return s * math.log(max(0.0, m + math.exp(x / s)))


# ── Preset parameters ────────────────────────────────────────────────────────

BASE_PRESET = dict(
    tn_Lg=18.0,
    tn_con=1.4,
    tn_sh=0.5,
    tn_toe=0.003,
    tn_off=0.0,
    tn_lcon_enable=False,
    tn_lcon=0.0,
    tn_lcon_w=0.5,
    tn_lcon_pc=1.0,
    tn_hcon_enable=False,
    tn_hcon=0.0,
    tn_hcon_pv=1.0,
    tn_hcon_st=4.0,
    rs_sa=0.35,
    rs_rw=0.25,
    rs_bw=0.5,
    pt_r=1.0,
    pt_g=2.0,
    pt_b=2.5,
    pt_rng_low=0.25,
    pt_rng_high=0.25,
    ptl_enable=True,
    ptm_enable=False,
    ptm_low=0.0,
    ptm_low_st=0.5,
    ptm_high=0.0,
    ptm_high_st=0.3,
    brl_enable=False,
    brl_r=0.0, brl_g=0.0, brl_b=0.0,
    brl_c=0.0, brl_m=0.0, brl_y=0.0,
    brl_rng=0.5,
)

DEFAULT_PRESET = dict(
    tn_Lg=18.0,
    tn_con=1.4,
    tn_sh=0.5,
    tn_toe=0.003,
    tn_off=0.005,
    tn_lcon_enable=True,
    tn_lcon=1.0,
    tn_lcon_w=0.5,
    tn_lcon_pc=1.0,
    tn_hcon_enable=False,
    tn_hcon=0.0,
    tn_hcon_pv=1.0,
    tn_hcon_st=4.0,
    rs_sa=0.35,
    rs_rw=0.25,
    rs_bw=0.55,
    pt_r=0.5,
    pt_g=2.0,
    pt_b=2.0,
    pt_rng_low=0.2,
    pt_rng_high=0.8,
    ptl_enable=True,
    ptm_enable=True,
    ptm_low=0.2,
    ptm_low_st=0.5,
    ptm_high=-0.8,
    ptm_high_st=0.3,
    brl_enable=True,
    brl_r=-0.5, brl_g=-0.4, brl_b=-0.2,
    brl_c=0.0, brl_m=0.0, brl_y=0.0,
    brl_rng=0.66,
)


def tonescale_params(tn_Lp: float, tn_gb: float, pt_hdr: float, p: dict) -> dict:
    """Pre-compute pixel-independent tonescale constants. CTL lines 516-531."""
    ts_x1 = 2.0 ** (6.0 * p["tn_sh"] + 4.0)
    ts_y1 = tn_Lp / 100.0
    ts_x0 = 0.18 + p["tn_off"]
    ts_y0 = (p["tn_Lg"] / 100.0) * (1.0 + tn_gb * math.log2(max(ts_y1, 1e-10)))
    ts_s0 = compress_toe_quadratic(ts_y0, p["tn_toe"], inv=True)
    ts_s10 = ts_x0 * (ts_s0 ** (-1.0 / p["tn_con"]) - 1.0)
    ts_m1 = ts_y1 / (ts_x1 / (ts_x1 + ts_s10)) ** p["tn_con"]
    ts_m2 = compress_toe_quadratic(ts_m1, p["tn_toe"], inv=True)
    ts_s = ts_x0 * ((ts_s0 / ts_m2) ** (-1.0 / p["tn_con"]) - 1.0)
    ts_dsc = 100.0 / tn_Lp

    # HDR purity blend
    pt_cmp_Lf = pt_hdr * min(1.0, (tn_Lp - 100.0) / 900.0)
    s_Lp100 = ts_x0 * ((p["tn_Lg"] / 100.0) ** (-1.0 / p["tn_con"]) - 1.0)
    ts_s1 = ts_s * pt_cmp_Lf + s_Lp100 * (1.0 - pt_cmp_Lf)

    return dict(ts_s=ts_s, ts_s1=ts_s1, ts_m2=ts_m2, ts_dsc=ts_dsc, ts_x0=ts_x0)


def transform(r: float, g: float, b: float,
              tn_Lp: float, tn_gb: float, pt_hdr: float,
              display_gamut: int,
              preset: dict | None = None) -> tuple[float, float, float]:
    """OpenDRT transform with full look modules.

    Args:
        r, g, b: scene-linear P3-D65 input
        tn_Lp: peak luminance (100 for SDR, 1000 for HDR)
        tn_gb: HDR grey boost (0.13 typical)
        pt_hdr: HDR purity (0.5 typical)
        display_gamut: 0=Rec.709, 2=Rec.2020

    Returns:
        (r, g, b) display-linear in target gamut, before EOTF, clamped [0,1]
    """
    p = preset or BASE_PRESET
    ts = tonescale_params(tn_Lp, tn_gb, pt_hdr, p)

    # ── Stage 1: Input already in P3-D65 ──

    # ── Stage 2: Rendering space saturation + offset (CTL 539-547) ──
    rs_w = np.array([p["rs_rw"], 1.0 - p["rs_rw"] - p["rs_bw"], p["rs_bw"]])
    rgb = np.array([r, g, b], dtype=np.float64)
    sat_L = float(np.dot(rgb, rs_w))
    rgb = sat_L * p["rs_sa"] + rgb * (1.0 - p["rs_sa"])

    # Offset
    rgb = rgb + p["tn_off"]

    # ── Contrast Low (CTL 553-585) ──
    if p["tn_lcon_enable"]:
        mcon_m = 2.0 ** (-p["tn_lcon"])
        mcon_w_half = p["tn_lcon_w"] / 4.0
        mcon_w = mcon_w_half * mcon_w_half

        # Normalize for ts_x0 intersection constraint (CTL 559)
        mcon_cnst_sc = compress_toe_cubic(ts["ts_x0"], mcon_m, mcon_w, inv=True) / ts["ts_x0"]
        rgb = rgb * mcon_cnst_sc

        # Ratio-preserving midtone contrast scale (CTL 563-564)
        rgb_clamped = np.maximum(rgb, 0.0)
        mcon_nm = float(np.sqrt(np.sum(rgb_clamped ** 2))) / SQRT3
        mcon_sc = (mcon_nm * mcon_nm + mcon_m * mcon_w) / (mcon_nm * mcon_nm + mcon_w)

        if p["tn_lcon_pc"] > 0.0:
            # Per-channel midtone contrast (CTL 570-573)
            mcon_rgb = np.array([
                compress_toe_cubic(rgb[0], mcon_m, mcon_w, inv=False),
                compress_toe_cubic(rgb[1], mcon_m, mcon_w, inv=False),
                compress_toe_cubic(rgb[2], mcon_m, mcon_w, inv=False),
            ])

            # Blend ratio-preserving vs per-channel (CTL 576-580)
            mcon_mx = float(np.max(rgb))
            mcon_mn = float(np.min(rgb))
            mcon_ch = max(0.0, min(1.0, 1.0 - sdivf(mcon_mn, mcon_mx)))
            mcon_ch = mcon_ch ** (4.0 * p["tn_lcon_pc"])
            rgb = rgb * mcon_sc * mcon_ch + mcon_rgb * (1.0 - mcon_ch)
        else:
            rgb = rgb * mcon_sc

    # ── Tonescale norm (CTL 588-594) ──
    rgb_clamped = np.maximum(rgb, 0.0)
    tsn = float(np.sqrt(np.sum(rgb_clamped ** 2))) / SQRT3

    # Purity compression norm (CTL 591)
    ts_pt = math.sqrt(max(0.0,
        rgb[0] * rgb[0] * p["pt_r"] +
        rgb[1] * rgb[1] * p["pt_g"] +
        rgb[2] * rgb[2] * p["pt_b"]))

    # RGB ratios (CTL 594)
    rgb_clamp_low = np.maximum(rgb, -2.0)
    if tsn > 0.0:
        rgb = rgb_clamp_low / tsn
    else:
        rgb = np.zeros(3)

    # ── High contrast (CTL 599-603) ──
    if p["tn_hcon_enable"]:
        hcon_p = 2.0 ** p["tn_hcon"]
        tsn = contrast_high(tsn, hcon_p, p["tn_hcon_pv"], p["tn_hcon_st"])
        ts_pt = contrast_high(ts_pt, hcon_p, p["tn_hcon_pv"], p["tn_hcon_st"])

    # ── Apply tonescale (CTL 605-607) ──
    tsn = compress_hyperbolic_power(tsn, ts["ts_s"], p["tn_con"])
    ts_pt = compress_hyperbolic_power(ts_pt, ts["ts_s1"], p["tn_con"])

    # ── Opponent space / achromatic distance (CTL 610-616) ──
    opp_cy = rgb[0] - rgb[2]
    opp_gm = rgb[1] - (rgb[0] + rgb[2]) / 2.0
    ach_d = math.sqrt(max(0.0, opp_cy * opp_cy + opp_gm * opp_gm)) / SQRT3
    # Smooth ach_d (CTL 616)
    ach_d = 1.25 * compress_toe_quadratic(ach_d, 0.25, inv=False)

    # ── Hue angle + RGB/CMY hue windows (CTL 618-634) ──
    hue = math.fmod(math.atan2(opp_cy, opp_gm) + PI + 1.10714931, 2.0 * PI)

    ha_rgb = [
        gauss_window(hue_offset(hue, 0.1), 0.9),
        gauss_window(hue_offset(hue, 4.3), 0.9),
        gauss_window(hue_offset(hue, 2.3), 0.9),
    ]
    ha_cmy = [
        gauss_window(hue_offset(hue, 3.3), 0.6),
        gauss_window(hue_offset(hue, 1.3), 0.6),
        gauss_window(hue_offset(hue, -1.2), 0.6),
    ]

    # ── Purity compression range (CTL 637-643) ──
    ts_pt_cmp = 1.0 - spowf(ts_pt, 1.0 / p["pt_rng_low"])

    pt_rng_high_f = min(1.0, ach_d / 1.2)
    pt_rng_high_f = pt_rng_high_f * pt_rng_high_f
    if p["pt_rng_high"] < 1.0:
        pt_rng_high_f = 1.0 - pt_rng_high_f

    ts_pt_cmp = (spowf(ts_pt_cmp, p["pt_rng_high"]) * (1.0 - pt_rng_high_f)
                 + ts_pt_cmp * pt_rng_high_f)

    # ── Brilliance (CTL 646-660) ──
    brl_f = 1.0
    if p["brl_enable"]:
        brl_f = (-p["brl_r"] * ha_rgb[0] - p["brl_g"] * ha_rgb[1] - p["brl_b"] * ha_rgb[2]
                 - p["brl_c"] * ha_cmy[0] - p["brl_m"] * ha_cmy[1] - p["brl_y"] * ha_cmy[2])
        brl_f = (1.0 - ach_d) * brl_f + 1.0 - brl_f
        brl_f = softplus(brl_f, 0.25, -100.0, 0.0)

        # Limit by tonescale (CTL 656-658)
        brl_ts = (1.0 - ts_pt) if brl_f > 1.0 else ts_pt
        brl_lim = spowf(brl_ts, 1.0 - p["brl_rng"])
        brl_f = brl_f * brl_lim + 1.0 - brl_lim
        brl_f = max(0.0, min(2.0, brl_f))

    # ── Mid-Range Purity (CTL 664-679) ──
    ptm_sc = 1.0
    if p["ptm_enable"]:
        # Mid Purity Low (CTL 672-673)
        ptm_ach_d_low = complement_power(ach_d, p["ptm_low_st"])
        ptm_sc = sigmoid_cubic(ptm_ach_d_low, p["ptm_low"] * (1.0 - ts_pt))

        # Mid Purity High (CTL 676-677)
        ptm_ach_d_high = complement_power(ach_d, p["ptm_high_st"]) * (1.0 - ts_pt) + ach_d * ach_d * ts_pt
        ptm_sc *= sigmoid_cubic(ptm_ach_d_high, p["ptm_high"] * ts_pt)
        ptm_sc = max(0.0, ptm_sc)

    # ── Apply brilliance (CTL 724-725) ──
    rgb = rgb * brl_f

    # ── Apply purity compression + mid purity (CTL 728-729) ──
    ts_pt_cmp *= ptm_sc
    rgb = rgb * ts_pt_cmp + (1.0 - ts_pt_cmp)

    # ── Inverse rendering space (CTL 731-733) ──
    sat_L = float(np.dot(rgb, rs_w))
    rgb = (sat_L * p["rs_sa"] - rgb) / (p["rs_sa"] - 1.0)

    # ── Display gamut conversion (CTL 736-752) ──
    # Creative white: cwp=0 (D65) → no-op
    if display_gamut == 0:  # Rec.709
        rgb = P3D65_TO_REC709 @ rgb

    # For display_gamut == 2 (Rec.2020): stays in P3, converted after clamp

    # ── Purity compress low (CTL 754-763) ──
    if p["ptl_enable"]:
        sum0 = (softplus(rgb[0], 0.2, -100.0, -0.3) +
                rgb[1] +
                softplus(rgb[2], 0.2, -100.0, -0.3))
        rgb[0] = softplus(rgb[0], 0.04, -0.3, 0.0)
        rgb[1] = softplus(rgb[1], 0.06, -0.3, 0.0)
        rgb[2] = softplus(rgb[2], 0.01, -0.05, 0.0)
        total = rgb[0] + rgb[1] + rgb[2]
        ptl_norm = min(1.0, sdivf(sum0, total))
        rgb = rgb * ptl_norm

    # ── Final tonescale: toe + display scale (CTL 765-771) ──
    tsn = tsn * ts["ts_m2"]
    tsn = compress_toe_quadratic(tsn, p["tn_toe"], inv=False)
    tsn = tsn * ts["ts_dsc"]

    # Return from RGB ratios to absolute values
    rgb = rgb * tsn

    # Clamp [0, 1]
    rgb = np.clip(rgb, 0.0, 1.0)

    # Rec.2020: P3 → Rec.2020 (CTL 777-781)
    if display_gamut == 2:
        rgb = np.maximum(rgb, 0.0)
        rgb = P3D65_TO_REC2020 @ rgb

    return float(rgb[0]), float(rgb[1]), float(rgb[2])


# ── Test vector generation ──────────────────────────────────────────────────

def generate_test_vectors() -> dict:
    """Generate comprehensive test vectors for Base and Default presets."""

    # Test inputs in P3-D65 scene-linear
    inputs = {
        # Grey ramp
        "grey_0.001": [0.001, 0.001, 0.001],
        "grey_0.005": [0.005, 0.005, 0.005],
        "grey_0.01":  [0.01, 0.01, 0.01],
        "grey_0.05":  [0.05, 0.05, 0.05],
        "grey_0.18":  [0.18, 0.18, 0.18],
        "grey_0.5":   [0.5, 0.5, 0.5],
        "grey_1.0":   [1.0, 1.0, 1.0],
        "grey_2.0":   [2.0, 2.0, 2.0],
        "grey_4.0":   [4.0, 4.0, 4.0],
        "grey_8.0":   [8.0, 8.0, 8.0],
        "grey_16.0":  [16.0, 16.0, 16.0],
        "grey_50.0":  [50.0, 50.0, 50.0],
        "grey_100.0": [100.0, 100.0, 100.0],

        # Pure primaries at key levels
        "red_0.18":   [0.18, 0.0, 0.0],
        "red_1.0":    [1.0, 0.0, 0.0],
        "red_4.0":    [4.0, 0.0, 0.0],
        "green_0.18": [0.0, 0.18, 0.0],
        "green_1.0":  [0.0, 1.0, 0.0],
        "green_4.0":  [0.0, 4.0, 0.0],
        "blue_0.18":  [0.0, 0.0, 0.18],
        "blue_1.0":   [0.0, 0.0, 1.0],
        "blue_4.0":   [0.0, 0.0, 4.0],

        # Secondaries
        "yellow_1.0":  [1.0, 1.0, 0.0],
        "cyan_1.0":    [0.0, 1.0, 1.0],
        "magenta_1.0": [1.0, 0.0, 1.0],

        # Near-neutral (purity compression edge)
        "near_neutral": [0.18, 0.17, 0.19],

        # Near-clip mixed
        "near_clip_neutral": [0.95, 0.95, 0.95],
        "near_clip_red":     [1.0, 0.2, 0.2],

        # Negative component
        "negative": [-0.01, 0.5, 0.3],

        # HDR values
        "hdr_mixed":    [5.0, 0.5, 0.1],
        "hdr_bright":   [10.0, 8.0, 6.0],
        "hdr_saturated": [20.0, 0.5, 0.5],
    }

    configs = {
        "sdr_709": dict(tn_Lp=100.0, tn_gb=0.13, pt_hdr=0.5, display_gamut=0),
        "hdr_2020": dict(tn_Lp=1000.0, tn_gb=0.13, pt_hdr=0.5, display_gamut=2),
    }

    presets = {
        "base": BASE_PRESET,
        "default": DEFAULT_PRESET,
    }

    vectors = []
    for preset_name, preset in presets.items():
        for input_name, rgb_in in inputs.items():
            for config_name, cfg in configs.items():
                r_out, g_out, b_out = transform(
                    rgb_in[0], rgb_in[1], rgb_in[2],
                    cfg["tn_Lp"], cfg["tn_gb"], cfg["pt_hdr"], cfg["display_gamut"],
                    preset=preset,
                )
                vectors.append({
                    "name": f"{input_name}__{config_name}__{preset_name}",
                    "preset": preset_name,
                    "input": rgb_in,
                    "config": config_name,
                    "tn_Lp": cfg["tn_Lp"],
                    "tn_gb": cfg["tn_gb"],
                    "pt_hdr": cfg["pt_hdr"],
                    "display_gamut": cfg["display_gamut"],
                    "expected": [r_out, g_out, b_out],
                })

    return {
        "generator": "opendrt_reference.py",
        "preset": "Base + Default (ART CTL)",
        "notes": "Input is P3-D65 scene-linear. Output is display-linear before EOTF.",
        "tolerance": 1e-5,
        "vectors": vectors,
    }


if __name__ == "__main__":
    vectors = generate_test_vectors()

    out_path = Path(__file__).parent / "opendrt_test_vectors.json"
    with open(out_path, "w") as f:
        json.dump(vectors, f, indent=2)

    print(f"Generated {len(vectors['vectors'])} test vectors → {out_path}")

    # Print a few spot checks
    for preset_name in ["base", "default"]:
        print(f"\nSpot checks — {preset_name} preset (SDR Rec.709):")
        for v in vectors["vectors"]:
            if v["preset"] == preset_name and v["config"] == "sdr_709" and v["name"].startswith(f"grey_0.18"):
                inp = v["input"][0]
                out = v["expected"]
                print(f"  {inp:8.3f} → [{out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f}]")
