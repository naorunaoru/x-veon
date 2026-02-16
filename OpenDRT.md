# OpenDRT Rust Implementation — Design Document

## Goal

Implement a display rendering transform based on OpenDRT in Rust, integrated into the
existing RAW processing pipeline. The transform produces two display-referred outputs
from scene-linear camera data:

- **SDR base** (sRGB / Rec.709, 8-bit) — for UltraHDR JPEG gain-map workflow
- **HDR render** (Rec.2100 HLG, 10-bit) — for AVIF HDR export

A per-pixel gain map is derived from the ratio of these two renders.

## Reference Implementations

| Source | Format | Notes |
|--------|--------|-------|
| `~/projects/open-display-transform/display-transforms/opendrt/OpenDRT.dctl` | DCTL | Canonical, ~1163 lines. All presets + full stickshift. Different purity compression than ART CTL. |
| `ref/opendrt_art.ctl` | CTL (1294 lines) | Reworked by agriggio for ART. Simplified purity compression. **Primary porting reference.** |
| OpenDRT.nk | Nuke node graph | Same algorithm as DCTL. Easier data flow visualization. |
| [pixls.us discussion](https://discuss.pixls.us/t/opendrt-for-art/49257) | Forum | Post #12 has corrected CTL with gamut fix. Middle grey discussion. |

### DCTL vs ART CTL Differences

The ART CTL is **not** a 1:1 port of the DCTL. Key differences:

| Aspect | DCTL (canonical) | ART CTL (porting reference) |
|---|---|---|
| Purity norm | `pt_lml/pt_lmh` with hue-angle modulation | `sqrt(r²*pt_r + g²*pt_g + b²*pt_b)` weighted norm |
| Purity range | Per-channel with Gaussian hue windows | `pt_rng_low/pt_rng_high` scalar controls |
| Base contrast | `tn_con = 1.66` | `tn_con = 1.4` |
| Base offset | `tn_off = 0.005` | `tn_off = 0.0` |
| Post-brilliance (Base) | ON (`brlp = -0.5`) | OFF |
| EOTF application | Inline (sRGB/HLG/PQ) | None — outputs linear, host applies EOTF |
| Output | Display-encoded in target gamut | Linear Rec.2020 (scaled by `Lp/100`) |

We use the **ART CTL** as primary reference because it's cleaner, the purity compression is
simpler, and the output format (linear) matches our pipeline (we apply EOTF ourselves).

## Pipeline Architecture

```
Camera RAW
    |
    v
Demosaic (existing X-Trans / Bayer neural net)
    |
    v
Camera Linear RGB (WB'd, camera native gamut)
    |
    v
[ color::cam_to_p3d65() ]  ← cam→XYZ→P3-D65
    |
    v
Scene-Linear P3-D65 (OpenDRT working space)
    |
    +----------------------------------+
    v                                  v
+---------------+             +---------------+
|  OpenDRT      |             |  OpenDRT      |
|  SDR path     |             |  HDR path     |
|  Lp=100       |             |  Lp=1000      |
|  Gamut=709    |             |  Gamut=2020   |
+-------+-------+             +-------+-------+
        |                             |
        v                             v
  display-linear               display-linear
  Rec.709 [0,1]               Rec.2020 [0,1]
        |                             |
   sRGB OETF                     HLG OETF
        |                             |
        v                             v
  sRGB 8-bit                   HLG 10-bit
        |                             |
        +--------+    +--------------+
                 v    v
           +--------------+
           |  Gain Map    |
           |  Derivation  |
           +------+-------+
                  |
        +---------+---------+
        v                   v
+----------------+    +----------+
| UltraHDR JPEG  |    | AVIF HLG |
| (SDR+gain map) |    | 10-bit   |
+----------------+    +----------+
```

### A/B Comparison

During development, both the legacy pipeline and OpenDRT must be available side by side:

- Add a `tone_map` parameter to `pipeline::encode()`: `"legacy"` | `"opendrt"`
- Expose as a UI toggle in the web frontend so exports can be compared visually
- Keep the legacy path until OpenDRT is validated across a range of images
- Remove the legacy path only after sign-off

## Working Space: P3-D65

OpenDRT works internally in **P3-D65** — standard DCI-P3 primaries with D65 white point.
Not a custom color space. All matrices are available in the reference implementations.

```rust
// From ART CTL lines 119, 133
pub const P3D65_TO_XYZ: Mat3 = [
    [0.486571133137, 0.265667706728, 0.198217317462],
    [0.228974640369, 0.691738605499, 0.079286918044],
    [0.0,            0.045113388449, 1.043944478035],
];

pub const XYZ_TO_P3D65: Mat3 = [
    [ 2.49349691194, -0.931383617919, -0.402710784451],
    [-0.829488969562, 1.76266406032,   0.023624685842],
    [ 0.035845830244,-0.076172389268,  0.956884524008],
];

// Display output matrices (from ART CTL lines 131-133, 142-143)
pub const XYZ_TO_REC709: Mat3 = [
    [ 3.2409699419, -1.53738317757, -0.498610760293],
    [-0.969243636281, 1.87596750151,  0.041555057407],
    [ 0.055630079697,-0.203976958889, 1.05697151424],
];

pub const XYZ_TO_REC2020: Mat3 = [
    [ 1.71665118797, -0.355670783776, -0.253366281374],
    [-0.666684351832, 1.61648123664,   0.015768545814],
    [ 0.017639857445,-0.042770613258,  0.942103121235],
];

// P3-D65 → display gamut (composed: P3→XYZ→display)
pub const P3D65_TO_REC709: Mat3 = [...]; // matrix_p3_to_rec709_d65 from ART CTL line 142
pub const P3D65_TO_REC2020: Mat3 = [...]; // matrix_p3_to_rec2020 from ART CTL line 143
```

Camera-specific at runtime:
```rust
pub fn cam_to_p3d65(xyz_to_cam: &Mat3) -> Mat3 {
    // row-normalize xyz_to_cam, invert → cam_to_xyz, then multiply by XYZ_TO_P3D65
    let cam_to_xyz = build_cam_to_xyz(xyz_to_cam);
    mul3x3(&XYZ_TO_P3D65, &cam_to_xyz)
}
```

## Color Space Helper Module (`color.rs`)

Consolidate all color space conversions into `color.rs`. Currently scattered across
ad-hoc matrix construction; unify with clear naming:

```rust
// All const matrices from reference (P3-D65 is the working space)
pub const P3D65_TO_XYZ: Mat3 = ...;
pub const XYZ_TO_P3D65: Mat3 = ...;
pub const XYZ_TO_SRGB: Mat3 = ...;   // same as XYZ_TO_REC709
pub const XYZ_TO_BT2020: Mat3 = ...;
pub const SRGB_TO_BT2020: Mat3 = ...;
pub const P3D65_TO_REC709: Mat3 = ...;
pub const P3D65_TO_REC2020: Mat3 = ...;

// Creative white point matrices (CAT02 adaptation, from ART CTL lines 136-143)
pub const P3D65_TO_REC709_D55: Mat3 = ...;
pub const P3D65_TO_REC709_D50: Mat3 = ...;
// etc.

// Camera-specific (built from DNG metadata at runtime)
pub fn build_cam_to_xyz(xyz_to_cam: &Mat3) -> Mat3;   // row-normalize + invert
pub fn cam_to_p3d65(xyz_to_cam: &Mat3) -> Mat3;       // cam→XYZ→P3-D65
pub fn build_cam_to_srgb(xyz_to_cam: &Mat3) -> Mat3;  // existing (legacy path)
pub fn build_cam_to_bt2020(xyz_to_cam: &Mat3) -> Mat3; // existing (legacy path)
```

## OpenDRT Module Breakdown

Sequential pipeline from ART CTL `transform()` function (lines 427-796).
Each stage operates on RGB triplets in P3-D65 working space.

### Stage 1 — Input Gamut Conversion (CTL lines 534-536)

```c
rgb = vdot(in_to_xyz, rgb);       // input gamut → XYZ
rgb = vdot(matrix_xyz_to_p3d65, rgb);  // XYZ → P3-D65
```

For us: single `cam_to_p3d65()` matrix, dot product per pixel.

### Stage 2 — Rendering Space Saturation + Offset (CTL lines 539-547)

```c
// Weighted luminance with custom red/blue weights
float3 rs_w = {rs_rw, 1.0 - rs_rw - rs_bw, rs_bw};  // Base: {0.25, 0.25, 0.5}
float sat_L = dot(rgb, rs_w);
rgb = sat_L * rs_sa + rgb * (1.0 - rs_sa);  // rs_sa = 0.35

// Scene-linear offset
rgb += tn_off;  // Base: 0.0 (ART) or 0.005 (DCTL)
```

### Stage 3 — Tonescale (CTL lines 509-606)

Pre-computed constraint solver (pixel-independent):
```c
ts_x1 = 2^(6*tn_sh + 4)                              // Shoulder intersection
ts_y1 = tn_Lp / 100.0                                 // Display peak normalized
ts_x0 = 0.18 + tn_off                                 // Grey point + offset
ts_y0 = (tn_Lg/100) * (1 + tn_gb * log2(ts_y1))      // Grey with HDR boost
ts_s0 = compress_toe_quadratic(ts_y0, tn_toe, inv=1)  // Inverse toe at grey
ts_s10 = ts_x0 * (ts_s0^(-1/tn_con) - 1)
ts_m1 = ts_y1 / (ts_x1/(ts_x1 + ts_s10))^tn_con
ts_m2 = compress_toe_quadratic(ts_m1, tn_toe, inv=1)
ts_s = ts_x0 * ((ts_s0/ts_m2)^(-1/tn_con) - 1)       // Final shape parameter
ts_dsc = 100.0 / tn_Lp                                // Display encoding scale
```

Per-pixel:
```c
// Tonescale norm = euclidean length of clamped RGB / sqrt(3)
tsn = hypot(max(rgb, 0)) / SQRT3

// Purity compression norm (weighted)
ts_pt = sqrt(r²*pt_r + g²*pt_g + b²*pt_b)  // Base: pt_r=1, pt_g=2, pt_b=2.5

// Extract RGB ratios (preserve ratios, apply tonescale to norm)
rgb = clamp_min(rgb, -2.0) / tsn

// Core compression: x/(x+s)^p
tsn = (tsn / (tsn + ts_s))^tn_con
ts_pt = (ts_pt / (ts_pt + ts_s1))^tn_con
```

Toe function (CTL line 344-352):
```c
// Forward: x² / (x + toe)
// Inverse: (x + sqrt(x*(4*toe + x))) / 2
```

### Stage 4 — Purity Compression (CTL lines 637-729)

ART CTL version (simpler than DCTL):
```c
// Purity compression factor from tonescale
ts_pt_cmp = 1.0 - ts_pt^(1/pt_rng_low)  // Base: pt_rng_low = 0.25

// Achromatic-distance modulation
pt_rng_high_f = min(1.0, ach_d/1.2)²
pt_rng_high_f = (pt_rng_high < 1.0) ? 1.0 - pt_rng_high_f : pt_rng_high_f
ts_pt_cmp = ts_pt_cmp^pt_rng_high * (1 - pt_rng_high_f) + ts_pt_cmp * pt_rng_high_f

// Application: lerp RGB ratios toward neutral (1.0)
rgb = rgb * ts_pt_cmp + (1.0 - ts_pt_cmp)
```

Purity Compress Low (CTL lines 754-763) — gamut boundary protection:
```c
// Softplus on individual channels to prevent hard clipping
rgb.x = softplus(rgb.x, 0.04, -0.3, 0.0)
rgb.y = softplus(rgb.y, 0.06, -0.3, 0.0)
rgb.z = softplus(rgb.z, 0.01, -0.05, 0.0)
// Normalize to preserve total energy
```

### Stage 5 — Brilliance (CTL lines 646-660)

Disabled in Base preset. Darkens saturated colors by modulating tonescale norm
based on hue-weighted achromatic distance. Phase 2.

### Stage 6 — Hue Adjustments (CTL lines 688-722)

Disabled in Base preset. Hue shift RGB/CMY and hue contrast using Gaussian
windows on opponent-space hue angle. Phase 3.

### Stage 7 — Display Encoding (CTL lines 731-795)

```c
// Inverse rendering space (undo saturation from Stage 2)
sat_L = dot(rgb, rs_w)
rgb = (sat_L * rs_sa - rgb) / (rs_sa - 1.0)

// Convert P3-D65 → display gamut
if display_gamut == Rec709:
    rgb = P3_TO_REC709 * rgb      // with optional creative whitepoint
if display_gamut == Rec2020:
    rgb = P3_TO_REC2020 * rgb

// Toe + display scale
tsn = compress_toe_quadratic(tsn * ts_m2, tn_toe, forward)
tsn *= ts_dsc

// Return from RGB ratios to absolute values
rgb *= tsn

// Clamp [0, 1]
```

We then apply sRGB OETF or HLG OETF from `transfer.rs`.

## Base Preset — Complete Parameter Values

From ART CTL lines 1028-1077:

```rust
// Tonescale
tn_Lg: 11.1,         // Grey luminance (nits)
tn_con: 1.4,         // Contrast (power exponent)
tn_sh: 0.5,          // Shoulder position
tn_toe: 0.003,       // Toe strength
tn_off: 0.0,         // Scene-linear offset

// Rendering space
rs_sa: 0.35,          // Saturation amount
rs_rw: 0.25,          // Red weight
rs_bw: 0.5,           // Blue weight

// Purity compression
pt_r: 1.0,            // Red weight for purity norm
pt_g: 2.0,            // Green weight
pt_b: 2.5,            // Blue weight
pt_rng_low: 0.25,     // Low range
pt_rng_high: 0.25,    // High range
ptl_enable: true,      // Purity compress low (gamut boundary)

// All disabled in Base:
// contrast_low, contrast_high, brilliance, hue_shift, hue_contrast, mid_purity
```

## Math Functions (for Rust port)

All from ART CTL lines 338-423:

```rust
/// Hyperbolic power compression: x/(x+s)^p
fn compress_hp(x: f32, s: f32, p: f32) -> f32 {
    spowf(x / (x + s), p)
}

/// Quadratic toe. Forward: x²/(x+toe). Inverse: (x+sqrt(x*(4*toe+x)))/2
fn compress_toe(x: f32, toe: f32, inverse: bool) -> f32

/// Cubic toe (for contrast low module — Phase 2)
fn compress_toe_cubic(x: f32, m: f32, w: f32, inverse: bool) -> f32

/// Softplus: smooth clamp. s*ln(m + exp(x/s))
fn softplus(x: f32, s: f32, x0: f32, y0: f32) -> f32

/// Gaussian window for hue angles
fn gauss_window(x: f32, w: f32) -> f32 { (-x*x/(w*w)).exp() }

/// Safe power: returns x unchanged if x <= 0
fn spowf(x: f32, p: f32) -> f32 { if x <= 0.0 { x } else { x.powf(p) } }
```

## Gain Map Derivation

Both SDR and HDR paths produce display-linear outputs before EOTF. To derive the gain map:

1. SDR path outputs display-linear Rec.709 [0,1] (before sRGB OETF)
2. HDR path outputs display-linear Rec.2020 [0,1] where 1.0 = Lp nits (before HLG OETF)
3. Per-pixel luminance: `Y = 0.2126*R + 0.7152*G + 0.0722*B`
4. Gain: `gain = log2((Y_hdr + offset) / (Y_sdr + offset))` with `offset = 1/64`
5. Quantize to 8-bit, encode as grayscale JPEG

Keep the `offset = 1/64` term for numerical stability in shadows (matches Adobe/Google
spec). Do not use raw `log2(hdr/sdr)`.

Note: Y_hdr luminance coefficients should technically use Rec.2020 weights
(0.2627, 0.6780, 0.0593), but since the gain map is single-channel and represents
a ratio, Rec.709 weights are close enough for 8-bit quantization.

## Implementation Strategy

### Phase 1 — Core (~400 LOC)

"Base" preset only. Active modules: input gamut, saturation, offset, tonescale+toe,
purity compression (high + low), display gamut, inverse EOTF.

Deliverables:
- `opendrt.rs` module with `OpenDrtConfig`, `TonescaleParams`, `process_pixel()`
- Updated `color.rs` with P3-D65 matrices, `cam_to_p3d65()`, display gamut matrices
- `pipeline.rs` branching on `tone_map` parameter (legacy vs opendrt)
- A/B comparison toggle in web UI

### Phase 2 — Look presets (~300 LOC)

Contrast Low/High (cubic toe, high contrast power), brilliance, mid purity.
Enables "Default" and other presets.

### Phase 3 — Creative controls (~200 LOC)

Hue shift RGB/CMY, hue contrast, creative white point (CAT02 adaptation).
Full stickshift parameter exposure.

### Phase 4 — Optimization

Deferred until correctness is validated. Possible paths:
- SIMD: `f32x4`/`f32x8` via `std::simd` (nightly) or `packed_simd`
- LUT: 1D LUT with cubic interpolation for tonescale+EOTF in preview path
- WASM: `wasm32-simd128` target feature for web builds
- No `rayon` in WASM (single-threaded), but pixels are independent so
  Web Workers can partition from JS side if needed

## Rust Struct Sketch

```rust
pub struct OpenDrtConfig {
    // Tonescale
    pub peak_luminance: f32,      // tn_Lp (100 SDR, 1000 HDR)
    pub grey_luminance: f32,      // tn_Lg (11.1 nits)
    pub hdr_grey_boost: f32,      // tn_gb
    pub contrast: f32,            // tn_con (1.4)
    pub shoulder: f32,            // tn_sh (0.5)
    pub toe: f32,                 // (0.003)
    pub offset: f32,              // tn_off (0.0)

    // Rendering space
    pub saturation: f32,          // rs_sa (0.35)
    pub sat_red_weight: f32,      // rs_rw (0.25)
    pub sat_blue_weight: f32,     // rs_bw (0.5)

    // Purity compression
    pub purity_weights: [f32; 3], // [pt_r, pt_g, pt_b] = [1.0, 2.0, 2.5]
    pub purity_rng_low: f32,      // 0.25
    pub purity_rng_high: f32,     // 0.25
    pub purity_hdr_blend: f32,    // pt_hdr
    pub purity_low_enable: bool,  // ptl_enable

    // Display
    pub display_gamut: DisplayGamut,
}

pub enum DisplayGamut { Rec709, Rec2020 }

/// Pre-computed tonescale constants (pixel-independent).
pub struct TonescaleParams {
    pub ts_s: f32,        // Main shape parameter
    pub ts_s1: f32,       // HDR purity shape
    pub ts_m2: f32,       // Toe scale
    pub ts_dsc: f32,      // Display encoding scale (100/Lp)
    pub s_lp100: f32,     // Reference at 100 nits
}

impl OpenDrtConfig {
    pub fn base_sdr() -> Self { /* Lp=100, Rec.709 */ }
    pub fn base_hdr() -> Self { /* Lp=1000, Rec.2020 */ }
}

impl TonescaleParams {
    pub fn new(config: &OpenDrtConfig) -> Self { /* constraint solver */ }
}

#[inline(always)]
pub fn process_pixel(
    rgb: [f32; 3],
    ts: &TonescaleParams,
    config: &OpenDrtConfig,
    cam_to_p3: &Mat3,
    p3_to_display: &Mat3,
) -> [f32; 3] {
    // Stage 1: cam → P3-D65
    let mut rgb = mat3_apply(cam_to_p3, rgb);
    // Stage 2: saturation + offset
    // Stage 3: tonescale
    // Stage 4: purity compression
    // Stage 7: inverse saturation, display gamut, toe, display scale
    rgb
}
```

## What This Replaces

| Current code | Replaced by |
|---|---|
| `color::build_cam_to_srgb()` / `build_cam_to_bt2020()` | `color::cam_to_p3d65()` + OpenDRT gamut conversion |
| Highlight blend hack (`pipeline.rs:82-89`) | OpenDRT purity compression (Stage 4) |
| HLG OETF + peak normalization (AVIF path) | OpenDRT tonescale (Lp=1000) + HLG OETF |
| HLG→inverse→sRGB roundtrip (JPEG-HDR SDR base) | OpenDRT tonescale (Lp=100) + sRGB OETF |
| Ad-hoc `scale = 1/peak` clipping (JPEG path) | OpenDRT tonescale shoulder compression |

### What stays unchanged

- Demosaic model (upstream)
- WB application to CFA (upstream)
- JPEG / AVIF / TIFF encoders (downstream)
- UltraHDR JPEG assembly (`encode_uhdr.rs` — XMP, MPF)
- EXIF rotation (runs after OpenDRT)

## Key Observations and Gotchas

### Middle grey placement
Maps 0.18 scene → ~0.11 display (~11 nits @ 100 nit). Cinema convention.
Photography expects ~0.18. Adjust `tn_Lg` or add exposure offset upstream.
Jed called this out explicitly: https://discuss.pixls.us/t/opendrt-for-art/49257/19

### SDR vs HDR consistency
Tonescale adapts to peak luminance. SDR and HDR renders are structurally related
through same image formation — ideal for gain map derivation. `tn_gb` controls
middle grey scaling with peak luminance.

### Purity compression is not optional
Without it, bright saturated colors clip with hard edges and hue shifts. Even "Base"
uses minimal chromaticity-linear compression. This replaces the highlight blending
hack — upstream CFA-level highlight reconstruction handles sensor clipping,
purity compression handles gamut mapping.

### The Abney effect
Chromaticity-linear desaturation causes perceived hue shifts (blue→purple, yellow→green).
Hue shift modules compensate. Absent in "Base" by design. Less critical for natural
photography than cinema (LED walls, CG).

### sRGB encoding vs EOTF
Jed flagged: sRGB ICC applies the sRGB encoding function (piece-wise with linear
segment), not pure 1/2.2 power. Slightly crushes shadows vs OpenDRT expectation.
Consider offering both sRGB encoding and pure gamma 2.2.

### ART CTL output convention
ART CTL outputs linear Rec.2020 scaled by `Lp/100`. We don't use this — our pipeline
outputs display-linear [0,1] in the target gamut and applies EOTF separately (from
`transfer.rs`). The `ts_dsc = 100/Lp` factor handles the [0,1] normalization.

## Resolved Decisions

- **Working space**: P3-D65 (confirmed from DCTL + ART CTL)
- **Porting reference**: ART CTL (`ref/opendrt_art.ctl`) — cleaner, simpler purity compression
- **Gain map packaging**: Manual JPEG APP2+XMP (already implemented in `encode_uhdr.rs`)
- **HLG vs PQ**: HLG — more forgiving for variable displays, already in use
- **Starting preset**: "Base" (minimal, photographer-oriented)
- **Gain map offset**: Keep `1/64` per Adobe/Google spec
- **Camera matrix source**: DNG ColorMatrix from RAF/CR2/NEF/ARW metadata (via rawpy)

## Testing Strategy

### Golden test vectors (automated, per-pixel exact)

`ref/opendrt_reference.py` — line-for-line Python port of the ART CTL `transform()`
for the Base preset. Generates 64 test vectors covering:

- Grey ramp (13 values from 0.001 to 100.0 — toe through shoulder)
- Pure primaries/secondaries at multiple exposures
- Near-neutral, near-clip, negative inputs, HDR values
- Both SDR (Lp=100, Rec.709) and HDR (Lp=1000, Rec.2020) configs

Output: `ref/opendrt_test_vectors.json` — input P3-D65 RGB + expected display-linear
output. Rust unit tests load the fixture and assert `max_abs_error < 1e-5` per channel.

Validated spot checks:
- Grey 0.18 SDR → 0.111 display-linear (= tn_Lg/100, exact grey mapping)
- Grey 0.18 HDR → 0.0159 display-linear (15.9 nits with grey boost)
- All grey values produce R=G=B (saturation/purity neutral for achromatic)

### A/B toggle in web UI (visual)

The `tone_map` parameter in `pipeline::encode()` switches between legacy and OpenDRT.
Side-by-side visual comparison on real photos for subjective validation.

## Open Questions

- Middle grey target for photography: `tn_Lg` = 11.1 nits (cinema default) vs ~18 nits?
  Start with 11.1, tune after visual comparison with legacy pipeline.
- WASM build: Confirm `powf` / `ln` / `exp` performance on wasm32 target.
