---
title: OpenDRT Tone Mapping and Color Grading
tags: [web, opendrt, tonemapping, grading, color-science]
scope: web/src/gl/opendrt-params.ts, web/src/gl/shaders/opendrt.wgsl, web/src/gl/renderer.ts
generated: 2026-03-22
commit: 347c8dd
---

# OpenDRT Tone Mapping and Color Grading

## Context

Camera sensor images demosaiced by the x-veon neural network arrive as
scene-referred linear RGB data with an essentially unbounded dynamic range.
Before these pixels can be displayed on a monitor or encoded to JPEG/TIFF/AVIF,
they must pass through a scene-referred-to-display-referred conversion that
compresses high dynamic range values into the displayable range while preserving
perceptual color relationships. This project uses **OpenDRT** (Open Display
Rendering Transform) -- a community-developed, open-source alternative to the
ACES Output Transform -- to perform that conversion entirely on the GPU via a
WGSL compute/fragment shader driven by the WebGPU `HdrRenderer` (see
[webgpu-renderer](./webgpu-renderer.md)).

The tone mapping pipeline must satisfy three constraints simultaneously:

1. **Real-time interactive preview** -- every slider change re-renders at
   display refresh rate with no visible latency.
2. **Lossless export fidelity** -- the identical transform (same config, same
   shader) produces float-precision display-linear pixels for encoding.
3. **HDR and SDR from one config** -- an SDR look preset can be promoted to HDR
   by changing a single scalar (`peak_luminance`), with the shader adapting
   automatically.

## Pattern / Approach

### OpenDrtConfig: the configuration struct

All OpenDRT behavior is controlled through a single TypeScript interface,
`OpenDrtConfig`, defined in `web/src/gl/opendrt-params.ts`. It mirrors the Rust
`OpenDrtConfig` struct field-for-field. Pre-processing adjustments (exposure,
white balance, sharpening) are defined in a separate `PreProcessConfig`
interface. The combined type `GradingConfig = OpenDrtConfig & PreProcessConfig`
is used at render time.

`OpenDrtConfig` groups into seven functional areas:

| Prefix   | Group                     | Fields                | Purpose |
|----------|---------------------------|-----------------------|---------|
| `tn_*`   | **Tonescale / contrast**  | `tn_lg`, `tn_con`, `tn_sh`, `tn_toe`, `tn_off`, `tn_lcon_*` (4), `tn_hcon_*` (4) | Global S-curve shape: mid-grey placement, contrast power, shoulder roll-off, shadow toe, low-contrast cubic modifier, high-contrast power extension |
| `rs_*`   | **Reach / saturation**    | `rs_sa`, `rs_rw`, `rs_bw` | Pre-tonescale desaturation using weighted luminance; controls how much color is pushed toward achromatic before the curve |
| `pt_*`   | **Per-channel purity tone** | `pt_r`, `pt_g`, `pt_b`, `pt_rng_low`, `pt_rng_high`, `ptl_enable`, `ptm_*` (5), `pt_hdr` | Per-channel weighting of the purity compression norm, controlling how individual channels desaturate toward the highlights; includes mid-range purity sculpting and a low-end softplus clamp |
| `brl_*`  | **Brilliance**            | `brl_enable`, `brl_r/g/b`, `brl_c/m/y`, `brl_rng` | Hue-selective brightness boost/cut applied via Gaussian hue windows in opponent space; six independent axes (RGB + CMY) |
| `hs_*`   | **Hue shift**             | `hs_rgb_enable`, `hs_r/g/b`, `hs_rgb_rng`, `hs_cmy_enable`, `hs_c/m/y` | Hue-selective chromatic rotation. RGB shifts increase with intensity; CMY shifts decrease. Controlled via Gaussian hue windows modulated by `ach_d * ts_pt`. |
| `hc_*` / `cwp*` | **Hue contrast / Creative white** | `hc_enable`, `hc_r`, `cwp`, `cwp_rng` | `hc_r` controls red-channel hue contrast (green/blue channels scaled by achromatic distance). `cwp` blends toward D50-adapted (warm) white point in highlights; `cwp_rng` controls the blend curve steepness. |
| *(top-level)* | **Display**          | `peak_luminance`, `grey_boost`, `pt_hdr` | Target display peak nits, HDR grey boost, HDR purity compression blend factor |

`PreProcessConfig` adds four fields applied before OpenDRT in the shader:

| Field | Purpose |
|-------|---------|
| `exposure` | EV stops (multiplied as `exp2(exposure)`) |
| `wb_temp` | White balance temperature correction: warm(+) / cool(-) |
| `wb_tint` | White balance tint correction: magenta(+) / green(-) |
| `sharpen_amount` | Unsharp mask strength (0 = off) |

### Look presets: base, default, colorful, umbra, flat

Five factory presets are provided as plain functions that return fully populated
`OpenDrtConfig` objects:

- **`baseSdr()`** -- Minimal creative intent. Low-contrast cubic, brilliance,
  hue shift, hue contrast, and creative white are all disabled. Serves as the
  spread-operator base for the other presets and as a neutral starting point
  for manual grading.
- **`defaultSdr()`** -- The recommended look. Extends `baseSdr()` with
  low-contrast enabled (`tn_lcon: 1.0`), brilliance enabled with negative R/G/B
  adjustments to tame over-bright primaries, mid-range purity sculpting active
  (`ptm_enable: true`), a slight toe offset (`tn_off: 0.005`), wider purity
  range (`pt_rng_high: 0.8`), hue shift RGB/CMY enabled, and hue contrast
  enabled (`hc_r: 1.0`).
- **`colorfulSdr()`** -- Higher contrast (`tn_con: 1.5`) with moderate
  low-contrast, active hue shift and hue contrast, and boosted mid-range purity.
- **`umbraSdr()`** -- High-contrast cinematic look (`tn_con: 1.8`, `tn_off:
  0.015`) with full low-contrast (`tn_lcon_w: 1.0`), wider hue shift ranges
  (`hs_rgb_rng: 2.0`), creative white point enabled (`cwp: 1.0`, `cwp_rng:
  0.25`) for warm highlight toning.
- **`flatSdr()`** -- A low-contrast, low-saturation variant (`tn_con: 1.15`,
  `rs_sa: 0.2`) intended for images that will be graded further downstream.

`configFromPreset(preset, hdrHeadroom?)` selects the appropriate factory
function (via a `LookPreset` key: `'default'`, `'colorful'`, `'umbra'`,
`'base'`, `'flat'`) and, when `hdrHeadroom > 1.0`, scales `peak_luminance`
accordingly.

### Tonescale presets

In addition to look presets, 13 `TonescalePreset` values are provided as
`TONESCALE_PRESETS`: a `Record<TonescalePreset, { label, overrides }>` mapping
that specifies tonescale-only overrides (`tn_con`, `tn_sh`, `tn_toe`, `tn_off`,
`tn_hcon_*`, `tn_lcon_*`). These can be applied on top of any look preset to
change the S-curve character without affecting color or brilliance settings.
Named presets include `low-contrast`, `medium-contrast`, `high-contrast`,
`arriba`, `sylvan`, `colorful`, `aery`, `dystopic`, `umbra`, `aces-1x`,
`aces-2`, `marvelous`, and `dagrinchi`.

### Config merging with configWithOverrides

Per-file user edits are stored in the Zustand store as
`Partial<OpenDrtConfig>` (`file.openDrtOverrides`) and
`Partial<PreProcessConfig>` (`file.preProcessOverrides`). At render time,
`configWithOverrides(base, overrides, preProcess?)` performs a shallow spread
merge (`{ ...DEFAULT_PREPROCESS, ...base, ...overrides, ...preProcess }`) and
returns a `GradingConfig`. It then auto-enables feature groups when their
parameters are explicitly set:

- Setting `tn_lcon` to a non-zero value auto-enables `tn_lcon_enable`.
- Setting any `brl_*` channel value auto-enables `brl_enable`.
- Setting any `hs_r/g/b` value auto-enables `hs_rgb_enable`.
- Setting any `hs_c/m/y` value auto-enables `hs_cmy_enable`.
- Setting `hc_r` to a non-zero value auto-enables `hc_enable`.

This ensures the user does not need to manually toggle boolean enable flags when
adjusting sliders in `GradingPanel`.

### HDR config derivation

`deriveHdrConfig(sdrConfig, peakLuminance)` creates an HDR variant by spreading
the `GradingConfig` and replacing `peak_luminance` (default 1000 nits). The
rest of the transform adapts automatically because `computeTonescaleParams`
derives the S-curve constants from `peak_luminance` and `grey_boost`. The
export path in `useExport.ts` calls this to produce dual SDR + HDR renders for
JPEG-HDR and AVIF output.

### Tonescale parameter derivation (precomputed shader constants)

The `TonescaleParams` interface holds five precomputed constants that the shader
needs but that are too expensive to recompute per-pixel:

```
ts_s    -- hyperbolic power scale (main tonescale)
ts_s1   -- blended scale for purity compression (HDR-aware)
ts_m2   -- tonescale maximum (after inverse quadratic toe)
ts_dsc  -- display scale factor: 100 / peak_luminance
ts_x0   -- mid-grey anchor: 0.18 + tn_off
```

`computeTonescaleParams(cfg)` derives these from the config using the same math
as the Rust `TonescaleParams::new` (lines 209-230 of `opendrt.rs`). The key
steps are:

1. Compute highlight rolloff point `ts_x1 = 2^(6*tn_sh + 4)`.
2. Compute display-referred peak `ts_y1 = peak_luminance / 100`.
3. Derive mid-grey display value `ts_y0` with logarithmic grey boost for HDR.
4. Solve for the hyperbolic power scale `ts_s` via inverse quadratic toe and
   `compress_hp` inversion.
5. Blend `ts_s` with a 100-nit reference scale using `pt_hdr` and a luminance
   ramp to produce `ts_s1` for the purity compression norm, ensuring purity
   behavior tracks smoothly from SDR to HDR.

These five floats are packed into the `ts` and `ts_x0_and_flags` uniform vec4s
and uploaded once per config change, not per frame.

### WGSL shader structure

The tone mapping shader (`web/src/gl/shaders/opendrt.wgsl`) is a single WGSL
module containing both vertex and fragment entry points. It is a line-for-line
port of the Rust `process_pixel` function.

**Uniform layout.** All parameters are packed into a `Uniforms` struct of vec4f
fields. The `HdrRenderer.applyOpenDrtUniforms()` method in `renderer.ts` writes
the packed floats into a `Float32Array` which is uploaded to the GPU uniform
buffer before each render pass. Color-space matrices (sRGB-to-P3 and
P3-to-display) are stored as three column vec4s each.

**Processing stages** (in order within the `opendrt()` function):

1. **Color space conversion** -- sRGB linear to P3-D65 via matrix multiply.
2. **Rendering-space saturation** -- weighted desaturation toward luminance
   (`rs_sa`, `rs_rw`, `rs_bw`) plus toe offset.
3. **Low-contrast modifier** -- optional cubic toe compression in norm-space,
   with a per-channel/achromatic blend controlled by `tn_lcon_pc`.
4. **Norm computation** -- Euclidean RGB norm (`sqrt(r^2 + g^2 + b^2) / sqrt(3)`)
   for the tonescale, and a weighted purity norm for per-channel tone.
5. **RGB ratio extraction** -- divide by norm to separate luminance from chromaticity.
6. **High-contrast extension** -- optional power curve above a pivot point.
7. **Tonescale application** -- `compress_hp(norm, ts_s, tn_con)` maps the
   scene-linear norm through the hyperbolic power S-curve.
8. **Opponent-space hue analysis** -- compute achromatic distance and hue angle
   for brilliance and purity modulation via Gaussian hue windows (R, G, B, C,
   M, Y).
9. **Purity compression range** -- compute purity blend factor from
   `pt_rng_low`/`pt_rng_high`, modulated by achromatic distance.
10. **Brilliance** -- hue-selective luminance scaling using six Gaussian windows
    (R, G, B, C, M, Y), gated by achromatic distance.
11. **Mid-range purity** -- optional purity sculpting via `ptm_low`/`ptm_high`
    sigmoid modifiers blended by achromatic distance and tonescale position.
12. **Hue contrast** -- optional red-channel hue contrast (`hc_r`): scales G/B
    ratios based on achromatic distance, hue angle proximity to red, and
    tonescale position.
13. **Hue shift RGB** -- optional per-channel hue rotation that increases with
    intensity (`ts_pt`), modulated by achromatic distance and `hs_rgb_rng`.
14. **Hue shift CMY** -- optional per-channel hue rotation that decreases with
    intensity (`1 - ts_pt`), using CMY hue windows.
15. **Apply brilliance + purity compression** -- multiply by brilliance factor,
    then blend RGB ratios toward achromatic using the purity + mid-purity factor.
16. **Inverse rendering-space saturation** -- undo the pre-tonescale
    desaturation to restore color relationships in display space.
17. **Display gamut conversion** -- P3-D65 to target gamut (Rec.709 for SDR,
    Rec.2020 for HDR export).
18. **Creative white point** -- optional blend toward D50-adapted (warm) display
    values in highlights, controlled by `cwp_rng` and tonescale position. Uses
    a precomputed CWP adaptation matrix from `color-matrices.ts`.
19. **Purity compress low** -- softplus clamp on negative display-gamut values
    to handle out-of-gamut colors without hard clipping.
20. **Final tonescale** -- apply `ts_m2` scaling, quadratic toe, and display
    scale `ts_dsc` to convert from the norm-relative domain to absolute display
    values.

**Fragment entry point** (`fs_main`): samples the scene-linear texture, applies
optional unsharp-mask sharpening, applies exposure and white-balance
corrections, calls `opendrt()`, and then either outputs display-linear (export
mode) or applies the sRGB OETF (display mode). HDR display mode skips the
`[0,1]` clamp to allow values above 1.0.

### Integration with the renderer

The `HdrRenderer` (documented in [webgpu-renderer](./webgpu-renderer.md))
manages two code paths that both consume `OpenDrtConfig` + `TonescaleParams`:

- **Display path** (`setOpenDrtMode` + `render`): called from
  `OutputCanvas.tsx` whenever the preset or overrides change. The component
  calls `configFromPreset` -> `configWithOverrides` -> `computeTonescaleParams`
  and passes the results to `renderer.setOpenDrtMode()`.
- **Export path** (`renderForExport`): called from `useExport.ts`. Computes
  merged SDR config, optionally derives HDR config via `deriveHdrConfig`, then
  renders to an off-screen float texture with `exportMode = 1` (no OETF).

### Python reference implementation

`tests/opendrt/opendrt_reference.py` is a NumPy-based line-for-line port of the
original OpenDRT CTL (Academy Color Transformation Language) code. It generates
golden test vectors for validating the Rust implementation, which in turn is the
source of truth for both the TypeScript config layer and the WGSL shader. The
reference operates on scene-linear P3-D65 input and outputs display-linear RGB
in Rec.709 or Rec.2020 before the EOTF.

## Rationale

### Why OpenDRT over ACES or a custom curve

OpenDRT was chosen for three reasons:

1. **Path-to-white preservation.** OpenDRT uses a per-channel purity compression
   approach that smoothly desaturates toward the highlights without the abrupt
   hue shifts (particularly in blue skies and warm highlights) that plague ACES
   RRT+ODT. For camera sensor imagery with a wide scene-referred gamut, this
   produces more natural results.
2. **Single-config HDR/SDR.** The tonescale is parameterized by
   `peak_luminance`, so the same creative look extends from 100-nit SDR to
   1000+-nit HDR by changing one number. ACES requires separate ODTs per display.
3. **Tunable look.** The parameters provide fine-grained creative control
   (brilliance per hue, hue shift, hue contrast, creative white point,
   mid-range purity sculpting, separate low/high contrast) without requiring
   LUTs or external files. All state is serializable as JSON.

### Preset design philosophy

The five presets follow a layered principle: `baseSdr` is the
mathematically-neutral starting point with all optional processing disabled;
`defaultSdr` adds the creative decisions that produce a pleasing image for
typical photographic content; `colorfulSdr` pushes saturation and contrast
higher; `umbraSdr` creates a dark, cinematic look with warm highlights via
creative white; `flatSdr` intentionally reduces contrast and saturation for
users who want a flatter starting point for manual grading. All presets use
spread-operator inheritance from `baseSdr`, so only the differing fields are
specified. This makes it straightforward to audit what each preset changes
relative to the base.

## Key Files

| File | Role |
|------|------|
| `web/src/gl/opendrt-params.ts` | `OpenDrtConfig`, `PreProcessConfig`, `GradingConfig`, `TonescaleParams` interfaces; preset factories; `TONESCALE_PRESETS`; `configFromPreset`, `configWithOverrides`, `deriveHdrConfig`, `computeTonescaleParams` |
| `web/src/gl/shaders/opendrt.wgsl` | WGSL tone mapping shader (vertex + fragment), 1:1 port of Rust `process_pixel` |
| `web/src/gl/renderer.ts` | `HdrRenderer.applyOpenDrtUniforms()` packs config into GPU uniform buffer; `setOpenDrtMode()` and `renderForExport()` entry points |
| `web/src/components/OutputCanvas.tsx` | Calls config pipeline (preset -> overrides -> tonescale) and feeds `HdrRenderer` on every state change |
| `web/src/components/GradingPanel.tsx` | UI sliders for tone and brilliance params; writes per-key overrides to Zustand store |
| `web/src/hooks/useExport.ts` | Export path: builds merged SDR config, optionally derives HDR config, renders via `renderForExport` |
| `web/src/store.ts` | Zustand store holds `openDrtOverrides: Partial<OpenDrtConfig>` per file |
| `tests/opendrt/opendrt_reference.py` | NumPy golden-vector generator for validating Rust OpenDRT against the original CTL |

## Antipatterns

- **Recomputing `TonescaleParams` in the shader.** The five tonescale constants
  involve iterative inverse-function evaluations (`compress_toe_quadratic_inv`,
  multi-step power solves). These are derived once on the CPU via
  `computeTonescaleParams` and uploaded as uniforms. Moving this math into the
  per-pixel shader path would be wasteful and risk floating-point divergence
  between CPU and GPU.

- **Mutating preset objects directly.** Preset factories return fresh objects on
  every call. Caching and mutating a returned preset would leak state between
  files. Always call the factory, then apply overrides with
  `configWithOverrides`.

- **Bypassing `configWithOverrides` for feature-group enables.** The function
  auto-enables `tn_lcon_enable`, `brl_enable`, `hs_rgb_enable`, `hs_cmy_enable`,
  and `hc_enable` when their dependent params are set. Manually spreading
  overrides without this logic can leave features configured but disabled,
  producing silent no-ops in the shader.

- **Assuming SDR clamp for HDR output.** The shader clamps to `[0, 1]` only
  when `hdrDisplay` is false. Export and HDR display paths must set the flag
  correctly; otherwise highlight detail is silently lost.
