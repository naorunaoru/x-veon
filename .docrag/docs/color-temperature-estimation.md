---
title: Color Temperature and Tint Estimation from White Balance Coefficients
tags: [web, color, white-balance, cct]
scope: web/src/pipeline/color-temperature.ts, web/src/components/GradingPanel.tsx
generated: 2026-03-22
commit: 347c8dd
---

## Context

Camera RAW files embed white balance (WB) coefficients -- per-channel multipliers
that neutralize the color cast of the scene illuminant. These coefficients are
numeric triplets (R, G, B gains) with no direct human meaning. Photo editors
universally present white balance as two intuitive controls:

- **Color temperature** (CCT, in Kelvin): the blue-amber axis (2000K = warm
  candlelight, 10000K = blue sky).
- **Tint**: the green-magenta axis, representing deviation from the Planckian
  (blackbody) locus.

x-veon needs to derive CCT and tint from camera WB coefficients for two reasons:

1. **Display**: show the user the effective color temperature and tint on the
   grading panel sliders, matching the conventions of Lightroom, Capture One, and
   similar tools.
2. **Inverse mapping**: when the user drags the temperature or tint slider to a
   target value (e.g. 6500K), compute the corresponding `wb_temp` / `wb_tint`
   shader parameter that achieves that CCT.

The WB coefficients and the camera-to-XYZ color matrix both originate from the
RAW decoding WASM module (see `raw-decoding-wasm.md`). The `RawImage` struct
produced by the WASM decoder carries `wbCoeffs` (the camera's auto-WB or
as-shot multipliers) and `camToXyz` (a 3x4 row-major matrix mapping camera RGB
to CIE XYZ). These two arrays are the sole inputs to the CCT/tint estimator.

## Pattern / Approach

All logic lives in a single module: `web/src/pipeline/color-temperature.ts`. It
exports three functions consumed by the rest of the application.

### Forward Path: WB Coefficients to CCT and Tint

The forward estimation converts a WB triplet into a (CCT, tint) pair. It
proceeds in four steps:

**Step 1 -- Recover illuminant color in camera space.**
The WB coefficients are inversely proportional to the illuminant's color as
seen by the sensor. If the camera reports WB multipliers `[R, G, B]`, the
illuminant's relative camera-space color is `[1/R, 1/G, 1/B]`.

**Step 2 -- Transform to CIE XYZ.**
Multiply the camera-space illuminant by the `camToXyz` matrix (a 3x4 matrix
stored as 12 floats in row-major order with stride 4) to obtain `(X, Y, Z)`
tristimulus values. The matrix rows are indexed at offsets 0, 4, and 8, with
the fourth column unused (it carries zeros in this context).

**Step 3 -- Compute CIE 1931 chromaticity and apply McCamy's CCT approximation.**
Derive chromaticity coordinates `x = X/(X+Y+Z)`, `y = Y/(X+Y+Z)`. Then apply
McCamy's third-degree polynomial approximation:

```
n = (x - 0.3320) / (0.1858 - y)
CCT = 449*n^3 + 3525*n^2 + 6823.3*n + 5520.33
```

The result is clamped to the range [1500, 25000] K. McCamy's formula is accurate
to within ~2 K for illuminants near the Planckian locus in the 2000-12500 K
range, which covers all common photographic lighting.

**Step 4 -- Compute tint as signed Planckian locus distance.**
Tint measures how far the illuminant's chromaticity lies from the Planckian
locus (the curve of ideal blackbody radiators). The computation:

1. Convert the illuminant's `(x, y)` to CIE 1960 UCS `(u, v)` using the
   standard transform: `u = 4x / (-2x + 12y + 3)`, `v = 6y / (-2x + 12y + 3)`.
2. Compute the Planckian locus point at the estimated CCT using Kang et al.
   (2002) polynomial approximations for `x(T)` and `y(x, T)`, with separate
   coefficient sets for T <= 4000 K and T > 4000 K. Convert that point to UCS
   `(u_p, v_p)`.
3. Compute the Euclidean distance `duv = sqrt((u - u_p)^2 + (v - v_p)^2)`.
4. Determine the sign by evaluating the cross product of the locus tangent
   vector (computed by finite difference at CCT and CCT+1) against the
   displacement vector. Positive means the illuminant is on the magenta side;
   negative means green.
5. Scale the signed `duv` by a factor of 3200 to map the raw distance into a
   range (roughly +/-150) that matches the tint scale used by photo editors.

The internal function `estimateRaw` performs steps 1-4 and returns unrounded
`{cct, tint}`. The public `estimateColorTemperature` wraps it, rounding CCT to
the nearest 50 K and tint to the nearest integer.

### Kang et al. Planckian Locus Approximation

Two helper functions (`planckianX` and `planckianY`) implement the Kang et al.
(2002) polynomial fit of the Planckian locus in CIE 1931 chromaticity space.
These are piecewise cubics: `planckianX` has one breakpoint at T = 4000 K (two pieces), while `planckianY` has breakpoints at T = 2222 K and T = 4000 K (three pieces):

- `planckianX(T)` returns the x-chromaticity of a blackbody at temperature T.
  Uses one set of coefficients for T <= 4000 and another for T > 4000.
- `planckianY(x, T)` returns the y-chromaticity given x and T. Uses three
  coefficient sets (T <= 2222, 2222 < T <= 4000, T > 4000).

These are used only for the tint calculation (computing the locus reference
point and its tangent), not for the CCT estimation itself.

### Inverse Path: Target CCT/Tint to Shader Parameter

The GPU shader applies white balance correction using two parameters:

- `wb_temp` (float): shifts R and B channels in opposite directions.
  `R *= 2^wb_temp`, `B *= 2^(-wb_temp)`. Green is unchanged.
- `wb_tint` (float): shifts the G channel. `G *= 2^(-wb_tint)`. R and B are
  unchanged.

Both default to 0 (no correction). Luminance is preserved by dividing by
`0.2126 * 2^wb_temp + 0.7152 * 2^(-wb_tint) + 0.0722 * 2^(-wb_temp)`.

When the user drags the temperature slider to a target CCT (e.g. 5500 K), the
UI must find the `wb_temp` value that, when applied to the camera's base WB
coefficients, produces that CCT. Two exported functions handle this:

**`findWbTempForCct(targetCct, baseWb, camToXyz)`** finds the `wb_temp` value
that produces a target CCT. It uses brute-force sampling over 401 evenly-spaced
values of `t` in the range [-4, +4]. For each candidate `t`, it constructs
modified WB coefficients `[baseWb[0] * 2^t, 1, baseWb[2] * 2^(-t)]` and calls
`estimateColorTemperature` to get the resulting CCT. It minimizes
`|CCT - targetCCT| + |t| * 100`, where the second term penalizes extreme
adjustments to prefer the solution closest to no correction.

**`findWbTintForTint(targetTint, baseWb, camToXyz)`** finds the `wb_tint` value
that produces a target tint. Same brute-force strategy: 401 samples of `t` in
[-4, +4], constructing `[baseWb[0], 2^(-t), baseWb[2]]` for each candidate and
minimizing `|tint - targetTint|`.

Both return the optimal `t` value, which is stored as the `wb_temp` or `wb_tint`
parameter in the `OpenDrtConfig` and passed to the WebGPU shader.

### Rounding Conventions

- CCT is rounded to the nearest 50 K (`Math.round(cct / 50) * 50`). This
  matches common photo editor conventions where temperature snaps to 50 K
  increments.
- Tint is rounded to the nearest integer (`Math.round(tint)`).
- The rounding happens in the public `estimateColorTemperature` function, not
  in `estimateRaw`. The inverse-path functions (`findWbTempForCct`,
  `findWbTintForTint`) call `estimateColorTemperature` (rounded) so that
  the brute-force search matches what the user sees.

### Decoupled Temperature and Tint Display

In `GradingPanel.tsx`, temperature and tint are estimated independently to avoid
cross-talk. The R/B gain adjustment (`wb_temp`) does not trace the Planckian
locus exactly, so applying both `wb_temp` and `wb_tint` simultaneously when
estimating CCT would cause the temperature readout to shift when the user
adjusts tint, and vice versa. Instead:

- The displayed CCT is estimated from WB coefficients that apply only the
  `wb_temp` slider: `[wb[0] * 2^temp, 1.0, wb[2] * 2^(-temp)]`.
- The displayed tint is estimated from WB coefficients that apply only the
  `wb_tint` slider: `[wb[0], 2^(-tint), wb[2]]`.

This decoupling means the readouts are slightly approximate (they do not reflect
the combined effect) but the user experience is far better: each slider
independently controls its own displayed value without interference.

### Degenerate Input Handling

If the XYZ tristimulus sum is zero or negative (which can happen with corrupt
metadata or an all-black image), `estimateRaw` returns a neutral fallback of
`{cct: 5500, tint: 0}` -- roughly daylight, zero tint.

## Rationale

### Why McCamy's Approximation over Robertson's Method

Robertson's method (1968) interpolates CCT from a table of isotherms in CIE
1960 UCS space. It is more accurate at extreme temperatures and handles
non-Planckian illuminants better. However:

- McCamy's formula is a single polynomial evaluation with no table lookups or
  iterative steps. This matters because the brute-force inverse functions call
  it 401 times per slider interaction.
- The accuracy difference (~2 K vs. ~1 K near the locus) is invisible at the
  50 K rounding granularity used for display.
- The implementation is five lines of arithmetic with no data tables, reducing
  bundle size and maintenance burden.

### Why Brute-Force Inverse Instead of Analytical or Bisection

McCamy's CCT approximation is not monotonic over the full `wb_temp` parameter
range. At extreme chromaticities (large `|t|` values), the polynomial can
produce the same CCT for very different WB settings. This means:

- An analytical inverse does not exist (McCamy maps chromaticity to CCT, not
  `wb_temp` to CCT, and the composed function has no closed form).
- Bisection fails because the function is not monotonic -- it can miss the
  correct root or oscillate.
- Brute-force with 401 samples is reliable and fast. At ~5 function evaluations
  per sample (one `estimateColorTemperature` call involving a matrix multiply,
  McCamy polynomial, and Planckian locus evaluation), the total cost is
  negligible for a UI callback.

The penalty term `|t| * 100` in `findWbTempForCct` resolves ambiguity when
multiple `t` values produce similar CCTs by preferring the one closest to
neutral (t = 0). This avoids jumping to an extreme white balance setting that
happens to produce the same rounded CCT.

### Tint Scale Factor

The raw Planckian locus distance `duv` is a small number (typically 0.001 to
0.05). Multiplying by 3200 maps this into the +/-150 range familiar from
photo editors. The factor 3200 was chosen empirically to match the tint scale
of Adobe Lightroom for a range of test images. There is no standard for tint
scaling -- different editors use different ranges -- but 3200 produces values
that feel familiar to photographers.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/color-temperature.ts` | All CCT/tint estimation logic: forward (WB to CCT/tint) and inverse (target CCT/tint to shader parameter) |
| `web/src/components/GradingPanel.tsx` | UI consumer: calls `estimateColorTemperature` for display, `findWbTempForCct` and `findWbTintForTint` for slider interaction |
| `web/src/hooks/useProcessFile.ts` | Pipeline consumer: calls `estimateColorTemperature` at decode time to store initial CCT/tint in `ProcessingResultMeta` |
| `web/src/pipeline/types.ts` | Defines `colorTemp` and `tint` fields on `ProcessingResultMeta.metadata` |
| `web/src/gl/opendrt-params.ts` | Defines `wb_temp` and `wb_tint` fields on `OpenDrtConfig` (the shader parameter struct) |
| `web/src/gl/shaders/opendrt.wgsl` | GPU shader that applies `wb_temp` / `wb_tint` as `2^t` gain adjustments with luminance compensation |

### Dependencies

The WB coefficients (`wbCoeffs`) and camera-to-XYZ matrix (`camToXyz`) that
feed into this module are produced by the RAW decoding WASM module. See
`raw-decoding-wasm.md` for details on how these values are extracted from
camera RAW files. The `wbCoeffs` are normalized to G=1 in `useProcessFile.ts`
before being passed to any estimation function.

## Antipatterns

### Do Not Couple Temperature and Tint Estimation

Applying both `wb_temp` and `wb_tint` adjustments to the WB coefficients before
calling `estimateColorTemperature` causes cross-talk: changing the tint slider
shifts the displayed CCT, and changing the temperature slider shifts the
displayed tint. This is because the R/B gain axis (`wb_temp`) does not align
with the Planckian locus, so moving along it has a tint component, and vice
versa. The correct approach, implemented in `GradingPanel.tsx`, is to estimate
CCT and tint from separate WB vectors that each apply only one slider's effect.

### Do Not Use Bisection for the Inverse CCT Lookup

It is tempting to replace the 401-sample brute-force search with a bisection
or Newton's method for better asymptotic complexity. This will silently break
at extreme white balance settings where McCamy's polynomial is non-monotonic in
the `wb_temp` parameter. The brute-force approach is correct by construction
and fast enough (sub-millisecond on modern hardware). If performance becomes a
concern, reduce the sample count (200 samples still gives < 50 K resolution)
rather than switching to an iterative solver.

### Do Not Skip the Penalty Term in findWbTempForCct

The score function `|CCT - targetCCT| + |t| * 100` includes a regularization
term that biases toward small `|t|` values. Removing it causes the function to
sometimes return extreme `t` values (e.g. t = 3.5) that produce the correct
rounded CCT but result in unnatural colors because the chromaticity is far from
the Planckian locus. The penalty ensures the returned `wb_temp` is the most
physically plausible solution.

### Do Not Round Inside estimateRaw

The internal `estimateRaw` function returns unrounded CCT and tint values. This
is intentional: the brute-force inverse functions need continuous (unrounded)
values for accurate minimization. Rounding inside `estimateRaw` would create
a step function that makes many different `t` values appear equally good,
degrading inverse-path accuracy. Rounding belongs exclusively in the public
`estimateColorTemperature` wrapper.
