---
title: Opposed-Channel Highlight Inpainting
tags: [web, highlights, inpainting, pipeline]
scope: web/src/pipeline/preprocessor.ts, web/src/pipeline/postprocess-gpu.ts, web/src/hooks/useProcessFile.ts, highlight_recovery.py, highlight_recovery_rgb.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

Digital camera sensors have a finite well capacity per photosite. When scene
luminance exceeds this capacity, the recorded value saturates at the sensor's
white level -- the pixel is "clipped." Clipping rarely affects all three color
channels simultaneously. A bright yellow highlight may saturate the red and
green CFA photosites while blue remains below its clip point. The result after
demosaicing is a channel imbalance: the clipped channels are capped at their
maximum while the unclipped channel continues to represent the true scene
intensity. This imbalance manifests as false color shifts in specular
highlights, light sources, and overexposed skin tones -- regions that should
appear as neutral bright white or a smooth saturated hue instead display harsh
magenta, yellow, or cyan fringes.

Highlight reconstruction recovers plausible values for the clipped channels by
extrapolating from the information that survives in unclipped channels. The
opposed-channel inpainting algorithm implemented in `reconstructHighlightsCfa`
is the first of two reconstruction passes in the x-veon web pipeline (the
second is the segmentation-based approach documented in
[highlight-reconstruction-segmentation](highlight-reconstruction-segmentation.md)).
It is adapted from darktable's "inpaint opposed" module and operates as a fast,
per-pixel pass over the raw CFA mosaic before any demosaicing takes place.

This document covers the opposed-channel algorithm. The algorithm has three
implementations:

1. **CPU/TypeScript** (`web/src/pipeline/preprocessor.ts`): The original
   `reconstructHighlightsCfa` function. Still defined but no longer called
   directly by `useProcessFile`. Uses dilated mask chrominance at 1/3 resolution.
2. **GPU/WGSL** (`web/src/pipeline/postprocess-gpu.ts`): The active
   implementation, running as passes 2-5 of the GPU postprocess pipeline after
   demosaicing. Operates on demosaiced RGB data (not CFA mosaic). Called by
   `gpuPostprocess()` which is invoked by `useProcessFile` for both neural-net
   and traditional demosaic paths.
3. **Python** (`highlight_recovery.py`, `highlight_recovery_rgb.py`): Reference
   implementations for training and offline inference. `highlight_recovery.py`
   operates on CFA mosaic data (matching the CPU TS implementation);
   `highlight_recovery_rgb.py` operates on demosaiced 3-channel RGB images.

For the broader preprocessing pipeline -- normalization, CFA alignment, tiling
-- see [cfa-preprocessing-web](cfa-preprocessing-web.md).

### Why Reconstruction Must Precede Demosaicing

Demosaicing algorithms interpolate missing color values at each pixel from
neighboring CFA samples. If those neighbors are clipped, the interpolation
propagates the clipped values into the reconstructed RGB image, baking the
channel imbalance into the output. Correcting this after demosaicing is far more
difficult because the clipped pixels are no longer identifiable -- they have
been blended with their neighbors. By running reconstruction on the raw CFA
mosaic, the algorithm can precisely identify which individual photosites are
clipped (comparing against per-channel thresholds) and replace them before
any inter-pixel blending occurs.

### Why Reconstruction Runs After White Balance

The pipeline applies white balance before highlight reconstruction. This is
deliberate. White balance multipliers equalize the channel responses so that
equal normalized values across R, G, and B correspond to equal scene luminance.
The opposed-channel algorithm computes cross-channel averages to estimate what a
clipped channel "should" read. These averages are only physically meaningful
when the channels are in balanced, scene-referred linear space. Without WB, a
red photosite at normalized value 0.8 and a blue photosite at 0.8 could
represent very different scene intensities (red sensors typically have lower
sensitivity and receive a WB multiplier of 1.5-2.5x). The clip thresholds
passed to the function are already in white-balanced space:
`clipNorm[c] * wb[c]`.

## Pattern / Approach

### Overview

`reconstructHighlightsCfa` modifies the CFA `Float32Array` in place. It
accepts the CFA data, image dimensions, the CFA pattern description (Bayer or
X-Trans), the pattern phase shift `(dy, dx)`, and per-channel clip thresholds.
The algorithm runs in two passes over the image:

1. **Pass 1 (chrominance calibration)**: computes a global per-channel
   chrominance correction from pixels that are near the clip point but not yet
   clipped.
2. **Pass 2 (reconstruction)**: replaces clipped pixel values using the
   opposed-channel reference average plus the chrominance correction.

Both passes skip the one-pixel image border to avoid bounds checks in the 3x3
neighborhood access.

### Clip Threshold Computation

Before calling the reconstruction function, the pipeline computes per-channel
clip thresholds via `channelClips` in `preprocessor.ts`. This function:

1. Maps the camera's per-CFA-position white levels to R/G/B channels using the
   first 2x2 block of the CFA pattern string, taking the minimum white level
   per channel (some cameras report different saturation points for different
   photosite positions of the same color).
2. Normalizes each white level: `(whiteLevel[c] - black) / range`.
3. Applies a 0.93 scaling factor (annotated `// HACK!` in the source) to
   trigger reconstruction 7% below the reported saturation point. This
   compensates for per-pixel variation near saturation and raw decoders that
   report white levels above actual clipping.
4. The caller then multiplies by the white balance coefficient:
   `clips[c] = clipNorm[c] * wb[c]`.

The 0.93 factor is safe because the reconstruction formula uses
`Math.max(clippedValue, estimate)` -- pixels incorrectly flagged as clipped
retain their original value if the estimate is lower.

### The calcRefavg Function: Opposed-Channel Reference Average

The core primitive is `calcRefavg(y, x, ch)`, which estimates the expected
brightness at pixel `(y, x)` of channel `ch` using the other two color
channels in the local neighborhood.

**Step 1: Gather 3x3 neighborhood.** For each of the up-to-9 pixels in the
3x3 window centered on `(y, x)`, the function accumulates values into
per-channel sums and counts. Channel identity is determined from the CFA
pattern: `getCh(ny, nx)` returns 0 (red), 1 (green), or 2 (blue) for the
photosite at `(ny, nx)`, applying the pattern phase shift `(dy, dx)`. Negative
CFA values are clamped to zero before accumulation.

**Step 2: Compute per-channel means in linear space.** Each channel's mean is
`mean[c] / cnt[c]`. This gives the average linear intensity of each color in
the local neighborhood.

**Step 3: Convert to cube-root (perceptual) space.** Each per-channel mean is
transformed via `Math.cbrt(mean)` -- the cube root function. This maps linear
intensity to a perceptual brightness scale (see Rationale below for why).

**Step 4: Average the two opposing channels.** For a pixel of channel `ch`,
the "opposed" reference is the mean of the other two channels' cube-root
values:

- For red (ch=0): `oppCr = 0.5 * (cr[1] + cr[2])` -- average of green and
  blue.
- For green (ch=1): `oppCr = 0.5 * (cr[0] + cr[2])` -- average of red and
  blue.
- For blue (ch=2): `oppCr = 0.5 * (cr[0] + cr[1])` -- average of red and
  green.

**Step 5: Convert back to linear space.** The opposed average is cubed to
return to linear intensity: `return oppCr * oppCr * oppCr`. This value
represents the expected linear intensity of the target channel if it had the
same perceptual brightness as the average of its neighbors' other two channels.

### Pass 1: Global Chrominance Correction

The chrominance correction captures the systematic difference between a
channel's actual value and what the opposed-channel reference predicts. This
difference encodes the color (chrominance) of the scene separately from its
brightness (luminance).

For every pixel `(y, x)` that satisfies both conditions:

- Its value is above the lower threshold: `val >= chromLo` (hardcoded to 0.2,
  filtering out dark pixels where the signal-to-noise ratio is poor).
- Its value is below its channel's clip threshold: `val < clips[ch]`.

The function accumulates `val - calcRefavg(y, x, ch)` into a per-channel sum.
After iterating the full image, the mean of these differences becomes the
chrominance correction for each channel:

```
chrom[c] = chromSum[c] / chromCnt[c]    (if chromCnt[c] > 100)
```

The minimum-count guard of 100 prevents unstable corrections from images with
very few qualifying pixels. When a channel has fewer than 100 unclipped
near-clip samples, its chrominance correction remains zero.

The chrominance correction is computed and stored in linear space, not
cube-root space. This is intentional -- the correction is added to the
opposed-channel reference average (which has been converted back to linear) in
Pass 2.

### Pass 2: Clipped Pixel Reconstruction

For every pixel `(y, x)` whose value meets or exceeds its channel's clip
threshold (`cfa[idx] >= clips[ch]`), the function computes:

```
estimate = calcRefavg(y, x, ch) + chrom[ch]
cfa[idx] = Math.max(cfa[idx], estimate)
```

The `Math.max` ensures the reconstruction only extends values upward. If the
opposed-channel estimate is lower than the already-clipped value, the pixel
retains its original (clipped) value. This prevents the algorithm from
darkening pixels near the clip boundary where the estimate might be slightly
below the true value due to noise.

The reconstructed value can exceed 1.0 (and often does, since the clip
thresholds are already above 1.0 after white balance scaling). This is correct
and expected: the downstream demosaicing and color correction pipeline operates
in unbounded linear space, and the final tone mapping or display transform
handles the compression to displayable range.

### CFA Pattern Handling

The algorithm is CFA-pattern-agnostic. The `getCh` closure resolves channel
identity at any `(y, x)` position using the pattern lookup table with modular
arithmetic and the phase shift `(dy, dx)`. This works identically for Bayer
(period 2, four patterns) and X-Trans (period 6, 36 patterns). In a Bayer
mosaic, the 3x3 neighborhood of a red pixel contains approximately 1 red,
4 green, and 4 blue samples (depending on position within the pattern). In
X-Trans, the distribution varies by position within the 6x6 repeating unit,
but the algorithm handles this naturally through per-channel counting.

### Pipeline Integration

**Active path (GPU):** Both the neural-net and traditional demosaic paths in
`useProcessFile.ts` call `gpuPostprocess()` after demosaicing. The GPU pipeline
runs highlight recovery on the demosaiced RGB data as passes 2-5:

- Pass 2: Compute per-pixel opposed-channel reference average + clip mask
- Pass 3: Accumulate chrominance correction from unclipped pixels within dilated mask regions
- Pass 4: Reduce chrominance sums to global per-channel corrections
- Pass 5: Reconstruct clipped pixels using refavg + chrominance

The GPU implementation operates on 3-channel demosaiced data (each pixel has
all RGB values) rather than on CFA mosaic data. The opposed-channel average
is computed from the 3x3 neighborhood of the same pixel in RGB space.

**CPU/TypeScript (inactive):** The `reconstructHighlightsCfa` function in
`preprocessor.ts` is still defined and tested but is not called by the main
processing pipeline. It operates on CFA mosaic data and uses dilated mask
chrominance at 1/3 resolution for adaptive spatial coverage.

**Python implementations:** `highlight_recovery.py` mirrors the CPU TS
implementation for offline use. `highlight_recovery_rgb.py` operates on
demosaiced RGB images, similar to the GPU path.

## Rationale

### Why Cube-Root Space for the Opposed Average

The cube root function `Math.cbrt(x)` provides a simple approximation of
perceptual brightness that compresses the highlight range while expanding the
shadow range. It serves a similar purpose to the CIE L* lightness curve (which
uses a conditional linear/cube-root function) but avoids the conditional
branch and scaling constants.

Averaging channel intensities in linear space would give excessive weight to
bright channels. In a scene where red is clipped at 2.0 (after WB) and blue
reads 0.5, a linear average of (2.0 + 0.5) / 2 = 1.25 overpredicts the
expected green value. The cube-root transform compresses the bright outlier:
cbrt(2.0) = 1.26, cbrt(0.5) = 0.79, average = 1.03, cubed back = 1.09. This
produces a more conservative, visually plausible estimate.

The choice of cube root over gamma (x^(1/3) vs x^(1/2.2)) is a pragmatic one:
`Math.cbrt` is a single standard library call, numerically stable for all
non-negative inputs, and the exponent 1/3 is close enough to 1/2.4 (sRGB
gamma) for the purposes of perceptual averaging in highlight regions where
precision matters less than avoiding gross color errors.

### Why a Global Chrominance Correction

The chrominance correction captures the color bias of the scene at a global
level. Without it, the opposed-channel average alone would produce achromatic
(gray) estimates for all clipped pixels, because the average of two opposing
channels naturally cancels chrominance. The correction re-introduces the color
difference measured from unclipped pixels near the clip boundary, where the
scene color is still fully observable.

Using a global (whole-image) average rather than a local per-pixel correction
is a simplification that works well for typical photographic scenes where
highlights share a common color (daylight-colored specular reflections, window
light, sky). The segmented reconstruction pass (step 7b) provides local
refinement for scenes where highlights have varying colors.

### Why This Step Precedes Segmented Reconstruction

The opposed inpainting pass is fast (O(pixels), with a small constant factor
from the 3x3 neighborhood) and provides a reasonable baseline estimate for
every clipped pixel individually. The segmented reconstruction pass
(`reconstructHighlightsSegmented`) is more expensive: it builds downsampled
color planes, runs flood-fill segmentation, performs morphological closing, and
searches for optimal candidate pixels per segment. Running the cheap opposed
pass first ensures that even if the segmented pass has poor coverage in some
regions (small isolated clipped pixels that do not form analyzable segments),
those pixels still receive a corrected value.

The segmented pass receives the original (pre-opposed) CFA as its reference
but writes into the already-inpainted CFA. Its reconstruction formula also
uses `Math.max`, so it can only improve upon (extend above) the opposed pass's
estimates, never degrade them.

### Why the chromLo Threshold Is 0.2

The lower chrominance threshold `chromLo = 0.2` excludes dark pixels from the
chrominance calibration. Dark CFA values have high relative noise, and their
chrominance (value minus opposed-channel estimate) is dominated by noise rather
than scene color. Including them would add noise to the global chrominance
correction without providing useful color information. The 0.2 threshold is
conservative -- it corresponds to roughly 20% of sensor saturation in
normalized space, well above the noise floor for most modern sensors.

### Why the Minimum Count Is 100

The chrominance correction for a channel is only applied when at least 100
qualifying pixels contributed to its calculation. This prevents numerically
unstable corrections in images where very few pixels of a given channel fall in
the near-clip range. With fewer than 100 samples, the mean chrominance could be
driven by a handful of outliers (hot pixels, noise spikes), producing a
correction that introduces color casts rather than removing them. Falling back
to zero chrominance (pure opposed average) is safer in this case.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/postprocess-gpu.ts` | **Active implementation.** GPU compute pipeline: passes 2-5 implement opposed-channel highlight recovery on demosaiced RGB data. Called by `gpuPostprocess()`. |
| `web/src/pipeline/preprocessor.ts` | **Inactive CPU implementation.** `reconstructHighlightsCfa` -- the opposed-channel inpainting function operating on CFA mosaic data; `channelClips` -- per-channel clip threshold computation; uses dilated mask chrominance at 1/3 resolution |
| `web/src/hooks/useProcessFile.ts` | Pipeline orchestration: calls `gpuPostprocess()` which invokes GPU highlight recovery for both neural-net and traditional demosaic paths |
| `web/src/pipeline/highlight-segments.ts` | `reconstructHighlightsSegmented` -- the second-pass segmented reconstruction; contains its own `calcRefavg` implementation following the same cube-root opposed pattern |
| `highlight_recovery.py` | Python CFA-domain opposed-channel implementation (mirrors CPU TS version) |
| `highlight_recovery_rgb.py` | Python RGB-domain opposed-channel implementation (mirrors GPU version) |

## Antipatterns

### Do Not Run Opposed Inpainting Before White Balance

The opposed-channel average assumes that equal channel values correspond to
equal scene luminance. Without white balance, the channels have unequal
sensitivity gains. A red photosite reading 0.8 and a blue photosite reading 0.8
may represent very different actual scene intensities (by a factor of 2x or
more, depending on the illuminant and sensor response). Computing the opposed
average on unbalanced data produces systematically biased estimates: channels
with low WB gain (typically green) are underestimated, and channels with high
gain (red, blue) are overestimated. This manifests as color casts in
reconstructed highlights -- exactly the artifact the algorithm exists to
prevent.

### Do Not Average in Linear Space Without the Cube-Root Transform

Removing the `Math.cbrt` / cube transform and computing the opposed average
directly in linear space produces estimates that are too high for highlights
with uneven channel clipping. Linear averaging gives disproportionate weight to
the brightest unclipped channel, pulling the estimate toward that channel's
value. The cube-root compression reduces this bias. A direct linear average
will produce visible color shifts at the boundary between fully clipped and
partially clipped regions, where the transition should be smooth.

### Do Not Apply the Chrominance Correction in Cube-Root Space

The chrominance correction (`chrom[c]`) is computed as the difference between
linear values: `val - refavg` where both `val` and `refavg` are in linear
space (the refavg is converted back from cube-root before subtraction). This
linear-space correction is then added to the linear-space refavg in Pass 2:
`estimate = ref + chrom[ch]`. Attempting to compute or apply the correction in
cube-root space would produce incorrect results because the cube-root transform
is nonlinear -- differences in cube-root space do not correspond to additive
offsets in linear space.

### Do Not Replace Math.max with Direct Assignment

The reconstruction formula uses `Math.max(cfa[idx], estimate)` rather than
direct assignment. This ensures reconstructed values never decrease a pixel's
intensity. Near the clip boundary, quantization noise in the opposing channels
can produce estimates slightly below the true value. Direct assignment would
darken these pixels, creating a visible intensity dip at the clip boundary --
a dark ring around specular highlights. The `Math.max` formulation guarantees
monotonic brightness increase from the clip edge outward into the fully clipped
region.

### Do Not Increase the 3x3 Neighborhood Size Without Profiling

The 3x3 neighborhood is the minimum viable window for computing per-channel
means in a Bayer mosaic (it guarantees at least one sample of each color
channel). Larger windows (5x5, 7x7) would provide more samples and smoother
estimates but at quadratic cost growth per pixel. For a 24-megapixel image,
the current implementation already iterates up to 9 pixels per clipped pixel
across two full-image passes. Increasing the window to 5x5 (25 pixels per
lookup) roughly triples the per-pixel work. The segmented reconstruction pass
handles cases where the 3x3 window is insufficient by analyzing larger coherent
regions, so enlarging the opposed pass's window is unnecessary.
