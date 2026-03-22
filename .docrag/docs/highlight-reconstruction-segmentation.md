---
title: Segmentation-Based Highlight Reconstruction
tags: [web, highlights, segmentation, darktable, pipeline]
scope: web/src/pipeline/highlight-segments.ts, web/src/hooks/useProcessFile.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

Opposed-channel inpainting (see
[highlight-reconstruction-opposed](highlight-reconstruction-opposed.md)) is the
first stage of highlight recovery. It estimates a clipped channel value from the
cube-root average of the two unclipped channels at each raw photosite. This
works well for small blown highlights -- individual specular dots or thin
overexposed edges -- because the surrounding unclipped channels provide a strong
local constraint.

Opposed inpainting alone is insufficient for larger clipped regions. When an
entire area of the sensor saturates across all three color channels
simultaneously (bright sky behind a dark subject, sun reflections on water,
overexposed clothing), no per-pixel opposed signal exists because *every*
neighboring channel is also clipped. The result is a flat, colorless blob with
abrupt transitions at its border. The visual artifacts are color fringing at
highlight boundaries, false magenta or green tints inside large blown areas, and
hard edges where the inpainted region meets valid data.

The segmentation-based reconstruction addresses this by shifting from per-pixel
reasoning to per-region reasoning. Instead of asking "what should this one pixel
be?" it asks "this clipped region used to have a certain color relationship with
its surroundings -- what unclipped pixel nearby has a similar relationship, and
can we borrow its chrominance?" This is the same strategy used by darktable's
`segmentation-based` highlight recovery mode.

## Pattern / Approach

The algorithm is implemented in `reconstructHighlightsSegmented()` in
`highlight-segments.ts`. It runs as the second pass in the highlight
reconstruction pipeline, after opposed inpainting has already filled in
single-channel clips. The function modifies the CFA array in-place.

### Step 1: Superpixel Color Planes at 1/3 Resolution

The full-resolution CFA mosaic is downsampled into three color planes (R, G, B)
at one-third resolution. Each superpixel is a 3x3 block of raw photosites
centered on every third row and column. Within each 3x3 block, the raw values
are grouped by their CFA color and averaged, then converted to cube-root space
(`Math.cbrt`).

For Bayer sensors with green at position (0,0), the 3x3 superpixel is centered
at `col % 3 == 1` (the `xshifter` variable). For other patterns including
X-Trans, it centers at `col % 3 == 2`. This centering is chosen so that the
3x3 block contains a favorable ratio of green samples (5:2:2 for G:R:B in
standard Bayer), providing better chroma stability in the superpixel average.

Plane dimensions are computed as:

```
pwidth  = roundEven(floor(width / 3))  + 2 * HL_BORDER
pheight = roundEven(floor(height / 3)) + 2 * HL_BORDER
```

where `HL_BORDER = 8` provides margin for the morphological and local-variance
kernels, and `roundEven` rounds up to the next even number (matching
darktable's `dt_round_size(n, 2)`).

Alongside each color plane, a **refavg plane** is computed. For channel `c`,
the refavg is the cube-root average of the two opposed channels. For red:
`refavg_R = 0.5 * (cbrt(meanG) + cbrt(meanB))`. This captures the local
chrominance context in a way that is independent of absolute brightness.

Any superpixel whose plane value meets or exceeds the cube-root clip threshold
(`Math.cbrt(clips[c])`) is marked as clipped by setting its segmentation data
entry to `1`.

### Step 2: Border Extension

The `extendBorder()` function clamps plane values at image edges by replicating
the nearest interior column/row into the `HL_BORDER`-wide margin. This prevents
out-of-bounds reads during the morphological and local-variance operations that
use diamond-shaped kernels up to radius 5.

### Step 3: Morphological Closing and Flood-Fill Segmentation

**Morphological closing** (dilate then erode) is applied to the binary clipped
mask before segmentation. This fills small gaps between nearby clipped
superpixels so they are grouped into a single segment rather than fragmenting
into many tiny ones.

The `segmentsCombine()` function implements this:

1. **Dilate** the binary mask with a diamond kernel of the given `combineRadius`
   (default 2, valid range 0--5). The `testDilate()` function checks whether any
   pixel within the diamond is nonzero.
2. **Erode** the dilated result, but with a reduced radius of
   `max(0, combineRadius - 3)`. When `combineRadius <= 3`, no erosion is
   performed and the dilated result is used directly. This asymmetry is
   intentional: the closing is designed to bridge gaps and slightly expand clipped
   regions rather than produce a true morphological close, which would restore the
   original boundary shape.

After closing, `segmentizePlane()` performs flood-fill segmentation on each
color plane independently. The flood-fill algorithm is a faithful port of
darktable's `_floodfill_segmentize` from `segmentation.c`:

- Seeds are scanned in raster order. Each pixel with value `1` (clipped, not yet
  assigned) becomes a seed.
- From the seed, the algorithm scans right then left along the row, pushing
  above/below spans onto an explicit stack (no recursion).
- Segments smaller than `MIN_SEGMENT_SIZE` (4 superpixels) are discarded by
  resetting their data entries back to `1`.
- Pixels adjacent to a segment but not part of it are marked with
  `SEG_ID_MASK | id` (the border ring). These border pixels become important
  during candidate selection.

Each segment stores its bounding box (`xmin/xmax/ymin/ymax`), pixel count
(`size`), and slots for the candidate values (`val1`, `val2`).

The `Segmentation` data structure uses a `Uint32Array` for per-pixel segment
IDs and a parallel `tmp` buffer for morphological scratch space. The maximum
number of segments is capped at `min(floor(width * height / 4000), 262142)`,
with a floor of 256.

### Step 4: Candidate Selection

For each segment, the algorithm searches within and around the segment bounding
box (expanded by 2 pixels) for the single best **candidate pixel** -- an
unclipped superpixel that best represents the color character of the region.

The candidate weight combines two factors:

- **Smoothness**: `max(0, 1 - 10 * sqrt(localStdDev))` where `localStdDev` is
  computed over a 21-tap cross-shaped kernel (5x5 diamond). Smooth regions score
  higher because they provide more reliable chrominance estimates.
- **Brightness**: `(min(clipval, avg3x3) / clipval)^2`, clamped to a minimum
  of 1.0. In practice, since unclipped candidates always have plane values at or below clipval,
  the squared ratio is always <= 1.0 and `Math.max(1, ...)` always returns 1.0, making this factor
  a no-op in the current implementation. The code is inherited from darktable's original design.

Pixels on the segment border ring (marked with `SEG_ID_MASK`) receive a 1.0
border multiplier, while interior segment pixels receive 0.75, slightly
favoring border pixels because they sit at the transition between clipped and
unclipped data.

A candidate is accepted only if its weight exceeds `1.0 - candidating` (with
the default `candidating = 0.5`, this threshold is 0.5). If accepted, the
candidate value is refined via a Gaussian-weighted 5x5 average (using only
unclipped neighbors), and stored alongside the refavg at the candidate
location:

- `seg.val1[id]` = Gaussian-averaged plane value at the candidate (clamped to
  `clipval`, rejected if below `0.125 * clipval`).
- `seg.val2[id]` = refavg at the candidate location.

### Step 5: Pseudo-Chrominance Transfer

The final reconstruction iterates over every clipped raw photosite at full
resolution. For each clipped pixel:

1. Map its position to the superpixel plane to find its segment ID.
2. If the segment has a valid candidate, compute a per-pixel `refavgHere` from
   the **original** (pre-opposed-inpainting) raw data using `calcRefavg()`. This
   3x3 neighborhood average of the two opposed channels in cube-root space
   captures the local chrominance at full resolution.
3. Apply the transfer formula:

```
output = (refavgHere + candidate - candReference) ^ HL_POWERF
```

where `HL_POWERF = 3.0` (cube, undoing the cube root). The expression
`candidate - candReference` is the chrominance offset at the reference location;
adding it to `refavgHere` transfers that chrominance to the current pixel.

4. The final value is `max(originalValue, output)` -- the reconstruction is
   only applied if it produces a value above the original clipped level, never
   darkening a pixel.

The use of `originalCfa` (the raw input before opposed inpainting) for the
per-pixel refavg and the clipping threshold matches darktable's behavior. The
opposed-inpainted `cfa` is used for building the superpixel planes (where its
gap-filling improves the downsampled averages), but the chrominance transfer
references the true sensor readings.

### Cube-Root Space

All superpixel averaging, candidate scoring, and chrominance transfer occur in
cube-root space (`x^(1/3)`) rather than linear light. The cube root is a
simple perceptual compression similar to the L* channel of CIELAB (which uses
`x^(1/3)` for `x > 0.008856`). Working in this space ensures that chrominance
differences are perceptually uniform, so the transfer formula produces smooth
gradients even across wide luminance ranges. The cube is applied only at the
very end when writing back to the linear CFA array via `^ HL_POWERF`.

## Rationale

### Why adapted from darktable

Darktable's segmentation-based highlight recovery (by Iain, garagecoder/gmic
team, and Hanno Schwalm) is one of the most advanced open-source
implementations. It handles both Bayer and X-Trans patterns, has been tested on
thousands of real-world images, and the algorithm design is well documented on
the pixls.us forums. Porting it to TypeScript provides equivalent quality in
the browser pipeline without reinventing the approach.

The port is intentionally faithful to darktable's C implementation -- the
flood-fill algorithm, morphological kernels, candidate weighting formula, and
chrominance transfer are all direct translations. This makes it possible to
validate the TypeScript output against darktable's C output pixel-for-pixel
using the test harness in `tests/hlrecon/`.

### Why 1/3 resolution superpixels

Operating at 1/3 resolution is a deliberate trade-off:

- **Performance**: The segmentation plane is 9x smaller than the raw image. For
  a 24 MP sensor, the plane is ~2.7 MP per channel. Flood-fill, morphological
  operations, and candidate search all run on this smaller grid.
- **Noise suppression**: Averaging 9 raw photosites per superpixel dramatically
  reduces shot noise, giving the segmentation cleaner boundaries and the
  candidate weighting more stable smoothness estimates.
- **CFA coherence**: A 3x3 block is the smallest region guaranteed to contain
  samples from all three color channels for a Bayer pattern (and a reasonable
  approximation for X-Trans period-6 patterns). This makes the per-superpixel
  color average meaningful regardless of CFA type.

### Why morphological closing before segmentation

Without closing, a large blown highlight with a few surviving unclipped
superpixels in its interior would fragment into many small segments. Each small
segment would independently search for candidates, often finding poor matches
or none at all. Morphological closing bridges these interior gaps so that the
entire highlight is treated as one coherent region with a single, well-chosen
candidate.

The asymmetric closing (full-radius dilate, reduced-radius erode) intentionally
leaves the segment slightly larger than the original clipped mask. This ensures
the border ring (used for candidate search) extends into solidly unclipped
territory rather than sitting exactly at the noisy clipped/unclipped boundary.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/highlight-segments.ts` | Full implementation: `Segmentation` data structure, flood-fill, morphological closing, candidate selection, chrominance transfer, `reconstructHighlightsSegmented()` entry point |
| `web/src/hooks/useProcessFile.ts` | Pipeline orchestration: calls `reconstructHighlightsCfa()` (opposed) then `reconstructHighlightsSegmented()` with `combineRadius=2`, `candidating=0.5` |
| `tests/hlrecon/test_ts.mjs` | Node.js test runner: loads binary fixture, runs both reconstruction stages, writes output for comparison |
| `tests/hlrecon/compare.py` | Validation script: compares C (darktable reference) and TS output, reports MAE/RMSE/percentile statistics on clipped pixels |

## Antipattern

Do not run segmentation-based reconstruction *without* running opposed
inpainting first. The superpixel plane builder reads from the `cfa` array after
it has been partially repaired by opposed inpainting. If opposed inpainting is
skipped, superpixels in regions where only one or two channels are clipped will
have inaccurate averages (the clipped channels contribute their saturated
values rather than estimates), leading to incorrect segment boundaries and
corrupted candidate chrominance. The pipeline in `useProcessFile.ts` enforces
this ordering: `reconstructHighlightsCfa()` always runs before
`reconstructHighlightsSegmented()`.
