---
title: CFA Preprocessing in the Web Pipeline
tags: [web, preprocessing, cfa, pipeline]
scope: web/src/pipeline/preprocessor.ts, web/src/hooks/useProcessFile.ts, web/src/pipeline/constants.ts, web/src/pipeline/types.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

CFA preprocessing transforms raw sensor mosaic data into the format expected by
the demosaicing stage -- whether that stage is a neural network or a traditional
algorithm such as AHD or Markesteijn. The input is a `RawImage` struct produced
by the WASM raw decoder (see [raw-decoding-wasm](raw-decoding-wasm.md)),
containing `Uint16Array` pixel data alongside metadata: black/white levels,
white balance coefficients, CFA pattern description, and crop rectangles. The
output is a set of 5-channel `Float32Array` tiles (or a single padded CFA plane
for traditional demosaic) with the CFA aligned to canonical phase so that
channel masks and pattern lookups can assume a (0, 0) origin.

This preprocessing pipeline lives in `web/src/pipeline/preprocessor.ts` and is
orchestrated by `web/src/hooks/useProcessFile.ts`. Every function operates on a
single-channel CFA plane -- the Bayer or X-Trans mosaic -- not on a
demosaiced RGB image. The pipeline is sequential: each step consumes the output
of the previous step and, where possible, releases the prior buffer to control
peak memory.

The preprocessing must handle both Bayer (period 2) and X-Trans (period 6)
sensors uniformly. CFA type detection and phase alignment are handled by the
functions documented in [cfa-alignment-detection](cfa-alignment-detection.md);
this document focuses on the data transformations that follow detection and
precede demosaicing.

## Pattern / Approach

### Step 1: cropToVisible

```
cropToVisible(rawData, fullWidth, fullHeight, crops) -> CroppedImage
```

Camera sensors include inactive border pixels outside the optically exposed
area. The WASM decoder provides a `crops: Uint16Array` with `[top, right,
bottom, left]` pixel counts for this border. `cropToVisible` extracts the
visible rectangle by copying row-by-row from the full sensor buffer into a new
`Uint16Array` of dimensions `(fullWidth - left - right) x (fullHeight - top -
bottom)`.

If all four crop values are zero (common for DNG files that have already been
cropped), the function returns the original buffer without allocation. This
fast-path avoids a redundant copy of the 20-50 million pixel array.

The crop offsets also affect CFA phase: `findPatternShift` accounts for `top`
and `left` when computing the visible pattern (see
[cfa-alignment-detection](cfa-alignment-detection.md)). It is essential that
cropping and phase detection use the same crop values; both receive the raw
`crops` array directly.

### Step 2a: calibrateWhiteLevels

```
calibrateWhiteLevels(rawData, width, height, whiteLevels) -> Uint16Array
```

Camera metadata white levels (from rawloader TOML / DNG tags) often report the
theoretical ADC maximum (e.g., 16383 for 14-bit) rather than the actual
photosite saturation, which can be significantly lower. `calibrateWhiteLevels`
detects the real saturation from the raw data:

1. For each of the four 2x2 CFA positions, finds the actual data maximum.
2. Counts how many pixels sit at that maximum (within 1 DN tolerance).
3. If a meaningful fraction of pixels (>= 0.01%) are at the maximum **and** the
   maximum is below the metadata white level, adopts the measured value as the
   effective white point for that position.

This produces a 4-element `Uint16Array` (one per 2x2 CFA position), which is
passed to `normalizeRawCfa` instead of the raw metadata `whiteLevels`.

### Step 2b: normalizeRawCfa

```
normalizeRawCfa(rawData, width, height, blackLevels, whiteLevels) -> Float32Array
```

Converts the integer sensor data (typically 12-14 bit values in a `Uint16Array`)
to a floating-point [0, 1] range using per-CFA-position black and white levels
(matching darktable's rawprepare module):

```
id = (y & 1) << 1 | (x & 1)    // 2×2 CFA position index (0-3)
out[y * w + x] = (rawData[y * w + x] - black[id]) / (white[id] - black[id])
```

Each of the four 2x2 photosite positions gets its own black subtraction and
range, so every channel clips at exactly 1.0 after normalization. This is why
`channelClips()` now returns a uniform `CLIP_MAGIC` (0.96) for all three
channels rather than per-channel values derived from white balance.

Values below the black level become negative; values at or above the white level
become >= 1.0. Neither case is clamped here -- downstream steps (highlight
reconstruction) rely on detecting values near or above the clip point.

The output `Float32Array` replaces the `Uint16Array` from `cropToVisible`. The
caller nulls the `CroppedImage` reference immediately after this step to free
the integer buffer.

### Step 3: Compute White Balance Coefficients

White balance coefficients are extracted from `raw.wbCoeffs` (a 3-element
`Float32Array` of `[R, G, B]` multipliers from the WASM decoder) and normalized
so that the green channel has unity gain:

```
wb = [wbCoeffs[0] / wbCoeffs[1], 1.0, wbCoeffs[2] / wbCoeffs[1]]
```

This normalization is performed in `useProcessFile.ts`, not in the preprocessor
module. The green-unity convention ensures that the green channel (which carries
luminance and dominates the mosaic) passes through unscaled, and only the red
and blue channels are adjusted.

### Step 4: findPatternShift

```
findPatternShift(cfaStr, cfaWidth, crops) -> CfaInfo { cfaType, pattern, period, dy, dx }
```

Determines the CFA type (Bayer or X-Trans) and the phase offset `(dy, dx)` of
the visible area relative to the canonical pattern. This step is fully
documented in [cfa-alignment-detection](cfa-alignment-detection.md). The
returned `CfaInfo` is consumed by subsequent CFA-aware operations: alignment
padding, mask generation, clip ratio computation, and traditional demosaic
dispatch.

### Step 5: White Balance (Deferred)

White balance is **not** applied to the CFA before demosaicing. The WB
coefficients are computed at this step (see Step 3) but the actual multiply
is deferred to the GPU postprocessing pipeline (`gpuPostprocess()` in
`postprocess-gpu.ts`), which applies WB to the demosaiced RGB image.

The function `applyWhiteBalance()` still exists in `preprocessor.ts` for
potential use by other code paths, but `useProcessFile.ts` does not call it.
The neural network is trained on un-white-balanced CFA data, so applying WB
before inference would shift the input distribution away from the training
data.

The WB-scaled clip thresholds (`clipNorm * wb`) are computed here and passed
to `gpuPostprocess()` for highlight recovery, which operates on WB'd data.

### Step 6: Highlight Reconstruction (Deferred to GPU)

Highlight reconstruction is **no longer** performed on the CFA before
demosaicing. Instead, `gpuPostprocess()` in `postprocess-gpu.ts` performs
inpaint-opposed highlight recovery on the demosaiced RGB image after white
balance has been applied. This GPU implementation (see
[color-correction-pipeline](color-correction-pipeline.md)) mirrors the logic of
`reconstructHighlightsCfa` but operates on full RGB data rather than the CFA
mosaic.

The CPU functions `reconstructHighlightsCfa()` and
`reconstructHighlightsSegmented()` still exist in `preprocessor.ts` and
`highlight-segments.ts` respectively, but are not called by `useProcessFile.ts`.

Per-channel clip thresholds are computed by `channelClips()`, which returns a
uniform value of `CLIP_MAGIC` (0.96) for all three channels. Because
`normalizeRawCfa` now uses per-CFA-position black/white calibration, every
channel clips at the same normalized level. The 0.96 factor (defined as
`CLIP_MAGIC` in the source) triggers reconstruction slightly below the true
saturation point, catching pixels that are near-clipped but not yet at the
sensor ceiling. The WB-scaled clip thresholds (`clipNorm * wb`) are passed to
`gpuPostprocess()` for use in the GPU highlight recovery passes.

### Step 7: padToAlignment

```
padToAlignment(cfa, width, height, dy, dx) -> PaddedImage
```

Prepends `dy` rows at the top and `dx` columns at the left, shifting the CFA
data so that pixel (0, 0) of the padded image aligns with position (0, 0) of
the canonical pattern. The border is filled with reflected (mirror) values:

```
srcY = y < padTop ? padTop - 1 - y : y - padTop
srcX = x < padLeft ? padLeft - 1 - x : x - padLeft
```

If `(dy, dx) = (0, 0)`, the original buffer is returned without copying.

After this step, all downstream operations can index the canonical pattern at
`pattern[y % period][x % period]` without any shift offset. This is why
`makeChannelMasks` generates masks with zero shift, and why the traditional
demosaic path passes `(0, 0)` to `runDemosaic`. The padding amounts are stored
in the `PaddedImage` struct (`padTop`, `padLeft`) and used later by
`cropToHWC` in the postprocessor to remove the padding from the demosaiced
output.

The maximum padding is `period - 1` pixels: 5 for X-Trans, 1 for Bayer. For a
6000x4000 image this adds negligible memory overhead. For a full discussion of
why the data is padded rather than shifting the masks, see
[cfa-alignment-detection](cfa-alignment-detection.md).

### Step 8: generateTiles

```
generateTiles(width, height, patchSize, overlap) -> TileGrid
```

Computes the overlapping tile grid for neural network inference. The tile
parameters are defined in `web/src/pipeline/constants.ts`:

- **PATCH_SIZE = 288**: the side length of each tile in pixels. This matches the
  neural network's expected spatial input dimension.
- **OVERLAP = 24**: the number of pixels shared between adjacent tiles on each
  edge. Overlap prevents seam artifacts at tile boundaries because the
  blending step applies a linear ramp weight that fades out the 24-pixel border
  region of each tile.
- **TILE_BATCH = 32**: the number of tiles processed per GPU batch submission.

The stride between tile origins is `patchSize - overlap = 264` pixels. The
function computes padded canvas dimensions that tile evenly and generates a
list of `{ x, y }` tile origins. For a 6000x4000 padded image, this produces
roughly 400 tiles.

The output `TileGrid` struct contains:

- `tiles`: array of `{ x, y }` origin coordinates
- `hPad`, `wPad`: padded canvas dimensions

Note that `generateTiles` no longer copies the CFA data into a padded buffer.
The CFA data remains in the `PaddedImage` from step 7 and is uploaded to the
GPU by `createGpuNNPipeline()`, or accessed directly by `fillBatchCfa()` on the
CPU path. Pixels beyond the original image extent are handled as zero reads.

This step is only used by the neural network demosaic path. Traditional
demosaic algorithms receive the full padded image from step 7 directly.

### Step 9: makeChannelMasks

```
makeChannelMasks(patchSize, pattern, period) -> ChannelMasks { r, g, b }
```

Generates three binary `Float32Array` masks of size `patchSize x patchSize`,
one per color channel. For each pixel position `(y, x)`, exactly one of
`r[y * patchSize + x]`, `g[y * patchSize + x]`, or `b[y * patchSize + x]` is
set to 1.0; the other two are 0.0. Channel assignment uses the canonical
pattern with zero shift: `pattern[y % period][x % period]`.

These masks are generated once and reused for all tiles in the image. This is
possible because `padToAlignment` has already shifted the CFA data to canonical
alignment, so every tile's CFA values line up with the same mask set regardless
of the tile's position within the image (tile origins are always multiples of
the stride, and the pattern period divides evenly into the mask).

For X-Trans the 288x288 mask contains 2304 repetitions of the 6x6 pattern;
for Bayer, 20736 repetitions of the 2x2 pattern.

### Step 10: Batch Tile Assembly (prefillBatchMasks + fillBatchCfa)

Tile assembly has been restructured for batched inference. Instead of assembling
one tile at a time via the former `buildTileInput()`, the pipeline now operates
on batches of tiles using two functions:

**`prefillBatchMasks(buf, masks, tileCount, patchSize)`** fills channels 1-3
(R/G/B masks) of every tile slot in a pre-allocated batch buffer. Called once per
batch buffer allocation. The channel layout per tile is
`[CFA, R_mask, G_mask, B_mask, clip_ratio]` with 5 channels total.

**`fillBatchCfa(buf, cfa, cfaWidth, cfaHeight, tiles, from, to, patchSize, clips, getCh)`**
fills channel 0 (CFA values) and channel 4 (clip ratio) for a range of tiles.
Called once per batch iteration. The clip ratio is computed as
`max(min(val / clipLevel, 1) * 2 - 1, 0)`, ramping from 0 (below 50% of clip)
to 1 (at clip). Tiles that extend beyond the CFA boundary are zero-filled.

The neural network expects input of shape `[batchSize, 5, patchSize, patchSize]`
in NCHW layout:

- **Channel 0**: the raw CFA tile values, extracted from the padded mosaic.
  Values are in normalized linear space (not yet white-balanced).
- **Channel 1**: the red channel mask (`masks.r`)
- **Channel 2**: the green channel mask (`masks.g`)
- **Channel 3**: the blue channel mask (`masks.b`)
- **Channel 4**: clip ratio -- proximity to saturation, enabling learned
  highlight handling.

Each tile occupies `5 * patchSize^2 = 5 * 82944 = 414720` floats
(approximately 1.6 MB).

On the GPU-resident path, these CPU batch functions are not used. Instead,
the extract shader in `tile-blend-gpu.ts` performs the equivalent operation
entirely on the GPU (see
[tiled-inference-blending](tiled-inference-blending.md)).

This 5-channel encoding allows the network to distinguish which color each CFA
pixel represents and how close it is to saturation. The network architecture
(see [unet-model-architecture](unet-model-architecture.md)) takes this tensor
as input and produces a 3-channel RGB output of the same spatial dimensions.

## Rationale

### Why This Specific Ordering

The preprocessing steps are ordered to satisfy data dependencies and numerical
correctness:

1. **Crop before normalize**: cropping operates on integer `Uint16Array` data
   with simple row-copy semantics. Normalizing first would require cropping
   `Float32Array` data, which is functionally equivalent but wastes a
   float-to-float copy when the crop can be done at the integer stage. More
   importantly, the crop dimensions feed into `findPatternShift`, which needs
   the visible-area geometry.

2. **Normalize to produce Float32Array**: normalization maps sensor values to
   [0, 1] using per-CFA-position black/white levels (after `calibrateWhiteLevels`).
   This produces the `Float32Array` that all subsequent operations consume.
   White balance multipliers are calibrated for this normalized space; they are
   computed here but applied later on the GPU.

3. **White balance and highlight reconstruction deferred to GPU**: both WB and
   highlight recovery now run on the demosaiced RGB image in `gpuPostprocess()`
   rather than on the CFA before demosaic. The neural network is trained on
   un-white-balanced CFA data, so the CFA is kept in normalized-only space
   through preprocessing. WB is applied first in the GPU pipeline, followed by
   highlight recovery (which requires WB'd data to compute inter-channel
   relationships correctly).

4. **Pad before tiling**: tiling assumes the CFA is canonically aligned so that
   a single set of channel masks works for all tiles. Padding must happen first
   to establish this alignment.

6. **Masks generated once, before the tile loop**: the masks depend only on the
   canonical pattern and the patch size, not on any per-tile data. Generating
   them inside the tile loop would redundantly allocate and fill 3 * 82944
   floats per tile.

### Why the 0.96 Clip Factor (CLIP_MAGIC)

The `channelClips` function returns `CLIP_MAGIC` (0.96) for all channels.
Because `normalizeRawCfa` now uses per-CFA-position black/white calibration
(with `calibrateWhiteLevels` detecting actual sensor saturation), every channel
clips at a uniform normalized level. The 0.96 factor triggers reconstruction
4% below the normalized saturation point, catching near-clipped pixels. This is
safe because the reconstruction formula uses `max(clippedValue, estimate)`, so
non-clipped pixels incorrectly flagged retain their original value.

### Why 288x288 Tiles with 24-Pixel Overlap

The 288x288 patch size matches the neural network's trained spatial dimension.
The 24-pixel overlap provides a blending zone wide enough for the linear ramp to
eliminate visible seams while minimizing tile count. The overlap was reduced from
48 to 24 because the batched GPU-resident inference produces more consistent
tile-boundary predictions than single-tile inference.

### Why 5-Channel Packing

Combining CFA values, channel masks, and clip ratio into a single 5-channel
tensor simplifies the ONNX inference interface (one input, one output) and
matches the model's training data format. The clip ratio channel (channel 4)
provides the network with saturation proximity information, enabling learned
highlight handling that improves reconstruction quality near sensor clipping.
Passing the pattern or clip information as separate uniforms would require model
architecture changes and complicate multi-backend inference.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/preprocessor.ts` | All preprocessing functions: `cropToVisible`, `calibrateWhiteLevels`, `normalizeRawCfa`, `applyWhiteBalance`, `padToAlignment`, `generateTiles`, `makeChannelMasks`, `prefillBatchMasks`, `fillBatchCfa`, `findPatternShift`, `reconstructHighlightsCfa`, `channelClips` |
| `web/src/hooks/useProcessFile.ts` | Pipeline orchestration: calls preprocessing steps in order, manages buffer lifecycle, routes to NN or traditional demosaic |
| `web/src/pipeline/constants.ts` | `PATCH_SIZE` (288), `OVERLAP` (24), `TILE_BATCH` (32), `XTRANS_PATTERN`, `BAYER_PATTERN` |
| `web/src/pipeline/types.ts` | `CroppedImage`, `PaddedImage`, `TileGrid`, `ChannelMasks`, `CfaInfo`, `RawImage` interfaces |
| `web/src/pipeline/highlight-segments.ts` | `reconstructHighlightsSegmented` -- second-pass highlight recovery for connected clipped regions |
| `web/src/pipeline/postprocessor.ts` | `createTileBlender` (consumes `TileGrid` output), `cropToHWC` (removes alignment padding) |
| `web/src/pipeline/inference.ts` | `runBatch` / `runBatchGpu` -- consumes 5-channel batched tensors, runs ONNX inference |

## Antipatterns

### Do Not Reorder Normalization and White Balance

Normalizing after white balance would apply the black/white level correction to
already-scaled values, producing incorrect results. The WB multipliers are
calibrated for the [0, 1] normalized space. Similarly, applying WB to raw
`Uint16Array` data would require integer-scaled multipliers and would truncate
fractional results. Always normalize first, then apply WB.

### Do Not Run Highlight Reconstruction Before White Balance

The opposed-channel averaging in highlight recovery computes relationships
between R, G, and B channel values. These relationships are only physically
meaningful in white-balanced space where equal values correspond to equal scene
luminance. In the current pipeline, `gpuPostprocess()` correctly applies WB
(Pass 1) before highlight recovery (Passes 2-5). Running reconstruction on
un-balanced data would systematically overestimate or underestimate the
replacement values for channels with high WB gain (typically red and blue),
producing color casts in highlight-recovered regions.

### Do Not Generate Per-Tile Channel Masks

The masks from `makeChannelMasks` are invariant across tiles because the CFA
data has been padded to canonical alignment and the tile stride is a whole
number of pixels. Creating masks per tile wastes allocation and CPU time. The
current design generates one mask set and reuses it for every tile -- either via
`prefillBatchMasks()` on the CPU path or by a single GPU upload in
`createGpuNNPipeline()`.

### Do Not Skip padToAlignment for Zero Shifts

Even when `(dy, dx) = (0, 0)`, the code path should go through
`padToAlignment` rather than branching around it. The function already handles
this case by returning the original buffer without allocation. Skipping the call
entirely risks downstream code receiving a `PaddedImage` with uninitialized
`padTop` / `padLeft` fields, which would corrupt the postprocessor's crop
calculation.

### Do Not Clamp Normalized Values to [0, 1]

Normalization intentionally produces values outside [0, 1] -- below zero for
pixels under the black level, and above 1.0 for near-saturated pixels. The
clip ratio channel (channel 4 of the tile input) and the GPU highlight recovery
depend on detecting values near or above the clip threshold. Clamping to [0, 1]
before tiling or demosaic would destroy the information needed to compute the
clip ratio and would prevent downstream highlight recovery from extending values
above the clip point.
