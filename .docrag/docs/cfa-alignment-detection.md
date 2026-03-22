---
title: CFA Phase Shift Detection and Alignment
tags: [cfa, alignment, preprocessing, pattern]
scope: cfa.py, infer.py, infer_hdr.py, web/src/pipeline/preprocessor.ts, web/src/hooks/useProcessFile.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

CFA phase shift detection and alignment is the process of determining how a camera sensor's physical color filter array is offset relative to the canonical pattern origin, and then compensating for that offset before demosaicing. This step is essential because the neural network model and the traditional demosaic algorithms both rely on channel masks or pattern lookups that assume the CFA tile starts at pixel (0, 0). If the raw image's CFA is phase-shifted — as it almost always is, due to sensor crop regions, camera-specific layouts, or visible-area offsets — the masks will map wrong colors to wrong pixels, producing catastrophic color artifacts in the output.

The problem applies to both CFA types the system supports: X-Trans (period 6, up to 36 possible phase offsets) and Bayer (period 2, up to 4 possible phase offsets corresponding to RGGB, GRBG, GBRG, BGGR). The codebase stores only one canonical pattern per CFA type (see [cfa-pattern-system](cfa-pattern-system.md)) and treats all variants as shifts of that canonical form.

Phase shift detection and alignment appears in three distinct pipeline paths:

1. **Python production inference** (`infer_hdr.py`): detects shift from `raw_colors_visible`, pads the CFA, runs tiled inference, crops the result.
2. **Python legacy inference** (`infer.py`): same logic but hardcoded to X-Trans.
3. **Web browser pipeline** (`web/src/pipeline/preprocessor.ts` and `web/src/hooks/useProcessFile.ts`): detects shift from the CFA string provided by the WASM raw decoder, pads, then routes to either neural network tiled inference or traditional WASM/GPU demosaicing.

## Pattern/Approach

### Phase Shift as (dy, dx)

A phase shift is represented as a tuple `(dy, dx)` where `dy` is the vertical offset and `dx` is the horizontal offset. The convention is: the raw image's CFA at pixel `(y, x)` corresponds to `canonical_pattern[(y + dy) % period][(x + dx) % period]`. When `(dy, dx) = (0, 0)`, the raw image is already aligned with the canonical pattern.

This convention is consistent across all implementations: Python's `find_pattern_shift()`, TypeScript's `findPatternShift()` / `matchShift()`, the WASM demosaic modules, and the WebGPU shader parameters.

### Detection Algorithm: Python

`cfa.py` provides two detection entry points:

**`detect_cfa_from_raw(raw_pattern)`** — Auto-detects the CFA type and returns `(cfa_name, canonical_pattern)`. Takes a rawpy `raw_pattern` or `raw_colors_visible` array. Remaps libraw's G2 (value 3) to G (value 1) to normalize from the 4-color libraw convention to the 3-color system convention. Tries X-Trans first (if the array is at least 6x6) by testing all 36 rolled versions of `XTRANS_PATTERN` against the top-left 6x6 block. Falls back to Bayer (all 4 rolled versions of `BAYER_PATTERN` against the top-left 2x2 block). Raises `ValueError` if neither matches.

**`find_pattern_shift(raw_pattern, reference)`** — Given a known reference pattern, returns just the `(dy, dx)` offset. Extracts the top-left `period x period` block from `raw_pattern`, remaps G2 to G, then brute-force tests all `period^2` rolled versions of the reference via `np.roll` on both axes. Returns the first matching `(dy, dx)`.

The typical inference call site in `infer_hdr.py` chains both:

```python
cfa_name, ref_pattern = detect_cfa_from_raw(raw_pattern)
period = cfa_period(ref_pattern)
dy, dx = find_pattern_shift(raw_pattern, ref_pattern)
```

The legacy `infer.py` has a self-contained `find_pattern_shift()` that is hardcoded to X-Trans (always uses period 6, always compares against `XTRANS_PATTERN`). This is the older version; new code should use the functions from `cfa.py`.

### Detection Algorithm: TypeScript

`web/src/pipeline/preprocessor.ts` exports `findPatternShift(cfaStr, cfaWidth, crops)` which returns a `CfaInfo` object containing `cfaType`, `pattern`, `period`, `dy`, and `dx`.

The TypeScript path differs from Python in its input format. The WASM raw decoder provides:

- `cfaStr`: a flat string of characters (`R`, `G`, `B`) encoding the full CFA tile
- `cfaWidth`: the tile width (6 for X-Trans, 2 for Bayer) — also used as the period
- `crops`: a `Uint16Array` of `[top, right, bottom, left]` crop offsets

The function first parses the CFA string into a 2D numeric array via `parseCfaStr()`, applying the same R=0, G=1, B=2 encoding. It then applies the crop offsets to derive the visible CFA pattern — this is a step Python does not need because `rawpy.raw_colors_visible` already reflects the visible region:

```typescript
for (let y = 0; y < period; y++) {
  for (let x = 0; x < period; x++) {
    vis[y][x] = rawPattern[(y + top) % period][(x + left) % period];
  }
}
```

The crop-adjusted visible pattern is then matched against the canonical reference via `matchShift()`, which is a brute-force loop over all `(dy, dx)` in `[0, period)^2`. For each candidate shift, it checks whether `canonical[(y + dy) % period][(x + dx) % period] === visible[y][x]` for all positions. The function dispatches on `cfaWidth`: 6 matches against `XTRANS_PATTERN`, 2 matches against `BAYER_PATTERN`.

### Padding Strategy: Python (infer_hdr.py)

Once the shift is known, the raw CFA data must be padded so that pixel (0, 0) of the padded image aligns with position (0, 0) of the canonical pattern. In `infer_hdr.py`:

```python
pad_top = (period - dy) % period
pad_left = (period - dx) % period
cfa_norm = np.pad(cfa_norm, ((pad_top, 0), (pad_left, 0)), mode='reflect')
```

The padding amounts are the *complement* of the shift modulo the period. If the raw image's CFA starts at phase `(dy, dx)` relative to the canonical pattern, then inserting `(period - dy) % period` rows at the top and `(period - dx) % period` columns at the left shifts the origin back to (0, 0). Mirror (reflect) padding is used so that the added border pixels contain plausible CFA values for the model's receptive field.

After inference, the padding is cropped away:

```python
rgb = output[:, pad_top:pad_top+h_raw, pad_left:pad_left+w_raw]
```

The legacy `infer.py` uses the same formula with `6` hardcoded in place of `period`.

### Padding Strategy: TypeScript (padToAlignment)

The web pipeline's `padToAlignment()` takes a different approach. Instead of computing the complement `(period - dy) % period`, it pads by exactly `(dy, dx)`:

```typescript
const padTop = dy;
const padLeft = dx;
```

This works because of how `(dy, dx)` is defined in the TypeScript path. The `matchShift()` function finds the shift such that `canonical[(y + dy) % period][(x + dx) % period] === visible[y][x]`. When you prepend `dy` rows and `dx` columns, the original pixel at `(y, x)` moves to `(y + dy, x + dx)` in the padded image. Then `canonical[(y + dy) % period][(x + dx) % period]` becomes `canonical[((y+dy) + 0) % period][((x+dx) + 0) % period]`, which means the padded image can be read with shift `(0, 0)` — exactly canonical alignment.

The padding fills the added border using reflected indexing:

```typescript
const srcY = y < padTop ? padTop - 1 - y : y - padTop;
const srcX = x < padLeft ? padLeft - 1 - x : x - padLeft;
```

After padToAlignment, the CFA data is canonical-aligned. This is why the subsequent `runDemosaic()` call for traditional algorithms passes `(0, 0)` as the shift:

```typescript
blended = await runDemosaic(padded.data, padded.width, padded.height, 0, 0, algorithm, flatCfa, period);
```

The neural network path also benefits: `makeChannelMasks()` generates masks at the canonical (unshifted) pattern, and after padding the CFA data matches those masks exactly.

After demosaicing, the padding is cropped away by `cropToHWC()` using the stored `padTop` and `padLeft` values.

### Shift Convention Difference: Python vs TypeScript

The Python and TypeScript shift values for the same raw file may differ because they encode complementary quantities:

- **Python** `find_pattern_shift()` finds `(dy, dx)` such that `np.roll(reference, dy, axis=0)` shifted by `(dy, dx)` equals the raw tile. Padding then uses `(period - dy) % period`.
- **TypeScript** `matchShift()` finds `(dy, dx)` such that `canonical[(y + dy) % period][(x + dx) % period] === visible[y][x]`. Padding then uses `(dy, dx)` directly.

These two conventions produce the same net alignment: both result in the padded image starting at canonical phase (0, 0). The Python approach says "how much to roll the reference to get the raw pattern" and pads by the complement; the TypeScript approach says "what offset to add to coordinates to look up the canonical pattern" and pads by the offset itself.

### Shift Propagation to Downstream Consumers

The `(dy, dx)` shift is not only used for padding. It flows through several CFA-aware operations that occur before padding:

1. **White balance** (`applyWhiteBalance` in `preprocessor.ts`): multiplies each CFA pixel by its channel's WB coefficient. Uses `pattern[(y + dy) % period][(x + dx) % period]` to determine which channel each pixel belongs to.

2. **Highlight reconstruction** (`reconstructHighlightsCfa` and `reconstructHighlightsSegmented` in `preprocessor.ts` and `highlight-segments.ts`): computes per-channel statistics and repairs clipped pixels. Uses the same `(dy, dx)`-offset pattern lookup to classify pixel channels.

3. **Traditional demosaic with strip splitting** (`demosaic-pool.ts`): when the web pipeline parallelizes demosaicing across horizontal strips, each strip receives an adjusted shift: `stripDy = (dy + startRow) % period`. This accounts for the fact that the strip starts at a different row within the CFA tile. The `dx` component is passed through unchanged since strips are horizontal.

4. **Bayer variant mapping** (`demosaic-pool.ts`): for Bayer sensors, the WASM demosaic functions accept a named variant (rggb, grbg, gbrg, bggr) rather than a shift tuple. The mapping is `BAYER_VARIANT_MAP[(dy % 2) * 2 + (dx % 2)]`, converting the 2D shift into one of four Bayer configurations.

5. **WebGPU demosaic shaders** (`demosaic-gpu.ts`): `(dy, dx)` is packed into a uniform buffer alongside width, height, and period. The shader uses `cfa_pattern[((y + params.dy) % p) * p + ((x + params.dx) % p)]` to look up the color channel at each pixel.

6. **WASM demosaic** (`web/wasm/demosaic/src/lib.rs`): receives `(dy, dx)` and applies it via `CfaPattern::xtrans_default().shift(dy, dx)`.

### The Additive Convention

All CFA-aware operations in the codebase use the additive convention for pattern lookup:

```
channel = pattern[(y + dy) % period][(x + dx) % period]
```

This is uniform across Python (`cfa.py` masks, `infer_hdr.py` WB application), TypeScript (`preprocessor.ts`, `highlight-segments.ts`, `demosaic-gpu.ts`), Rust WASM, and C test code. The additive convention means `(dy, dx)` are added to the pixel coordinates before indexing the canonical pattern.

## Rationale

### Why Phase Matters

The neural network's 4-channel input tensor encodes channel identity in the mask planes (channels 1-3). These masks are generated once per patch size from the canonical pattern with zero shift. If the CFA data is not aligned to canonical phase, a red-filter pixel in the raw data might be labeled as green by the mask. The network would then attempt to reconstruct missing red and blue channels based on a "green" input value that is actually red, producing severe color cast, moire, and detail loss.

Traditional demosaic algorithms (AHD, PPG, Markesteijn, DHT) similarly depend on knowing the exact CFA layout to perform directional interpolation. A phase error causes the algorithm to interpolate in the wrong color plane.

### Why Not Shift the Masks Instead of the Data

An alternative to padding the CFA data would be to generate per-image shifted masks. The codebase chose to pad the data for two reasons:

1. **Masks are generated once and reused across all tiles.** The neural network path generates a single set of channel masks at the configured `PATCH_SIZE` (288) and reuses them for every tile in the image. Shifting the data to canonical alignment means one mask set works for all images regardless of camera model.

2. **Padding is cheap.** The pad amounts are at most `period - 1` pixels (5 for X-Trans, 1 for Bayer). For a 6000x4000 image, padding by 5 rows and 5 columns adds negligible memory (< 0.01%) and the reflect fill has no quality impact on the model's output.

### Why Reflect Padding

Both Python (`np.pad(..., mode='reflect')`) and TypeScript (manual mirror indexing) use reflect padding. Zero padding would introduce a dark border at CFA positions where the model expects sensor signal. Reflect padding provides plausible neighboring values for the receptive field of border pixels, avoiding edge artifacts. The padded border is cropped from the final output, so its exact values only matter insofar as they influence the model's predictions near the image edge.

## Key Files

| File | Role |
|------|------|
| `cfa.py` | Canonical `find_pattern_shift(raw_pattern, reference)` and `detect_cfa_from_raw(raw_pattern)`. CFA-agnostic, handles both X-Trans and Bayer. All Python inference code should import from here. |
| `infer_hdr.py` | Production Python inference. Calls `detect_cfa_from_raw` then `find_pattern_shift`, computes `pad_top = (period - dy) % period` / `pad_left = (period - dx) % period`, applies `np.pad(..., mode='reflect')`, crops after inference. |
| `infer.py` | Legacy Python inference. Contains a local X-Trans-only `find_pattern_shift()` and `align_cfa_pattern()`. Pending migration to use `cfa.py` functions. |
| `web/src/pipeline/preprocessor.ts` | TypeScript detection (`findPatternShift`, `matchShift`, `parseCfaStr`) and alignment (`padToAlignment`). Also contains CFA-aware white balance and highlight reconstruction that consume `(dy, dx)`. |
| `web/src/pipeline/constants.ts` | TypeScript mirror of `XTRANS_PATTERN` and `BAYER_PATTERN` (see [cfa-pattern-system](cfa-pattern-system.md)). |
| `web/src/hooks/useProcessFile.ts` | Orchestrates the web pipeline: calls `findPatternShift`, applies WB and highlight reconstruction with `(dy, dx)`, calls `padToAlignment`, then routes to NN or traditional demosaic. Passes `(0, 0)` to demosaic after padding. |
| `web/src/pipeline/demosaic-pool.ts` | Strip-parallel WASM demosaic. Adjusts `dy` per strip: `stripDy = (dy + startRow) % period`. Maps `(dy, dx)` to Bayer variant string for Bayer WASM functions. |
| `web/src/pipeline/demosaic-gpu.ts` | WebGPU demosaic shaders. Receives `(dy, dx)` as uniform parameters for pattern lookup. |
| `web/src/pipeline/types.ts` | Defines `CfaInfo` interface (`cfaType`, `pattern`, `period`, `dy`, `dx`) and `PaddedImage` interface (`padTop`, `padLeft`). |
| `tests/hlrecon/extract_fixture.py` | Test fixture extraction. Uses `find_pattern_shift` from `cfa.py` and serializes `(dy, dx)` into binary fixture headers. |

## Antipatterns

### Do Not Skip Shift Detection

Every raw file must go through shift detection before demosaicing. Even cameras of the same model may have different effective shifts depending on crop mode, firmware version, or raw decoder behavior. Hardcoding `(dy, dx) = (0, 0)` will produce correct results only by coincidence and will fail silently — the output will have plausible structure but wrong colors, which is easy to miss in automated testing.

### Do Not Confuse the Two Padding Conventions

The Python path computes `pad_top = (period - dy) % period` while the TypeScript path uses `padTop = dy` directly. These are not bugs — they reflect different shift conventions (see "Shift Convention Difference" above). Do not "fix" one to match the other. If porting logic between Python and TypeScript, verify which convention the shift value was computed under.

### Do Not Apply Shift After Padding

After `padToAlignment()` (TypeScript) or the equivalent `np.pad()` (Python), the data is canonical-aligned. Any subsequent operation should use shift `(0, 0)` for pattern lookup. The web pipeline correctly passes `(0, 0)` to `runDemosaic` after padding. Passing the original `(dy, dx)` after padding would double-apply the correction, producing the same class of color errors as skipping detection entirely.

### Do Not Forget Shift When Splitting Strips

When parallelizing demosaic across horizontal strips, the vertical shift must be adjusted per strip: `stripDy = (dy + startRow) % period`. Forgetting this adjustment causes each strip to use the wrong CFA phase, producing visible color seams at strip boundaries. The `dx` component does not change because strips are horizontal. The `demosaic-pool.ts` implementation handles this correctly.

### Do Not Use infer.py's Local find_pattern_shift for New Code

The `find_pattern_shift()` in `infer.py` is X-Trans-only and does not remap G2. The canonical version in `cfa.py` handles both CFA types and performs the G2-to-G remap. The `infer.py` version exists for historical reasons and the file is legacy code pending cleanup.
