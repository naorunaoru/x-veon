---
title: Python Highlight Recovery Implementations
tags: [highlight-recovery, cfa, rgb, demosaic, darktable, inpaint-opposed, segmentation]
scope: highlight_recovery.py, highlight_recovery_rgb.py, infer_hdr.py, ui.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

Digital camera sensors clip highlights when photosites reach saturation. Because
Bayer and X-Trans color filter arrays sample one color per pixel, clipping rarely
affects all three channels simultaneously -- red often clips before green and blue
due to white-balance multipliers. This means unclipped neighboring channels carry
recoverable information about the true color of blown highlights.

X-Veon provides two Python modules that implement a two-pass highlight
reconstruction algorithm adapted from darktable. They share identical algorithmic
structure but differ in the domain they operate on:

| Module | Domain | Input shape | When it runs |
|---|---|---|---|
| `highlight_recovery.py` | CFA (pre-demosaic) | `(H, W)` single-channel | Before neural demosaic, after WB applied to CFA |
| `highlight_recovery_rgb.py` | RGB (post-demosaic) | `(H, W, 3)` | After neural demosaic + WB applied to RGB |

Both are invoked from `infer_hdr.py` via the `--hlrecon` flag, which accepts
`cfa` or `rgb`. The Gradio UI in `ui.py` exposes this as a radio button labeled
"Highlight Reconstruction" with choices `cfa` (default) and `rgb`.

## Pattern / Approach

### Two recovery modes

**CFA-domain (`--hlrecon cfa`).** This follows the traditional darktable pipeline
order: white balance is applied to the raw CFA mosaic first, then highlight
recovery runs on the single-channel `(H, W)` array, and finally the neural
network demosaics. The clip levels are the per-channel WB multipliers themselves
(since after WB-on-CFA, a correctly exposed pixel of channel `c` saturates at
`wb[c]`). The CFA pattern array is required so the algorithm knows which color
each pixel belongs to.

**RGB-domain (`--hlrecon rgb`).** The neural network demosaics the raw CFA
*without* prior white balance. After demosaic produces an `(H, W, 3)` image, WB
is applied as a simple per-channel multiply on the RGB output. Highlight recovery
then operates on the three-channel result. Clip levels are `wb * 1.0` (the WB
multipliers scaled by the sensor-normalized white point of 1.0).

### Shared two-pass algorithm

Both modules implement the same two passes, run sequentially by their respective
`reconstruct_highlights()` entry point:

```
Pass 1: reconstruct_opposed()    -- inpaint-opposed, fast baseline
Pass 2: reconstruct_segmented()  -- segmentation-based, structured recovery
```

The output of Pass 1 feeds into Pass 2. Pass 2 also receives the *original*
(pre-Pass-1) data so that its reference averages come from untouched values.

### Cube-root perceptual linearization

Both passes convert pixel values into cube-root space (`cbrt(x)`, i.e.
`x^(1/3)`) before computing reference averages. This compresses the dynamic range
in a way that approximates perceptual uniformity, preventing bright unclipped
pixels from dominating the average. The constant `HL_POWERF = 3.0` is the
exponent used to convert back from cube-root space to linear (`x^3`). Both
modules define this constant identically.

### Pass 1: Inpaint-opposed

The inpaint-opposed pass estimates clipped pixel values from the *opposed*
(other-color) channels in each pixel's 3x3 neighborhood. It proceeds in five
steps:

1. **Clip detection.** Pixels are considered clipped when they exceed
   `clip_level * CLIP_MAGIC`. `CLIP_MAGIC` is 0.987 for CFA mode and 0.96 for
   RGB mode. A low-clip threshold at 20% of the clip level excludes dim pixels
   from chrominance estimation.

2. **Clip mask at 1/3 resolution.** A per-channel binary mask is built at 1/3
   the image dimensions. Any 3x3 cell containing at least one clipped pixel of a
   given channel sets that channel's mask bit. Borders are zeroed. In CFA mode
   this uses `cv2.filter2D` convolution then downsamples; in RGB mode it is a
   direct `cv2.resize` with `INTER_AREA`.

3. **Adaptive dilation.** Each channel's mask is dilated with an elliptical
   structuring element whose size scales with how early that channel clips. The
   ratio `max_clip / clip[c]` determines dilation size (clamped to odd values
   7-21). Channels that clip first (lower clip level, typically red with high
   WB multiplier) get wider dilation to find enough unclipped chrominance samples
   nearby.

4. **Chrominance estimation.** For each channel, unclipped pixels within the
   dilated mask (excluding a 3-pixel border) contribute to a global chrominance
   correction term: the mean difference between the actual pixel value and its
   opposed-channel reference average (computed in linear space after cubing the
   cube-root refavg). At least 100 samples are required per channel.

5. **Extension.** Each clipped pixel is replaced with `max(original,
   refavg_linear + chroma[c])`, ensuring the reconstructed value never falls
   below the original clipped value.

**CFA vs. RGB difference in refavg computation.** In CFA mode,
`_compute_refavg_cr` uses per-channel convolution masks (`full_pat == c`) to
extract each color's contribution from the interleaved mosaic, computing
per-channel 3x3 means, then cube-rooting each. In RGB mode, the three channels
are already separated, so it simply runs `cv2.filter2D` on each channel directly
and divides by 9 (the fixed kernel sum).

### Pass 2: Segmentation-based reconstruction

The segmentation pass identifies contiguous clipped regions and inpaints them
using a "best candidate" pixel found at the region boundary. It operates at 1/3
resolution in cube-root space. Steps:

1. **Downsampled color planes.** In CFA mode, 3x3 superpixel averages are
   computed per channel via convolution, then sampled at superpixel centers
   (row % 3 == 1, col % 3 == `xshifter` where `xshifter` depends on whether
   green occupies position (0,0) in the Bayer pattern). In RGB mode, each channel
   is resized to 1/3 dimensions via `cv2.resize` with `INTER_AREA`. Values are
   stored in cube-root space.

2. **Opposed reference at 1/3 resolution.** For each downsampled position, the
   reference average is the mean of the other two channels' cube-root values at
   that same position.

3. **Clipped pixel marking.** Downsampled values exceeding `cbrt(clip_level)`
   are flagged in the segmentation data structure (value = 1).

4. **Morphological closing.** The binary clipped mask is dilated then eroded
   (closing) using elliptical structuring elements. The dilation radius is the
   `combine_radius` parameter (default 2); the erosion radius is
   `max(radius - 1, 1)`. This bridges small gaps between clipped fragments,
   producing more coherent segments. Borders of width `HL_BORDER` (8 pixels) are
   zeroed before and after.

5. **Flood-fill segmentation.** A raster-scan seeds flood-fill from each
   remaining marked pixel (value == 1). The fill assigns a unique segment ID and
   tracks bounding box and pixel count. Segments smaller than
   `MIN_SEGMENT_SIZE` (4 pixels) are reverted. Border pixels adjacent to
   segments are tagged with `SEG_ID_MASK | sid` to identify the segment boundary.

6. **Candidate selection.** For each segment, the algorithm scans the bounding
   box (expanded by 2 pixels) for unclipped pixels belonging to the segment.
   Each candidate is scored by `weight = smoothness * brightness`:
   - **Smoothness:** `max(0, 1 - 10 * local_std_dev^0.5)` where `local_std_dev`
     is computed over a 21-pixel cross-shaped 5x5 neighborhood.
   - **Brightness:** `max(1, (min(clipval, mean_3x3) / clipval)^2)`.
   - Border pixels (tagged with `SEG_ID_MASK`) receive a 1.0 multiplier vs. 0.75
     for interior pixels, favoring edge candidates that transition smoothly.
   The best candidate's value is Gaussian-averaged over a 5x5 window (only
   unclipped contributors). It must exceed `0.125 * clipval` to be accepted,
   controlled by the `candidating` threshold (default 0.5).

7. **Chrominance transfer.** For each clipped raw pixel (at full resolution),
   its plane coordinates map it to a segment. If the segment has a valid
   candidate, the reconstructed value is:
   ```
   output = (refavg_here + candidate_value - candidate_refavg) ^ HL_POWERF
   ```
   where `refavg_here` is the full-resolution opposed reference average (in
   cube-root space) at the clipped pixel, and the difference
   `candidate_value - candidate_refavg` transfers the pseudo-chrominance from
   the candidate location. The cube (`^ 3.0`) converts back to linear space.
   The final value is `max(original, output)`.

### Integration with infer_hdr.py

In `process_raw()`, the `hlrecon` parameter controls the pipeline order:

**`hlrecon="cfa"` (default):**
- WB multipliers are applied to the normalized CFA: `cfa_norm *= wb_map`
- `reconstruct_highlights(cfa_norm, raw_pattern, clip_levels=wb)` runs both
  passes on the single-channel mosaic
- The neural network then demosaics the highlight-recovered CFA
- No further highlight processing after demosaic

**`hlrecon="rgb"`:**
- The CFA is *not* white-balanced before demosaic
- The neural network demosaics the raw CFA (clip levels = `[1, 1, 1]`)
- After demosaic, WB is applied to the RGB output: `rgb *= wb`
- `reconstruct_highlights_rgb(rgb, clip_levels=wb)` runs both passes on the
  three-channel image

After highlight recovery (either mode), `apply_color_correction()` converts from
camera RGB to BT.2020 using the camera's XYZ-to-camera matrix.

The `--hlrecon` flag is exposed on the CLI `argparse` with
`choices=["cfa", "rgb"]` and default `"cfa"`. In the Gradio UI (`ui.py`), it
appears as a `gr.Radio` widget labeled "Highlight Reconstruction" with info text
"cfa = pre-demosaic (darktable), rgb = post-demosaic". The selected value is
passed directly to `process_raw()` via the `run_inference()` callback.

### Relationship to TypeScript implementations

The web pipeline has parallel TypeScript implementations:

- **`preprocessor.ts`** contains `reconstructHighlightsCfa()`, a CFA-domain
  inpaint-opposed pass (Pass 1 only). It operates on the raw CFA before ONNX
  inference and uses the same algorithm: 1/3-resolution clip masks, adaptive
  per-channel dilation, cube-root opposed reference averages, global chrominance
  correction, and `max(original, estimate)` extension. No segmentation pass is
  included in the preprocessor.

- **`postprocess-gpu.ts`** implements the RGB-domain inpaint-opposed pass as
  WebGPU compute shaders (WGSL). The pipeline runs: white balance, refavg +
  clip mask computation, downsample clip mask to 1/3 resolution, adaptive
  elliptical dilation, workgroup-reduced chrominance accumulation (using
  `atomicAddF32` for cross-workgroup aggregation), and a finalize shader that
  applies HL extension + color correction + DR gain. This is Pass 1 only.

- **`highlight-segments.ts`** provides the full segmentation-based
  reconstruction (Pass 2) in TypeScript, matching the Python `_Segmentation`
  class, flood-fill, morphological closing, candidate selection with the same
  smoothness/brightness weighting, and chrominance transfer formula. This runs
  on the CPU in a web worker.

The TypeScript CFA-domain opposed pass and the WebGPU RGB-domain opposed pass
are structural ports of the Python code. The segmentation module is a faithful
port including the `SEG_ID_MASK` tagging convention, `MIN_SEGMENT_SIZE` threshold,
and identical candidate scoring logic.

## Rationale

- **Two passes complement each other.** The opposed pass is fast (fully
  vectorized, no iteration over segments) and handles the common case of small
  scattered clipped pixels. The segmentation pass handles large structured
  highlights (sun reflections, specular gloss) where a single global chrominance
  correction is insufficient -- it finds per-region chrominance from the best
  nearby unclipped boundary pixel.

- **Cube-root space** approximates the human visual system's response curve and
  prevents blown channels from pulling averages upward. Without it, the opposed
  reference average would be dominated by the brightest channel, producing
  incorrect hue in reconstructed highlights.

- **CFA-domain recovery preserves mosaic structure.** Running before demosaic
  means the neural network sees more physically plausible input in highlight
  regions, which can improve demosaic quality at highlight boundaries. This is
  why CFA mode is the default.

- **RGB-domain recovery as fallback.** Some workflows or models may produce
  better results when the network sees the raw unmodified CFA. The RGB path
  allows highlight recovery to clean up WB-induced clipping after demosaic
  without altering the network's input.

- **Adaptive dilation** compensates for WB-induced asymmetric clipping. Red
  typically clips first (highest WB multiplier), so its chrominance samples are
  farther from clipped regions and need a wider search radius.

## Key Files

| File | Role |
|---|---|
| `highlight_recovery.py` | CFA-domain two-pass reconstruction (pre-demosaic) |
| `highlight_recovery_rgb.py` | RGB-domain two-pass reconstruction (post-demosaic) |
| `infer_hdr.py` | Inference pipeline; `process_raw()` dispatches on `hlrecon` parameter |
| `ui.py` | Gradio web UI; `hlrecon_radio` widget and `run_inference()` callback |
| `web/src/pipeline/preprocessor.ts` | TypeScript CFA-domain opposed pass (Pass 1) |
| `web/src/pipeline/postprocess-gpu.ts` | WebGPU RGB-domain opposed pass (Pass 1, WGSL shaders) |
| `web/src/pipeline/highlight-segments.ts` | TypeScript segmentation pass (Pass 2, CPU) |

## Antipatterns

- **Do not run both CFA and RGB recovery on the same image.** They are mutually
  exclusive paths selected by `--hlrecon`. Running CFA recovery and then RGB
  recovery would double-process highlights, producing artifacts.

- **Do not change `CLIP_MAGIC` without understanding both modules.** The CFA
  module uses 0.987 (darktable's original) while the RGB module uses 0.96. These
  values were tuned independently. The RGB threshold is lower because
  post-demosaic interpolation can spread clipping, requiring earlier detection.

- **Do not feed WB-adjusted CFA to the RGB path.** The RGB path expects the
  neural network to receive raw CFA without WB. If WB is applied to the CFA
  *and* later to the RGB output, white balance is effectively applied twice.

- **Do not skip Pass 1 when using Pass 2.** The segmentation pass expects its
  input to already have opposed-inpainted data. The `reconstruct_highlights()`
  entry point enforces this ordering.

- **Avoid modifying the segmentation data structure layout.** The TypeScript port
  in `highlight-segments.ts` mirrors the Python `_Segmentation` class field by
  field. Changes to slot indexing, `SEG_ID_MASK`, or `nr` semantics must be
  synchronized across both implementations.
