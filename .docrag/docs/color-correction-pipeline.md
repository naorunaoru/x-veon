---
title: Color Correction and Camera-to-sRGB Transform
tags: [web, color, matrix, postprocessing]
scope:
  - web/src/pipeline/postprocessor.ts
  - web/src/pipeline/postprocess-gpu.ts
  - web/src/gl/color-matrices.ts
generated: 2026-03-22
commit: 347c8dd
dependsOn:
  - tiled-inference-blending
---

# Color Correction and Camera-to-sRGB Transform

Camera sensors record light in a device-specific color space defined by
the spectral response of each color filter. This document covers how
x-veon converts demosaiced linear camera RGB into linear sRGB (and
optionally BT.2020) using a per-camera 3x3 color correction matrix,
including Fujifilm dynamic range gain compensation.

The color correction stage runs immediately after tiled inference
blending and CHW-to-HWC conversion (see `tiled-inference-blending`).
All operations happen in the linear domain, before tone mapping or
gamma encoding. The primary path is now a GPU compute pipeline
(`postprocess-gpu.ts`) that performs white balance, highlight recovery,
color correction, and DR gain in a single command buffer. The CPU path
in `postprocessor.ts` still exists for `buildColorMatrix`,
`applyColorCorrection`, and `cropToHWC` but is used only for the
traditional demosaic path or as a fallback.


## Pipeline Position

The color correction sits after demosaicing in the web processing pipeline
(`useProcessFile.ts`). There are two paths:

**Neural-net path (GPU-resident):** After tile blending produces a cropped HWC
`GPUBuffer` (via `GpuNNPipeline.finalize()`), the entire postprocessing chain
runs on GPU via `gpuPostprocess()`:

```
tile blending → GPU finalize+crop → gpuPostprocess(WB → HL → CC → DR → RGBA) → renderer
```

**Traditional demosaic path:** After `runDemosaic()` produces CHW output, the
CPU `cropToHWC()` converts to HWC, then `gpuPostprocess()` uploads the data
and performs the same GPU pipeline:

```
runDemosaic (CHW) → cropToHWC → gpuPostprocess(WB → HL → CC → DR → RGBA) → renderer
```

In both cases, `buildColorMatrix()` (CPU, in `postprocessor.ts`) constructs the
3x3 camera-to-sRGB matrix before passing it to `gpuPostprocess()`.


## cropToHWC: Layout Conversion with Padding Removal

After tile blending, the demosaiced image lives in a CHW (channel-height-width)
planar buffer that includes alignment padding added during CFA preprocessing.
Cropping and layout conversion happen differently depending on the path:

**GPU-resident path (neural-net):** The crop is performed by a WGSL compute
shader as the final step of `GpuNNPipeline.finalize()` in `tile-blend-gpu.ts`.
The shader reads from the padded HWC blend buffer and writes the unpadded region
directly to a new `GPUBuffer`. This output is already in HWC layout because the
GPU accumulate shader writes in HWC format. No CPU `cropToHWC` call is needed.

**Traditional demosaic / CPU fallback path:** `cropToHWC()` in
`postprocessor.ts` performs the operation on CPU. It reads from three separate
channel planes at stride `planeSize = hPad * wPad` and writes interleaved
triplets:

```typescript
function cropToHWC(
  output: Float32Array,   // CHW planar, padded
  hPad: number, wPad: number,
  padTop: number, padLeft: number,
  hOrig: number, wOrig: number,
): Float32Array
```

For each output pixel `(y, x)` the source index is
`(y + padTop) * wPad + (x + padLeft)` into each of the three channel
planes. The result is a contiguous `Float32Array` of length
`hOrig * wOrig * 3` in R-G-B interleaved order.


## buildColorMatrix: Constructing Camera-to-sRGB

The `buildColorMatrix` function in `postprocessor.ts` takes the
camera-specific `xyzToCam` matrix (a 3x3 `Float32Array` from the raw
file's DNG ColorMatrix tag, sourced via rawpy or the WASM raw decoder)
and produces a 3x3 Camera-to-sRGB matrix. It follows the dcraw
convention used across virtually all open-source raw processors.

### Algorithm

**Step 1 -- Build the forward matrix (sRGB-to-Camera):**

```
srgbToXyz = invert(XYZ_TO_SRGB)
srgbToCam = xyzToCam * srgbToXyz
```

`XYZ_TO_SRGB` is the standard IEC 61966-2-1 matrix (D65 whitepoint),
stored in `web/src/pipeline/constants.ts`:

```
 3.2404542  -1.5371385  -0.4985314
-0.9692660   1.8760108   0.0415560
 0.0556434  -0.2040259   1.0572252
```

This is the same constant defined in `infer_hdr.py` as `XYZ_TO_SRGB`.

**Step 2 -- Row-normalize the forward matrix:**

Each row of `srgbToCam` is divided by its row sum:

```typescript
for (let i = 0; i < 3; i++) {
  const sum = srgbToCam[i*3] + srgbToCam[i*3+1] + srgbToCam[i*3+2];
  srgbToCam[i*3]   /= sum;
  srgbToCam[i*3+1] /= sum;
  srgbToCam[i*3+2] /= sum;
}
```

**Step 3 -- Invert to get Camera-to-sRGB:**

```typescript
return invert3x3(srgbToCam);  // camToSrgb
```

### Why Row-Normalize

Row normalization ensures that sRGB white `[1, 1, 1]` maps to camera
neutral `[1, 1, 1]` in the forward direction. Without this, the color
matrix would bake in a white-point shift because `xyzToCam` rows do not
inherently sum to 1.0. Since white balance has already been applied to
the CFA data before demosaicing (each pixel multiplied by its channel's
WB coefficient), the color matrix must assume a neutral input -- i.e.,
a correctly white-balanced scene where equal-energy white produces equal
sensor response. Row normalization enforces this contract.

This is the exact same convention used by dcraw, darktable, and RawTherapee.
The Python implementation in `apply_color_correction` mirrors this logic:

```python
srgb_to_cam = xyz_to_cam @ np.linalg.inv(XYZ_TO_SRGB)
row_sums = srgb_to_cam.sum(axis=1, keepdims=True)
srgb_to_cam = srgb_to_cam / row_sums
cam_to_srgb = np.linalg.inv(srgb_to_cam)
```


## applyColorCorrection: Per-Pixel Matrix Multiply

`applyColorCorrection` applies the 3x3 camera-to-sRGB matrix to every
pixel in the HWC buffer, in place:

```typescript
function applyColorCorrection(
  hwc: Float32Array,
  numPixels: number,
  matrix: Float32Array,   // 3x3 row-major
): void
```

For each pixel at index `i`:

```
R' = max(0, m[0]*R + m[1]*G + m[2]*B)
G' = max(0, m[3]*R + m[4]*G + m[5]*B)
B' = max(0, m[6]*R + m[7]*G + m[8]*B)
```

The `Math.max(0, ...)` clamp prevents negative values that can arise
from out-of-gamut colors after the matrix transform. These negatives
would otherwise cause artifacts in subsequent tone mapping. The Python
path applies the same clamp after the matrix multiply:

```python
result_full = rgb_flat @ combined.T
# ... later in save_hdr_avif:
rgb = np.maximum(rgb, 0)
```

### In-Place Operation

The function modifies the `hwc` buffer directly rather than allocating
a new array. This is deliberate: the HWC buffer for a 26-megapixel image
is roughly 300 MB as `Float32Array`, so avoiding allocation pressure
matters for browser performance. The old RGB values are read into local
variables (`r`, `g`, `b`) before the new values are written back.


## GPU Postprocessing Pipeline (postprocess-gpu.ts)

The primary postprocessing path is now `gpuPostprocess()` in
`postprocess-gpu.ts`, which runs the entire chain as GPU compute shaders in a
single `GPUCommandEncoder`. It accepts either a `Float32Array` (uploaded to GPU)
or a `GPUBuffer` already on the device (from the GPU-resident tile blending
pipeline). The passes are:

1. **White balance (Pass 1):** Per-pixel in-place multiply of R/G/B by the
   normalized WB coefficients.

2. **Refavg + clip mask (Pass 2):** Computes per-pixel opposed-channel reference
   average (3x3 neighborhood, cube-root space) and generates a per-pixel bitmask
   of which channels are clipped. This is the GPU equivalent of the first steps
   of `reconstructHighlightsCfa`.

3. **Downsample clip mask (Pass 3):** Downsamples the clip bitmask to 1/3
   resolution by OR-ing 3x3 blocks. Border pixels are zeroed.

4. **Dilate (Pass 4):** Per-channel elliptical dilation of the downsampled mask
   with adaptive radii (7-21 px at 1/3 resolution). Channels that clip at lower
   thresholds get wider dilation to find more unclipped chrominance samples.

5. **Chrominance accumulation (Pass 5):** Workgroup-parallel reduction that
   accumulates `(value - refavg)` for unclipped pixels within the dilated mask.
   Uses atomic float-add (CAS loop) to combine workgroup partial sums into a
   global 6-element buffer (3 sums + 3 counts).

6. **Finalize (Pass 6):** Per-pixel highlight extension (`max(value, refavg +
   chroma)`), color correction (3x3 matrix multiply), DR gain multiply, and
   output to RGBA32F format. The alpha channel stores the clip ratio
   (`max(r/clip_r, g/clip_g, b/clip_b)`, clamped to [0,1]) for downstream use.

If no pixels are clipped (detected by a CPU pre-scan when input is a
`Float32Array`, or conservatively assumed for `GPUBuffer` input), passes 2-5
are skipped entirely and a simplified finalize shader (CC + DR only) is used.
This saves approximately 1 GB of VRAM that would otherwise be allocated for the
highlight recovery intermediate buffers.

The output is a `PostprocessResult` containing a `GPUBuffer` in RGBA32F layout
with row-padding for 256-byte `bytesPerRow` alignment, ready for
`copyBufferToTexture` into the renderer's display texture.


## Fujifilm Dynamic Range Gain Compensation

Fujifilm cameras have a "Dynamic Range" mode (DR200, DR400) that
deliberately underexposes the sensor by 1 or 2 stops to preserve
highlights, expecting the raw developer to compensate. The gain factor
is extracted from EXIF makernote tag `0x1403` (DevelopmentDynamicRange):

| EXIF Value | DR Mode | Gain Multiplier |
|-----------|---------|-----------------|
| 100       | DR100   | 1.0 (no-op)     |
| 200       | DR200   | 2.0             |
| 400       | DR400   | 4.0             |

**Web pipeline** (`useProcessFile.ts`): the WASM raw decoder extracts
`drGain` via Rust code in `exif_parse.rs` that parses the Fuji makernote
IFD. The gain is passed to `gpuPostprocess()` which applies it as a scalar
multiply per channel in the finalize shader, after highlight recovery and
color correction:

```wgsl
r *= params.dr_gain;
g *= params.dr_gain;
b *= params.dr_gain;
```

DR gain is applied after color correction because it is a scene-referred
exposure compensation, not a camera-space adjustment. The color matrix
operates on normalized sensor values; scaling afterward preserves the
matrix's white-point normalization.


## Python vs. TypeScript: BT.2020 Extension

The Python `apply_color_correction` has an additional `to_bt2020`
parameter. When `True` (the default for HDR AVIF output), it chains a
second matrix multiply to convert from linear sRGB to BT.2020:

```python
if to_bt2020:
    combined = SRGB_TO_BT2020 @ cam_to_srgb
```

The web pipeline does not do this in the postprocessor (CPU or GPU). Instead,
downstream color space conversions (sRGB to P3-D65 for OpenDRT tone
mapping, then P3-D65 to Rec.709 or Rec.2020 for display) happen in the
WebGPU rendering shader pipeline using precomputed matrices from
`web/src/gl/color-matrices.ts`:

| Matrix              | Purpose                                    |
|--------------------|--------------------------------------------|
| `SRGB_TO_P3D65`   | Linear sRGB to P3-D65 for OpenDRT input    |
| `P3D65_TO_REC709` | P3-D65 back to Rec.709 for SDR display     |
| `P3D65_TO_REC2020`| P3-D65 to Rec.2020 for HDR display         |
| `IDENTITY_3X3`    | Pass-through when display is already P3    |
| `P3D65_TO_REC709_D50` | P3-D65 to Rec.709 with D50 white point adaptation (CAT02, from OpenDRT CTL) |
| `P3D65_TO_P3D65_D50`  | P3-D65 with D50 adaptation for HDR P3 output |

`computeCwpAdaptMatrix(isHdr)` computes the creative white point (CWP)
adaptation matrix by multiplying the D50-adapted display matrix by the inverse
of the D65 display matrix. This allows the renderer to apply D50 white
adaptation in the display-referred domain.

These matrices share the D65 whitepoint with sRGB, so the off-diagonal
cross-terms between sRGB and P3 are small -- `SRGB_TO_P3D65` has
near-zero off-diagonals in the G-B quadrant because the two gamuts
differ primarily in red primary placement.


## Helper Functions: invert3x3 and mul3x3

`postprocessor.ts` includes hand-written 3x3 matrix inversion and
multiplication (no external linear algebra library). Both use row-major
flat `Float32Array` layout matching WebGL/WebGPU conventions:

- `invert3x3(m)`: Cofactor expansion with scalar `1/det`.
- `mul3x3(a, b)`: Standard `O(n^3)` row-by-column multiply.

The Python path uses `numpy.linalg.inv` and the `@` operator but
promotes to `float64` during the inversion to avoid precision issues
with `float32` matrix conditioning, then casts the final result back to
`float32`. The TypeScript version operates in `Float32Array` throughout
-- this has not caused visible precision issues because the matrices are
well-conditioned (determinant magnitude on the order of 1.0 for typical
camera profiles).


## Why Color Correction Runs After Demosaicing

Color correction is a per-pixel 3x3 matrix multiply that requires all
three RGB channels at each pixel position. Before demosaicing, each
pixel has only one channel value (its CFA color). Applying color
correction before demosaicing would require interpolating all three
channels first -- which is exactly what demosaicing does. Therefore the
matrix multiply must follow the demosaic step.

Additionally, the neural network is trained on white-balanced camera-space
RGB. Applying a color matrix before inference would transform the input
distribution away from what the model learned, degrading reconstruction
quality at edges and fine detail.


## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/postprocess-gpu.ts` | `gpuPostprocess()`: GPU compute pipeline for WB, highlight recovery, CC, DR gain, and RGBA output. 7 WGSL shaders (wb, refavgClip, downsampleClip, dilate, chroma, finalize, finalizeSimple). |
| `web/src/pipeline/postprocessor.ts` | `buildColorMatrix`, `applyColorCorrection` (CPU fallback), `cropToHWC` (CPU fallback), `invert3x3`, `mul3x3` |
| `web/src/pipeline/constants.ts` | `XYZ_TO_SRGB` (D65), `SRGB_TO_BT2020` |
| `web/src/gl/color-matrices.ts` | `SRGB_TO_P3D65`, `P3D65_TO_REC709`, `P3D65_TO_REC2020`, `P3D65_TO_REC709_D50`, `P3D65_TO_P3D65_D50`, `computeCwpAdaptMatrix` for GPU-side gamut conversion |
| `web/src/hooks/useProcessFile.ts` | Orchestrates the postprocessing sequence: calls `buildColorMatrix` on CPU, then `gpuPostprocess()` for the full GPU chain |
| `web/src/pipeline/types.ts` | `RawImage.xyzToCam`, `RawImage.drGain` type definitions |
| `web/wasm/rawloader/src/exif_parse.rs` | Fuji DR gain extraction from EXIF makernote (tag `0x1403`) |
| `infer_hdr.py` | Python equivalent: `apply_color_correction`, `extract_dr_gain`, `XYZ_TO_SRGB`, `XYZ_TO_BT2020` |


## Antipatterns

**Applying color correction before white balance.** The row-normalization
step assumes white `[1,1,1]` maps to camera neutral. If white balance
has not been applied, the input white point is wrong and the matrix
produces a color cast. x-veon applies WB to the CFA before demosaicing
(multiplying each pixel by its channel's WB coefficient), so by the time
`applyColorCorrection` runs, the data is already white-balanced.

**Skipping row normalization.** Building `camToSrgb` as
`inv(xyzToCam) * XYZ_TO_SRGB` without normalizing the intermediate
forward matrix produces a matrix that does not map white-balanced neutral
to sRGB white. The resulting images exhibit a global color shift that
varies by camera model.

**Applying DR gain before color correction.** The DR gain is a simple
scalar exposure boost. Applying it before the color matrix would not
change the mathematical result (scaling commutes with matrix multiply),
but the pipeline convention is to apply it after to keep the color
matrix operating on the same value range regardless of DR mode. This
matches the Python pipeline ordering and makes debugging easier: the
color-corrected image can be inspected at a consistent exposure level
before the DR boost.

**Clamping negative values too aggressively.** The `Math.max(0, ...)`
clamp in `applyColorCorrection` zeros out negative channel values from
the matrix transform. A more sophisticated approach would use gamut
mapping to redistribute the out-of-gamut energy. The current approach
is acceptable because the downstream tone mapper (OpenDRT) handles
gamut compression in the display-referred domain.

**Allocating a second HWC buffer for color correction.** Because the
matrix multiply reads all three channels before writing, it might seem
necessary to allocate a separate output buffer. The implementation
avoids this by reading R, G, B into local variables before writing
back, saving a 300 MB allocation for a typical 26 MP image.
