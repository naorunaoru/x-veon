---
title: WebGPU HdrRenderer for Real-Time Preview
tags: [web, webgpu, renderer, preview, hdr]
scope: web/src/gl/renderer.ts, web/src/gl/hdr-display.ts, web/src/gl/color-matrices.ts
generated: 2026-03-22
commit: 347c8dd
---

# WebGPU HdrRenderer for Real-Time Preview

## Context

The x-veon web frontend needs to display demosaiced camera sensor images with
full-range HDR tone mapping at interactive frame rates. Images arrive as linear
RGB Float32 arrays (HWC layout, three channels) from the neural network
demosaicing pipeline and must be tone-mapped through OpenDRT before reaching the
screen. The same tone mapping must also produce lossless float-precision exports
for JPEG, TIFF, and AVIF encoding, and compute per-channel histograms for the
grading UI.

The `HdrRenderer` class in `web/src/gl/renderer.ts` is the single GPU entry
point for all three tasks: display rendering, float-texture export, and
histogram computation.

## Pattern / Approach

### HdrRenderer class lifecycle

`HdrRenderer` uses an async factory pattern. Construction is private; callers
use the static `HdrRenderer.create(canvas, opts?)` method, which:

1. Obtains the module-level singleton `GPUDevice` via `getDevice()`. The device
   is created lazily, cached in `devicePromise`, and auto-reset on device loss.
   Alternatively, `setSharedDevice(device)` can inject an externally-created
   device (e.g., ONNX Runtime's WebGPU device) before `create()` is called, so
   that the renderer and inference engine share a single `GPUDevice` for
   zero-copy buffer interop.
2. Configures the canvas WebGPU context -- `rgba16float` + `display-p3` color
   space + `toneMapping: 'extended'` for HDR, or the preferred format + `srgb`
   for SDR.
3. Builds two render pipelines from the same WGSL shader module
   (`opendrt.wgsl`): one targeting the canvas format (display), one targeting a
   float texture (export).
4. Creates a compute pipeline from `histogram-hdr.wgsl` for scene-referred
   histogram binning.
5. Creates a histogram reduce compute pipeline (`histogram-reduce.wgsl`) that
   scans the 4x256 bin buffer to find the occupied range and peak count.
6. Creates a histogram visualization render pipeline (`histogram-viz.wgsl`)
   that draws RGB/Luma/EV histogram charts directly on a second WebGPU canvas,
   eliminating CPU readback of histogram data.
7. Allocates a uniform buffer (400 bytes / 100 floats), histogram
   storage/readback buffers, reduce result buffer, viz config buffer, and a
   reduced-resolution (320x320) float texture for display-referred histograms.

A static `HdrRenderer.isSupported()` gate checks `'gpu' in navigator` before
any canvas is created.

### Dual render pipelines: display vs. export

The core design separates display rendering from pixel-accurate export:

**Display pipeline** -- targets whatever `GPUTextureFormat` the canvas was
configured with (`bgra8unorm` on SDR, `rgba16float` on HDR). A single
`render()` call uploads the uniform buffer, draws a fullscreen triangle, and
optionally dispatches the histogram compute pass in the same command encoder
submission. The fullscreen triangle is generated procedurally from
`vertex_index` (no vertex buffer needed).

**Export pipeline** -- targets an off-screen `GPUTexture` in `rgba32float`
(or `rgba16float` when the `float32-blendable` feature is unavailable). The
`renderForExport(cfg, ts, gamut)` method:

1. Calls `ensureExportResources(w, h)` to lazily create or resize the export
   texture and a MAP_READ readback buffer.
2. Overrides uniform flags: sets `exportMode = 1.0`, clears `hdrDisplay`, and
   swaps the P3-to-display gamut matrix to either `P3D65_TO_REC709` (SDR
   JPEG/TIFF) or `P3D65_TO_REC2020` (HDR AVIF).
3. Renders to the export texture, copies it to the readback buffer (with
   WebGPU's mandatory 256-byte row alignment via `padTo256()`), and
   `mapAsync`s the buffer.
4. Strips RGBA back to RGB HWC Float32Array (with an `f16ToF32` fallback
   decoder for the `rgba16float` path).
5. Calls `restoreDisplayState()` to put the uniform buffer back to display
   mode and re-render the canvas.

This dual-pipeline pattern ensures the display canvas is never blocked by
export readback, and that export always operates at the full image resolution
regardless of canvas dimensions or DPI.

### Uniform buffer layout

All OpenDRT tone mapping parameters, preprocessing controls, and color
matrices are packed into a single 400-byte (100 float) uniform buffer uploaded
once per frame. The layout uses vec4f-aligned slots:

| Offset (floats) | Name           | Contents                                   |
|------------------|----------------|--------------------------------------------|
| 0                | U_TS           | ts_s, ts_s1, ts_m2, ts_dsc                 |
| 4                | U_FLAGS        | ts_x0, hdrDisplay, exportMode, ptl_enable  |
| 8                | U_ODRT_TONE    | tn_con, tn_sh, tn_toe, tn_off              |
| 12               | U_ODRT_RS      | rs_sa, rs_rw, rs_bw, 0                     |
| 16               | U_ODRT_PT      | pt_r, pt_g, pt_b, pt_rng_low               |
| 20               | U_ODRT_PT2     | pt_rng_high, ptm_high_st, 0, 0             |
| 24               | U_ODRT_LCON    | enable, tn_lcon, tn_lcon_w, tn_lcon_pc     |
| 28               | U_ODRT_HCON    | enable, tn_hcon, tn_hcon_pv, tn_hcon_st    |
| 32               | U_ODRT_BRL_RGB | brl_r, brl_g, brl_b, brl_rng               |
| 36               | U_ODRT_BRL_CMY | brl_c, brl_m, brl_y, enable                |
| 40               | U_ODRT_PTM     | enable, ptm_low, ptm_low_st, ptm_high      |
| 44               | U_PREPROCESS   | exposure, wb_temp, wb_tint, sharpen_amount  |
| 48               | U_TEXEL        | texel_w, texel_h, show_clip_overlay, 0      |
| 52               | U_SRGB_P3_C0  | sRGB-to-P3 matrix, 3 x vec4 (cols 0-2)     |
| 64               | U_P3_DSP_C0   | P3-to-display matrix, 3 x vec4 (cols 0-2)  |
| 76               | U_ODRT_HS_RGB  | hs_rgb_enable, hs_r, hs_g, hs_b            |
| 80               | U_ODRT_HS_ETC  | hs_rgb_rng, hs_cmy_enable, hc_enable, hc_r |
| 84               | U_ODRT_HS_CMY  | hs_c, hs_m, hs_y, cwp_rng                  |
| 88               | U_CWP_C0       | CWP adaptation matrix, 3 x vec4 (cols 0-2) |

The 3x3 color matrices are stored as three column-major vec4f values (row 3
zeroed) so the WGSL shader can reconstruct a `mat3x3<f32>` without padding
issues. The `setMat3()` private helper transposes from the row-major
`Float32Array` constants in `color-matrices.ts` into this column-major layout.

The `applyOpenDrtUniforms()` method writes all OpenDRT config fields into the
shadow `Float32Array` (no GPU upload -- that happens at `render()` time via
`writeBuffer`), keeping uniform updates CPU-cheap for real-time slider
interaction.

### Histogram pipeline

The histogram system has three stages: binning, reduce, and visualization.
Three channel modes are supported: `rgb`, `luma`, and `ev`. Combined with
scene vs. display source, this gives six effective modes.

**Stage 1: Binning** (`histogram-hdr.wgsl`) -- A compute shader reads the
source texture and bins pixel values into 4 x 256 `atomic<u32>` storage
slots (R, G, B, luminance). Scene histograms read from the full-resolution
`imageTex` with stride-4 subsampling and apply exposure/WB corrections
matching the display shader. Display histograms read from a 320x320
`rgba32float` texture (`dispHistTex`) that the renderer populates by
rendering the tonemapped scene with `exportMode = 1.0` in a separate
submission.

**Stage 2: Reduce** (`histogram-reduce.wgsl`) -- A single-workgroup compute
shader scans the 1024 bins (4 channels x 256) to find the occupied bin range
(lo, hi) with 5% padding and the peak count as `log(1 + max)`. Output is a
`vec4f` result buffer consumed by the viz shader. For RGB/luma modes,
`force_zero_lo = 1` anchors the range at bin 0; for EV mode it auto-ranges.

**Stage 3: Visualization** (`histogram-viz.wgsl`) -- A fullscreen-triangle
render pipeline draws the histogram chart directly onto a second WebGPU canvas
(`histVizCanvas`) configured with `alphaMode: 'premultiplied'`. Three display
modes:

- **RGB** -- Three overlapping channel fills (R/G/B) with screen blending and
  per-pixel linear interpolation between adjacent bins.
- **Luma** -- Single luminance channel with grey fill.
- **EV** -- Luminance channel mapped to EV stops (-8 to +8), with tick lines
  at major EV intervals and a zone bar showing per-zone luma distribution at
  the bottom of the chart.

The viz shader also draws an HDR clip overlay (tinted region + vertical line)
at the bin corresponding to value 1.0, visible in RGB/luma modes when the
image contains HDR content.

The reduce + viz passes are appended to the same command encoder as the scene
histogram (single submission), or issued after the display histogram's
separate binning submission. `setHistogramCanvas(canvas)` configures the
viz output canvas; passing `null` disables visualization rendering.

### HDR display detection

`hdr-display.ts` exports `probeHdrDisplay()` which returns an
`HdrDisplayInfo` with three fields: `supported`, `headroom`, and `accurate`.
It probes three sources in priority order:

1. **Window Management API** (`getScreenDetails()`) -- returns real
   nit-based headroom (e.g. 4.0 for a 400-nit peak display). Requires a user
   gesture for the permission prompt; `requestWindowManagementHeadroom()` is
   exposed separately for click handlers.
2. **`screen.highDynamicRangeHeadroom`** -- a proposed API not yet shipped in
   most browsers; checked as a forward-compatibility path.
3. **Media query `(dynamic-range: high)`** -- confirms HDR support but cannot
   report headroom, so a conservative fallback of 2.0 is used with
   `accurate: false`.

The headroom value flows into OpenDRT's tonescale computation. When HDR is
active, the canvas is configured with `display-p3` color space and the
P3-to-display gamut matrix is set to `IDENTITY_3X3` (no gamut compression
needed since the canvas is already P3).

### Color matrices

`color-matrices.ts` provides precomputed 3x3 matrices as flat row-major
`Float32Array` constants, plus helper functions for creative white adaptation:

- **`SRGB_TO_P3D65`** -- converts the demosaiced linear sRGB data into the
  P3-D65 working space used by OpenDRT.
- **`P3D65_TO_REC709`** -- converts OpenDRT output back to Rec.709/sRGB for
  SDR display and export.
- **`P3D65_TO_REC2020`** -- converts to Rec.2020 for HDR AVIF export.
- **`IDENTITY_3X3`** -- used for HDR display mode where the canvas is already
  configured for P3 output.
- **`P3D65_TO_REC709_D50`** / **`P3D65_TO_P3D65_D50`** -- CAT02 chromatic
  adaptation matrices for creative white point (CWP) warm-shift, converting
  from D65 to D50 adapted display gamuts.
- **`computeCwpAdaptMatrix(isHdr)`** -- computes the CWP adaptation matrix as
  `P3->display_D50 * inv(P3->display_D65)`. When applied to D65
  display-referred RGB, it yields the D50-adapted result for warm highlight
  blending in the shader's creative white stage.

Near-zero off-diagonal elements are explicitly zeroed (sRGB and P3 share the
D65 white point, so these terms are floating-point noise at ~1e-8).

### Image upload

Two upload paths are provided:

**CPU path** -- `uploadImage(hwc, width, height, clipMask?)` takes a
three-channel HWC Float32Array, pads it to RGBA (WebGPU has no RGB-only
texture format), and writes to an `rgba32float` `GPUTexture` via
`writeTexture`. The optional `clipMask` (per-pixel float) is packed into the
alpha channel for clip overlay visualization in the shader.

**GPU path** -- `uploadImageFromBuffer(buffer, width, height, bytesPerRow)`
takes an already-populated `GPUBuffer` in RGBA32F format (with 256-byte row
alignment) and copies it directly to the image texture via
`copyBufferToTexture`. This is the zero-copy path used by the GPU buffer
handoff from `hwc-handoff.ts`. The buffer is destroyed after the copy is
queued.

Both paths recreate the render and histogram bind groups to reference the new
texture, and update texel size uniforms (`1/width`, `1/height`) for the
sharpening kernel.

The `getDevice()` function requests the adapter's `maxBufferSize` limit because
the default 256 MB cap is insufficient for large camera images -- a 6252x4176
RGBA32F image needs approximately 398 MB for the `writeTexture` staging buffer.

### Shared device pattern

`setSharedDevice(device)` overrides the module-level `devicePromise` with an
externally-created `GPUDevice`. This is called from `useInit` after ONNX
Runtime initializes its WebGPU execution provider: `getInferenceDevice()`
returns ORT's device, and `setSharedDevice(ortDevice)` ensures that the
renderer, inference engine, and GPU postprocessing pipeline all share a single
device. Sharing the device enables zero-copy `GPUBuffer` transfers between
inference output, postprocessing, and the renderer's `uploadImageFromBuffer`.

### Integration with React

`OutputCanvas.tsx` manages the renderer lifecycle in a `useEffect`. A
`rendererKey` string (combining fileId, dimensions, HDR mode, and HDR headroom) determines
when the renderer must be recreated versus reused. The pixel data is obtained
via `takeGpuResult(fileId)` from the GPU buffer handoff and uploaded via
`uploadImageFromBuffer` (zero-copy GPU-to-GPU), then `setOpenDrtMode` +
`render` are called. A separate `useEffect` watches `lookPreset` and
`openDrtOverrides` for cheap re-renders (uniform update + draw only, no
texture re-upload).

`useExport.ts` calls `renderForExport` with the appropriate gamut. For
JPEG-HDR, two sequential export renders are performed: SDR at Rec.709 and HDR
at Rec.2020.

## Rationale

### Why WebGPU over WebGL

WebGPU provides three capabilities that WebGL2 cannot:

1. **Compute shaders** -- histograms run as GPU compute dispatches rather than
   CPU loops over readback pixels. This keeps histogram updates synchronous
   with the render frame at negligible cost.
2. **`rgba16float` canvas with extended tone mapping** -- WebGPU's canvas
   `toneMapping: { mode: 'extended' }` allows fragment shader output values
   above 1.0 to drive HDR display hardware. WebGL has no equivalent; the
   compositor clamps to [0,1].
3. **Typed buffer readback with `mapAsync`** -- export reads back float
   textures without blocking the main thread. WebGL's `readPixels` is
   synchronous and stalls the pipeline.

A secondary benefit is explicit resource lifecycle (`destroy()`) and a cleaner
binding model (bind groups vs. individual `uniform*` / `bindTexture` calls).

### Why dual pipelines share one shader module

Both the display and export pipelines use the same `opendrt.wgsl` shader with
behavior controlled by the `exportMode` uniform flag. This avoids shader
duplication and ensures pixel-exact parity between what the user sees on screen
and what gets exported. The only differences are:

- The display pipeline targets the canvas format (which may be `bgra8unorm`);
  the export pipeline targets `rgba32float` or `rgba16float`.
- The export pipeline forces `hdrDisplay = 0` and swaps the gamut matrix to
  match the target color space.

### Why a module-level device singleton with shared-device override

WebGPU device creation is expensive and triggers adapter selection. Caching the
device promise across `HdrRenderer.create()` calls avoids redundant GPU
initialization, particularly under React strict mode where effects may fire
twice. The `device.lost` handler resets the cache so the next `create()` call
re-acquires cleanly. The `setSharedDevice()` API allows the inference engine's
device to be reused, which is critical for zero-copy GPU buffer transfers
between ONNX Runtime inference, GPU postprocessing, and the display renderer.

## Key Files

| File | Role |
|------|------|
| `web/src/gl/renderer.ts` | `HdrRenderer` class: pipelines, uniforms, render, export, histogram; `setSharedDevice()` / `getDevice()` |
| `web/src/gl/hdr-display.ts` | HDR display capability probing (headroom, Window Management API) |
| `web/src/gl/color-matrices.ts` | Precomputed sRGB/P3/Rec.709/Rec.2020/D50 conversion matrices + CWP adaptation |
| `web/src/gl/shaders/opendrt.wgsl` | Combined vertex + fragment shader (OpenDRT tone mapping) |
| `web/src/gl/shaders/histogram-hdr.wgsl` | Compute shader for scene/display histogram binning |
| `web/src/gl/shaders/histogram-reduce.wgsl` | Compute shader: scans bins to find range + peak for visualization |
| `web/src/gl/shaders/histogram-viz.wgsl` | Render shader: draws RGB/Luma/EV histogram charts on GPU canvas |
| `web/src/lib/hwc-handoff.ts` | GPU buffer handoff: `setGpuResult` / `takeGpuResult` for zero-copy upload |
| `web/src/components/OutputCanvas.tsx` | React integration: renderer lifecycle, GPU buffer upload, re-render |
| `web/src/hooks/useExport.ts` | Export flow: calls `renderForExport` with SDR/HDR gamut selection |
| `web/src/store.ts` | Zustand store: holds `rendererRef` for cross-component access |

## Antipatterns

### Do not call `context.unconfigure()` in `dispose()`

The `dispose()` method intentionally skips `context.unconfigure()`. In React
strict mode, two `HdrRenderer.create()` calls can race on the same canvas. The
cancelled renderer's `dispose()` would unconfigure the context that the
surviving renderer depends on, causing a blank canvas. The comment in `dispose`
documents this. The next `create()` call re-configures the context, and browser
GC handles stale context state.

### Do not upload uniforms eagerly in `applyOpenDrtUniforms()`

Uniform data is written to a CPU-side `Float32Array` shadow buffer, not to the
GPU. The actual `writeBuffer` happens once in `render()`. Do not add
`writeBuffer` calls inside `applyOpenDrtUniforms` or `setMat3` -- batching
the upload avoids redundant GPU traffic when multiple parameters change between
frames (e.g., preset switch updates dozens of fields at once).

### Do not use filtering samplers with `rgba32float` textures

The image texture uses `unfilterable-float` sample type and a `non-filtering`
(nearest) sampler. This is a WebGPU requirement: `rgba32float` textures cannot
use linear filtering unless the device has the `float32-filterable` feature.
Switching to a filtering sampler without checking this feature will cause
pipeline creation to fail at runtime.

### Do not remove the `padTo256` alignment on export readback

WebGPU mandates that `bytesPerRow` in `copyTextureToBuffer` is a multiple of
256. The `padTo256()` utility enforces this. The readback loop then accounts
for the padded stride when extracting pixel data. Removing this alignment or
assuming a tightly-packed row layout will produce corrupted exports on images
whose `width * bytesPerPixel` is not already 256-aligned.
