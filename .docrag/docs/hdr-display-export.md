---
title: HDR Display Detection and Dual-Render Export
tags: [web, hdr, export, display]
scope: web/src/gl/hdr-display.ts, web/src/hooks/useExport.ts, web/src/pipeline/encoder.ts, web/src/pipeline/encoder-worker.ts, web/wasm/encoder/src/pipeline.rs
generated: 2026-03-22
commit: 347c8dd
---

# HDR Display Detection and Dual-Render Export

## Context

Modern browsers on HDR-capable displays can render extended-range content
through WebGPU's `toneMapping: { mode: 'extended' }` canvas configuration,
which allows fragment shader outputs above 1.0 to drive display hardware beyond
the SDR white point. However, the ecosystem for _exporting_ HDR images from the
browser is fragmented: no single format is universally viewable, and the most
widely supported option -- JPEG -- has no native HDR encoding. The Ultra HDR
JPEG specification (ISO 21496-1 gain map) bridges this gap by embedding a
secondary gain map image alongside the SDR baseline, allowing HDR-capable
viewers to reconstruct extended-range content while SDR viewers see a
fully-graded 8-bit JPEG.

This document covers two tightly coupled subsystems:

1. **HDR display detection** -- runtime probing of the physical display's peak
   luminance headroom, which feeds into OpenDRT tone mapping parameters (see
   [opendrt-tone-mapping](./opendrt-tone-mapping.md)).
2. **Dual-render export pipeline** -- the strategy for producing SDR and HDR
   pixel data from the WebGPU renderer (see
   [webgpu-renderer](./webgpu-renderer.md)) and encoding it into three export
   formats: Ultra HDR JPEG, AVIF HLG, and 16-bit TIFF.

## Pattern / Approach

### Three-tier HDR display probing

`probeHdrDisplay()` in `web/src/gl/hdr-display.ts` returns an `HdrDisplayInfo`
object with three fields:

- `supported` -- whether the display has any HDR headroom (headroom > 1.0).
- `headroom` -- the peak-to-SDR luminance ratio (e.g. 4.0 means 400 nit peak
  at 100 nit SDR reference).
- `accurate` -- whether the headroom value is precisely known or a conservative
  fallback.

The probe uses a three-tier strategy, falling through on failure:

**Tier 1: Window Management API** (`getScreenDetails()`) -- the most accurate
source. Returns real nit-based headroom from `currentScreen.highDynamicRangeHeadroom`.
This API requires a user gesture for the permission prompt, so
`requestWindowManagementHeadroom()` is exported separately for click handlers.
If the permission is denied or no user gesture is available, the probe falls
through silently.

**Tier 2: `screen.highDynamicRangeHeadroom`** -- a proposed Screen API property
not yet shipped in most browsers (checked as a forward-compatibility path). When
available it provides the same nit-based headroom as Tier 1. Both Tier 1 and
Tier 2 return `accurate: true`.

**Tier 3: Media query `(dynamic-range: high)`** -- confirms the display
supports HDR but cannot report the actual headroom value. A conservative
fallback of `headroom: 2.0` is used with `accurate: false`. This triggers a UI
flow: if `hasWindowManagementApi()` returns true, the Zustand store sets
`hdrPermissionNeeded: true`, prompting the user to grant the Window Management
permission for an accurate reading.

If none of the three tiers report HDR, the function returns
`{ supported: false, headroom: 1.0, accurate: true }`.

### Headroom flow into tone mapping

The detected headroom value propagates through the system:

1. `useInit` calls `probeHdrDisplay()` at app startup and stores results in
   Zustand via `setDisplayHdr(true, headroom)`.
2. `OutputCanvas.tsx` reads `displayHdr` and `displayHdrHeadroom` from the
   store and passes headroom to `configFromPreset()`, which sets
   `peak_luminance = headroom * 100` in the OpenDRT config.
3. The `HdrRenderer` constructor receives `isHdr` and `headroom`, configuring
   the WebGPU canvas with `rgba16float` format, `display-p3` color space, and
   `toneMapping: { mode: 'extended' }` when HDR is active.
4. When rendering on an HDR display, `ts_dsc` (display-scale) is forced to 1.0
   in `applyOpenDrt()` so the full dynamic range is preserved rather than
   compressed to SDR.

### Dual-render export strategy

Export is orchestrated by the `useExport` hook in `web/src/hooks/useExport.ts`.
The core challenge is that Ultra HDR JPEG requires _two_ pixel buffers rendered
with different tone mapping configurations and output gamuts. The export path
branches by format:

**JPEG-HDR (Ultra HDR JPEG)** -- two sequential `renderForExport` calls:

1. SDR render: OpenDRT with the user's SDR config, Rec.709 output gamut.
   Produces the primary JPEG base layer.
2. HDR render: OpenDRT with a derived HDR config (`deriveHdrConfig` sets
   `peak_luminance = 1000` nits), Rec.2020 output gamut. Provides the source
   for gain map computation.

Both renders go through the same WGSL shader with `exportMode = 1.0` and
`hdrDisplay = 0.0`, differing only in tonescale parameters and the P3-to-display
gamut matrix (`P3D65_TO_REC709` vs. `P3D65_TO_REC2020`).

**AVIF** -- a single HDR render at Rec.2020 gamut with the 1000-nit HDR config.
No SDR fallback is needed since AVIF natively supports wide color gamut and HLG
transfer.

**TIFF** -- a single SDR render at Rec.709 gamut with the user's SDR config.
Output is uncompressed 16-bit linear sRGB.

The `renderForExport` method on `HdrRenderer` returns a `Float32Array` in HWC
layout (width x height x 3 channels) at the original unrotated image dimensions.
After rendering, it calls `restoreDisplayState()` to put the uniform buffer back
to display mode and re-render the canvas so the user sees no flicker.

### Encoder worker architecture

The encoding pipeline uses a Web Worker to keep heavy pixel processing off the
main thread:

```
useExport.ts
  -> encodeImage()                  [encoder.ts, main thread]
       -> encodeViaWorker()         [posts Float32Arrays via transferable buffers]
            -> encoder-worker.ts    [Web Worker]
                 -> init()          [lazy WASM init on first encode]
                 -> encode_image()  [wasm-bindgen call to Rust]
                      -> pipeline::encode()  [format dispatch]
```

`encoder.ts` manages a lazily-created singleton `Worker`. The
`encodeViaWorker()` function copies both the SDR and HDR `Float32Array`s via
`.slice()`, then transfers ownership of the copies to the worker using
`postMessage` with a transfer list. This avoids holding the original GPU
readback data while encoding proceeds asynchronously.

`encoder-worker.ts` lazily initializes the `xtrans_encoder_wasm` WASM module on
its first message. It calls the Rust `encode_image` function (exported via
`wasm-bindgen`) and posts the result buffer back with transfer semantics.

The Rust encoder in `web/wasm/encoder/src/pipeline.rs` dispatches by format:

### Ultra HDR JPEG gain map encoding

The JPEG-HDR path in `pipeline.rs` performs four steps:

1. **SDR quantization**: applies the sRGB OETF to the Rec.709 display-linear
   data and quantizes to 8-bit RGB for the primary JPEG.

2. **Gain map computation**: for each pixel and each of 3 channels, computes
   `gain = log2((hdr_value * peak_ratio + offset) / (sdr_value + offset))`.
   A small offset (1/64) prevents division by zero in black regions. Per-channel
   min/max gain values are tracked for XMP metadata normalization. The gains are
   linearly mapped from `[gain_min, gain_max]` to `[0, 255]` per channel,
   producing a 3-channel 8-bit gain map image.

3. **Dual JPEG encoding**: both the primary SDR image and the gain map are
   independently JPEG-compressed (gain map at a fixed quality of 85).

4. **Ultra HDR container assembly** (`encode_uhdr.rs`): the final file is
   constructed as:
   - `SOI` marker
   - EXIF `APP1` segment (orientation tag)
   - XMP `APP1` segment (Container directory with `hdrgm:Version="1.0"` and a
     `GainMap` item entry referencing the gain map size)
   - MPF `APP2` segment (Multi-Picture Format index pointing to the gain map
     image's byte offset)
   - Primary JPEG body (everything after its SOI)
   - Gain map JPEG (complete, with its own XMP `APP1` containing per-channel
     `hdrgm:GainMapMin`, `hdrgm:GainMapMax`, `hdrgm:OffsetSDR`,
     `hdrgm:OffsetHDR`, `hdrgm:HDRCapacityMax`, and `hdrgm:Gamma`)

The MPF (Multi-Picture Format) structure uses little-endian TIFF encoding with
a two-entry MP Entry table. Image 2's offset is calculated relative to the MPF
TIFF header start, following the ISO 21496-1 / Google Ultra HDR convention.

### AVIF HLG encoding

The AVIF path in `pipeline.rs` and `encode_avif.rs`:

1. Physically rotates the image (AVIF's `irot` box is not used).
2. Applies the inverse of a 1.2 gamma ramp followed by the BT.2100 HLG OETF,
   quantizing to 10-bit.
3. Encodes via `rav1e` (AV1 intra-frame) at 4:4:4 chroma, full pixel range,
   with CICP tags: BT.2020 primaries, HLG transfer, Identity matrix
   coefficients (meaning planes carry G, B, R per IEC 61966-2-1).
4. Wraps the AV1 bitstream in an AVIF container via `avif-serialize`.

### 16-bit TIFF encoding

The TIFF path quantizes display-linear Rec.709 data to 16-bit unsigned integer
(linear sRGB) after physical rotation, and encodes via the `tiff` crate. No
transfer function is applied -- values are linear.

### Export format selection UI

`ExportDialog.tsx` presents three options:

| UI Label | `ExportFormat` value | Gamut | Transfer | Bit depth |
|----------|---------------------|-------|----------|-----------|
| Ultra HDR JPEG | `jpeg-hdr` | SDR: Rec.709, HDR: Rec.2020 | sRGB OETF + gain map | 8-bit + 8-bit gain |
| AVIF (BT.2020 / HLG) | `avif` | Rec.2020 | HLG OETF | 10-bit |
| TIFF (Linear sRGB) | `tiff` | Rec.709 | Linear | 16-bit |

A quality slider (mapped to JPEG quality or rav1e quantizer) is shown for JPEG
and AVIF but hidden for TIFF (lossless).

## Rationale

### Why dual render for JPEG-HDR instead of deriving the gain map from a single HDR render

The gain map specification requires a ratio between the SDR base and the HDR
representation. Computing this ratio accurately demands that both versions are
rendered through the same OpenDRT pipeline with matched artistic intent. Simply
dividing the HDR output by `peak_ratio` would discard the nonlinear tonescale
curve shape, producing gain maps that exaggerate highlights or crush shadows.
The dual-render approach ensures the SDR image looks exactly like the user's
graded preview, and the gain map captures only the true luminance difference.

The performance cost is acceptable: `renderForExport` renders to an off-screen
float texture and reads back via `mapAsync`, taking roughly 10-30ms per render
on a midrange GPU. The two renders are sequential (uniforms must change between
them), but the total time is dominated by the Wasm JPEG encoding, not the GPU
work.

### Why Ultra HDR JPEG over a single HDR-native format

Ultra HDR JPEG (ISO 21496-1 gain map JPEG) provides the widest compatibility:

- **SDR fallback**: any JPEG viewer displays the SDR base image. Users can share
  the file without worrying about viewer support.
- **HDR upgrade**: Android, Chrome, and iOS 18+ reconstruct the HDR version from
  the gain map when displayed on an HDR screen.
- **File size**: the gain map typically adds 15-30% over the base JPEG, compared
  to a full second image.

AVIF HLG is offered as an alternative for workflows that need native wide-gamut
HDR without a gain map layer, but it lacks the universal fallback of JPEG.

### Why a 3-channel gain map

The gain map is computed and encoded per-channel (R, G, B) rather than as a
single luminance channel. This preserves color-dependent headroom differences
that arise when the OpenDRT tonescale compresses different hue ranges at
different rates. A luminance-only gain map would desaturate or shift hues in
highlight recovery regions where the neural network has reconstructed
per-channel detail. The per-channel `hdrgm:GainMapMin` and `hdrgm:GainMapMax`
XMP attributes signal to decoders that three independent gain channels are
present.

### Why a Web Worker for encoding

JPEG and AV1 encoding in Wasm is CPU-intensive (100-500ms for a 26 MP image).
Running this on the main thread would block React rendering and input handling.
The Worker receives transferred `ArrayBuffer`s (zero-copy handoff) and returns
the encoded bytes the same way. The lazy singleton pattern avoids Worker startup
cost on subsequent exports.

### Why the encoder Worker uses `.slice()` before transfer

`encodeViaWorker` copies the `Float32Array`s before transferring because the
original buffers may still be referenced by the caller (e.g., for a second
export render or for display restore). Transferring the original would detach
it, causing subsequent reads to throw. The `.slice()` + transfer pattern
provides zero-copy semantics for the Worker at the cost of one memcpy on the
main thread.

## Key Files

| File | Role |
|------|------|
| `web/src/gl/hdr-display.ts` | Three-tier HDR display probe: `probeHdrDisplay()`, `requestWindowManagementHeadroom()` |
| `web/src/hooks/useExport.ts` | Export orchestration: dual render for JPEG-HDR, single render for AVIF/TIFF |
| `web/src/pipeline/encoder.ts` | Worker management, `encodeImage()` entry point, MIME type / extension mapping |
| `web/src/pipeline/encoder-worker.ts` | Web Worker: lazy Wasm init, calls Rust `encode_image` via wasm-bindgen |
| `web/wasm/encoder/src/pipeline.rs` | Rust format dispatch: SDR quantization, gain map computation, rotation |
| `web/wasm/encoder/src/encode_uhdr.rs` | Ultra HDR JPEG assembly: XMP, MPF, container layout |
| `web/wasm/encoder/src/encode_avif.rs` | AVIF encoding via rav1e: HLG OETF, 10-bit 4:4:4, CICP BT.2020 |
| `web/wasm/encoder/src/encode_tiff.rs` | 16-bit linear TIFF via the `tiff` crate |
| `web/wasm/encoder/src/transfer.rs` | Transfer functions: `srgb_oetf`, `hlg_oetf` |
| `web/src/gl/opendrt-params.ts` | `deriveHdrConfig()`: promotes SDR config to HDR by setting peak luminance |
| `web/src/hooks/useInit.ts` | Startup HDR probe integration, permission-needed flag |
| `web/src/components/ExportDialog.tsx` | Export format and quality selection UI |
| `web/src/components/OutputCanvas.tsx` | HDR headroom integration into live preview rendering |
| `web/src/gl/renderer.ts` | `HdrRenderer.renderForExport()`: off-screen float texture render + readback |
| `web/src/pipeline/types.ts` | `ExportFormat` type: `'jpeg-hdr' | 'avif' | 'tiff'` |

## Antipatterns

### Do not skip the SDR render for JPEG-HDR export

It may be tempting to derive the SDR JPEG by tone-mapping the HDR data on the
CPU (e.g., a simple clip-and-gamma). This produces an SDR base image that does
not match the user's graded preview, breaking the "what you see is what you
export" guarantee. Always perform the full SDR render through the GPU OpenDRT
pipeline with the user's actual config and Rec.709 gamut matrix.

### Do not transfer the original Float32Array to the Worker

The `renderForExport` readback buffer may be reused for subsequent renders (the
display restore render happens after the export data is obtained). Transferring
the original buffer detaches it, causing `restoreDisplayState()` or a second
export render to fail with a detached-buffer error. Always `.slice()` before
transferring.

### Do not assume headroom is always accurate after probing

When `probeHdrDisplay()` returns `accurate: false`, the headroom value (2.0) is
a conservative guess. Do not use it to make irreversible decisions (e.g.,
hard-coding a peak luminance into exported metadata). The UI should prompt for
Window Management permission to obtain the real value. Export always uses a
fixed `HDR_PEAK_LUMINANCE = 1000` nits for the HDR render, independent of the
display headroom, so exported files are not affected by probe inaccuracy.

### Do not encode the gain map at the primary JPEG's quality setting

The gain map represents smooth luminance ratios, not high-frequency image
detail. It compresses efficiently at a lower quality (hardcoded at 85 in
`encode_uhdr.rs`). Using the primary JPEG's quality (which may be 95+) wastes
bytes on gain map JPEG quantization that produces no visible improvement in the
reconstructed HDR image.

### Do not assume AVIF viewers handle Identity matrix coefficients

The AVIF encoder uses `MatrixCoefficients::Identity`, which means AV1 planes
carry G, B, R instead of Y, Cb, Cr. This is correct for 4:4:4 RGB content but
some older AVIF decoders may misinterpret it as YCbCr, producing color-shifted
output. This is a known limitation; the alternative (converting to YCbCr) would
introduce chroma subsampling artifacts at 4:2:0 or unnecessary quality loss at
4:4:4.
