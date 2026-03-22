---
title: Pipeline Type Contracts
tags: [pipeline, types, interfaces, data-flow, RawImage, ProcessingResult, CfaInfo, TileGrid, OpenDrtConfig]
scope: web/src/pipeline/types.ts, web/src/store.ts, web/src/gl/opendrt-params.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

The x-veon demosaicing pipeline moves data through a strict sequence of typed stages: RAW decode, CFA preprocessing, tiling, neural-net or traditional demosaic inference, postprocessing (blend + color correction), and finally display/export via WebGPU. Each stage produces a typed struct consumed by the next. This document catalogs every pipeline type contract -- the TypeScript interfaces that define data flowing between stages -- so that agents encountering type errors or wiring new stages can immediately answer: "what fields does this interface have, which stage produces it, and which stages consume it?"

All pipeline interfaces are defined in `web/src/pipeline/types.ts` except `OpenDrtConfig` (in `web/src/gl/opendrt-params.ts`) and `QueuedFile` (in `web/src/store.ts`).

---

## RawImage

The decoded sensor data plus all camera metadata needed by downstream preprocessing. Every field originates from the WASM rawloader module.

| Field | Type | Description |
|---|---|---|
| `data` | `Uint16Array` | Raw Bayer/X-Trans mosaic pixel values, row-major, full sensor dimensions |
| `width` | `number` | Full sensor width in pixels (before crop) |
| `height` | `number` | Full sensor height in pixels (before crop) |
| `wbCoeffs` | `Float32Array` | White balance multipliers [R, G, B] from camera metadata |
| `blackLevels` | `Uint16Array` | Per-channel black level offsets (typically 4 elements) |
| `whiteLevels` | `Uint16Array` | Per-channel saturation levels (typically 4 elements) |
| `xyzToCam` | `Float32Array` | 3x3 CIE XYZ-to-camera color matrix (row-major) |
| `camToXyz` | `Float32Array` | 3x4 camera-to-XYZ matrix (row-major, stride 4; 4th column unused) |
| `orientation` | `string` | EXIF orientation tag: `"Normal"`, `"Rotate90"`, `"Rotate180"`, `"Rotate270"` |
| `make` | `string` | Camera manufacturer (e.g. `"Fujifilm"`) |
| `model` | `string` | Camera model (e.g. `"X-T5"`) |
| `cfaStr` | `string` | CFA pattern as characters (`R`, `G`, `B`), length = cfaWidth * cfaHeight |
| `cfaWidth` | `number` | CFA pattern period width: 6 for X-Trans, 2 for Bayer |
| `crops` | `Uint16Array` | Sensor crop margins [top, right, bottom, left] |
| `drGain` | `number` | Fuji dynamic range gain (e.g. 2.0 for DR200); used to undo deliberate underexposure after demosaic |
| `exposureBias` | `number` | EXIF exposure bias in EV stops |
| `lensModel` | `string` | Lens model name from EXIF metadata |
| `focalLength` | `number` | Focal length in mm from EXIF |
| `fNumber` | `number` | Aperture f-number from EXIF |

**Producer:** `decodeRaw()` in `web/src/pipeline/raf-decoder.ts`. Calls the WASM `decode_image` function and assembles the struct from individual getters.

**Consumers:** `useProcessFile` hook (`web/src/hooks/useProcessFile.ts`) is the sole consumer. It destructures `RawImage` fields across pipeline steps 1-13: `data`/`width`/`height`/`crops` feed `cropToVisible`; `blackLevels`/`whiteLevels` feed `normalizeRawCfa` and `channelClips`; `cfaStr`/`cfaWidth`/`crops` feed `findPatternShift`; `wbCoeffs` builds the WB vector; `xyzToCam` feeds `buildColorMatrix`; `camToXyz`/`orientation`/`make`/`model` are carried forward into `ProcessingResultMeta`.

**Lifecycle:** Transient. Created at the start of `processFile`, consumed across multiple preprocessing calls, then garbage-collected when the function completes. The `ArrayBuffer` source is explicitly nulled after decode to reduce peak memory.

---

## CroppedImage

Visible-area pixel data after removing sensor border crops.

| Field | Type | Description |
|---|---|---|
| `data` | `Uint16Array` | Cropped pixel data (may alias `RawImage.data` if crops are all zero) |
| `width` | `number` | Visible width after crop |
| `height` | `number` | Visible height after crop |

**Producer:** `cropToVisible()` in `web/src/pipeline/preprocessor.ts`.

**Consumers:** `useProcessFile` reads `width`/`height` as `visWidth`/`visHeight` (used throughout the rest of the pipeline), then passes `data` to `normalizeRawCfa`. The reference is nulled immediately after normalization.

**Lifecycle:** Transient. If all crops are zero, `data` aliases `RawImage.data` (no copy). Explicitly nulled in `useProcessFile` after normalization to free memory.

---

## CfaInfo

Identifies the CFA type, canonical pattern, period, and alignment shift relative to the visible area.

| Field | Type | Description |
|---|---|---|
| `cfaType` | `CfaType` (`'xtrans' \| 'bayer'`) | Sensor type discriminator |
| `pattern` | `readonly (readonly number[])[]` | Canonical 2D pattern array (values: 0=R, 1=G, 2=B) |
| `period` | `number` | Pattern repetition period (6 for X-Trans, 2 for Bayer) |
| `dy` | `number` | Vertical shift to align visible area to canonical pattern |
| `dx` | `number` | Horizontal shift to align visible area to canonical pattern |

**Producer:** `findPatternShift()` in `web/src/pipeline/preprocessor.ts`. Parses the CFA string, applies crop offsets, and brute-force matches against `XTRANS_PATTERN` or `BAYER_PATTERN`.

**Consumers:** `useProcessFile` destructures all five fields. `cfaType` selects the neural-net model variant in `runTile`. `pattern`/`period`/`dy`/`dx` are passed to: `applyWhiteBalance`, `reconstructHighlightsCfa`, `reconstructHighlightsSegmented`, `padToAlignment` (uses `dy`/`dx`), `generateTiles` (indirectly via padded data), `makeChannelMasks` (uses `pattern`/`period`), `flattenPattern` (for GPU/WASM demosaic), and `runDemosaic`.

**Lifecycle:** Transient. Lives for the duration of `processFile`. The `pattern` field references a module-level constant (`XTRANS_PATTERN` or `BAYER_PATTERN`), so it is not garbage-collected with the `CfaInfo` object.

---

## PaddedImage

CFA data padded so that the top-left pixel aligns to the canonical pattern origin (shift=0,0). This eliminates the need to pass `dy`/`dx` to downstream tile and demosaic stages.

| Field | Type | Description |
|---|---|---|
| `data` | `Float32Array` | Normalized, WB-applied, highlight-reconstructed CFA with alignment padding |
| `width` | `number` | Padded width (`visWidth + dx`) |
| `height` | `number` | Padded height (`visHeight + dy`) |
| `padTop` | `number` | Number of rows added at top (equals `dy`) |
| `padLeft` | `number` | Number of columns added at left (equals `dx`) |

**Producer:** `padToAlignment()` in `web/src/pipeline/preprocessor.ts`. Returns the input array directly (no copy) when `dy === 0 && dx === 0`.

**Consumers:** `useProcessFile` passes `data`/`width`/`height` to either `generateTiles` (neural-net path) or `runDemosaic` (traditional path). `padTop`/`padLeft` are saved and later passed to `cropToHWC` in the postprocessor to strip padding from the blended output. The reference is nulled after tiling/demosaic begins.

**Lifecycle:** Transient. May alias the pre-padding CFA `Float32Array` if no padding is needed. Nulled explicitly in `useProcessFile`.

---

## TileGrid

The tiled decomposition of the padded CFA for neural-net inference.

| Field | Type | Description |
|---|---|---|
| `tiles` | `Array<{ x: number; y: number }>` | Top-left coordinates of each tile in the padded canvas |
| `hPad` | `number` | Total height of the padded tile canvas |
| `wPad` | `number` | Total width of the padded tile canvas |

**Producer:** `generateTiles()` in `web/src/pipeline/preprocessor.ts`. Computes stride from `patchSize - overlap`, calculates the padded dimensions for an integer number of tiles, and enumerates tile coordinates. Note: `generateTiles` no longer produces or stores the padded CFA data itself -- it only computes tile coordinates and canvas dimensions.

**Consumers:** `useProcessFile` (neural-net path only). `tiles` coordinates are passed to `createGpuNNPipeline` which extracts tile data on the GPU. `hPad`/`wPad` are used for blend buffer allocation and `cropToHWC`. `tiles.length` drives the progress counter and is stored as `tileCount` in `ProcessingResultMeta.metadata`.

**Lifecycle:** Transient. Only created on the neural-net path. Lightweight (just coordinate arrays and two numbers).

---

## ChannelMasks

Binary per-channel masks indicating which CFA positions correspond to R, G, or B. Used to construct the 5-channel neural-net input tensor (CFA + R/G/B masks + clip ratio).

| Field | Type | Description |
|---|---|---|
| `r` | `Float32Array` | Binary mask: 1.0 at red CFA positions, 0.0 elsewhere (length = patchSize^2) |
| `g` | `Float32Array` | Binary mask: 1.0 at green CFA positions |
| `b` | `Float32Array` | Binary mask: 1.0 at blue CFA positions |

**Producer:** `makeChannelMasks()` in `web/src/pipeline/preprocessor.ts`. Generated once per image from the canonical pattern and period (no shift, since padding already aligned the CFA).

**Consumers:** `prefillBatchMasks()` and `fillBatchCfa()` in `web/src/pipeline/preprocessor.ts`. The masks are pre-filled as channels 1-3 of the 5-channel `[cfa, r, g, b, clip_mask]` input tensor for each tile batch. Channel 4 (clip ratio) is computed per-tile in `fillBatchCfa`.

**Lifecycle:** Transient. Allocated once per `processFile` call, reused across all tiles, then garbage-collected.

---

## ExportData

Full-resolution demosaiced pixel data plus the metadata needed for color-managed export. This is the "heavy" result that includes the pixel buffer.

| Field | Type | Description |
|---|---|---|
| `hwc` | `Float32Array` | Demosaiced pixels in HWC (height-width-channel) layout, 3 channels (linear RGB), row-major |
| `width` | `number` | Image width (pre-rotation) |
| `height` | `number` | Image height (pre-rotation) |
| `xyzToCam` | `Float32Array \| null` | Color matrix (null if color correction was already applied) |
| `wbCoeffs` | `Float32Array` | White balance coefficients used during processing |
| `orientation` | `string` | EXIF orientation for display rotation |

**Producer:** Conceptually assembled in `useProcessFile`, though `ExportData` as an interface is only used by `ProcessingResult` (see below). In practice, `useProcessFile` writes the `hwc` buffer to OPFS separately and constructs `ProcessingResultMeta` (which omits `hwc`).

**Consumers:** Referenced by `ProcessingResult.exportData`. The `hwc` field is the data written to OPFS via `writeHwc`. Not stored in Zustand or IndexedDB due to its size.

---

## ExportDataMeta

Lightweight version of `ExportData` that omits the heavy `hwc` pixel buffer. Stored in Zustand. Adds `camToXyz` for use by the WebGPU renderer's color pipeline.

| Field | Type | Description |
|---|---|---|
| `width` | `number` | Image width (pre-rotation, matches HWC texture dimensions) |
| `height` | `number` | Image height (pre-rotation) |
| `xyzToCam` | `Float32Array \| null` | Color matrix (null after CC is baked in) |
| `wbCoeffs` | `Float32Array` | White balance coefficients |
| `camToXyz` | `Float32Array` | 3x4 camera-to-XYZ matrix (row-major); used by the renderer for scene-referred color transforms |
| `orientation` | `string` | EXIF orientation string |

**Producer:** `useProcessFile` constructs this inline when building `ProcessingResultMeta`. Sets `xyzToCam` to `null` because color correction is applied before storage.

**Consumers:** `OutputCanvas` reads `width`/`height` (for WebGPU texture dimensions and pan/zoom), `orientation` (for CSS rotation). The WebGPU renderer (`web/src/gl/renderer.ts`) receives dimensions via `uploadImage`. Export logic uses `camToXyz` and `wbCoeffs`.

---

## ProcessingResult

The full result of a demosaic run, containing both pixel data and metadata. Defined in types but never stored directly -- the pipeline immediately splits it into OPFS storage (hwc) and `ProcessingResultMeta` (metadata).

| Field | Type | Description |
|---|---|---|
| `exportData` | `ExportData` | Full pixel buffer + color metadata |
| `metadata` | `object` | Processing metadata (see field table below) |

Metadata sub-fields:

| Field | Type | Description |
|---|---|---|
| `make` | `string` | Camera manufacturer |
| `model` | `string` | Camera model |
| `width` | `number` | Final display width (after orientation swap) |
| `height` | `number` | Final display height (after orientation swap) |
| `tileCount` | `number` | Number of tiles processed (1 for traditional demosaic) |
| `inferenceTime` | `number` | Wall-clock processing time in seconds |
| `backend` | `string` | Inference backend (`"webgpu"`, `"wasm"`, or the demosaic method name for traditional) |
| `exposureBias` | `number` | EXIF exposure bias in EV |
| `lensModel` | `string` | Lens model name from EXIF |
| `focalLength` | `number` | Focal length in mm |
| `fNumber` | `number` | Aperture f-number |
| `colorTemp` | `number` | Estimated illuminant color temperature in Kelvin |
| `tint` | `number` | Estimated green/magenta tint |
| `modelSize` | `ModelSize \| undefined` | Neural-net model size (`'S'`, `'M'`, `'L'`); undefined for traditional demosaic |

**Note:** `ProcessingResult` itself is not instantiated at runtime. `useProcessFile` constructs `ProcessingResultMeta` directly. The `metadata` type is reused by `ProcessingResultMeta` and `SerializableResultMeta` via `ProcessingResult['metadata']`.

---

## ProcessingResultMeta

The lightweight result stored in Zustand state. Contains `ExportDataMeta` (no pixel buffer) plus the shared metadata block.

| Field | Type | Description |
|---|---|---|
| `exportData` | `ExportDataMeta` | Dimensions, color matrices, orientation (no pixels) |
| `metadata` | `ProcessingResult['metadata']` | Same metadata shape as `ProcessingResult` |

**Producer:** `useProcessFile` constructs this at the end of processing and passes it to `useAppStore.getState().setFileResult()`.

**Consumers:**
- **Zustand store** (`web/src/store.ts`): stored as `QueuedFile.result` and cached in `QueuedFile.cachedResults`.
- **OutputCanvas** (`web/src/components/OutputCanvas.tsx`): reads `metadata.width`/`metadata.height` for pan/zoom, `exportData.width`/`exportData.height`/`exportData.orientation` for canvas sizing and CSS rotation.
- **IDB persistence**: serialized via `serializeResultMeta()` before storage; deserialized via `deserializeResultMeta()` on restore.

---

## SerializableResultMeta

IndexedDB-safe version of `ProcessingResultMeta`. Converts `Float32Array` fields to `number[]` for structured-clone compatibility.

| Field | Type | Description |
|---|---|---|
| `exportData.width` | `number` | Image width |
| `exportData.height` | `number` | Image height |
| `exportData.xyzToCam` | `number[] \| null` | Serialized color matrix |
| `exportData.wbCoeffs` | `number[]` | Serialized WB coefficients |
| `exportData.camToXyz` | `number[] \| undefined` | Serialized cam-to-XYZ (optional for backward compat with older persisted data) |
| `exportData.orientation` | `string` | Orientation string |
| `metadata` | `ProcessingResult['metadata']` | Unchanged -- already JSON-safe |

**Producer:** `serializeResultMeta()` in `web/src/pipeline/types.ts`, called by `fileToPersistedFile()` in the store.

**Consumers:** `deserializeResultMeta()` in `web/src/pipeline/types.ts`, called by `persistedToQueued()` in `web/src/hooks/useInit.ts` during session restore. Note: `camToXyz` uses a 3x4 identity fallback if the field is missing (backward compatibility with data persisted before this field existed).

---

## QueuedFile

The per-file state object stored in the Zustand `files` array. Tracks the full lifecycle from drop to export.

| Field | Type | Description |
|---|---|---|
| `id` | `string` | UUID (from `crypto.randomUUID()`) |
| `file` | `File \| null` | Original `File` handle; `null` for restored sessions (data is in OPFS) |
| `name` | `string` | Display name (filename without extension) |
| `originalName` | `string` | Original filename including extension |
| `thumbnailUrl` | `string \| null` | Blob URL for the embedded JPEG thumbnail |
| `metadata` | `QuickMetadata \| null` | Quick metadata: `{ camera, lensModel, focalLength, fNumber }` |
| `cfaType` | `CfaType \| null` | `'xtrans'` or `'bayer'`; inferred from file extension at add time |
| `status` | `FileStatus` | `'queued' \| 'processing' \| 'done' \| 'error'` |
| `error` | `string \| null` | Error message when `status === 'error'` |
| `progress` | `{ current: number; total: number } \| null` | Tile processing progress (neural-net path only) |
| `result` | `ProcessingResultMeta \| null` | Active demosaic result metadata |
| `resultMethod` | `DemosaicMethod \| null` | Which method produced `result` |
| `lensProfile` | `LensProfile \| null` | Matched LensFun lens correction profile |
| `lookPreset` | `LookPreset` | Active tone-mapping preset: `'default' \| 'colorful' \| 'umbra' \| 'base' \| 'flat'` |
| `openDrtOverrides` | `Partial<OpenDrtConfig>` | Per-file sparse overrides on top of the look preset |
| `preProcessOverrides` | `Partial<PreProcessConfig>` | Per-file exposure, WB, sharpening overrides |

**Producer:** `addFiles()` action in the Zustand store creates entries from dropped `File` objects. `persistedToQueued()` in `web/src/hooks/useInit.ts` reconstructs entries from IndexedDB on session restore.

**Consumers:**
- **useProcessFile**: reads `file`/`id` to obtain the ArrayBuffer for decoding.
- **Store actions**: `updateFileStatus`, `updateFileProgress`, `setFileResult`, `setFileLensProfile`, `setFileLookPreset`, `setFileOpenDrtOverride`, `setFilePreProcessOverride` all mutate fields.
- **UI components**: `FileListItem` renders `name`/`status`/`thumbnailUrl`/`progress`. `OutputCanvas` reads `result`/`lookPreset`/`openDrtOverrides`. `GradingPanel` reads and writes `lookPreset`/`openDrtOverrides`.
- **IDB persistence**: `fileToPersistedFile()` converts to `PersistedFile` for IndexedDB storage.

---

## OpenDrtConfig

The configuration struct for the OpenDRT tone-mapping shader. Controls tonescale, saturation, purity, brilliance, hue shifts, and creative white point. Pre-processing parameters (exposure, WB, sharpening) are now in a separate `PreProcessConfig` interface; the combined type is `GradingConfig = OpenDrtConfig & PreProcessConfig`.

| Group | Key Fields | Description |
|---|---|---|
| Tonescale | `tn_lg`, `tn_con`, `tn_sh`, `tn_toe`, `tn_off` | Mid-grey luminance, contrast, shoulder, toe compression, offset |
| Low contrast | `tn_lcon_enable`, `tn_lcon`, `tn_lcon_w`, `tn_lcon_pc` | Local contrast boost (enable gate + strength/width/pivot) |
| High contrast | `tn_hcon_enable`, `tn_hcon`, `tn_hcon_pv`, `tn_hcon_st` | High-end contrast (enable gate + strength/pivot/steepness) |
| Saturation | `rs_sa`, `rs_rw`, `rs_bw` | Global saturation, red weight, blue weight |
| Purity | `pt_r`, `pt_g`, `pt_b`, `pt_rng_low`, `pt_rng_high`, `ptl_enable`, `ptm_enable`, `ptm_low`, `ptm_low_st`, `ptm_high`, `ptm_high_st` | Per-channel purity twists and masking |
| Brilliance | `brl_enable`, `brl_r/g/b`, `brl_c/m/y`, `brl_rng` | Per-hue luminance adjustments |
| Hue shifts (RGB) | `hs_rgb_enable`, `hs_r`, `hs_g`, `hs_b`, `hs_rgb_rng` | Per-primary hue rotation with range |
| Hue shifts (CMY) | `hs_cmy_enable`, `hs_c`, `hs_m`, `hs_y` | Per-secondary hue rotation |
| Hue/chroma | `hc_enable`, `hc_r` | Hue-dependent chroma control |
| Creative white | `cwp`, `cwp_rng` | Creative white point: 0=D65/off, 1=D50/full warm |
| Display | `peak_luminance`, `grey_boost`, `pt_hdr` | Display target luminance and HDR purity blend |

`PreProcessConfig` (separate interface, merged into `GradingConfig`):

| Field | Type | Description |
|---|---|---|
| `exposure` | `number` | EV exposure shift |
| `wb_temp` | `number` | White balance temperature correction: warm(+) / cool(-) |
| `wb_tint` | `number` | White balance tint correction: magenta(+) / green(-) |
| `sharpen_amount` | `number` | Unsharp mask strength (0 = off) |

**Producer:** `configFromPreset()` in `web/src/gl/opendrt-params.ts` creates a full config from a `LookPreset` (`'default'`, `'colorful'`, `'umbra'`, `'base'`, `'flat'`). `configWithOverrides()` merges sparse `Partial<OpenDrtConfig>` and `Partial<PreProcessConfig>` overrides on top, returning a `GradingConfig`. `deriveHdrConfig()` creates an HDR variant for export.

**Consumers:**
- **WebGPU renderer** (`web/src/gl/renderer.ts`): `setOpenDrtMode()` writes all config fields into the GPU uniform buffer. `renderForExport()` accepts a config for export renders.
- **OutputCanvas**: calls `configFromPreset` + `configWithOverrides` + `computeTonescaleParams` on every grading change, then passes to `renderer.setOpenDrtMode()`.
- **GradingPanel** (`web/src/components/GradingPanel.tsx`): reads individual fields for slider values, writes overrides via `setFileOpenDrtOverride`.
- **Zustand store**: persists `Partial<OpenDrtConfig>` as `QueuedFile.openDrtOverrides`; serialized to IDB as `Record<string, number | boolean>`.

---

## TonescaleParams

Precomputed tonescale curve constants derived from `OpenDrtConfig`. Computed on the CPU, uploaded to the GPU as uniforms.

| Field | Type | Description |
|---|---|---|
| `ts_s` | `number` | Tonescale slope parameter |
| `ts_s1` | `number` | HDR-blended slope (interpolated between SDR and HDR purity) |
| `ts_m2` | `number` | Tonescale maximum (toe-decompressed) |
| `ts_dsc` | `number` | Display scale factor: `100 / peak_luminance` |
| `ts_x0` | `number` | Mid-grey input: `0.18 + tn_off` |

**Producer:** `computeTonescaleParams()` in `web/src/gl/opendrt-params.ts`.

**Consumers:** `renderer.setOpenDrtMode()` (written to uniform buffer offset `U_TS`). `OutputCanvas.applyOpenDrt()` computes and passes to the renderer. Export paths call it independently.

---

## Data Ownership Model

### Zustand (in-memory, reactive)

- `QueuedFile[]` -- the file queue with all UI-facing state
- `ProcessingResultMeta` -- lightweight result metadata (no pixel data)
- `Partial<OpenDrtConfig>` -- per-file grading overrides
- Global settings: `demosaicMethod`, `exportFormat`, `exportQuality`, `displayHdr`

### OPFS (Origin-Private File System, persistent)

- RAW file bytes (`writeRaw` / `readRaw`) -- the original camera file
- Thumbnails (`writeThumbnail` / `readThumbnail`) -- extracted JPEG preview

Note: HWC pixel buffers are no longer stored in OPFS. The pipeline uses
GPU-resident buffers during a session and re-processes files on demand after
session restore.

### IndexedDB (persistent, structured-clone safe)

- `PersistedFile` -- JSON-safe mirror of `QueuedFile` with `SerializableResultMeta` (Float32Arrays converted to number arrays)
- `AppSetting` -- key-value pairs for global settings

### Transient (GC'd after processFile)

- `RawImage`, `CroppedImage`, `PaddedImage`, `TileGrid`, `ChannelMasks`, `CfaInfo` -- all live only during a single `processFile()` invocation. Several are explicitly nulled to reduce peak memory.

---

## Antipatterns

**Holding `RawImage.data` and `PaddedImage.data` simultaneously.** Both are full-sensor-resolution buffers. The pipeline explicitly nulls intermediate references (`visible = null!`, `cfa = null`, `padded = null!`) to avoid doubling peak memory. New stages must follow this discipline.

**Storing `Float32Array` in IndexedDB.** Typed arrays survive structured clone but not all IDB implementations handle them reliably. The codebase uses `SerializableResultMeta` (with `number[]` fields) and the `serializeResultMeta`/`deserializeResultMeta` pair. Always go through these functions.

**Passing `ExportData` (with `hwc`) through Zustand.** The `hwc` buffer is 50-200 MB. It is transient (GPU-resident or re-generated on demand), not in state. Zustand holds only `ExportDataMeta` / `ProcessingResultMeta`.

**Assuming `xyzToCam` is non-null in `ExportDataMeta`.** After color correction is applied in `useProcessFile`, `xyzToCam` is set to `null` to signal "already baked in." Code that consumes `ExportDataMeta` must handle the null case.

**Forgetting `camToXyz` backward compatibility.** `SerializableResultMeta.exportData.camToXyz` is optional (`number[] | undefined`). `deserializeResultMeta` provides a 3x4 identity fallback for data persisted before this field was added. Do not assume it is always present in persisted data.

**Using `dy`/`dx` after padding.** Once `padToAlignment` has shifted the CFA data, downstream stages (tiling, channel masks, demosaic) operate at shift `(0, 0)`. Passing non-zero `dy`/`dx` to `runDemosaic` after padding would double-shift the pattern. The traditional demosaic path explicitly passes `(0, 0)`.

**Confusing display dimensions with texture dimensions.** `ExportDataMeta.width`/`height` are pre-rotation texture dimensions. `ProcessingResult.metadata.width`/`height` are post-rotation display dimensions (swapped for 90/270 orientations). The WebGPU canvas uses texture dimensions; layout/zoom uses display dimensions.

**Mutating `CfaInfo.pattern` elements.** The `pattern` field references module-level constant arrays (`XTRANS_PATTERN`, `BAYER_PATTERN`) which are `readonly`. Attempting mutation will fail at compile time due to the `readonly` modifier.

---

## Key Files

| File | Role |
|---|---|
| `web/src/pipeline/types.ts` | All pipeline interfaces: `RawImage`, `CroppedImage`, `PaddedImage`, `CfaInfo`, `TileGrid`, `ChannelMasks`, `ExportData`, `ExportDataMeta`, `ProcessingResult`, `ProcessingResultMeta`, `SerializableResultMeta`; type aliases `CfaType`, `DemosaicMethod`, `ModelSize`, `ExportFormat`, `LookPreset`; serialization functions |
| `web/src/store.ts` | `QueuedFile` interface, Zustand store actions that produce/consume `ProcessingResultMeta` |
| `web/src/gl/opendrt-params.ts` | `OpenDrtConfig`, `PreProcessConfig`, `GradingConfig`, `TonescaleParams` interfaces; preset factories; `configFromPreset`, `configWithOverrides`, `computeTonescaleParams` |
| `web/src/pipeline/preprocessor.ts` | Produces `CroppedImage`, `CfaInfo`, `PaddedImage`, `TileGrid`, `ChannelMasks` |
| `web/src/pipeline/postprocessor.ts` | `cropToHWC` removes padding; `buildColorMatrix`/`applyColorCorrection` consume `xyzToCam` |
| `web/src/pipeline/tile-blend-gpu.ts` | GPU-resident tile blend pipeline: `createGpuNNPipeline` for extract/accumulate/finalize |
| `web/src/pipeline/postprocess-gpu.ts` | GPU post-demosaic: WB, highlight recovery, color correction, DR gain |
| `web/src/hooks/useProcessFile.ts` | Orchestrates the full pipeline; produces `ProcessingResultMeta`; sole consumer of `RawImage` |
| `web/src/hooks/useInit.ts` | Session restore: deserializes `SerializableResultMeta` to `ProcessingResultMeta`, reconstructs `QueuedFile` from `PersistedFile` |
| `web/src/components/OutputCanvas.tsx` | Consumes `ProcessingResultMeta` for WebGPU display; reads HWC from OPFS |
| `web/src/lib/idb-storage.ts` | `PersistedFile` interface (IDB schema); consumes `SerializableResultMeta` |
| `web/src/lib/opfs-storage.ts` | OPFS operations for RAW file and thumbnail storage |
| `web/src/lib/hwc-handoff.ts` | GPU buffer handoff between processing pipeline and renderer |
