---
title: "End-to-End Web RAW Processing Pipeline Flow"
tags: [pipeline, flow, web, useProcessFile, orchestration, raf, demosaic, inference, rendering]
scope: web
generated: 2026-03-22
commit: 347c8dd
---

## Context

This document traces the **end-to-end web RAW processing pipeline** from the moment a user drops a RAF file to the final rendered preview on screen. It is a flow document, not a concept document. Its purpose is to show an agent the exact ordered sequence of module handoffs so they can locate where any given operation fits in the full chain. The single source of truth for ordering is the `useProcessFile` hook.

For concept-level deep dives on individual stages, follow the cross-reference links at each step.

## Initialization (useInit)

**Module:** `web/src/hooks/useInit.ts` -- `useInit()`

Before any file can be processed, `useInit` runs once on app mount inside a `useEffect`. It launches several subsystems in parallel via `Promise.all`:

| Task | Function | Module | What It Does |
|---|---|---|---|
| WASM decoder | `initWasm()` | `pipeline/raf-decoder.ts` | Loads and instantiates the Rust rawloader WASM module |
| ONNX models | `initModels()` | `pipeline/inference.ts` | Loads `xtrans.onnx` and `bayer.onnx` via ONNX Runtime (tries WebGPU EP first, falls back to WASM EP) |
| GPU demosaic | `initDemosaicGpuSafe()` | `pipeline/demosaic.ts` | Initializes WebGPU compute pipelines for bilinear/DHT demosaicing |
| Session restore | `getAllFiles()` | `lib/idb-storage.ts` | Reads persisted file entries from IndexedDB |
| Settings restore | `getSetting()` (x4) | `lib/idb-storage.ts` | Restores `demosaicMethod`, `exportFormat`, `exportQuality`, `selectedFileId` |

After parallel init completes, `useInit` performs sequential post-processing:

1. **Restore Zustand state** -- Converts persisted files to `QueuedFile` objects (loading thumbnails from OPFS), then calls `useAppStore.getState().restoreFromDb()` with the file list and settings.
2. **Share WebGPU device** -- `getInferenceDevice()` retrieves ORT's WebGPU device; if available, `setSharedDevice(ortDevice)` injects it into the renderer's module-level cache so that inference, GPU postprocessing, and display rendering all share one `GPUDevice` for zero-copy buffer transfers.
3. **Probe HDR display** -- `probeHdrDisplay()` queries display headroom via the Screen API / Window Management API. Updates store with `setDisplayHdr()` and conditionally sets `setHdrPermissionNeeded()`.
4. **Mark initialized** -- `setInitialized(backend)` flips the app into the ready state.
5. **Lens matching** -- For restored files with lens metadata but no profile, asynchronously attempts lens matching via `matchLens()`.
6. **Orphan cleanup** -- Fire-and-forget: removes OPFS entries (raw + thumbnails) that have no matching IndexedDB record.
7. **Persistent storage** -- Best-effort `navigator.storage.persist()` call.

Cross-reference: [session-persistence](session-persistence.md), [zustand-state-management](zustand-state-management.md)

## Pipeline Steps

All steps below execute sequentially inside `useProcessFile` > `processFile(fileId)`. The hook holds a `lockRef` to prevent concurrent processing.

**Module:** `web/src/hooks/useProcessFile.ts` -- `processFile()`

---

### Step 1: Decode RAW

**Function:** `decodeRaw(arrayBuffer)`
**Module:** `pipeline/raf-decoder.ts`
**Input:** `ArrayBuffer` -- the raw file bytes (from `File.arrayBuffer()` for fresh drops, or `readRaw()` from OPFS for restored sessions)
**Output:** `RawImage` -- a struct containing:
- `data: Uint16Array` -- single-channel mosaic pixel data
- `width, height: number` -- full sensor dimensions
- `wbCoeffs: Float32Array` -- camera white balance [R, G, B]
- `blackLevels, whiteLevels: Uint16Array` -- per-channel pedestal and saturation
- `xyzToCam, camToXyz: Float32Array` -- color matrices (3x3 and 3x4 row-major)
- `cfaStr: string`, `cfaWidth: number` -- CFA pattern descriptor
- `crops: Uint16Array` -- [top, right, bottom, left] active area margins
- `orientation: string` -- EXIF orientation (`"Normal"`, `"Rotate90"`, etc.)
- `drGain: number` -- Fuji dynamic range compensation multiplier
- `make, model: string`, `exposureBias: number` -- metadata

**Side effects:** The `ArrayBuffer` reference is nulled after decoding to free memory.

Cross-reference: [raw-decoding-wasm](raw-decoding-wasm.md)

---

### Step 2: Crop to Visible Area

**Function:** `cropToVisible(raw.data, raw.width, raw.height, raw.crops)`
**Module:** `pipeline/preprocessor.ts`
**Input:** `Uint16Array` (full sensor), dimensions, crop margins
**Output:** `CroppedImage { data: Uint16Array, width: number, height: number }` -- the active pixel region with masked borders removed

Cross-reference: [cfa-preprocessing-web](cfa-preprocessing-web.md)

---

### Step 3: White-Point Calibration

**Function:** `calibrateWhiteLevels(visible.data, visWidth, visHeight, raw.whiteLevels)`
**Module:** `pipeline/preprocessor.ts`
**Input:** `Uint16Array` (cropped visible data), dimensions, metadata white levels
**Output:** `Uint16Array` -- calibrated per-channel white levels derived from actual sensor saturation, which may differ from metadata values

---

### Step 4: Normalize to Float

**Function:** `normalizeRawCfa(visible.data, visWidth, visHeight, raw.blackLevels, whiteLevels)`
**Module:** `pipeline/preprocessor.ts`
**Input:** `Uint16Array` (cropped), black levels, calibrated white levels (from step 3)
**Output:** `Float32Array` of length `visWidth * visHeight` -- each pixel normalized as `(raw - black) / (white - black)`, producing values nominally in [0, 1]

Cross-reference: [cfa-preprocessing-web](cfa-preprocessing-web.md)

---

### Step 5: Compute White Balance Coefficients

**Inline in `processFile`** (no separate function)
**Input:** `raw.wbCoeffs: Float32Array` [R, G, B]
**Output:** `wb: Float32Array` [R/G, 1.0, B/G] -- coefficients normalized so green = 1.0

Note: White balance is **not** applied to the CFA before demosaicing. The neural network is trained on raw (un-WB'd) CFA data. WB is applied post-demosaic during GPU postprocessing.

---

### Step 6: Detect CFA Pattern and Shift

**Function:** `findPatternShift(raw.cfaStr, raw.cfaWidth, raw.crops)`
**Module:** `pipeline/preprocessor.ts`
**Input:** CFA string descriptor, CFA width, crop offsets
**Output:** `CfaInfo { cfaType: CfaType, pattern: number[][], period: number, dy: number, dx: number }` -- identifies X-Trans (period 6) or Bayer (period 2) and the sub-pixel shift of the visible area relative to the canonical pattern

The function applies crop offsets to the raw CFA pattern, then matches against the canonical `XTRANS_PATTERN` or `BAYER_PATTERN` from `pipeline/constants.ts` to find the rotation/shift (dy, dx).

Cross-reference: [cfa-alignment-detection](cfa-alignment-detection.md), [cfa-pattern-system](cfa-pattern-system.md)

---

### Step 7: Compute Channel Clips

**Function:** `channelClips()`
**Module:** `pipeline/preprocessor.ts`
**Output:** `[number, number, number]` -- per-channel normalized clip thresholds (all 0.987 after per-CFA-position normalization). WB-scaled clip thresholds (`clipsWb`) are computed inline by multiplying by WB coefficients, for use in GPU postprocessing.

---

### Step 8: Pad for CFA Alignment

**Function:** `padToAlignment(cfa, visWidth, visHeight, dy, dx)`
**Module:** `pipeline/preprocessor.ts`
**Input:** `Float32Array` (normalized CFA), pattern shift (dy, dx)
**Output:** `PaddedImage { data: Float32Array, width: number, height: number, padTop: number, padLeft: number }` -- image padded by (dy, dx) pixels using mirror reflection so that the CFA pattern starts at canonical (0,0) alignment. If dy=dx=0, returns the input unchanged (no copy).

Cross-reference: [cfa-alignment-detection](cfa-alignment-detection.md)

---

### Step 9: Demosaic (Branching Path)

The demosaic method is read from the Zustand store (`useAppStore.getState().demosaicMethod`). The pipeline branches into two paths:

#### Path A: Neural Network (`method === 'neural-net'`) -- GPU-Resident

The entire NN path stays on the GPU. No CPU-side pixel buffers are created after tile generation. The key modules are `pipeline/tile-blend-gpu.ts` (tile extraction, blend accumulation, crop/finalize on GPU) and `pipeline/postprocess-gpu.ts` (WB, highlight recovery, color correction on GPU).

**Step 9a: Generate Tiles**
**Function:** `generateTiles(cfaW, cfaH, PATCH_SIZE, OVERLAP)`
**Module:** `pipeline/preprocessor.ts`
**Input:** Padded CFA dimensions, patch size, overlap
**Output:** `TileGrid { tiles: Array<{x, y}>, hPad: number, wPad: number }` -- tile coordinate list and padded dimensions

**Step 9b: Make Channel Masks**
**Function:** `makeChannelMasks(PATCH_SIZE, pattern, period)`
**Module:** `pipeline/preprocessor.ts`
**Output:** `ChannelMasks { r: Float32Array, g: Float32Array, b: Float32Array }` -- binary masks of size `PATCH_SIZE^2` indicating R/G/B photosite positions

**Step 9c: Create GPU NN Pipeline**
**Function:** `createGpuNNPipeline(device, cfaData, cfaW, cfaH, masks, clipNorm, tiles, hPad, wPad, PATCH_SIZE, OVERLAP, padTop, padLeft, visHeight, visWidth, TILE_BATCH)`
**Module:** `pipeline/tile-blend-gpu.ts`
**Input:** Shared `GPUDevice`, CFA data (uploaded to GPU), channel masks, clip thresholds, tile coordinates, dimensions, batch size
**Output:** `GpuNNPipeline` object with methods `extractBatch`, `accumulateBatch`, `finalize`

The pipeline uploads the full CFA to a GPU storage buffer once. All subsequent tile extraction and blend accumulation happens on-GPU.

**Step 9d: Batched Tile Loop**
Tiles are processed in batches of `TILE_BATCH` (32):

1. `gpu.extractBatch(batchStart, count)` -- GPU compute shader extracts `count` tiles from the CFA storage buffer into a 5-channel NCHW `GPUBuffer` (CFA + R/G/B masks, matching the model's input format)
2. `runBatchGpu(cfaType, inputBuf, count, PATCH_SIZE)` -- ONNX Runtime WebGPU inference, `GPUBuffer` in / `GPUBuffer` out (no CPU readback). Input shape `[count, 4 or 5, PATCH_SIZE, PATCH_SIZE]`, output `Float32Array(count * 3 * PATCH_SIZE^2)` in CHW
3. `gpu.accumulateBatch(inferBuf, batchStart, count)` -- GPU compute shader accumulates inference output into the blend buffer using triangular overlap-add weights

**Step 9e: Finalize + Crop on GPU**
`gpu.finalize()` -- GPU compute shader divides by weight sums, crops to visible dimensions, and reorders CHW to HWC RGBA (with clip mask in alpha). Returns a `GPUBuffer` in RGBA32F format with 256-byte row alignment, ready for GPU postprocessing.

Cross-reference: [tiled-inference-blending](tiled-inference-blending.md), [unet-model-architecture](unet-model-architecture.md)

#### Path B: Traditional Demosaic (`method !== 'neural-net'`)

**Function:** `runDemosaic(padded.data, padded.width, padded.height, 0, 0, algorithm, flatCfa, period)`
**Module:** `pipeline/demosaic.ts`
**Input:** `Float32Array` (aligned CFA), dimensions, dy=0, dx=0 (post-alignment), algorithm name, flattened CFA pattern, period
**Output:** `Float32Array(3 * width * height)` in CHW layout

The function attempts WebGPU compute first (for `bilinear` and `dht` via `runBilinearGpu` / `runDhtGpu`), falling back to a WASM worker pool (`DemosaicPool`) for all algorithms.

**Then:** `cropToHWC(blended, hPad, wPad, padTop, padLeft, visHeight, visWidth)` reorders CHW to HWC and crops to visible dimensions. The CPU-side `Float32Array` is passed to `gpuPostprocess` for the same GPU postprocessing as the NN path.

Cross-reference: [demosaic-algorithm-dispatch](demosaic-algorithm-dispatch.md), [webgpu-compute-demosaicing](webgpu-compute-demosaicing.md), [worker-pool-parallelism](worker-pool-parallelism.md)

---

### Step 10: GPU Postprocessing

**Function:** `gpuPostprocess(device, hwcBuf, visWidth, visHeight, wb, clipsWb, ccMatrix, drGain)`
**Module:** `pipeline/postprocess-gpu.ts`
**Input:** `GPUDevice`, HWC pixel data (either a `GPUBuffer` from NN path or a `Float32Array` from traditional path), dimensions, WB coefficients, WB-scaled clip thresholds, camera-to-sRGB color correction matrix, Fuji DR gain
**Output:** `PostprocessResult { buffer: GPUBuffer, bytesPerRow: number }` -- RGBA32F GPU buffer with 256-byte row alignment

The GPU postprocessing pipeline runs in a single command buffer:

1. **White balance** -- per-pixel multiply by WB coefficients (R, G, B)
2. **Highlight recovery** -- opposed-channel inpainting on clipped pixels (pass 1 only, no segmentation), using reference averages and chrominance correction
3. **Color correction** -- camera-to-sRGB 3x3 matrix multiplication (when `xyzToCam` is present)
4. **DR gain** -- Fuji dynamic range compensation multiply
5. **Finalize** -- pack to RGBA with clip mask in alpha channel

This replaces the previous CPU-side `applyColorCorrection` + DR gain + `writeHwc` flow.

---

### Step 11: Compute Display Dimensions

**Inline in `processFile`**
Applies EXIF orientation to determine final display width/height. For `Rotate90` and `Rotate270`, width and height are swapped. The actual pixel data is not rotated; orientation is handled at render time via CSS transforms.

---

### Step 12: Estimate Color Temperature

**Function:** `estimateColorTemperature(wb, raw.camToXyz)`
**Module:** `pipeline/color-temperature.ts`
**Input:** G-normalized WB coefficients `[R/G, 1.0, B/G]` (from Step 5), camera-to-XYZ matrix (3x4 row-major, stride 4)
**Output:** `{ temp: number, tint: number }` -- correlated color temperature (rounded to 50K) and tint (signed green/magenta deviation)

The function derives the illuminant's chromaticity from `1/wb` transformed through `camToXyz`, uses McCamy's approximation for CCT, and computes Planckian locus distance in CIE 1960 UCS for tint.

Cross-reference: [color-temperature-estimation](color-temperature-estimation.md)

---

### Step 13: GPU Buffer Handoff and Store Results

**Function:** `setGpuResult(fileId, gpuResult)`
**Module:** `lib/hwc-handoff.ts`
**Input:** File ID, `PostprocessResult { buffer: GPUBuffer, bytesPerRow: number }`
**Output:** Stores the GPU buffer in a single-slot handoff for immediate consumption by `OutputCanvas`

**Store update:** `setFileResult(fileId, resultMeta, method)` writes the following to the Zustand store:
- `ProcessingResultMeta.exportData` -- width, height, orientation, WB, color matrices (xyzToCam set to null since CC is already applied on GPU)
- `ProcessingResultMeta.metadata` -- make, model, dimensions, tile count, inference time, backend, exposure bias, lens info, color temperature, tint, model size

**Cleanup:** `destroyDemosaicPool()` is called in the `finally` block to terminate WASM worker threads.

Cross-reference: [opfs-pixel-storage](opfs-pixel-storage.md), [zustand-state-management](zustand-state-management.md)

## Rendering: GPU Buffer to Screen

**Components:** `OutputCanvas` (`components/OutputCanvas.tsx`) and `HdrRenderer` (`gl/renderer.ts`)

Once `setFileResult` marks a file as `done`, the `OutputCanvas` component mounts for the selected file. The rendering path:

1. **Create renderer** -- `HdrRenderer.create(canvas, { hdr, headroom })` obtains the shared `GPUDevice` (injected by `setSharedDevice` during init, or created lazily). Configures a `webgpu` canvas context with either `rgba16float` (HDR, display-p3 color space, extended tone mapping) or the preferred canvas format (SDR, srgb color space). Creates display and export render pipelines, histogram compute/reduce/viz pipelines, and compiles the OpenDRT WGSL shader.

2. **Claim GPU buffer** -- `takeGpuResult(fileId)` retrieves the `GPUBuffer` from the single-slot handoff (set by `useProcessFile` in step 13).

3. **Upload texture (zero-copy)** -- `renderer.uploadImageFromBuffer(buffer, width, height, bytesPerRow)` copies the GPU buffer directly to an `rgba32float` `GPUTexture` via `copyBufferToTexture` (no CPU readback). The buffer is destroyed after the copy. Bind groups are rebuilt to reference the new texture.

4. **Configure tone mapping** -- `applyOpenDrt(renderer, preset, overrides, headroom)` computes OpenDRT parameters from the look preset (e.g., `"default"`, `"base"`) merged with per-file overrides, then calls `renderer.setOpenDrtMode(ts, cfg)` to write tonescale and grading uniforms.

5. **Render** -- `renderer.render()` uploads the uniform buffer, creates a render pass targeting the canvas swap chain texture, draws a full-screen triangle (3 vertices, no vertex buffer) through the OpenDRT fragment shader, and optionally appends histogram compute + reduce + viz passes in the same submission. The shader performs: exposure adjustment, white balance temperature/tint shift, sharpening (Laplacian), sRGB-to-P3 gamut conversion, OpenDRT tone mapping (including perceptual tonescale, chroma compression, hue shift, hue contrast, creative white, purity compression), P3-to-display gamut conversion, clip overlay, and HDR headroom scaling.

6. **EXIF orientation** -- Handled purely via CSS `transform` on the canvas element (rotate 90/180/270 degrees). No pixel data is rotated.

7. **Pan/zoom** -- The `usePanZoom` hook applies CSS transforms for interactive pan and zoom. At zoom > 1x, `image-rendering: pixelated` is set for sharp pixel inspection.

When look preset or per-file overrides change, only the uniform update + draw call is re-executed (no texture re-upload).

Cross-reference: [webgpu-renderer](webgpu-renderer.md), [opendrt-tone-mapping](opendrt-tone-mapping.md), [hdr-display-export](hdr-display-export.md)

## Export: Dual-Render Path

**Hook:** `useExport` (`hooks/useExport.ts`)
**Encoder:** `encodeImage` (`pipeline/encoder.ts`)

The export path re-uses the already-uploaded `HdrRenderer` texture but renders through an offscreen pipeline with different tone mapping parameters and gamut targets.

### Export Flow

1. **Read state** -- Retrieves the active `HdrRenderer` from `state.rendererRef`, the file's `exportData`, `lookPreset`, and `openDrtOverrides` from the Zustand store.

2. **Compute SDR config** -- `configFromPreset(lookPreset)` + `configWithOverrides(base, overrides, preProcessOverrides)` + `computeTonescaleParams(sdrConfig)`.

3. **Branch by format:**
   - **JPEG / TIFF (SDR):** Single render pass -- `renderer.renderForExport(sdrConfig, sdrTs, 'rec709')` returns `Float32Array` in HWC (width x height x 3) in Rec.709 primaries.
   - **AVIF (HDR):** Single render pass with HDR config -- `deriveHdrConfig(sdrConfig, 1000)` scales peak luminance to 1000 nits, renders in Rec.2020 gamut.
   - **JPEG-HDR:** Dual render -- SDR pass in Rec.709 + HDR pass in Rec.2020. Both `Float32Array` buffers are passed to the encoder.

4. **`renderForExport` internals** -- Sets `exportMode=1` and `hdrDisplay=0` uniforms, switches the P3-to-display matrix to `P3D65_TO_REC709` or `P3D65_TO_REC2020`, renders to an offscreen `GPUTexture` (`rgba32float` or `rgba16float`), copies texture to a readback `GPUBuffer`, maps it async, converts RGBA to HWC RGB `Float32Array`, then restores display uniforms and re-renders the preview.

5. **Encode** -- `encodeImage(data, hdrData, width, height, orientation, format, quality, peakLuminance)` delegates to a dedicated Web Worker (`encoder-worker.ts`) via `postMessage` with transferable buffers. The worker handles JPEG, TIFF, AVIF encoding and EXIF orientation embedding.

6. **Download** -- `triggerDownload(blob, filename)` creates a temporary object URL and programmatically clicks an anchor element.

Cross-reference: [hdr-display-export](hdr-display-export.md), [opendrt-tone-mapping](opendrt-tone-mapping.md)

## Data Flow Diagram

```
 User drops RAF file
        |
        v
 +-----------------+    ArrayBuffer
 | decodeRaw       |--------------------> RawImage
 | (raf-decoder)   |                      { Uint16Array, metadata }
 +-----------------+
        |
        v
 +-----------------+    Uint16Array (full sensor)
 | cropToVisible   |--------------------> CroppedImage
 | (preprocessor)  |                      { Uint16Array, visW, visH }
 +-----------------+
        |
        v
 | calibrateWhiteLevels |--------------> calibrated whiteLevels
        |
        v
 +-----------------+    Uint16Array (cropped)
 | normalizeRawCfa |--------------------> Float32Array [0..1]
 | (preprocessor)  |                      (single-channel CFA, no WB)
 +-----------------+
        |
        v
 +-----------------+
 | findPatternShift|--------------------> CfaInfo { cfaType, pattern,
 | (preprocessor)  |                                period, dy, dx }
 +-----------------+
        |
        v
 +-----------------+    Float32Array (CFA)
 | padToAlignment  |--------------------> PaddedImage
 | (preprocessor)  |                      { Float32Array, padW, padH }
 +-----------------+
        |
        v
 +=====================================================+
 || Neural Net Path (GPU-resident)   | Traditional    ||
 ||----------------------------------|----------------||
 || createGpuNNPipeline              | runDemosaic    ||
 ||   (uploads CFA to GPU once)      |  (GPU or WASM ||
 || for each batch of TILE_BATCH:    |   worker pool) ||
 ||   gpu.extractBatch (GPU)         |                ||
 ||   runBatchGpu (ORT, GPU→GPU)     | cropToHWC      ||
 ||   gpu.accumulateBatch (GPU)      |  (CPU)         ||
 || gpu.finalize (GPU→GPUBuffer)     |                ||
 +=====================================================+
        |                        |
        +--- GPUBuffer or -------+
        |    Float32Array (HWC)
        v
 +----------------------------+
 | gpuPostprocess             |    GPU compute pipeline:
 | (postprocess-gpu.ts)       |    WB → HL recovery →
 |                            |    CC → DR gain → RGBA
 +----------------------------+
        |
        v  PostprocessResult { buffer: GPUBuffer, bytesPerRow }
        |
 +---------------------------+
 | estimateColorTemperature  |----------> { temp, tint }
 | (color-temperature)       |
 +---------------------------+
        |
        v
 +---------------------+    GPUBuffer (RGBA32F)
 | setGpuResult        |----> Single-slot GPU handoff
 | (hwc-handoff.ts)    |
 +---------------------+
        |
        v
 +-----------------+
 | setFileResult   |--------------------> Zustand store updated
 | (store)         |                      (status = 'done')
 +-----------------+

 === RENDERING ===

 +-----------------+    GPUBuffer from handoff
 | OutputCanvas    |
 | takeGpuResult() |----> HdrRenderer.uploadImageFromBuffer()
 |                 |        |  (zero-copy GPU buffer → texture)
 |                 |        v  rgba32float GPUTexture
 |                 |      HdrRenderer.render()
 |                 |        |  (OpenDRT fragment shader
 |                 |        |   + histogram compute/viz)
 |                 |        v
 |                 |      Canvas swap chain
 |                 |        + CSS rotation + pan/zoom
 +-----------------+

 === EXPORT ===

 +-----------------+    HdrRenderer (existing texture)
 | useExport       |
 |  renderForExport|----> Offscreen GPUTexture
 |  (SDR / HDR)    |        |
 |                 |        v  Float32Array (HWC, Rec.709 or Rec.2020)
 |  encodeImage    |----> encoder-worker.ts
 |                 |        |
 |                 |        v  Blob (JPEG / TIFF / AVIF)
 |  triggerDownload|----> Browser download
 +-----------------+
```
