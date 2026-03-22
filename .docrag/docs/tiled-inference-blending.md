---
title: Tiled Neural Network Inference with Overlap Blending
tags: [web, inference, onnx, tiling, blending]
scope: web/src/pipeline/inference.ts, web/src/pipeline/postprocessor.ts, web/src/pipeline/preprocessor.ts, web/src/pipeline/constants.ts, web/src/pipeline/tile-blend-gpu.ts, web/src/hooks/useProcessFile.ts
generated: 2026-03-22
commit: 347c8dd
depends_on:
  - demosaic-algorithm-dispatch
  - onnx-export
---

# Tiled Neural Network Inference with Overlap Blending

Tiled inference, overlap blending, ONNX Runtime Web, WebGPU execution provider,
WASM fallback, accumulate/finalize pattern, weighted tile merging, linear ramp
window, patch stride, seam-free reconstruction, GPU-resident pipeline,
batched inference, tile-blend-gpu, GpuNNPipeline, model size switching.

## Context

Camera sensor images routinely exceed 6000x4000 pixels (24 MP). Running a neural
network over the full image in a single pass is infeasible in a browser: GPU
memory is limited, shader dispatch has per-tile overhead, and ONNX Runtime Web
allocates intermediate tensors proportional to spatial dimensions. Even on native
hardware, memory pressure grows quadratically with image side length.

Tiled inference solves this by splitting the image into small fixed-size patches,
running the model on each patch independently, and reassembling the results.
Naive stitching produces visible seams at tile boundaries because the network
lacks context beyond the tile edge and may produce slightly different predictions
for the same pixel when it appears in adjacent tiles. Overlap blending eliminates
these seams by running adjacent tiles with shared border regions and merging the
overlapping predictions through a smooth weighting function that ramps to zero at
tile edges.

## Pattern / Approach

### ONNX Model Loading: WebGPU with WASM Fallback

Model loading is handled by `initModels()` in `inference.ts`. It is called once
during application startup by `useInit.ts`, in parallel with WASM decoder
initialization and GPU demosaic setup. Models are loaded from a `models.json`
manifest: for the selected model size (S/M/L), one model is loaded per CFA type
(X-Trans and Bayer). The CFA type determines which session is selected at
inference time. See [onnx-export](onnx-export.md) for model export details and
[demosaic-algorithm-dispatch](demosaic-algorithm-dispatch.md) for how the
neural-net path is selected versus traditional demosaicing.

`createSession()` implements a three-tier backend fallback chain:

1. **WebGPU** -- preferred. Offers GPU-accelerated inference with lowest latency.
   When WebGPU is active, the session is created with
   `preferredOutputLocation: 'gpu-buffer'` so inference outputs stay on GPU.
2. **Multi-threaded WASM** -- fallback when WebGPU is unavailable (e.g., Firefox
   as of early 2026, or devices without WebGPU support). Uses
   `navigator.hardwareConcurrency` threads.
3. **Single-threaded WASM** -- last resort when multi-threaded WASM fails (some
   environments block `SharedArrayBuffer`). Sets `ort.env.wasm.numThreads = 1`
   before retrying.

The first session that loads successfully determines the backend, which is stored
in a module-level variable and exposed via `getBackend()`. All subsequent
sessions reuse the same execution provider. The backend string is surfaced in the
UI for diagnostic purposes.

When the WebGPU backend is active, `createSession()` also captures ORT's
internal `GPUDevice` reference (via `ort.env.webgpu.device`) and stores it in a
module-level variable. This device is exposed to the rest of the pipeline via
`getInferenceDevice()`, enabling GPU compute shaders in `tile-blend-gpu.ts` and
`postprocess-gpu.ts` to share the same device and exchange `GPUBuffer` objects
with ONNX Runtime without CPU readback.

### Model Manifest and Size Switching

Models are no longer loaded from hardcoded filenames. `initModels()` fetches a
`models.json` manifest from the checkpoints directory that maps model keys (e.g.,
`xtrans_w16_hl`, `bayer_w32_base`) to metadata including epoch, PSNR, parameter
count, and filename. Three model sizes are supported:

| Size | Base Width | Manifest Key Pattern |
|------|-----------|---------------------|
| S    | 16        | `{cfa}_w16_{variant}` |
| M    | 32        | `{cfa}_w32_{variant}` |
| L    | 64        | `{cfa}_w64_{variant}` |

The `ModelSize` type (`'S' | 'M' | 'L'`) is defined in `types.ts`.
`resolveModelKey()` prefers the `_hl` (highlight-head) variant over `_base` when
both exist in the manifest. Sessions are cached in a `Map<string, ModelEntry>`;
`switchModelSize(size)` loads the new sessions on demand (reusing previously
loaded ones) and updates the `active` map so that subsequent inference calls use
the new size. `getAvailableSizes(cfaType)` returns which sizes have entries in
the manifest for a given CFA type.

### Tile Parameters

Defined in `web/src/pipeline/constants.ts`:

```
PATCH_SIZE = 288
OVERLAP    = 24
TILE_BATCH = 32
```

The stride (step between tile origins) is `PATCH_SIZE - OVERLAP = 264`.
`TILE_BATCH` controls how many tiles are processed per GPU batch in the
GPU-resident pipeline (see below).

These values are shared between ONNX export (which bakes a fixed
`(1, 5, 288, 288)` input shape into the model) and browser-side inference.
Changing `PATCH_SIZE` requires re-exporting the ONNX model.

### Tile Grid Generation

`generateTiles()` in `preprocessor.ts` produces the tile grid. It takes the
image dimensions (from the CFA already padded by `padToAlignment()`) and
computes a padded canvas size that tiles evenly:

```
stride = patchSize - overlap              // 264
hPad   = ceil((height - overlap) / stride) * stride + patchSize
wPad   = ceil((width  - overlap) / stride) * stride + patchSize
```

Tile positions are emitted as `{ x, y }` pairs in raster-scan order
(top-to-bottom, left-to-right), along with the padded dimensions `hPad` and
`wPad`. The function no longer copies CFA data into a padded buffer; pixels
beyond the original image extent are handled as zero reads by the GPU extract
shader or by `fillBatchCfa()` on the CPU path.

### Per-Tile 5-Channel Input

Each tile is laid out in NCHW order (channel-major) with 5 channels:

| Channel | Content                                                     |
|---------|-------------------------------------------------------------|
| 0       | CFA sensor values (raw, normalized, not yet white-balanced) |
| 1       | Binary mask: 1 where the CFA position is red, 0 elsewhere  |
| 2       | Binary mask: 1 where the CFA position is green, 0 elsewhere|
| 3       | Binary mask: 1 where the CFA position is blue, 0 elsewhere |
| 4       | Clip ratio: 0 below 50% of clip level, ramps 0-1 from 50%-100% |

The clip ratio channel tells the network how close each pixel is to saturation,
enabling learned highlight handling. The clip ratio is computed as
`max(min(val / clipLevel, 1) * 2 - 1, 0)`, where `clipLevel` is the per-channel
clip threshold from `channelClips()`.

`makeChannelMasks()` generates the three binary masks once per image (they
depend on the CFA pattern and period, not on tile position, because the CFA data
has already been shifted to canonical alignment).

On the GPU path, the extract shader in `tile-blend-gpu.ts` fills all 5 channels
directly from the CFA buffer and precomputed masks. On the CPU fallback path,
`prefillBatchMasks()` pre-fills channels 1-3 into a reusable batch buffer, and
`fillBatchCfa()` fills channel 0 (CFA values) and channel 4 (clip ratio) per
batch of tiles.

### Batched Inference: runBatch() and runBatchGpu()

`inference.ts` provides two inference entry points, both operating on batches
of tiles rather than single tiles:

**`runBatch(cfaType, batchInput, batchSize, patchSize)`** -- CPU-accessible
path. Wraps the `Float32Array` as an `ort.Tensor` with shape
`[batchSize, 5, patchSize, patchSize]`, calls `session.run()`, and returns the
output as a `Float32Array`. Used as a fallback when the WebGPU backend is not
available.

**`runBatchGpu(cfaType, inputBuffer, batchSize, patchSize)`** -- GPU-resident
path. Takes a `GPUBuffer` (created on ORT's shared device) and wraps it via
`ort.Tensor.fromGpuBuffer()` with shape `[batchSize, 5, patchSize, patchSize]`.
The output tensor's `gpuBuffer` is returned directly, along with a `dispose()`
callback to release ORT's reference. No CPU readback occurs -- the output buffer
is passed directly to the accumulate step in `tile-blend-gpu.ts`.

Both paths select the active `InferenceSession` for the given `CfaType` from the
`active` map (set by `initModels()` or `switchModelSize()`). The output tensor
is `[batchSize, 3, patchSize, patchSize]` -- three channels (linear RGB) at the
same spatial resolution as the input.

### GPU-Resident Pipeline: GpuNNPipeline (tile-blend-gpu.ts)

The primary inference path keeps all data on the GPU. `createGpuNNPipeline()` in
`tile-blend-gpu.ts` returns a `GpuNNPipeline` object with four operations:

1. **`extractBatch(startIdx, count)`** -- A WGSL compute shader tiles the CFA
   image into 5-channel NCHW format on the GPU. It reads from the uploaded CFA
   buffer and precomputed channel masks, computes the clip ratio per pixel, and
   writes to a `GPUBuffer` with `STORAGE | COPY_SRC` usage (required by
   `ort.Tensor.fromGpuBuffer`). Dispatched as
   `(ceil(ps/16), ceil(ps/16), count)`.

2. **`accumulateBatch(inferOut, startIdx, count)`** -- Scatters each tile's
   3-channel inference output into the padded blend buffer using the same linear
   ramp weight grid as the CPU path. Each tile is dispatched as a separate
   compute pass within a single command buffer (no atomics needed because
   dispatches are serialized).

3. **`finalize()`** -- Two compute passes in sequence: (a) divide each pixel's
   accumulated RGB by its weight sum (same `1e-8` epsilon guard as CPU), and
   (b) crop the padded buffer to original dimensions, outputting HWC layout. The
   cropped `GPUBuffer` is returned directly for consumption by
   `gpuPostprocess()`. All intermediate buffers are destroyed.

4. **`destroy()`** -- Releases all GPU resources if `finalize()` was not called
   (error cleanup).

The pipeline uses ORT's `GPUDevice` (obtained via `getInferenceDevice()`) so
that the extract output buffer is directly usable by `runBatchGpu()` and the
inference output buffer is directly usable by `accumulateBatch()` -- no
CPU-GPU transfers occur during the tile loop.

### createTileBlender: CPU Fallback with Linear Ramp Window

`createTileBlender()` in `postprocessor.ts` implements the CPU fallback for
weighted overlap blending, used when WebGPU is not available. It allocates two
buffers over the full padded canvas:

- `output`: `Float32Array(3 * hPad * wPad)` -- accumulated weighted RGB values
  in HWC interleaved layout (3 floats per pixel).
- `weights`: `Float32Array(hPad * wPad)` -- accumulated blending weights per
  pixel.

**Window function.** A 1D weight array `w1d` of length `patchSize` is
constructed:

- The center region (indices `overlap` through `patchSize - 1 - overlap`) has
  weight 1.0.
- The overlap region on each side ramps linearly from 0 to 1:
  `w1d[i] = i / overlap` for `i` in `[0, overlap)`, and symmetrically at the
  trailing end.

The 2D weight for pixel `(px, py)` within a tile is the separable product
`w1d[py] * w1d[px]`. This is a **linear ramp** (triangular window), not a cosine
or Hann window. The ramp reaches exactly 0 at the tile boundary and 1 at the
inner edge of the overlap zone. When two adjacent tiles overlap, their linear
ramps sum to 1 in the overlap region, producing a smooth crossfade.

**`accumulate(tile, tx, ty)`** iterates over every pixel in the tile, computes
the separable weight `w`, and adds `tile[c][py][px] * w` to the corresponding
position in the output buffer for each of the 3 color channels. It also adds `w`
to the weight buffer.

**`finalize()`** divides each output pixel by its accumulated weight (guarded by
a `1e-8` epsilon to avoid division by zero). The result is an HWC `Float32Array`
covering the full padded canvas. Note: `createTileBlender` is currently not
called by `useProcessFile.ts` (the GPU-resident pipeline handles the neural-net
path entirely). It exists as potential CPU fallback code.

### Orchestration in useProcessFile.ts

The neural-net path in `useProcessFile.ts` uses the fully GPU-resident pipeline:

```
const tileGrid = generateTiles(cfaW, cfaH, PATCH_SIZE, OVERLAP);
const masks    = makeChannelMasks(PATCH_SIZE, pattern, period);
const device   = getInferenceDevice() ?? await getDevice();
const gpu      = createGpuNNPipeline(
  device, cfaData, cfaW, cfaH,
  masks, clipNorm, tiles,
  hPad, wPad, PATCH_SIZE, OVERLAP,
  padTop, padLeft, visHeight, visWidth, TILE_BATCH,
);

for (let b = 0; b < tiles.length; ) {
  const end   = Math.min(b + TILE_BATCH, tiles.length);
  const count = end - b;
  const inputBuf = gpu.extractBatch(b, count);
  const { buffer: inferBuf, dispose } = await runBatchGpu(cfaType, inputBuf, count, PATCH_SIZE);
  gpu.accumulateBatch(inferBuf, b, count);
  dispose();
  b = end;
}

const hwcBuf = await gpu.finalize();
```

Key observations:

- **Tiles are processed in batches of `TILE_BATCH` (32)**. Each batch iteration
  performs three GPU operations (extract, infer, accumulate) with a single
  `await` on `runBatchGpu`. This batching amortizes GPU dispatch overhead while
  keeping memory bounded.
- **Everything stays on GPU** -- the CFA data, masks, and clip thresholds are
  uploaded once during `createGpuNNPipeline()`. No CPU-GPU transfers occur in the
  tile loop. The output `GPUBuffer` from `finalize()` is passed directly to
  `gpuPostprocess()`.
- **Channel masks are computed once** and reused for every tile because CFA
  alignment padding has already shifted the data to canonical pattern origin.
- **Memory is released eagerly**: `padded` and `cfa` are nulled after
  `createGpuNNPipeline` uploads them; `gpu.finalize()` destroys all intermediate
  GPU buffers.
- After the GPU-resident tile loop, the pipeline continues with GPU
  postprocessing (`gpuPostprocess()`) which applies white balance, highlight
  recovery, color correction, and DR gain -- all on GPU. See
  [color-correction-pipeline](color-correction-pipeline.md).

### Note on Python Reference

The Python inference script `infer.py` has been removed from the repository. The
web pipeline is now the sole implementation. The web pipeline uses a linear ramp
window with 24 px overlap, which is sufficient to prevent visible seams in
practice.

## Rationale

### Why 288x288 Patches

- **Divisibility by CFA period.** 288 is divisible by both 6 (X-Trans period)
  and 2 (Bayer period), ensuring that every tile starts on a CFA period boundary
  when the stride (264) is also divisible by both periods. This avoids partial
  CFA patterns at tile edges that would confuse the network.
- **ONNX fixed-shape optimization.** The ONNX model is exported with a fixed
  `(1, 5, 288, 288)` input shape. ONNX Runtime Web's WebGPU backend optimizes
  kernel selection and fusion for fixed tensor shapes. Dynamic axes would
  defeat these optimizations. See [onnx-export](onnx-export.md) for details.
- **GPU memory budget.** 288x288x5 channels x 4 bytes = 1.6 MB per input
  tensor, well within WebGPU limits even on mobile GPUs. The intermediate
  activations of the UNet scale with spatial dimensions; 288 keeps peak
  allocation manageable.
- **Receptive field coverage.** The UNet's effective receptive field at 288x288 is
  large enough to capture sufficient spatial context for demosaicing, including
  the multi-scale features needed for edge-directed interpolation.

### Why 24 px Overlap

- **24 is divisible by 6 and 2**, preserving CFA alignment across tile
  boundaries.
- **24 px is 4 CFA periods (X-Trans)**, providing enough context overlap for the
  network to produce consistent predictions where tiles meet.
- The stride of 264 (`288 - 24`) means the web pipeline generates roughly
  `(H / 264) * (W / 264)` tiles. The smaller overlap (reduced from the previous
  48 px) decreases tile count by approximately 20%, which is significant for
  GPU-resident batched inference where total batch submissions matter more than
  per-tile overhead. The 24 px overlap is sufficient because the GPU-resident
  pipeline's batched inference produces more consistent tile-boundary predictions
  than single-tile inference.

### Why Linear Ramp Instead of Hann Window

- **Simplicity.** The linear ramp is trivially constructed with a loop; no
  trigonometric functions needed. The same weight grid is shared between CPU and
  GPU paths.
- **Sufficient quality.** At 24 px overlap, the transition zone is narrow enough
  that the difference between linear and cosine blending is not perceptible in
  the final image.
- **Predictable normalization.** Two overlapping linear ramps sum to exactly 1.0
  across the overlap zone. With more than two tiles overlapping (at corners,
  where four tiles meet), the accumulated weight is still well-behaved and
  normalization handles it correctly.

### WebGPU-First Strategy

WebGPU provides the fastest execution path by running ONNX operator kernels as
GPU compute shaders. The fallback to WASM is necessary because WebGPU adoption
is not universal: Firefox support is still maturing, and some embedded or older
GPUs lack the required feature tier. The three-tier fallback (WebGPU ->
multi-threaded WASM -> single-threaded WASM) ensures the application works
everywhere, with graceful degradation rather than failure. The chosen backend is
recorded once and exposed via `getBackend()` so the UI can inform the user
which execution path is active.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/constants.ts` | Defines `PATCH_SIZE = 288`, `OVERLAP = 24`, and `TILE_BATCH = 32`. |
| `web/src/pipeline/inference.ts` | `initModels()` loads ONNX sessions from manifest with backend fallback; `runBatch()` / `runBatchGpu()` execute batched inference; `switchModelSize()` swaps S/M/L models; `getInferenceDevice()` exposes ORT's WebGPU device; `getBackend()` reports active backend. |
| `web/src/pipeline/tile-blend-gpu.ts` | `createGpuNNPipeline()` returns GPU-resident `GpuNNPipeline` with `extractBatch`, `accumulateBatch`, `finalize`, `destroy`; WGSL shaders for extract, accumulate, finalize-blend, and crop. |
| `web/src/pipeline/postprocessor.ts` | `createTileBlender()` CPU fallback for linear-ramp blending; `cropToHWC()` extracts original-size HWC output. |
| `web/src/pipeline/preprocessor.ts` | `generateTiles()` computes padded canvas and tile grid; `prefillBatchMasks()` and `fillBatchCfa()` assemble 5-channel batched tensors; `makeChannelMasks()` builds binary CFA masks. |
| `web/src/hooks/useProcessFile.ts` | Orchestrates the GPU-resident tile loop: creates `GpuNNPipeline`, iterates batches via `extractBatch` / `runBatchGpu` / `accumulateBatch`, finalizes. |
| `web/src/hooks/useInit.ts` | Calls `initModels()` at app startup in parallel with other initialization. |

## Antipatterns

**Do not use dynamic ONNX input shapes for browser-deployed models.** Although
the underlying UNet is fully convolutional and can accept arbitrary spatial
dimensions, ONNX Runtime Web's WebGPU backend optimizes kernel dispatch and
memory planning for fixed tensor shapes. Adding `dynamic_axes` to the export
call defeats these optimizations, increasing latency per tile. The browser
always sends tiles at `PATCH_SIZE`, so dynamic axes provide no benefit. If a
different tile size is needed, re-export the model with the new fixed shape.

**Do not submit overlapping batches concurrently.** The current implementation
`await`s each `runBatchGpu()` call before starting the next batch. Concurrent
batch dispatch would overwhelm the WebGPU command queue, spike GPU memory usage
(multiple batches of intermediate activation tensors alive simultaneously), and
risk ORT internal state corruption. The `TILE_BATCH = 32` size already provides
good GPU utilization within a single submission.

**Do not reduce overlap below 24 px.** Smaller overlaps cause visible banding at
tile seams because the blending ramp becomes too narrow to mask prediction
discontinuities. The overlap must also remain divisible by the CFA period (6 for
X-Trans, 2 for Bayer) to maintain pattern alignment across tile boundaries.

**Do not skip the weight normalization step in `finalize()`.** At image borders
and corners, tiles may not overlap symmetrically, and the accumulated weight
per pixel varies. Dividing by the weight sum is necessary even with a linear
ramp that sums to 1 in the interior, because edge pixels may only be covered by
a single tile's ramp. Omitting normalization produces brightness variation at
image edges.
