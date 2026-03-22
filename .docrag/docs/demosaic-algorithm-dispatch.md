---
title: Demosaic Algorithm Selection and Dispatch
tags: [web, demosaic, dispatch, pipeline]
scope: web/src/pipeline/demosaic.ts, web/src/pipeline/demosaic-gpu.ts, web/src/pipeline/demosaic-pool.ts, web/src/pipeline/demosaic-worker.ts, web/src/pipeline/inference.ts, web/src/pipeline/tile-blend-gpu.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

Demosaicing is the stage that reconstructs a full-color RGB image from the
single-channel CFA (Color Filter Array) mosaic. x-veon supports eight demosaic
methods spanning three quality/speed tiers: a neural network (branded "X-veon"),
traditional high-quality algorithms (Markesteijn, AHD, DHT, PPG, MHC), and a
fast bilinear interpolator. Each method has different CFA compatibility (Bayer
only, X-Trans only, or both), different computational characteristics, and
different runtime backend requirements. The dispatch system decides which
backend executes a given method and provides a fallback chain when the preferred
backend is unavailable.

The dispatch system consumes the preprocessed CFA data produced by the
preprocessing pipeline documented in
[cfa-preprocessing-web](cfa-preprocessing-web.md). At the point of dispatch,
the CFA has already been cropped, normalized, white-balanced, highlight-
reconstructed, and padded to canonical alignment with `(dy, dx) = (0, 0)`.

## Pattern / Approach

### The DemosaicMethod Union Type

All algorithm selection flows through the `DemosaicMethod` union type defined in
`web/src/pipeline/types.ts`:

```
type DemosaicMethod =
  | 'neural-net'
  | 'bilinear'
  | 'markesteijn3'
  | 'markesteijn1'
  | 'dht'
  | 'ahd'
  | 'ppg'
  | 'mhc';
```

The type is partitioned by CFA compatibility:

| Method | CFA Type | Description |
|--------|----------|-------------|
| `neural-net` | Bayer + X-Trans | Learned demosaic via tiled ONNX inference; separate models per CFA type |
| `bilinear` | Bayer + X-Trans | Simple averaging of same-color neighbors in a 5x5 window |
| `markesteijn3` | X-Trans only | Markesteijn 3-pass refinement, highest-quality traditional X-Trans method |
| `markesteijn1` | X-Trans only | Markesteijn 1-pass, faster but lower quality than 3-pass |
| `dht` | X-Trans | Directional Homogeneity Test with color-difference interpolation |
| `ahd` | Bayer only | Adaptive Homogeneity-Directed interpolation |
| `ppg` | Bayer only | Patterned Pixel Grouping |
| `mhc` | Bayer only | Malvar-He-Cutler linear interpolation |

A derived type, `TraditionalMethod`, is defined as
`Exclude<DemosaicMethod, 'neural-net'>` and covers the seven non-neural methods.
This type is used in the dispatch function signature to enforce that neural-net
routing is handled separately.

A companion type `ModelSize = 'S' | 'M' | 'L'` (also in `types.ts`) controls
which neural network model variant is used. The size maps to different UNet base
widths (S=16, M=32, L=64) via `inference.ts`'s `SIZE_TO_WIDTH` table. See
[tiled-inference-blending](tiled-inference-blending.md) for model manifest and
size switching details.

### CFA-Aware Method Filtering in the UI

The `SettingsPanel` component maintains a `DEMOSAIC_OPTIONS` array where each
entry optionally specifies a `cfa` field (`'bayer'` or `'xtrans'`). Methods
without a `cfa` field (neural-net, bilinear) appear for all sensors. When the
user loads an image, the UI filters options to those compatible with the
detected CFA type. If the currently selected method is incompatible (e.g., AHD
selected but an X-Trans file was loaded), a `useEffect` hook automatically
resets the selection to `'neural-net'`.

### Top-Level Dispatch: Neural-Net vs. Traditional

The primary dispatch branch is in `useProcessFile.ts` at step 9 of the
processing pipeline:

```
if (method === 'neural-net') {
  // Tiled ONNX inference path
} else {
  // Traditional demosaic path via runDemosaic()
}
```

This is a hard split with fundamentally different data flows:

**Neural-net path**: The preprocessed CFA is tiled into overlapping 288x288
patches by `generateTiles()`. The primary path is fully GPU-resident: tiles are
extracted into 5-channel NCHW tensors (CFA values + 3 channel masks + clip
ratio) on the GPU by `GpuNNPipeline.extractBatch()`, then fed in batches of
`TILE_BATCH` (32) to `runBatchGpu()` which runs ONNX inference with GPU buffer
interop. Tile outputs are accumulated and blended by GPU compute shaders in
`tile-blend-gpu.ts`. The ONNX runtime has a backend fallback chain: WebGPU
first, then multi-threaded WASM, then single-threaded WASM. Models are loaded
from a manifest (`models.json`) with S/M/L size variants, and the correct
CFA-type model is selected based on the image's `CfaType`.

**Traditional path**: The full padded CFA plane (not tiled) is passed to
`runDemosaic()` in `web/src/pipeline/demosaic.ts`, which handles sub-dispatch
across WebGPU compute shaders and WASM workers.

### Traditional Demosaic Sub-Dispatch

The `runDemosaic()` function implements a two-tier dispatch with fallback:

**Tier 1 -- WebGPU compute shaders** (attempted first for eligible methods):

- `bilinear`: Single-pass WGSL compute shader (`BILINEAR_WGSL`). Each thread
  processes one pixel. For each of the three color channels, if the pixel's CFA
  position matches the channel, the known value is used directly. For missing
  channels, same-color neighbors within a 5x5 window are averaged. Workgroup
  size is 16x16.

- `dht`: Two-pass WGSL compute shader. Pass 1 (`DHT_GREEN_WGSL`) computes
  directional green estimates -- horizontal and vertical -- using
  inverse-distance-squared-weighted averaging of green neighbors within +/-3 pixels in
  the respective direction. Pass 2 (`DHT_RESOLVE_WGSL`) performs homogeneity
  selection between the two directional estimates using a 3x3 window variance
  test, then interpolates the missing R and B channels via color-difference
  averaging in a 5x5 window. The two passes share the CFA input buffer and CFA
  pattern buffer; an intermediate `greenHvBuffer` (2 planes: green_h and
  green_v) connects them. Both passes are encoded into a single command buffer
  and submitted together.

**Tier 2 -- WASM worker pool** (fallback for all methods):

If the GPU path is unavailable or if the method has no GPU implementation (all
methods other than bilinear and DHT), execution falls through to
`DemosaicPool.run()`. This includes markesteijn3, markesteijn1, ahd, ppg,
and mhc, as well as bilinear and dht when WebGPU is not available.

The dispatch logic in `runDemosaic()`:

```
if (algorithm === 'bilinear' && gpuAvailable()) {
  try { return await runBilinearGpu(...); }
  catch { /* fall through */ }
}
if (algorithm === 'dht' && gpuAvailable()) {
  try { return await runDhtGpu(...); }
  catch { /* fall through */ }
}
// All methods: WASM worker pool
return await getPool().run(...);
```

Each GPU path is wrapped in a try/catch that catches runtime errors (e.g.,
buffer size exceeds `maxStorageBufferBindingSize` for very large images) and
falls back to the WASM path. The fallback is logged via `console.warn`.

### GPU Initialization and Availability

GPU initialization happens at application startup in `useInit.ts`, called as
`initDemosaicGpuSafe()` alongside WASM init and model loading:

```
await Promise.all([
  initWasm(),
  initModels(),
  initDemosaicGpuSafe(),
  ...
]);
```

The `initDemosaicGpu()` function in `demosaic-gpu.ts`:

1. Checks `navigator.gpu` exists (WebGPU API availability).
2. Requests a GPU adapter.
3. Requests a device with maximum buffer limits (`maxStorageBufferBindingSize`,
   `maxBufferSize`) to handle large sensor images.
4. Pre-compiles all three compute pipelines: bilinear, DHT green pass, and
   DHT resolve pass.

If any step fails, the function returns `false` and the module-level `device`
variable remains `null`. The `gpuAvailable()` check used in the dispatch simply
tests `device !== null && pipeline !== null`.

The "safe" wrapper `initDemosaicGpuSafe()` exists so that GPU init failure
does not prevent application startup -- it silently returns `false`.

### WASM Worker Pool Architecture

The `DemosaicPool` class in `demosaic-pool.ts` manages a pool of Web Workers
for CPU-based demosaicing. Key design points:

**Lazy initialization**: Workers are not spawned until the first call to
`run()`. The pool calls `ensureWorkers()` which creates workers on demand.

**Pool sizing**: Worker count is `min(navigator.hardwareConcurrency, 8)`. The
`MAX_WORKERS` cap of 8 prevents over-subscription on high-core-count machines
where memory would be the bottleneck.

**Horizontal strip splitting**: For images tall enough, the CFA plane is
divided into horizontal strips distributed across workers. Each strip includes
overlap regions (3x the CFA period on each side) so that algorithms with
spatial neighborhoods produce correct results at strip boundaries. After all
workers finish, the inner (non-overlap) rows from each strip are stitched into
the final output buffer.

Strip overlap factor of 3 (`STRIP_OVERLAP_FACTOR = 3`) is sufficient for
Markesteijn's 3-pass refinement and its border requirements. Strips smaller
than `MIN_STRIP_HEIGHT = 128` rows are not created; if the image is too short
to split, a single worker processes the whole image.

**Worker message protocol**: Each worker loads the WASM demosaic module on first
use, then receives messages containing the CFA strip data (transferred, not
copied), dimensions, CFA shift, and algorithm name. The worker dispatches to
one of two WASM functions:

- `demosaic_bayer(cfa, width, height, bayerVariant, algorithm)` for Bayer
  sensors. The `bayerVariant` string (`'rggb'`, `'grbg'`, `'gbrg'`, `'bggr'`)
  is derived from the CFA shift by `bayerVariantForShift(dy, dx)`.
- `demosaic_image(cfa, width, height, dy, dx, algorithm)` for X-Trans sensors.

The output `Float32Array` in CHW planar format is transferred back to the main
thread.

**Lifecycle management**: The pool is destroyed after each file finishes
processing (`destroyDemosaicPool()` in the `finally` block of
`useProcessFile`). This terminates all workers and releases their memory.
The pool is recreated lazily on the next processing run.

### ONNX Inference Backend Fallback (Neural-Net Path)

The `inference.ts` module loads ONNX models with its own three-tier fallback:

1. **WebGPU**: `ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'], preferredOutputLocation: 'gpu-buffer' })`.
   Preferred for GPU-accelerated inference. When active, captures ORT's
   `GPUDevice` for compute shader interop via `getInferenceDevice()`.
2. **Multi-threaded WASM**: Falls back if WebGPU is not available. Thread count
   is set to `navigator.hardwareConcurrency`.
3. **Single-threaded WASM**: Final fallback if `SharedArrayBuffer` is not
   available (e.g., missing cross-origin isolation headers). Sets
   `ort.env.wasm.numThreads = 1`.

The backend is determined by the first model loaded and reused for all
subsequent models. The resolved backend name is stored and reported in the
processing result metadata.

Models are loaded from a `models.json` manifest keyed by
`{cfaType}_w{baseWidth}_{variant}`. The `initModels(size)` function loads the
initial pair (one per CFA type); `switchModelSize(size)` swaps to a different
S/M/L variant at runtime. See
[tiled-inference-blending](tiled-inference-blending.md) for full details.

### Data Format Convention

All demosaic backends produce output in CHW (Channel-Height-Width) planar
format as a `Float32Array` of length `3 * width * height`. The three planes
are laid out contiguously: `[R plane | G plane | B plane]`. This is consistent
across the GPU shaders, the WASM workers, and the ONNX model outputs. The
postprocessor's `cropToHWC` converts from CHW to HWC (Height-Width-Channel,
interleaved) format for display and export.

### Result Caching by Method

The Zustand store maintains a `cachedResults` map per file, keyed by
`DemosaicMethod`. When a user switches between methods, previously computed
results can be restored without reprocessing via `restoreCachedResult()`. The
pixel data itself is stored in OPFS (Origin Private File System) keyed by
`hwcKey(fileId, method)`, so only lightweight metadata lives in the Zustand
store. Neural-net results are always cached; traditional method results follow
the same caching path.

## Rationale

### Why Three Backends

The three execution backends (ONNX inference, WebGPU compute, WASM workers)
exist because no single backend covers all requirements:

- **ONNX via onnxruntime-web** is the only option for neural network inference.
  It cannot execute hand-written compute shaders for traditional algorithms.

- **WebGPU compute shaders** provide the fastest execution for algorithms that
  can be expressed as per-pixel parallel operations. Bilinear interpolation is
  embarrassingly parallel, and DHT's two-pass structure maps cleanly to compute
  passes. However, writing complex algorithms like Markesteijn (which involves
  multi-pass refinement with data-dependent branching across a 6x6 tile) in
  WGSL is impractical and error-prone.

- **WASM workers** are the universal fallback. The Rust WASM module implements
  all seven traditional algorithms with full correctness, including Markesteijn's
  three-pass X-Trans refinement. WASM runs on any browser regardless of GPU
  capabilities. The worker pool provides parallelism through horizontal strip
  splitting, partially compensating for the lack of GPU acceleration.

The dispatch system tries the fastest available backend and falls back
gracefully. A user on a device without WebGPU still gets correct results via
WASM, just slower.

### Algorithm Quality/Speed Characteristics

From highest to lowest quality: neural-net (X-veon) provides learned demosaic
with best edge preservation; Markesteijn 3-pass is the highest-quality
traditional X-Trans method; AHD and DHT are directional homogeneity methods
(DHT has a WebGPU path); PPG and MHC are mid-quality Bayer methods; bilinear
is fastest but lowest quality, useful for previews.

### Why GPU Only for Bilinear and DHT

Bilinear and DHT were chosen for GPU implementation because:

- **Bilinear** is trivially parallel: each output pixel depends only on a small
  fixed neighborhood of input pixels with no data-dependent control flow. The
  WGSL shader is compact (63 lines) and correct by construction.

- **DHT** decomposes into two independent parallel passes with a simple
  intermediate buffer. The first pass computes directional green estimates per
  pixel; the second reads those estimates and resolves the final color. Each
  pixel's computation is independent within a pass.

Algorithms like Markesteijn involve iterative refinement where pass N reads
results written by pass N-1 in complex patterns across a 6x6 neighborhood.
AHD builds homogeneity maps that require full-image reduction steps.
Implementing these in WGSL would be complex, hard to validate, and unlikely to
provide a speedup proportional to the engineering effort, given that the WASM
worker pool already parallelizes across CPU cores.

### Why Destroy the Worker Pool After Each Image

The `destroyDemosaicPool()` call in the `finally` block of `useProcessFile`
terminates all workers after each file completes. This is deliberate: each
worker holds a loaded WASM instance with its own linear memory, and keeping
8 idle workers alive wastes tens of megabytes of memory. Since demosaic
processing is an infrequent batch operation (not a continuous real-time loop),
the cost of re-spawning workers for the next image is negligible compared to
the memory savings.

### Why the CFA Pattern Is Flattened

The `flattenPattern()` helper in `useProcessFile.ts` converts the 2D
`pattern: readonly (readonly number[])[]` into a `Uint32Array` for GPU and
WASM consumption. WebGPU storage buffers require contiguous typed arrays, and
the WASM boundary cannot receive nested JavaScript arrays. The flat layout
`flat[y * period + x]` is the same indexing convention used in the WGSL shaders'
`cfa_ch()` function.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/demosaic.ts` | Top-level dispatch: `runDemosaic()` routes traditional methods to WebGPU or WASM pool; `initDemosaicGpuSafe()` wraps GPU init; `destroyDemosaicPool()` tears down workers |
| `web/src/pipeline/demosaic-gpu.ts` | WebGPU compute shaders for bilinear and DHT; `initDemosaicGpu()` creates device and pipelines; `gpuAvailable()` check; `runBilinearGpu()` and `runDhtGpu()` |
| `web/src/pipeline/demosaic-pool.ts` | `DemosaicPool` class: worker lifecycle, horizontal strip splitting with overlap, result stitching |
| `web/src/pipeline/demosaic-worker.ts` | Web Worker entry point: loads WASM, dispatches to `demosaic_bayer()` or `demosaic_image()` |
| `web/src/pipeline/inference.ts` | ONNX model loading from manifest, `runBatch()` / `runBatchGpu()` for neural-net path; `switchModelSize()`; `getInferenceDevice()`; backend fallback chain (WebGPU -> WASM -> single-threaded WASM) |
| `web/src/pipeline/tile-blend-gpu.ts` | `createGpuNNPipeline()`: GPU-resident tile extraction, accumulation, blending, and cropping for neural-net path |
| `web/src/pipeline/types.ts` | `DemosaicMethod` union type, `ModelSize` type, `CfaType`, `CfaInfo`, and related interfaces |
| `web/src/hooks/useProcessFile.ts` | Pipeline orchestration: reads `demosaicMethod` from store, branches neural-net vs. traditional, calls `runDemosaic()` or tile loop |
| `web/src/components/SettingsPanel.tsx` | UI method selector: `DEMOSAIC_OPTIONS` array with CFA-type filtering, auto-fallback to `'neural-net'` |
| `web/src/store.ts` | Zustand store: `demosaicMethod` state, `cachedResults` per-file map keyed by method, persistence to IndexedDB |

## Antipatterns

### Do Not Add a GPU Path Without a WASM Fallback

Every WebGPU compute path in `runDemosaic()` is wrapped in try/catch with a
fallback to the WASM worker pool. WebGPU may be unavailable (no GPU, browser
without WebGPU support, insufficient buffer limits for large images). If you add
a new GPU-accelerated algorithm, it must follow the same pattern: attempt GPU,
catch errors, fall through to pool. Never make GPU execution a hard requirement
for any traditional method.

### Do Not Call runDemosaic with 'neural-net'

The `runDemosaic()` function accepts `TraditionalMethod`, which is
`Exclude<DemosaicMethod, 'neural-net'>`. The neural-net path has completely
different data flow (tiled inference with blending) and is handled separately in
`useProcessFile.ts`. Attempting to pass `'neural-net'` to `runDemosaic()` is a
type error. The two paths should remain separate; do not try to unify them
behind a single dispatch function.

### Do Not Keep Workers Alive Between Processing Runs

The pool is destroyed after each image via `destroyDemosaicPool()`. Do not
"optimize" by keeping workers alive between runs. The WASM instances in idle
workers consume memory disproportionate to their utility, and the worker spawn
cost (~10ms per worker) is negligible relative to the multi-second demosaic
operation. The lazy `ensureWorkers()` pattern handles re-creation transparently.

### Do Not Exceed Strip Overlap Limits

The worker pool's strip overlap is `3 * period` pixels on each side. This is
calibrated for Markesteijn 3-pass, which has the largest spatial footprint of
any implemented algorithm. If you add a new algorithm with a larger
neighborhood (e.g., requiring 4x the CFA period of context), you must increase
`STRIP_OVERLAP_FACTOR`. Insufficient overlap causes visible banding artifacts at
strip boundaries that are difficult to diagnose because they appear only on
images tall enough to be multi-stripped.

### Do Not Share GPU Buffers Across Invocations

The `runBilinearGpu()` and `runDhtGpu()` functions create and destroy all GPU
buffers per invocation. Do not cache or reuse buffers across calls -- image
dimensions vary between files, and buffer reuse introduces subtle bugs when a
smaller image follows a larger one (stale data in unwritten regions). Buffer
allocation cost is negligible relative to compute dispatch and readback.

### Do Not Assume a Fixed CFA Period in Shaders

The WGSL shaders parameterize the CFA period via a uniform buffer rather than
hardcoding it. The `cfa_ch()` helper function in both shaders uses
`cfa_pattern[((y + dy) % period) * period + ((x + dx) % period)]` to look up
channel identity. This supports both Bayer (period 2) and X-Trans (period 6)
patterns with the same shader code. Do not add period-specific branching to the
shaders; the dynamic lookup handles all cases.
