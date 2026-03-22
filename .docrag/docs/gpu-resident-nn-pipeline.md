---
title: GPU-Resident Neural Network Pipeline
tags: [web, webgpu, compute, inference, tiling, blending, postprocess, highlight-recovery]
scope: web/src/pipeline/tile-blend-gpu.ts, web/src/pipeline/postprocess-gpu.ts, web/src/hooks/useProcessFile.ts, web/src/pipeline/constants.ts
generated: 2026-03-22
commit: 347c8dd
depends_on:
  - tiled-inference-blending
  - highlight-reconstruction-opposed
  - webgpu-compute-demosaicing
---

# GPU-Resident Neural Network Pipeline

GPU-resident processing, zero-copy tile extraction, NCHW batch layout,
weighted scatter accumulate, overlap blending, WGSL compute shaders,
GpuNNPipeline interface, gpuPostprocess, inpaint-opposed highlight recovery,
clip-skip optimization, 256-byte row alignment, WebGPU storage buffers,
copyBufferToTexture, white balance, color correction, chroma reconstruction,
RGBA32F output, buffer lifecycle management.

## Context

Neural network inference on a GPU is fast -- often an order of magnitude faster
than the CPU-to-GPU and GPU-to-CPU data transfers that bracket it. In a naive
pipeline, every batch of tiles would follow this path:

1. CPU builds a Float32Array of 5-channel NCHW tile data.
2. `device.queue.writeBuffer` uploads it to the GPU.
3. ONNX Runtime Web runs inference (GPU-resident).
4. ORT reads the output buffer back to CPU.
5. CPU scatters tile pixels into the blend buffer.
6. After all tiles, CPU uploads the blended image for postprocessing.

Steps 2, 4, and 6 dominate wall-clock time for large images. A 24 MP sensor
processed with PATCH_SIZE=288 and TILE_BATCH=32 generates ~70 batches; each
round-trip transfer stalls the pipeline waiting on PCIe or shared-memory copies.

The GPU-resident path eliminates every transfer between extraction and final
RGBA output. The CFA image, channel masks, and tile positions are uploaded once
at pipeline creation. From that point forward, tile extraction, inference
input/output, overlap blending, cropping, white balance, highlight recovery,
color correction, and RGBA conversion all execute as GPU compute dispatches.
The only readback is the final RGBA32F buffer consumed by the display renderer
via `copyBufferToTexture`.

## Pattern / Approach

### GpuNNPipeline Interface

`createGpuNNPipeline` in `tile-blend-gpu.ts` returns an object implementing
four methods that the orchestrator (`useProcessFile`) calls in sequence:

```
interface GpuNNPipeline {
  extractBatch(startIdx: number, count: number): GPUBuffer;
  accumulateBatch(inferOut: GPUBuffer, startIdx: number, count: number): void;
  finalize(): Promise<GPUBuffer>;
  destroy(): void;
}
```

**extractBatch** -- Dispatches the extract shader to tile the CFA image into a
GPU buffer shaped `[count, 5, ps, ps]` in NCHW layout. The five channels are:
CFA value, R mask, G mask, B mask, and clip ratio. The returned buffer has
`STORAGE | COPY_SRC` usage flags, which satisfies ORT's requirement for
`ort.Tensor.fromGpuBuffer`. Each tile's data is aligned to 256 bytes
(`extractTileStride`) so that dynamic offsets into the buffer satisfy WebGPU's
minimum offset alignment constraint.

**accumulateBatch** -- Takes the inference output buffer (still on GPU, never
read back) and scatters each tile into the padded blend buffer using weighted
accumulation. Each tile gets its own compute pass within a single command
encoder. This serialization within the command buffer eliminates the need for
atomics -- dispatches are sequenced by the GPU command processor.

**finalize** -- Submits two final compute passes (blend finalize + crop) in a
single command encoder, waits for GPU completion via
`device.queue.onSubmittedWorkDone()`, destroys all intermediate buffers, and
returns the cropped HWC GPUBuffer. This buffer passes directly to
`gpuPostprocess` without any CPU-side readback.

**destroy** -- Releases all GPU resources. Called as a safety net if the
pipeline is abandoned before `finalize` (error paths, cancellation).

### Orchestration Flow

In `useProcessFile.ts`, the neural-net demosaic path operates as follows:

1. Preprocessing runs on CPU: decode RAW, crop, normalize CFA, compute WB
   coefficients and clip thresholds, pad to tile alignment, generate tile grid.
2. `createGpuNNPipeline` uploads CFA data, masks, tile positions, and clips to
   the GPU once.
3. A batch loop calls `extractBatch` -> `runBatchGpu` -> `accumulateBatch` for
   each chunk of TILE_BATCH (32) tiles. All three calls operate on GPU buffers.
4. `gpu.finalize()` produces a cropped HWC GPUBuffer.
5. `gpuPostprocess` takes that buffer directly (the `rawHwc` parameter accepts
   both `Float32Array` and `GPUBuffer`) and runs 6 postprocessing passes.
6. The RGBA32F output buffer is handed off via `setGpuResult` for zero-copy
   display.

### WGSL Shaders: Tile Blend (tile-blend-gpu.ts)

All shaders use a workgroup size of 16x16 (`WG = 16`).

**SHADER_EXTRACT** -- Converts the flat CFA image into 5-channel NCHW tiles.
For each pixel `(px, py)` in each tile `tile_idx`, it reads the CFA value at
the tile's absolute position, reads the corresponding R/G/B channel masks
(precomputed from the CFA pattern), and computes a clip ratio. The clip ratio
ramps from 0 to 1 as the CFA value goes from 50% to 100% of the channel's
clip threshold. Dispatched as `(ceil(ps/WG), ceil(ps/WG), count)` -- one
z-slice per tile in the batch.

**SHADER_ACCUMULATE** -- Weighted scatter of one tile into the padded output
buffer. Each thread reads one pixel from the tile's CHW output (3 channels),
multiplies by the 2D blending weight `w2d[ti]`, and adds to the output buffer
at the tile's absolute position. The weight buffer is also accumulated for
later normalization. The 2D weight grid is the outer product of a 1D linear
ramp that rises from 0 to 1 over the overlap region (OVERLAP=24 pixels) and
holds at 1.0 across the interior. Dispatched as `(ceil(ps/WG), ceil(ps/WG))`
per tile; tiles are serialized via separate compute passes.

**SHADER_FINALIZE_BLEND** -- Divides each pixel in the padded output buffer by
its accumulated weight. Dispatched over the full padded image dimensions
`(wPad, hPad)`. Guards against division by zero with a `1e-8` threshold.

**SHADER_CROP** -- Extracts the original (unpadded) region from the padded
buffer. Copies pixels from `(padLeft, padTop)` origin to a destination buffer
in HWC layout with the original `(wOrig, hOrig)` dimensions. The output is
directly consumable by `gpuPostprocess`.

### WGSL Shaders: Postprocess (postprocess-gpu.ts)

The `gpuPostprocess` function encodes up to 6 compute passes in a single
command buffer. Workgroup size is 16x16 throughout.

**Pass 1 -- SHADER_WB (white balance):** In-place multiplication of each pixel
by per-channel WB coefficients `[wb_r, wb_g, wb_b]` normalized to G=1.
Operates on the HWC data buffer.

**Pass 2 -- SHADER_REFAVG_CLIP (reference average + clip detection):** For
each pixel, computes a 3x3 neighborhood average in cube-root space (opposing
channel reconstruction), then cubes the result back. Simultaneously generates
a per-pixel clip mask with bit flags: bit 0 = R clipped, bit 1 = G clipped,
bit 2 = B clipped. The cube-root / cube transform compresses the dynamic range
for averaging, which improves highlight color fidelity when one channel is
saturated while its neighbors are not.

**Pass 3 -- SHADER_DOWNSAMPLE_CLIP:** 3x downsampling of the clip mask via
bitwise OR over 3x3 blocks. Boundary cells are zeroed to prevent edge
artifacts. Produces a reduced-resolution mask for the dilation pass.

**Pass 4 -- SHADER_DILATE:** Per-channel circular dilation of the downsampled
clip mask. Each channel has an independent radius computed from the ratio of
maximum clip to per-channel clip (`7 * maxClip / clip`, clamped to 7..21,
rounded to odd). The kernel uses a circular footprint (`kx^2 + ky^2 <= r^2`)
and early-exits once a set bit is found.

**Pass 5 -- SHADER_CHROMA:** Accumulates per-channel chroma correction offsets
using workgroup-level parallel reduction followed by global atomic
`atomicAddF32` (CAS loop on `atomic<u32>` with `bitcast`). For each unclipped
pixel that falls within a dilated clipped region and exceeds a low threshold
(20% of clip), it contributes `(value - refavg)` to the sum. The `chroma_buf`
stores 6 uint32 values: 3 atomic sums (bitcast as f32) and 3 atomic counts.

**Pass 6 -- SHADER_FINALIZE:** For each pixel, decodes the chroma means from
the atomic buffer, extends highlight values using
`max(original, refavg + chroma_mean)` for clipped channels, applies the 3x3
camera-to-sRGB color correction matrix, applies DR exposure gain, and writes
RGBA32F output. The alpha channel stores the clip ratio
`min(max(r/clip_r, g/clip_g, b/clip_b), 1.0)` for downstream highlight
visualization.

### The Clip-Skip Optimization

When `gpuPostprocess` receives a `Float32Array` (traditional demosaic path),
it scans the CPU-side data to check whether any pixel exceeds the WB-scaled
clip thresholds. If no pixel clips, `anyClipped` is set to `false`, and
passes 2-5 are skipped entirely. The finalize pass uses
`SHADER_FINALIZE_SIMPLE` instead, which applies only color correction and DR
gain -- no refavg buffer, no clip mask, no chroma buffer. This saves
approximately 1 GB of VRAM for images without clipped highlights (the
refavg buffer alone is `width * height * 3 * 4` bytes).

When the input is already a `GPUBuffer` (GPU-resident NN path), CPU-side
scanning is impossible. The pipeline conservatively assumes clipping may be
present and always runs the full 6-pass path. The `SHADER_FINALIZE_SIMPLE`
pipeline is still compiled and cached but only used for the traditional
demosaic code path.

### 256-Byte Row Alignment

WebGPU's `copyBufferToTexture` requires `bytesPerRow` to be a multiple of 256.
For RGBA32F output (4 channels x 4 bytes = 16 bytes per pixel), this means
the image width must be padded to a multiple of 16 pixels:

```
paddedW = ceil(width / 16) * 16
bytesPerRow = paddedW * 16
```

The `padWidth16` helper computes this. The finalize shaders write to
`(y * out_stride + x) * 4` where `out_stride = paddedW`, leaving unused
pixels at the right edge. The `PostprocessResult` type bundles the buffer with
its `bytesPerRow` so the renderer can issue a correct `copyBufferToTexture`
call without recomputing alignment.

### Buffer Management

Buffers fall into three lifecycle categories:

**Uploaded once (pipeline creation):** `cfaBuf` (full CFA image), `masksBuf`
(R/G/B channel masks, 3 x ps^2), `tilePosBuf` (all tile positions), `clipsBuf`
(3 clip thresholds), `w2dBuf` (2D blending weight grid). These are allocated
with `STORAGE | COPY_DST` and written via `device.queue.writeBuffer` during
`createGpuNNPipeline`. They persist across all batches.

**Per-batch (transient):** `batchPosBuf` (tile positions for current batch
slice), `extractOut` (5-channel NCHW output buffer per batch). `extractOut` is
allocated with `STORAGE | COPY_SRC` (ORT requirement) and tracked in an array
for cleanup. The accumulate uniform `batchParamBuf` is reused across batches
(pre-allocated for `maxBatch` tiles with 256-byte stride between entries).

**Blend accumulators (pipeline lifetime):** `blendOutBuf` (padded HWC float32,
3 channels), `weightsBuf` (padded scalar float32). These are zero-initialized
at creation (default GPU buffer behavior) and accumulated across all batches.
Destroyed during `finalize` after the blend and crop passes complete.

**Postprocess intermediates:** All HL recovery buffers (`refavgBuf`,
`clipMaskBuf`, `maskDsBuf`, `dilatedDsBuf`, `chromaBuf`, and their uniform
buffers) are allocated inside `gpuPostprocess`, tracked in the `hlBufs` array,
and destroyed immediately after the command buffer is submitted. Only
`outputBuf` survives -- its ownership transfers to the caller.

The `finalize` method explicitly destroys everything except `cropOutBuf` before
returning. The `destroy` method is a safety net that attempts to destroy all
tracked buffers, catching errors for already-destroyed buffers.

## Rationale

**Single command encoder for accumulate:** Each tile's accumulate dispatch is a
separate compute pass within one command encoder. This guarantees execution
order without atomics or barriers. Dispatching all tiles in one `submit` call
also reduces driver overhead versus submitting per tile.

**Tile stride alignment:** ORT's WebGPU execution provider binds tensors at
byte offsets into a GPUBuffer. WebGPU requires storage buffer offsets to be
256-byte aligned. The `extractTileStride` and `outTileStride` values are
rounded up to 256 bytes so that each tile in a batch can be bound as a
sub-range of the batch buffer.

**Cube-root averaging in refavg:** Cube-root compression before neighborhood
averaging reduces the influence of very bright pixels in the reference
computation. This produces more accurate opposing-channel estimates for
highlight reconstruction compared to linear averaging, where a single bright
pixel can dominate the 3x3 window.

**Circular dilation with per-channel radii:** Channels with lower clip
thresholds (typically blue after WB scaling) need wider search regions to find
enough unclipped reference pixels. The radius formula
`7 * maxClip / channelClip` scales inversely with headroom, clamped to a
practical range.

**WeakMap pipeline cache:** Compute pipelines (`GPUComputePipeline`) are
expensive to create because of shader compilation. Both `tile-blend-gpu.ts` and
`postprocess-gpu.ts` cache their pipelines in a `WeakMap<GPUDevice, Pipelines>`
so they are compiled once per device and garbage-collected if the device is
lost and replaced.

**Workgroup-level reduction + global atomics for chroma:** The chroma shader
aggregates `(value - refavg)` across the entire image into a single 6-element
buffer. A two-level approach is used: each 16x16 workgroup performs a parallel
reduction in shared memory, then workgroup 0 writes the partial sum to the
global buffer via an atomic CAS loop that emulates `atomicAddF32` using
`bitcast<f32>` / `bitcast<u32>`. This reduces global atomic contention by a
factor of 256.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/tile-blend-gpu.ts` | GpuNNPipeline: extract, accumulate, finalize, crop shaders and buffer management |
| `web/src/pipeline/postprocess-gpu.ts` | gpuPostprocess: 6-pass WB + HL recovery + CC + DR + RGBA pipeline |
| `web/src/hooks/useProcessFile.ts` | Orchestrator: drives the extract-infer-accumulate loop and hands off results |
| `web/src/pipeline/constants.ts` | PATCH_SIZE (288), OVERLAP (24), TILE_BATCH (32) |
| `web/src/pipeline/inference.ts` | `runBatchGpu`: ORT inference accepting and returning GPUBuffers |
| `web/src/lib/hwc-handoff.ts` | `setGpuResult`: zero-copy handoff of PostprocessResult to renderer |

## Antipatterns

**Reading back inference output to CPU.** Earlier versions called
`ort.Tensor.getData()` to pull inference results into a Float32Array, scattered
tiles on the CPU, then re-uploaded the blended image. This doubled total
processing time for large images. The current path keeps ORT's output buffer on
the GPU and passes it directly to `accumulateBatch`.

**Allocating HL recovery buffers unconditionally.** The refavg, clip mask,
downsampled mask, dilated mask, and chroma buffers together consume roughly
`width * height * 20` bytes. For a 24 MP image, that is ~480 MB of VRAM. The
clip-skip optimization avoids this allocation entirely when no pixel exceeds
clip thresholds after WB. On the GPU-resident path, where CPU scanning is not
possible, these buffers are allocated conservatively.

**Using a single GPUBuffer for all batch tiles without stride alignment.** ORT
requires `fromGpuBuffer` offsets to be 256-byte aligned. Packing tiles
contiguously without stride padding causes binding failures on some WebGPU
implementations. Both the extract (5-channel) and accumulate (3-channel) tile
layouts round up per-tile byte size to 256 via `roundUp(bytes, 256)`.

**Submitting one command buffer per tile.** The accumulate phase serializes
tiles across separate compute passes inside a single command encoder, but
submits only once. Submitting a separate command buffer per tile would incur
per-submit driver overhead (fence synchronization, command buffer scheduling)
that adds up across hundreds of tiles.

**Forgetting 256-byte row alignment on the output buffer.** The RGBA32F output
must satisfy `copyBufferToTexture`'s bytesPerRow alignment constraint. Writing
output at `(y * width + x) * 4` without padding the stride leads to a
validation error on non-16-aligned widths. The `padWidth16` function and
`out_stride` uniform prevent this.
