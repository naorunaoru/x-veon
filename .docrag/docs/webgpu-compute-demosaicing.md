---
title: WebGPU Compute Shader Demosaicing
tags: [web, webgpu, compute, demosaic]
scope: web/src/pipeline/demosaic-gpu.ts, web/src/pipeline/demosaic.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

GPU-accelerated demosaicing provides a high-throughput path for reconstructing
full-color images from single-channel CFA mosaic data. Two traditional demosaic
algorithms -- bilinear interpolation and DHT (Directional Homogeneity Testing)
-- have WebGPU compute shader implementations that run entirely on the GPU,
bypassing the WASM worker pool used by other traditional algorithms.

This module is a backend within the dispatch system documented in
[demosaic-algorithm-dispatch](demosaic-algorithm-dispatch.md). The dispatcher in
`demosaic.ts` checks `gpuAvailable()` at runtime and routes `'bilinear'` and
`'dht'` algorithm requests to the GPU path when the device is ready. If the GPU
path throws (buffer size exceeded, device lost), the dispatcher catches the
error and falls back to the WASM worker pool transparently.

WebGPU availability depends on the browser and hardware. `initDemosaicGpu()`
probes `navigator.gpu`, requests an adapter, and negotiates a device with the
adapter's maximum storage buffer limits. This initialization runs in parallel
with WASM and model loading during app startup via `useInit`.

## Pattern / Approach

### Initialization: Adapter, Device, and Shader Compilation

`initDemosaicGpu()` performs a three-stage setup:

1. **Adapter request** -- calls `navigator.gpu.requestAdapter()`. Returns
   `false` if WebGPU is unsupported or no adapter is available.

2. **Device request** -- calls `adapter.requestDevice()` with `requiredLimits`
   set to the adapter's own `maxStorageBufferBindingSize` and `maxBufferSize`.
   This ensures the device can handle the largest buffers the hardware supports,
   which is critical for high-resolution sensor data (24+ megapixel images
   produce storage buffers in the hundreds of megabytes).

3. **Pipeline compilation** -- creates three `GPUComputePipeline` objects from
   inline WGSL shader source strings:
   - `pipeline` from `BILINEAR_WGSL` -- single-pass bilinear demosaic.
   - `dhtGreenPipeline` from `DHT_GREEN_WGSL` -- DHT pass 1 (directional green
     interpolation).
   - `dhtResolvePipeline` from `DHT_RESOLVE_WGSL` -- DHT pass 2 (homogeneity-
     weighted resolve with color-difference interpolation).

All three pipelines use `layout: 'auto'`, letting the WebGPU implementation
derive the bind group layout from the WGSL declarations.

The module stores the `GPUDevice` and pipeline objects at module scope.
`gpuAvailable()` returns `true` only when both `device` and `pipeline` are
non-null, providing a synchronous readiness check for the dispatcher.

### Bilinear Algorithm: Single-Pass 5x5 Window

The bilinear shader (`BILINEAR_WGSL`) is a single compute pass with workgroup
size `(16, 16)`. Each thread processes one pixel.

**Uniform parameters** (the `Params` struct):
- `width`, `height` -- image dimensions.
- `dy`, `dx` -- CFA phase offset (sub-pattern alignment within the mosaic
  period).
- `period` -- CFA repeat period (2 for Bayer, 6 for X-Trans).

**CFA lookup** -- the `cfa_pattern` storage buffer holds a flattened
`period x period` channel map. The `cfa_ch(y, x)` helper applies `dy`/`dx`
offsets with modular arithmetic to determine which channel (0=R, 1=G, 2=B) is
captured at each pixel position. This design handles arbitrary CFA patterns
without hardcoding Bayer or X-Trans geometry.

**Interpolation logic**:
- The known channel (where `cfa_ch(y, x)` matches the channel index) copies
  the input value directly.
- Each missing channel averages all same-channel neighbors within a 5x5 window
  centered on the current pixel, with bounds checking to handle edges. This is
  a simple mean filter -- no gradient weighting or directional awareness.

**Output format** -- CHW planar `Float32Array`. The output buffer is laid out
as three contiguous planes of `width * height` floats: R plane, G plane, B
plane, in that order. This planar layout matches the convention used downstream
by the rendering pipeline.

**Host-side buffer management** (`runBilinearGpu`):
1. Validates `outputBytes` against `device.limits.maxStorageBufferBindingSize`.
2. Creates five GPU buffers: input (STORAGE+COPY_DST), output (STORAGE+COPY_SRC),
   params (UNIFORM+COPY_DST), CFA pattern (STORAGE+COPY_DST), staging
   (COPY_DST+MAP_READ).
3. Uploads CFA data and parameters via `device.queue.writeBuffer()`.
4. Dispatches `ceil(width/16) x ceil(height/16)` workgroups.
5. Copies the output buffer to the staging buffer within the same command
   encoder.
6. Submits the command buffer, awaits `stagingBuffer.mapAsync(READ)`, and copies
   the mapped range into a new `Float32Array`.
7. Destroys all five buffers after reading the result.

### DHT Algorithm: Two-Pass Directional Green + Homogeneity Resolve

DHT splits the problem into two sequential compute passes encoded in a single
command buffer. Both passes share workgroup size `(16, 16)`.

#### Pass 1: Directional Green Interpolation (`DHT_GREEN_WGSL`)

Produces a planar buffer `green_hv` with two planes: horizontal green estimates
and vertical green estimates, each `width * height` floats.

- **Green pixels** (channel 1): both H and V planes store the known green value
  directly.
- **Red/Blue pixels**: the shader interpolates green separately along the
  horizontal and vertical directions using inverse-distance-squared weighting
  over a +/-3 pixel window. For horizontal, it scans the same row; for
  vertical, the same column. Only neighbors that are green according to
  `cfa_ch()` contribute. The weight function is `1 / (1 + d^2)` where `d` is
  the absolute offset in pixels.

This directional separation is the core of the DHT approach: edges aligned with
one axis will produce a more accurate estimate in the perpendicular direction.

#### Pass 2: Homogeneity Selection and Color-Difference Resolve (`DHT_RESOLVE_WGSL`)

Reads both the original CFA input and the `green_hv` intermediate buffer.
Produces the final CHW planar RGB output.

**Homogeneity test**: For each pixel, the shader computes a smoothness score
for each direction by summing absolute green differences between the center
pixel and its 3x3 neighbors (excluding the center itself). Lower sum indicates
more homogeneous (smoother) green reconstruction in that direction. The clamped
neighbor indexing avoids out-of-bounds reads at image edges.

**Direction blending**: Rather than a hard binary choice, the shader computes a
soft blending weight:

```
alpha = hom_v / (hom_h + hom_v + epsilon)
green = alpha * green_h + (1 - alpha) * green_v
```

When the vertical homogeneity score is high (rough), `alpha` is large, so the
horizontal estimate (which is smoother) gets more weight. The `1e-6` epsilon
prevents division by zero in perfectly flat regions.

**Red/Blue reconstruction via color-difference interpolation**: Once green is
established, missing R and B channels are recovered using a color-difference
model. The `interp_cd()` function scans a 5x5 window around the target pixel,
finds neighbors where the target channel is known, computes the color
difference `(channel_value - green_avg)` at each, and takes an
weighted average using `1 / (1 + Manhattan_distance)`. The final value is
`green + weighted_average_color_difference`. The `green_avg` helper averages
the H and V green estimates for each neighbor, providing a direction-neutral
green reference for the color-difference calculation.

**Green pixels** are handled as a special case: the green channel uses the known
CFA value directly, and R/B are interpolated via color-difference using the
known green as the reference.

#### Buffer Layout for DHT

The DHT path allocates six GPU buffers:

| Buffer          | Size                    | Usage                            |
|-----------------|-------------------------|----------------------------------|
| inputBuffer     | W * H * 4               | STORAGE + COPY_DST               |
| paramsBuffer    | 32 bytes                | UNIFORM + COPY_DST               |
| cfaPatternBuffer| period^2 * 4            | STORAGE + COPY_DST               |
| greenHvBuffer   | 2 * W * H * 4           | STORAGE (internal only)          |
| outputBuffer    | 3 * W * H * 4           | STORAGE + COPY_SRC               |
| stagingBuffer   | 3 * W * H * 4           | COPY_DST + MAP_READ              |

The `greenHvBuffer` is the largest intermediate allocation, holding two full
green planes. It is created with only `GPUBufferUsage.STORAGE` since it is never
copied to or from the CPU -- it exists only as the link between pass 1 and
pass 2. Both passes are encoded sequentially within a single `GPUCommandEncoder`
and submitted as one command buffer. WebGPU guarantees that pass 1 completes
before pass 2 begins because they are sequential compute passes within the same
submission.

### Output Format: CHW Planar Float32Array

Both GPU paths return a `Float32Array` of length `3 * width * height` in CHW
(channel-height-width) planar order:

- Offset `0` to `W*H - 1`: Red channel
- Offset `W*H` to `2*W*H - 1`: Green channel
- Offset `2*W*H` to `3*W*H - 1`: Blue channel

This matches the output contract of the WASM worker path, so the dispatcher can
return the result without transformation regardless of which backend executed.

### Dispatch Integration

The `runDemosaic()` function in `demosaic.ts` implements the GPU-first fallback
chain:

```
if (algorithm === 'bilinear' && gpuAvailable()) -> runBilinearGpu()
if (algorithm === 'dht' && gpuAvailable())      -> runDhtGpu()
else                                             -> DemosaicPool (WASM workers)
```

Each GPU call is wrapped in a try/catch. On failure (buffer too large, device
lost, validation error), it logs a warning and falls through to the WASM path.
This makes the GPU path a transparent optimization: the caller never needs to
know which backend ran.

`initDemosaicGpuSafe()` is a thin wrapper around `initDemosaicGpu()` called
from `useInit`. It runs in `Promise.all` alongside WASM initialization and
model loading. If initialization fails, `gpuAvailable()` returns `false` for
the lifetime of the session, and all demosaic requests route to WASM workers.

## Rationale

### Why Only Bilinear and DHT Have GPU Paths

The choice of which algorithms to port to WebGPU reflects a tradeoff between
implementation complexity and performance impact:

- **Bilinear**: The simplest algorithm, naturally expressed as a single compute
  pass with no dependencies between pixels. It is the default preview algorithm
  and benefits most from GPU throughput since it is used during interactive
  parameter adjustment where latency matters.

- **DHT**: A moderate-complexity algorithm that decomposes cleanly into two
  independent passes with a well-defined intermediate buffer. The directional
  green interpolation and homogeneity resolve are both per-pixel operations
  with fixed-size local neighborhoods, mapping well to compute shader
  workgroups.

- **Algorithms without GPU paths** (Markesteijn, AHD, PPG, MHC): These involve
  iterative refinement, multi-pass adaptive thresholding, or complex control
  flow that does not map efficiently to GPU compute. AHD and Markesteijn in
  particular use homogeneity maps with variable-radius neighborhoods and
  iterative convergence that would require many passes and synchronization
  barriers. The engineering cost of porting them exceeds the expected benefit
  since they are used less frequently (primarily for final export) and the WASM
  worker pool already parallelizes across CPU cores.

- **Neural network**: Runs via ONNX Runtime with its own WebGPU backend,
  entirely separate from this module.

### Performance Characteristics

The GPU path eliminates CPU-side per-pixel iteration. For bilinear on a 24MP
image, the shader dispatches roughly 93,750 workgroups (6000x4000 / 256
threads), completing in a single GPU submission with no CPU round-trips during
computation. The main overhead is the CPU-GPU memory transfer: uploading the
input CFA and downloading the output RGB via the staging buffer.

DHT adds a second compute pass and the `greenHvBuffer` intermediate, but both
passes are in the same command buffer submission. The GPU handles the
pass-to-pass synchronization internally, avoiding any `mapAsync` or CPU-side
fence between passes.

The `maxStorageBufferBindingSize` negotiation at init time is important: the
default WebGPU limit (128MB or 256MB depending on the implementation) is too
small for the output buffer of a 50MP+ image at `3 * W * H * 4` bytes. By
requesting the adapter's actual limit, the module supports sensors up to the
hardware's VRAM capacity.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/demosaic-gpu.ts` | WGSL shaders, GPU init, bilinear and DHT GPU runners |
| `web/src/pipeline/demosaic.ts` | Dispatch: GPU-first with WASM fallback, `initDemosaicGpuSafe` wrapper |
| `web/src/hooks/useInit.ts` | App startup: calls `initDemosaicGpuSafe()` in parallel with WASM/model init |
| `web/src/pipeline/types.ts` | `DemosaicMethod` union type definition |

### Related Documents

- [demosaic-algorithm-dispatch](demosaic-algorithm-dispatch.md) -- dispatch
  logic, WASM worker pool, algorithm selection rules.
- [cfa-pattern-system](cfa-pattern-system.md) -- CFA pattern representation
  consumed by the `cfa_pattern` storage buffer.
- [webgpu-renderer](webgpu-renderer.md) -- the separate WebGPU rendering
  pipeline that consumes the demosaiced output.

## Antipatterns

### Do Not Re-create the Device Per Frame

The module initializes the `GPUDevice` once at startup and reuses it across all
demosaic calls. Re-requesting an adapter and device for each image would add
hundreds of milliseconds of latency and risk shader recompilation. The pipelines
are compiled once and remain valid for the device's lifetime. If the device is
lost (e.g., GPU driver crash, tab backgrounding), `gpuAvailable()` will return
`false` and the fallback chain handles it.

### Do Not Keep Staging Buffers Mapped Across Calls

Each `runBilinearGpu` and `runDhtGpu` call creates, maps, reads, unmaps, and
destroys its staging buffer within a single invocation. Holding a mapped staging
buffer across calls would block the GPU command queue and risk validation errors
if the buffer is used in a subsequent submission while still mapped. The
create-use-destroy pattern trades a small allocation overhead for correctness
and simplicity.

### Do Not Use Shared Mutable Module State for Buffers

The only module-level mutable state is the `GPUDevice` and the three pipeline
objects. Per-invocation buffers (input, output, staging, params, CFA pattern,
greenHv) are created and destroyed within each function call. Reusing buffers
across calls would require size tracking, invalidation on dimension changes, and
careful synchronization between concurrent invocations. The current approach
avoids these hazards with negligible overhead since buffer allocation is fast
relative to compute dispatch and memory transfer.

### Do Not Hardcode CFA Geometry in Shaders

The shaders parameterize the CFA pattern via the `cfa_pattern` storage buffer
and `period` uniform rather than hardcoding Bayer or X-Trans offsets. This means
the same shader code handles any CFA layout. Hardcoding a 2x2 Bayer assumption
would break X-Trans support and require separate shader variants for each sensor
type.
