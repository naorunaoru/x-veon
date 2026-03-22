---
title: WASM Worker Pool with Strip-Based Parallelism
tags: [web, workers, wasm, parallelism]
scope: web/src/pipeline/demosaic-pool.ts, web/src/pipeline/demosaic-worker.ts, web/src/pipeline/demosaic.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

Traditional demosaicing algorithms (AHD, PPG, MHC, DHT, Markesteijn) run in
WebAssembly and are single-threaded within a single WASM instance. A 24 MP raw
image can take several hundred milliseconds to demosaic with a multi-pass
algorithm like Markesteijn or AHD, and during that time the browser's main
thread would be entirely blocked if the work happened there. Even offloaded to a
single Web Worker, the latency stays proportional to pixel count because WASM
does not expose shared-memory threading in the way native pthreads does.

The worker pool solves both problems simultaneously: it moves WASM execution off
the main thread to keep the UI responsive, and it splits each image into
horizontal strips processed in parallel across multiple workers to reduce
wall-clock time roughly by the number of cores available. This is the CPU
fallback path for algorithms that the GPU dispatch layer (documented in
[demosaic-algorithm-dispatch](demosaic-algorithm-dispatch.md)) cannot handle or
when WebGPU is unavailable.

## Pattern / Approach

### DemosaicPool Class

`DemosaicPool` (exported from `demosaic-pool.ts`) owns an array of Web Workers,
each running an independent WASM demosaic instance. A singleton is managed by
the facade module `demosaic.ts` via `getPool()` / `destroyDemosaicPool()`. The
pool is lazily initialized: workers are only spawned on the first call to
`run()`, avoiding startup cost when the GPU path succeeds.

```
demosaic.ts  (facade, singleton pool)
  └─ DemosaicPool           (strip splitting, stitching)
       └─ demosaic-worker.ts (per-worker WASM init + dispatch)
```

### Worker Count

The pool size is determined at construction time:

```
this.size = Math.min(navigator.hardwareConcurrency ?? 4, MAX_WORKERS);
```

`MAX_WORKERS` is capped at **8**. Beyond 8 workers the overhead of strip
stitching, memory duplication, and contention on the memory bus outweighs the
parallelism gains for the image sizes camera raws typically produce. When
`navigator.hardwareConcurrency` is unavailable (some privacy-hardened browsers),
the fallback is 4.

### Strip Splitting

The image is divided into **horizontal strips** (full-width slices of rows)
rather than tiles. Each worker receives one strip of the CFA, demosaics it
independently, and returns a planar RGB result.

Key parameters:

| Constant             | Value | Purpose                                        |
|----------------------|-------|------------------------------------------------|
| `MIN_STRIP_HEIGHT`   | 128   | Minimum row count per strip; prevents over-splitting on small images |
| `STRIP_OVERLAP_FACTOR` | 3   | Multiplied by CFA period to compute overlap    |
| `MAX_WORKERS`        | 8     | Hard ceiling on concurrency                    |

The effective worker count for a given image is:

```
effectiveWorkers = min(pool.size, floor(height / MIN_STRIP_HEIGHT))
```

If the image is too small for meaningful splitting (effective workers <= 1), the
fast path dispatches the entire CFA to a single worker with no splitting or
stitching overhead.

### Overlap Calculation

Each strip extends beyond its "owned" row range by an overlap margin on both the
top and bottom edges:

```
stripOverlap = STRIP_OVERLAP_FACTOR * period
startRow = max(0, ownedStart - stripOverlap)
endRow   = min(height, ownedEnd + stripOverlap)
```

The overlap factor of 3 CFA periods is chosen to satisfy two constraints:

1. **CFA border artifacts.** Demosaic algorithms need neighborhood context. For
   Bayer sensors (period = 2) the overlap is 6 rows; for X-Trans (period = 6)
   the overlap is 18 rows. Multi-pass algorithms like Markesteijn 3-pass and AHD
   read up to 5 rows of context. Three full CFA periods guarantee that even the
   most aggressive interpolation window finds valid CFA data around every owned
   pixel row.

2. **CFA phase alignment.** Overlap rows must be a multiple of the CFA period so
   that the sub-image presented to the WASM demosaicer begins on a
   pattern-aligned boundary. Because `stripOverlap` is `3 * period`, subtracting
   it preserves alignment modulo the period.

Only the "inner" (non-overlap) rows from each strip's output are copied into the
final image, so overlap pixels are computed but discarded. This eliminates
boundary seam artifacts entirely at the cost of a small amount of redundant work.

### Bayer Variant Detection from CFA Shift

The Rust WASM module exposes two entry points: `demosaic_image` (X-Trans, takes
raw `dy`/`dx` shift) and `demosaic_bayer` (Bayer, takes a variant string like
`rggb`). When the strip start row shifts relative to the full image, the
effective Bayer variant changes. The pool computes per-strip phase:

```
stripDy = (dy + startRow) % period
```

The `bayerVariantForShift` function maps `(dy % 2, dx % 2)` to one of four
variant strings via a lookup table:

```
const BAYER_VARIANT_MAP = ['rggb', 'grbg', 'gbrg', 'bggr'] as const;
```

Index: `(dy % 2) * 2 + (dx % 2)`. The pool passes `isBayer` (derived as
`period === 2` in the facade) to decide which WASM entry point the worker
should call.

### Worker Message Protocol

Communication uses a simple two-message protocol over `postMessage`:

**Main -> Worker (request):**
```
{
  type: 'demosaic',
  cfa: ArrayBuffer,          // transferred, not copied
  width: number,
  height: number,
  dy: number,
  dx: number,
  algorithm: string,
  bayerVariant?: string,     // present only for Bayer sensors
}
```

**Worker -> Main (response):**
```
{ type: 'done', data: ArrayBuffer }   // transferred back
// or
{ type: 'error', message: string }
```

### Transferable Buffers

Both directions use the Transferable protocol to avoid copying large
`Float32Array` buffers:

- **Outbound:** `runOne` calls `stripCfa.slice()` to create a copy of the strip
  region (because `subarray` shares the underlying buffer and cannot be
  transferred independently), then transfers `cfaCopy.buffer` in the second
  argument to `postMessage`.

- **Inbound:** The worker transfers `result.buffer` back. The main thread wraps
  the received `ArrayBuffer` into a new `Float32Array`.

This means each pixel crosses the thread boundary exactly once in each direction
via zero-copy ownership transfer, keeping memory overhead proportional to the
image size rather than doubling it.

### Stitching Non-Overlapping Regions

After all workers resolve, the pool assembles the final planar RGB output
(`Float32Array` of length `3 * width * height`) by copying only the owned rows
from each strip:

```
for each channel c in [0, 1, 2]:
  srcOff = c * stripPixels + innerStart * width
  dstOff = c * npix + ownedStart * width
  output.set(result.subarray(srcOff, srcOff + innerRows * width), dstOff)
```

The output layout is planar (all R values, then all G, then all B), matching
what the downstream tone-mapping and rendering pipeline expects. The `innerStart`
/ `innerEnd` bookkeeping ensures that overlap rows on both sides of each strip
are silently discarded.

### WASM Initialization in Workers

Each worker lazily initializes its own WASM module on first message. The `init()`
call (generated by wasm-bindgen) downloads and compiles the `.wasm` binary once
per worker. A `ready` flag prevents re-initialization on subsequent messages.
Because workers persist for the pool's lifetime, this cost is paid only once even
if the user processes multiple files.

## Rationale

### Why Horizontal Strips, Not Tiles

Horizontal strips (full-width, partial-height) are the natural decomposition for
CFA demosaicing for three reasons:

1. **CFA periodicity is 2D but small.** Both Bayer (2x2) and X-Trans (6x6)
   patterns repeat in both axes. A full-width strip preserves the complete
   horizontal period, so the WASM algorithm can operate as if it received a
   shorter but complete image. Tiles would require overlap on all four edges,
   quadrupling boundary waste for small tiles.

2. **Row-major memory layout.** CFA data is stored in row-major order.
   `Float32Array.subarray()` can extract a contiguous horizontal strip in O(1),
   while extracting a tile would require a per-row copy loop.

3. **Simpler stitching.** Strips require copying contiguous row slices per
   channel plane. Tile stitching needs 2D offset arithmetic and produces more
   cache-unfriendly scatter writes.

### Why 3x CFA Period Overlap

The overlap must satisfy the widest interpolation kernel among all supported
algorithms. Markesteijn 3-pass uses a 5-row lookahead in its gradient estimation
step; AHD uses homogeneity comparison across a window of similar size. Setting
the overlap to `3 * period` (i.e. 6 rows for Bayer, 18 for X-Trans) provides
headroom above the worst-case kernel radius while keeping the overlap a multiple
of the CFA period for phase alignment. A smaller overlap (1x or 2x period) would
produce visible color fringing at strip boundaries for multi-pass algorithms.

### Why Cap at 8 Workers

Each worker holds its own WASM linear memory (typically 16-32 MB for a 24 MP
strip). At 8 workers this is 128-256 MB of WASM heap alone. Mobile browsers and
lower-end devices would start thrashing or hitting memory pressure at higher
counts. Additionally, strip granularity becomes counterproductive when strip
height approaches the CFA kernel radius, as the ratio of overlap rows to owned
rows grows.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/demosaic-pool.ts` | `DemosaicPool` class: strip splitting, overlap computation, stitching, transferable buffer management |
| `web/src/pipeline/demosaic-worker.ts` | Web Worker entry point: lazy WASM init, dispatch to `demosaic_bayer` or `demosaic_image` |
| `web/src/pipeline/demosaic.ts` | Facade: singleton pool lifecycle, GPU-vs-WASM dispatch (see [demosaic-algorithm-dispatch](demosaic-algorithm-dispatch.md)) |
| `web/src/pipeline/types.ts` | `DemosaicMethod` union type defining all algorithm names |
| `web/src/hooks/useProcessFile.ts` | Consumer: calls `runDemosaic()` and `destroyDemosaicPool()` on cleanup |

## Antipatterns

### Sharing a Single WASM Instance Across Workers

Web Workers do not share memory by default; each WASM instantiation gets its own
linear memory. Attempting to share a `WebAssembly.Memory` with
`shared: true` and `SharedArrayBuffer` would require the WASM module to be
compiled with threading support (e.g. Emscripten pthreads or wasm-bindgen
`--target web` with atomics), the server to send `Cross-Origin-Opener-Policy`
and `Cross-Origin-Embedder-Policy` headers, and the Rust demosaic code to be
thread-safe. The current architecture avoids this entire class of complexity by
giving each worker its own WASM instance and splitting work at the image level.

### Transferring Subarrays Directly

`Float32Array.subarray()` returns a view into the parent buffer. Passing
`subarray(...).buffer` to `postMessage` with a transfer list would transfer the
**entire** parent `ArrayBuffer`, neutering it for all other strips. The pool
correctly calls `.slice()` to create an independent copy before transfer. Failing
to do this would corrupt concurrent strip data and produce cryptic
"ArrayBuffer is detached" errors.

### Splitting Below MIN_STRIP_HEIGHT

Without the `MIN_STRIP_HEIGHT` guard, a small crop preview (e.g. 200x200) could
be split into 8 strips of 25 rows each. At 25 rows with 6-18 rows of overlap on
each side, the overlap-to-payload ratio becomes absurd and the WASM startup cost
per strip dominates total latency. The 128-row minimum ensures that parallelism
is only used when it provides a net speedup.
