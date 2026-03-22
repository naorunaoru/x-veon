---
title: RAW File Decoding via WebAssembly
tags: [web, wasm, decoding, raw]
scope: web/src/pipeline, web/wasm/rawloader
generated: 2026-03-22
commit: 347c8dd
---

# RAW File Decoding via WebAssembly

RAW image decoding, WASM module initialization, RawImage metadata extraction,
and embedded thumbnail retrieval in the x-veon browser pipeline.

## Context

x-veon performs neural-network demosaicing entirely in the browser. The first
step of the pipeline is decoding proprietary camera RAW files (RAF, ARW, CR2,
NEF, DNG, RW2, ORF, PEF, and others) into linear sensor data with associated
metadata. This decoding must happen client-side because no server is involved --
the app is deployed as a static site on GitHub Pages.

RAW formats are binary, undocumented, and vendor-specific. Re-implementing
parsers in TypeScript would be impractical and error-prone. Instead, x-veon
compiles the Rust `rawloader` crate to WebAssembly and calls it from the
TypeScript pipeline. This gives the browser access to a mature, well-tested
RAW decoder with support for hundreds of camera models.

The decoder produces a `RawImage` struct that carries everything the downstream
pipeline needs: the raw Bayer/X-Trans mosaic as `Uint16Array` pixel data, white
balance coefficients, black/white levels, CFA pattern description, color
matrices, crop rectangles, orientation, and Fujifilm dynamic-range gain.

## Pattern / Approach

### WASM Module Architecture

The decoding WASM module lives at `web/wasm/rawloader/`. It is a Rust crate
(`rawloader-wasm`) that wraps the external `rawloader` library with
`wasm-bindgen` bindings. The crate compiles to `cdylib` for WASM and uses
`wasm-pack` with `--target web` to produce ES module output in
`web/wasm/rawloader/pkg/`.

Key Rust dependencies in `web/wasm/rawloader/Cargo.toml`:

- `rawloader` -- the actual RAW format parser (path dependency to a fork at
  `naorunaoru/rawloader`)
- `wasm-bindgen` / `js-sys` -- JS interop layer
- `kamadak-exif` -- EXIF parsing for exposure bias and Fujifilm DR tags
- `console_error_panic_hook` -- converts Rust panics to readable JS errors

The release profile uses `opt-level = "s"` and LTO for small binary size.

### Build Pipeline

WASM is built before the Vite bundle. Three separate WASM modules exist in the
project (decoder, encoder, demosaic); each is built independently:

```
npm run build:wasm:decoder
  -> wasm-pack build wasm/rawloader --target web --release
```

The `setup` script chains everything:
`rustup target add wasm32-unknown-unknown && npm install && npm run build:wasm`.

In CI (`.github/workflows/deploy.yml`), the workflow:

1. Checks out the `naorunaoru/rawloader` fork into `rawloader-wasm/`
2. Creates a symlink so the Cargo path dependency resolves
3. Installs the `wasm32-unknown-unknown` Rust target
4. Runs `npx wasm-pack build wasm/rawloader --target web --release`
5. Builds the remaining WASM modules (demosaic, encoder)
6. Runs `npx vite build`

Vite integrates WASM via the `vite-plugin-wasm` plugin, which handles the
`.wasm` binary import and instantiation at bundle time.

### WASM Initialization (Lazy, One-Shot)

The TypeScript entry point is `web/src/pipeline/raf-decoder.ts`. It holds a
module-level nullable reference to the loaded WASM:

```ts
let wasmModule: Awaited<typeof import('../../wasm/rawloader/pkg/rawloader_wasm.js')> | null = null;

export async function initWasm(): Promise<void> {
  wasmModule = await import('../../wasm/rawloader/pkg/rawloader_wasm.js');
  await wasmModule.default();
}
```

`initWasm()` is called once during application startup in `useInit.ts`. It runs
in parallel with model loading and GPU demosaic initialization via
`Promise.all`:

```ts
await Promise.all([
  initWasm(),
  initModels(),
  initDemosaicGpuSafe(),
  // ... IndexedDB restoration
]);
```

After `initWasm()` resolves, the `wasmModule` reference is non-null for the
lifetime of the page. The `decodeRaw` function guards against premature calls
with a synchronous null check.

### decodeRaw Flow

`decodeRaw(arrayBuffer: ArrayBuffer): RawImage` is the main decode entry point.
It is synchronous from the caller's perspective (the WASM call itself blocks).

The flow:

1. **Guard**: throws if WASM is not initialized.
2. **Copy to WASM memory**: wraps the `ArrayBuffer` in a `Uint8Array` and passes
   it to the Rust side.
3. **Rust decode**: `decode_image()` in `lib.rs` calls
   `rawloader::decode_file_vec(&vec)` which identifies the format, parses the
   container, and extracts the mosaic data.
4. **EXIF extraction**: the Rust side also calls `exif_parse::extract_dr_gain()`
   and `exif_parse::extract_exposure_bias()` on the raw bytes to pull Fujifilm
   dynamic range settings and exposure compensation. These use the `kamadak-exif`
   crate, with a manual Fuji makernote fallback parser for DR tags.
5. **Data marshalling**: the Rust `Image` struct exposes typed-array getters
   (`get_data()`, `get_wb_coeffs()`, etc.) via `#[wasm_bindgen]`. The
   TypeScript side calls each getter and assembles a plain `RawImage` object.
6. **Memory release**: once the `RawImage` is constructed, the caller sets
   `arrayBuffer = null` to allow GC of the original file bytes.

The Rust `decode_image` function rejects float-format raw data (only integer
sensor data is supported) and maps `rawloader::Orientation` variants to string
constants.

### RawImage Structure

Defined in `web/src/pipeline/types.ts`:

```ts
export interface RawImage {
  data: Uint16Array;         // raw mosaic pixel data (full sensor)
  width: number;             // sensor width in pixels
  height: number;            // sensor height in pixels
  wbCoeffs: Float32Array;    // white balance multipliers [R, G, B]
  blackLevels: Uint16Array;  // per-channel black level
  whiteLevels: Uint16Array;  // per-channel saturation level
  xyzToCam: Float32Array;    // 3x3 color matrix (XYZ -> camera RGB)
  camToXyz: Float32Array;    // 3x4 inverse color matrix
  orientation: string;       // EXIF orientation as string enum
  make: string;              // camera manufacturer
  model: string;             // camera model
  cfaStr: string;            // CFA pattern string (e.g., "RGGBRGGB...")
  cfaWidth: number;          // CFA pattern repeat width
  crops: Uint16Array;        // [top, right, bottom, left] active area
  drGain: number;            // Fuji DR gain (1.0, 2.0, or 4.0)
  exposureBias: number;      // EXIF exposure compensation in EV
}
```

Notable design decisions:

- **Typed arrays throughout**: `Uint16Array` for 14/16-bit sensor data and
  levels, `Float32Array` for matrices and WB coefficients. This avoids
  conversion overhead and matches downstream GPU/WASM consumers.
- **CFA as string + width**: the CFA pattern is a flat string of characters
  (R/G/B) with a separate width. For X-Trans sensors this is a 6-wide string
  of length 36; for Bayer it is 2-wide with length 4. The
  `findPatternShift()` function in the preprocessor interprets this.
- **Crops as TRBL**: top/right/bottom/left ordering, which `cropToVisible()`
  uses to extract the active sensor area.
- **drGain and exposureBias**: Fuji-specific fields. `drGain` compensates for
  deliberate underexposure in DR200/DR400 modes. The processing pipeline
  multiplies the final linear RGB by this factor after color correction.

### Thumbnail Extraction (Pure TypeScript, No WASM)

Thumbnail extraction is handled separately in `web/src/pipeline/raf-thumbnail.ts`
and does not use WASM. This is intentional: thumbnails are needed immediately
when files are dropped into the queue, before WASM initialization may have
completed, and before the expensive full decode.

Two public functions:

- `extractRafThumbnail(buffer: ArrayBuffer): Blob | null` -- returns the
  embedded JPEG preview.
- `extractRafQuickMetadata(buffer: ArrayBuffer): { camera: string } | null` --
  returns the camera make/model string.

Both try format-specific extraction first, then fall back to generic TIFF/EXIF
parsing:

**Fujifilm RAF path**: reads the JPEG offset/length from fixed header positions
(0x54 and 0x58), validates the JPEG SOI marker (0xFFD8), and slices the
`ArrayBuffer` into a `Blob`. Camera name is at offset 0x1C (32 bytes, null-
terminated).

**TIFF-based path** (ARW, CR2, NEF, DNG, RW2, ORF, PEF): detects endianness
from the TIFF byte-order marker (`II` or `MM`), then walks the IFD chain
(up to 10 IFDs) looking for:

- `JPEGInterchangeFormat` (0x0201) / `JPEGInterchangeFormatLength` (0x0202) for
  embedded JPEG thumbnails
- `SubIFD` (0x014A) entries with `NewSubfileType` bit 0 set (reduced-resolution
  images), which often contain larger JPEG previews
- `StripOffsets` (0x0111) / `StripByteCounts` (0x0117) as an alternative JPEG
  source in SubIFDs
- `Make` (0x010F) / `Model` (0x0110) for metadata

The parser selects the largest available JPEG across all IFDs and SubIFDs,
validates the SOI marker, and returns it as a `Blob`.

### Data Flow Summary

```
File drop / OPFS restore
        |
        v
  ArrayBuffer
        |
        +---> extractRafThumbnail()   [TypeScript, immediate]
        |          |
        |          v
        |     Blob (JPEG preview) --> <img> in queue UI
        |
        +---> decodeRaw()             [WASM, after initWasm()]
                   |
                   v
              RawImage {data, wbCoeffs, blackLevels, ...}
                   |
                   v
             cropToVisible -> normalizeRawCfa -> WB -> HL reconstruct
                   |
                   v
             padToAlignment -> demosaic (NN or traditional)
                   |
                   v
             color correction -> export
```

## Rationale

### Why Rust/WASM for Decoding

- **Ecosystem reuse**: the `rawloader` crate already handles the binary parsing
  for 700+ camera models. Porting this logic to JavaScript would be a
  multi-year effort and a maintenance burden.
- **Performance**: RAW files are 20-60 MB. Rust compiled to WASM decodes a
  50 MB RAF file in roughly the same time as native, and significantly faster
  than a pure JS implementation would manage for the byte-level parsing.
- **Correctness**: RAW format quirks (endianness, vendor extensions, variable
  CFA layouts) are already battle-tested in the Rust crate.

### Why Lazy Initialization

The WASM module is loaded via dynamic `import()` rather than a static top-level
import. This has two benefits:

1. **Non-blocking startup**: the WASM binary (~400 KB gzipped) is fetched and
   compiled asynchronously. The UI shell renders immediately.
2. **Parallel init**: `initWasm()` runs concurrently with ONNX model loading
   and IndexedDB restoration, which together dominate startup time.

### Why Thumbnails Skip WASM

Thumbnail extraction uses pure TypeScript for two reasons:

1. **Independence from init order**: thumbnails must display instantly when
   files are queued, potentially before WASM has finished loading.
2. **Minimal scope**: extracting a JPEG blob from a known offset is trivial
   binary parsing. Pulling in the full WASM decoder for this would be wasteful.

### Why Synchronous decodeRaw

`decodeRaw` is synchronous (not async) because `wasm-bindgen` functions called
from the main thread are inherently synchronous. The caller (`useProcessFile`)
is already in an async context, so the blocking WASM call does not affect the
API shape. Moving decoding to a Web Worker is a possible future optimization
but is not currently necessary because the decode time (~100-300 ms) is small
relative to the neural-network inference that follows.

## Key Files

| File | Role |
|------|------|
| `web/src/pipeline/raf-decoder.ts` | WASM init + `decodeRaw()` entry point |
| `web/src/pipeline/raf-thumbnail.ts` | Pure-TS thumbnail and quick metadata extraction |
| `web/src/pipeline/types.ts` | `RawImage` interface and related types |
| `web/wasm/rawloader/src/lib.rs` | Rust WASM bindings: `decode_image()`, `Image` struct |
| `web/wasm/rawloader/src/exif_parse.rs` | EXIF parsing for DR gain and exposure bias |
| `web/wasm/rawloader/Cargo.toml` | Crate config, dependencies, release profile |
| `web/src/hooks/useInit.ts` | App startup: parallel `initWasm()` call |
| `web/src/hooks/useProcessFile.ts` | Processing pipeline: calls `decodeRaw()` at step 1 |
| `web/vite.config.ts` | Vite config with `vite-plugin-wasm` |
| `web/package.json` | `build:wasm:decoder` script |
| `.github/workflows/deploy.yml` | CI: rawloader checkout, symlink, wasm-pack build |

## Antipatterns

### Do not call decodeRaw before initWasm resolves

`decodeRaw` throws synchronously if the WASM module is null. There is no
retry or queuing mechanism. Always ensure the `useInit` hook has completed
(check `initialized` in the store) before triggering file processing.

### Do not use WASM for thumbnail extraction

The thumbnail extraction code intentionally avoids WASM. Adding a WASM
dependency to the thumbnail path would create a circular init problem: the
queue UI needs thumbnails before WASM is ready. Keep `raf-thumbnail.ts`
self-contained with only `DataView` / `TextDecoder` APIs.

### Do not hold the full ArrayBuffer after decoding

The processing pipeline explicitly nulls the `arrayBuffer` reference after
`decodeRaw()` returns. RAW files are large (20-60 MB) and the decoded
`RawImage.data` is similarly large. Retaining both simultaneously doubles peak
memory consumption. Always release the source buffer as soon as the `RawImage`
is constructed.

### Do not add float RAW data support without updating the Rust side

The Rust `decode_image` function explicitly rejects `RawImageData::Float` and
returns an error. If a camera produces float data, both the Rust marshalling
and the TypeScript `RawImage` type (`Uint16Array` data field) would need
changes.

### Do not skip the rawloader fork symlink in CI

The `rawloader` path dependency in `Cargo.toml` points to
`../../../../rawloader-wasm`, which CI resolves via a symlink from
`$GITHUB_WORKSPACE/../rawloader-wasm`. Removing or reordering the checkout
and symlink steps in the deploy workflow will cause the WASM build to fail
with an unresolved path dependency.
