---
title: OPFS Storage and GPU Buffer Handoff
tags: [web, storage, opfs, gpu-handoff]
scope: web/src/lib/opfs-storage.ts, web/src/lib/hwc-handoff.ts, web/src/store.ts, web/src/hooks/useProcessFile.ts, web/src/hooks/useInit.ts, web/src/components/OutputCanvas.tsx
generated: 2026-03-22
commit: 347c8dd
---

## Context

The x-veon web app processes camera RAW files into full-color demosaiced images. RAW input files are 50-100 MB each. The app uses the browser's Origin Private File System (OPFS) to persist original RAW files and thumbnails across page reloads, keeping large binary data off the JS heap. OPFS provides a sandboxed virtual filesystem accessible via `navigator.storage.getDirectory()`.

Demosaiced pixel results (previously cached to OPFS with byte-shuffle + gzip compression) are no longer stored in OPFS. Reprocessing from the cached RAW file is now faster than reading, decompressing, and deserializing a ~288 MB HWC buffer from disk. Instead, the processing pipeline hands off a GPU-resident buffer directly to the renderer via a single-slot GPU buffer handoff (`web/src/lib/hwc-handoff.ts`), avoiding any CPU readback.

The OPFS storage layer lives in `web/src/lib/opfs-storage.ts`, which exposes typed read/write/delete functions for two categories of data: original RAW files and thumbnails.

## OPFS Directory Structure

Two subdirectories are created lazily under the OPFS root:

| Directory      | Contents                          | Key format              |
|----------------|-----------------------------------|-------------------------|
| `raw/`         | Original RAW file bytes (uncompressed) | `{fileId}`          |
| `thumbnails/`  | JPEG thumbnail blobs from embedded RAW previews | `{fileId}` |

Directory handles are cached as module-level singletons (`rawDir`, `thumbDir`) and initialized on first access via `getRawDir()` and `getThumbDir()`.

## GPU Buffer Handoff (`hwc-handoff.ts`)

Demosaiced pixel results are no longer persisted to OPFS. Instead, the processing pipeline produces a GPU-resident RGBA32F `GPUBuffer` (via `postprocess-gpu.ts`) and passes it directly to the display renderer through a single-slot handoff in `web/src/lib/hwc-handoff.ts`.

### Interface

```typescript
export interface GpuHandoff {
  buffer: GPUBuffer;
  bytesPerRow: number;
}
```

### API

- **`setGpuResult(key, handoff)`** -- Called by `useProcessFile` after GPU postprocessing completes. Stores the `GPUBuffer` and its row stride. If a previous unclaimed buffer exists for a different key, it is destroyed to prevent GPU memory leaks.
- **`takeGpuResult(key)`** -- Called by `OutputCanvas` to claim the buffer. Returns the `GpuHandoff` and **clears the slot** (consumed on first read). Returns `null` if the key does not match. The consumed buffer is then uploaded to the renderer's image texture via `uploadImageFromBuffer()` (zero-copy GPU-to-GPU copy, no CPU readback).

### Design constraints

- The slot holds **at most one buffer** at a time. Each new `setGpuResult` call destroys any unclaimed previous buffer.
- Unlike the old OPFS handoff cache, `takeGpuResult` **consumes** the buffer on read (the slot is nulled). This is safe because the renderer's `uploadImageFromBuffer` copies the data into a texture and then destroys the buffer, so there is no need for a second read.
- The handoff is a module-level variable, not React state, so it is accessible from both the processing hook and the rendering component without prop drilling.

### Flow

```
useProcessFile                          OutputCanvas
     |                                       |
     | gpuPostprocess(...)                   |
     |   -> returns {buffer, bytesPerRow}    |
     | setGpuResult(fileId, gpuResult)       |
     |                                       |
     | setFileResult(...)                    |
     |   -> Zustand state update             |
     |                     triggers -------> |
     |                                       | takeGpuResult(fileId)
     |                                       |   -> returns GPUBuffer
     |                                       | uploadImageFromBuffer(buf, w, h, bpr)
     |                                       |   (zero-copy GPU buffer → texture)
```

## RAW File Caching

Original RAW file bytes are stored uncompressed in the `raw/` directory. This serves session persistence: when the user reloads the page, the `File` object from the original drag-and-drop is gone, but the RAW bytes remain in OPFS.

- **Write**: `store.addFiles()` calls `writeRaw(entry.id, buf)` fire-and-forget after reading the `File` into an `ArrayBuffer`.
- **Read**: `useProcessFile` checks `fileEntry.file` first (available for fresh drops), then falls back to `readRaw(fileEntry.id)` for restored sessions.
- **Delete**: `deleteRawForFile(fileId)` removes the single file from `raw/`.

RAW files are not compressed because they are already efficiently packed (16-bit sensor data) and the write must complete quickly to avoid blocking the processing pipeline.

## Thumbnail Storage

Embedded JPEG thumbnails extracted from RAW files are stored as `Blob`s in the `thumbnails/` directory. These are small (typically 100-300 KB) and used for the file sidebar.

- **Write**: `store.addFiles()` calls `extractRafThumbnail(buf)` and then `writeThumbnail(entry.id, thumbBlob)`.
- **Read**: `useInit` calls `readThumbnail(p.id)` during session restore and creates an object URL for display.
- **Delete**: `deleteThumbnailForFile(fileId)` removes the thumbnail blob.

## Cleanup

### Per-File Deletion

`deleteAllForFile(fileId)` runs two deletions in parallel via `Promise.all`:

1. `deleteRawForFile(fileId)` -- single file in `raw/`
2. `deleteThumbnailForFile(fileId)` -- single file in `thumbnails/`

Called from `store.removeFile()` when the user deletes a file from the queue.

### Orphan Cleanup

On app initialization, `useInit` runs `cleanupOrphans(knownIds)` as a fire-and-forget operation. It:

1. Lists all file IDs present in OPFS via `listRawFileIds()`.
2. For any ID not in the set of IDB-persisted file IDs, calls `deleteAllForFile(id)`.

This handles the case where the user's browser cleared IndexedDB but not OPFS, or where a crash interrupted cleanup.

### Session Restore

During `useInit`, files restored from IndexedDB with status `"done"` and valid `resultMeta` are restored directly. Since HWC pixel data is no longer cached in OPFS, there is no HWC existence check. When the user selects a restored file, the app reprocesses from the cached RAW bytes (which remain in OPFS) to regenerate the display buffer.

## Persistent Storage Request

After initialization, the app calls `navigator.storage.persist()` (best-effort) to request that the browser not evict OPFS data under storage pressure. This is non-blocking and may be denied by the browser.

## Rationale

### OPFS over IndexedDB for Large Buffers

IndexedDB can store `ArrayBuffer` and `Blob` values, but has significant drawbacks for large data:

- **Serialization overhead**: IDB's structured clone algorithm copies the entire buffer during writes. For a 50-100 MB RAW file, this doubles peak memory usage during the write transaction.
- **Transaction blocking**: Large IDB writes can block the main thread or cause transaction timeouts.
- **Read amplification**: IDB reads deserialize the entire value into a new JS heap allocation. There is no streaming or partial read API.

OPFS avoids these issues. `createWritable()` streams data to disk without a structured clone. File reads return `File` objects whose `arrayBuffer()` method maps from disk. The browser manages the underlying filesystem, and data does not transit through IDB's transactional layer.

The project uses a **split storage strategy**: lightweight metadata (file names, processing status, settings, serialized result metadata) goes to IndexedDB via `web/src/lib/idb-storage.ts`, while large binary data (RAW files, thumbnails) goes to OPFS. This keeps IDB transactions small and fast.

### Why GPU Buffer Handoff Replaced OPFS HWC Caching

The previous approach cached demosaiced HWC results to OPFS using byte-shuffle + gzip compression (3-6x compression, 50-90 MB on disk per image). This was removed because:

1. **Reprocessing is faster than decompressing.** With the GPU-resident neural-net pipeline (batched inference + GPU postprocessing), regenerating the display buffer from cached RAW bytes takes less time than reading 50-90 MB from OPFS + gzip decompression + byte-unshuffle.
2. **GPU buffer handoff is zero-copy.** The `GPUBuffer` from `gpuPostprocess` is passed directly to the renderer's `uploadImageFromBuffer`, which copies buffer-to-texture on the GPU without any CPU readback. This eliminates the 288 MB CPU-side Float32Array allocation that OPFS caching required.
3. **Simpler state management.** No need to track HWC cache validity, validate existence on session restore, or manage the `hwc-cache/` directory.

## Key Files

| File | Role |
|------|------|
| `web/src/lib/opfs-storage.ts` | OPFS read/write/delete for RAW files and thumbnails |
| `web/src/lib/hwc-handoff.ts` | Single-slot GPU buffer handoff (`setGpuResult` / `takeGpuResult`) |
| `web/src/store.ts` | Zustand store; calls `writeRaw`, `writeThumbnail`, `deleteAllForFile` |
| `web/src/hooks/useProcessFile.ts` | Processing pipeline; calls `readRaw`, `setGpuResult` |
| `web/src/hooks/useInit.ts` | Session restore; runs orphan cleanup |
| `web/src/components/OutputCanvas.tsx` | Rendering; calls `takeGpuResult` to get GPU buffer, then `uploadImageFromBuffer` |
| `web/src/pipeline/postprocess-gpu.ts` | GPU postprocessing; produces the `GPUBuffer` + `bytesPerRow` handed off |
| `web/src/pipeline/types.ts` | Defines `ProcessingResultMeta` (lightweight metadata stored in Zustand, not OPFS) |
| `web/src/lib/idb-storage.ts` | IndexedDB layer for metadata (complementary to OPFS for binary data) |

## Antipatterns

**Do not store HWC Float32Arrays in Zustand or React state.** A single demosaiced image is ~288 MB. The `ProcessingResultMeta` type exists specifically as the lightweight in-memory proxy: it carries dimensions, orientation, color matrices, and metadata, but explicitly omits the pixel buffer. The GPU buffer handoff ensures pixel data stays on the GPU and never materializes as a JS heap allocation.

**Do not cache demosaiced results to OPFS.** This was the previous approach (byte-shuffle + gzip to `hwc-cache/`). With the current GPU-resident pipeline, reprocessing from cached RAW bytes is faster than OPFS read + decompress, and the GPU buffer handoff avoids the 288 MB CPU allocation entirely.

**Do not compress RAW files in OPFS.** RAW sensor data is 12-14 bit packed or 16-bit integer. It does not benefit meaningfully from compression, and the write must complete quickly to avoid blocking the processing pipeline.

**Do not use synchronous OPFS access handles on the main thread.** The `createSyncAccessHandle()` API is only available in Web Workers and would block the thread. The current implementation uses async `createWritable()` and `getFile().arrayBuffer()` exclusively, keeping the main thread responsive.
