---
title: IndexedDB Persistence and Session Restore
tags: [web, persistence, indexeddb, session]
scope: web/src/lib/idb-storage.ts, web/src/hooks/useInit.ts, web/src/store.ts, web/src/pipeline/types.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

x-veon is a fully client-side web application -- there is no server backend. All
user data (imported RAW files, processing results, application settings) must
survive page reloads, browser restarts, and tab crashes without any round-trip to
a remote service. The browser's IndexedDB API provides the structured persistence
layer for this requirement.

IndexedDB stores serializable metadata about every imported file and every
user-facing setting. It works in tandem with OPFS, which holds the large binary
payloads (raw sensor data and thumbnails). IndexedDB records are small (a few
KB each) and reference OPFS entries by file ID. The Zustand store (see
`zustand-state-management`) is the in-memory source of truth at runtime; IDB
is its on-disk shadow.

The system guarantees that a user who processes five RAW files, closes the tab,
and reopens it later will see the same file list, the same selected file, and
the same demosaic method. Files that were previously processed are restored with
their result metadata intact; re-processing is triggered on demand rather than
automatically at restore, since HWC pixel data is no longer persisted to OPFS.

## Pattern / Approach

### Two IDB Object Stores

The database `xtrans-demosaic` (version 1) contains two object stores:

| Store      | Key path | Purpose                                       |
|------------|----------|-----------------------------------------------|
| `files`    | `id`     | One record per imported file (PersistedFile)   |
| `settings` | `key`    | Key-value pairs for global application state   |

The `files` store holds `PersistedFile` records with fields that mirror the
in-memory `QueuedFile` type, minus non-serializable data (`File` object handle,
`progress`, blob URLs). `PersistedFile` additionally stores `lensProfile`,
`preProcessOverrides`, lens metadata (`lensModel`, `focalLength`, `fNumber`),
`fileSize`, and `addedAt` fields. The deprecated `cachedMethods` array is kept
for schema compatibility but is always empty.

The `settings` store holds five keys:

- `demosaicMethod` -- the active demosaic algorithm (`'neural-net'`, etc.)
- `modelSize` -- the neural-net model variant (`'S'`, `'M'`, `'L'`)
- `exportFormat` -- the chosen export codec (`'jpeg-hdr'`, etc.)
- `exportQuality` -- JPEG/AVIF quality slider value (number)
- `selectedFileId` -- which file is selected in the sidebar (string or null)

The database is opened lazily via `openDb()`, which caches the connection promise
in module-level state. If the connection fails, the promise is nulled so the next
call retries. Schema creation happens in the `onupgradeneeded` handler, which
creates both object stores on first visit.

### Serialization: serializeResultMeta / deserializeResultMeta

`ProcessingResultMeta` contains `Float32Array` fields (`xyzToCam`, `wbCoeffs`,
`camToXyz`) that IndexedDB can store via structured clone but that are fragile
across serialization boundaries. To keep persistence explicit and
forward-compatible, the store converts these to plain number arrays before
writing:

- **`serializeResultMeta`** (`web/src/pipeline/types.ts`): Converts
  `Float32Array` fields to `number[]` via `Array.from()`, producing a
  `SerializableResultMeta` that is a plain JSON-safe object.
- **`deserializeResultMeta`** (`web/src/pipeline/types.ts`): Reconstructs
  `Float32Array` instances from the stored arrays. Includes a fallback for the
  `camToXyz` field -- records persisted before this field was added get a 3x4
  identity matrix.

This conversion happens at the store-to-IDB boundary (`fileToPersistedFile` in
`store.ts` calls `serializeResultMeta`) and at the IDB-to-store boundary
(`persistedToQueued` in `useInit.ts` calls `deserializeResultMeta`).

### CRUD API

`idb-storage.ts` exports five functions:

| Function           | Store      | Operation | Notes                            |
|--------------------|------------|-----------|----------------------------------|
| `getAllFiles()`    | `files`    | getAll    | Returns all PersistedFile records |
| `putFile(file)`   | `files`    | put       | Upsert by `id`                   |
| `deleteFile(id)`  | `files`    | delete    | Remove by `id`                   |
| `getSetting<T>()` | `settings` | get       | Generic typed read by key         |
| `putSetting()`    | `settings` | put       | Upsert by `key`                  |

All functions are async, open a single transaction, and resolve/reject based on
the IDB request lifecycle. There is no batching or multi-record transaction API
-- each call is an independent transaction. This is intentional: writes are
fire-and-forget from the Zustand store's perspective, and IDB auto-commits
single-operation transactions efficiently.

### Debounced Writes

Slider-driven changes (look preset selection, OpenDRT color grading overrides)
can fire many state updates per second. Writing to IDB on every change would
create unnecessary transaction overhead.

`debouncedPutFile(file, delayMs = 300)` solves this with a per-file timer map
(`pendingTimers: Map<string, timeout>`). Each call clears any pending timer for
that file ID and schedules a new one. Only the last update within a 300ms window
actually hits IDB. The timer map is keyed by file ID so concurrent slider
adjustments on different files do not interfere.

In `store.ts`, two wrappers choose the appropriate strategy:

- `persistFile(f)` -- calls `putFile()` immediately. Used for discrete events:
  file added, processing completed, error recorded, lens profile matched,
  overrides reset.
- `persistFileDebounced(f)` -- calls `debouncedPutFile()`. Used for continuous
  adjustments: `setFileLookPreset`, `setFileOpenDrtOverride`,
  `setFilePreProcessOverride`.

Settings writes (`putSetting`) are not debounced because they are triggered by
discrete user actions (switching demosaic method, changing export format).

### Session Restore Flow in useInit

`useInit` (`web/src/hooks/useInit.ts`) runs once on app mount inside a
`useEffect`. It orchestrates the full session restore sequence:

```
1. Promise.all (parallel):
   - initWasm()             -- load RAF decoder WASM module
   - initModels()           -- load ONNX neural network models
   - initDemosaicGpuSafe()  -- initialize GPU demosaic pipeline
   - getAllFiles()           -- read all PersistedFile records from IDB
   - getSetting(...)  x4    -- read demosaicMethod, exportFormat,
                                exportQuality, selectedFileId

2. For each persisted file:
   a. deserializeResultMeta() to reconstruct ProcessingResultMeta
   b. readThumbnail() from OPFS to restore sidebar previews
   c. If status == 'done' && resultMeta exists: keep as 'done'
   d. Otherwise: status becomes 'queued' (file will be re-processed)

3. restoreFromDb(files, settings)
   - Writes the validated file list and settings into the Zustand store
     in a single atomic `set()` call
   - Also restores modelSize from the selected file's result metadata

4. Share ORT's WebGPU device with the renderer for zero-copy buffer interop

5. Probe HDR display capabilities

6. setInitialized(backend)
   - Marks the app as ready, UI renders

7. Match lenses for restored files with metadata but no profile yet

8. cleanupOrphans(knownIds) -- fire-and-forget (see below)

9. navigator.storage.persist() -- best-effort request for durable storage
```

The `cancelled` flag guards against React strict-mode double-invocation: if the
effect cleanup runs before `init()` completes, no state is written.

Every IDB read in step 1 has a `.catch(() => defaultValue)` fallback so that a
corrupted or empty database does not block initialization. The app starts in a
clean state rather than crashing.

### Orphan Cleanup

Two storage systems (IDB and OPFS) can drift out of sync. This happens when:

- The user deletes a file but the tab crashes before OPFS cleanup completes.
- A bug writes to OPFS but fails to create the corresponding IDB record.
- The user manually clears IDB from DevTools but not OPFS (or vice versa).

The `cleanupOrphans` function in `useInit.ts` handles **OPFS orphans** (OPFS
entries with no matching IDB record):

1. `listRawFileIds()` enumerates all file IDs present in the OPFS `raw/`
   directory.
2. Any ID not in the `knownIds` set (built from IDB records that survived
   validation) is cleaned up via `deleteAllForFile(id)`, which removes all OPFS
   entries (raw and thumbnail) for that file.
3. Errors are silently caught -- orphan cleanup is non-critical and must not
   block the app.

Note: HWC pixel buffers are no longer stored in OPFS. The pipeline re-processes
files on demand or uses GPU buffers directly. The orphan cleanup therefore only
needs to check the `raw/` directory.

### Write Points in the Zustand Store

The store (`web/src/store.ts`) calls IDB at these points:

| Action                     | IDB call              | Timing     |
|----------------------------|-----------------------|------------|
| `addFiles`                 | `putFile`, `putSetting('selectedFileId')` | Immediate |
| `removeFile`               | `deleteFile`, `putSetting('selectedFileId')` | Immediate |
| `selectFile`               | `putSetting('selectedFileId')` | Immediate |
| `updateFileStatus` (error) | `putFile`             | Immediate  |
| `setFileResult`            | `putFile`             | Immediate  |
| `setFileLensProfile`       | `putFile`             | Immediate  |
| `setFileLookPreset`        | `debouncedPutFile`    | Debounced  |
| `setFileOpenDrtOverride`   | `debouncedPutFile`    | Debounced  |
| `resetFileOpenDrtOverrides`| `putFile`             | Immediate  |
| `setFilePreProcessOverride`| `debouncedPutFile`    | Debounced  |
| `resetFilePreProcessOverrides`| `putFile`          | Immediate  |
| `setModelSize`             | `putSetting`          | Immediate  |
| `setDemosaicMethod`        | `putSetting`          | Immediate  |
| `setExportFormat`          | `putSetting`          | Immediate  |
| `setExportQuality`         | `putSetting`          | Immediate  |

Note that `updateFileStatus` only persists on `'error'` status. The `'processing'`
status is transient and never written to IDB. If persisted, a `'processing'`
status would be ambiguous on restore (was the process interrupted?). Instead,
`fileToPersistedFile` maps `'processing'` to `'queued'` as a safety net.

## Rationale

### Why Two Object Stores Instead of One

File metadata and application settings have different access patterns. On
restore, all files are loaded with `getAll()` while settings are loaded by
individual key lookups. During runtime, file records are updated frequently
(every processing completion, every slider change) while settings change rarely.
Separate stores keep transactions scoped to the relevant data, avoiding
unnecessary locking between file writes and setting writes.

A single `settings` record containing the entire app state was considered but
rejected because it would require read-modify-write for every change, creating
a serialization bottleneck during rapid slider interactions.

### Why Debounce Instead of Throttle

Debouncing (write only after N ms of silence) was chosen over throttling (write
at most once per N ms) because the final slider value is the only one that
matters for persistence. Intermediate values are visible in the Zustand store and
the rendered canvas but have no archival value. A 300ms debounce means the IDB
write happens at most ~333ms after the user lifts their finger, which is
imperceptible.

### Why Orphan Cleanup Runs on Every Startup

OPFS storage is quota-managed by the browser and counts against the origin's
storage budget. Orphaned RAW files can consume significant space. Running
cleanup on every startup keeps storage hygiene automatic. The cost is a single
directory enumeration (fast, O(n) in file count) and is fire-and-forget so it
never blocks rendering.

## Key Files

| File | Role |
|------|------|
| `web/src/lib/idb-storage.ts` | IDB connection management, CRUD for files and settings, debounced write |
| `web/src/hooks/useInit.ts` | Session restore orchestrator: load IDB, validate OPFS, rebuild store, orphan cleanup |
| `web/src/store.ts` | Zustand store with persistence side-effects on every mutation |
| `web/src/pipeline/types.ts` | `serializeResultMeta` / `deserializeResultMeta` for Float32Array round-tripping |
| `web/src/lib/opfs-storage.ts` | OPFS operations referenced during restore validation and orphan cleanup |

## Antipatterns

**Do not use Zustand's built-in `persist` middleware for this store.** The
`persist` middleware serializes the entire store to a single IDB/localStorage
entry on every state change. This is unsuitable for x-veon because:

1. The store contains non-serializable values (`File` objects, `HTMLCanvasElement`
   refs, `HdrRenderer` instances) that would need complex partitioning or custom
   serializers.
2. A single serialized blob means every slider tick writes the full state,
   defeating the per-file debounce strategy.
3. Restore logic needs to cross-validate against OPFS (checking whether HWC
   buffers exist), which requires async orchestration that the middleware does
   not support.

The current approach -- explicit `putFile`/`putSetting` calls at each mutation
site -- is more verbose but gives precise control over what is persisted, when,
and with what error handling.

**Do not persist `'processing'` status to IDB.** A file recorded as
`'processing'` in IDB has ambiguous semantics on restore: was it mid-inference
when the tab closed, or did it complete but fail to write? The
`fileToPersistedFile` function maps `'processing'` to `'queued'` so that
interrupted files are automatically re-processed on the next session.

**Do not assume `done` files have cached pixel data.** HWC pixel buffers are no
longer stored in OPFS. Restored `done` files have metadata but no cached pixels.
The pipeline re-processes files on demand when display is needed, or uses
GPU-resident buffers during a session. Do not add code that assumes OPFS
contains HWC data.
