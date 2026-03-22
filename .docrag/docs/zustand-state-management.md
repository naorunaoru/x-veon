---
title: Zustand Store and Per-File Grading State
tags: [web, state, zustand, ui]
scope: web/src/store.ts, web/src/lib/idb-storage.ts, web/src/components/GradingPanel.tsx
generated: 2026-03-22
commit: 347c8dd
---

## Context

x-veon is a browser-based RAW image processor that demosaics camera sensor data
using neural networks and traditional algorithms. Users can drop multiple RAW
files, process them independently, switch between files, and apply per-file
color grading adjustments -- all within a single-page application that persists
state across sessions via IndexedDB and OPFS.

The core challenge is managing heterogeneous per-file state (processing
lifecycle, result metadata, lens profiles, OpenDRT color-science overrides,
pre-processing overrides) alongside global application settings (model size,
demosaic method, export format, HDR display mode). State must survive page reloads, handle concurrent
async operations (OPFS reads, inference, thumbnail extraction), and drive
efficient re-renders in a WebGPU rendering pipeline where grading slider changes
must update GPU uniforms on every frame without triggering React reconciliation
of unrelated components.

## Pattern / Approach

### Single Zustand Store

All application state lives in a single `create<AppState>()` store exported as
`useAppStore` from `web/src/store.ts`. The store is divided into four logical
regions:

1. **Initialization** -- `initialized`, `initError`, `backend`
2. **File queue** -- `files: QueuedFile[]`, `selectedFileId`
3. **Global processing/export settings** -- `modelSize`, `demosaicMethod`,
   `exportFormat`, `exportQuality`, `displayHdr`, `displayHdrHeadroom`
4. **UI state** -- `showClipMask`, `hdrPermissionNeeded`
5. **Transient refs** -- `canvasRef`, `rendererRef` (mutable handles to the
   WebGPU canvas and `HdrRenderer` instance, stored in Zustand so hooks like
   `useExport` can read them without prop drilling)

### QueuedFile Shape

Each file in the queue is represented by the `QueuedFile` interface:

```ts
interface QueuedFile {
  id: string;                // crypto.randomUUID()
  file: File | null;         // null after restore from IDB (raw bytes in OPFS)
  name: string;              // display name (extension stripped)
  originalName: string;      // original filename with extension
  thumbnailUrl: string | null;
  metadata: QuickMetadata | null;  // { camera, lensModel, focalLength, fNumber }
  cfaType: CfaType | null;  // 'xtrans' | 'bayer'
  status: FileStatus;
  error: string | null;
  progress: { current: number; total: number } | null;
  result: ProcessingResultMeta | null;
  resultMethod: DemosaicMethod | null;
  lensProfile: LensProfile | null;
  lookPreset: LookPreset;
  openDrtOverrides: Partial<OpenDrtConfig>;
  preProcessOverrides: Partial<PreProcessConfig>;
}
```

Key design decisions:

- `file: File | null` -- On first drop the browser `File` handle is held for
  immediate processing. After persistence, restored files have `file: null`
  because the RAW bytes live in OPFS; the processing hook reads from OPFS when
  `file` is null.
- `lensProfile` -- Matched LensFun lens correction profile (distortion, TCA,
  vignetting coefficients). Automatically populated from camera/lens metadata
  via `matchLens()` after file add or processing.
- `preProcessOverrides` -- Per-file sparse `Partial<PreProcessConfig>` for
  exposure, white balance, and sharpening adjustments applied before the
  OpenDRT tone mapping pipeline.
- `lookPreset` and `openDrtOverrides` -- Per-file grading state (see below).

### FileStatus Lifecycle

```
queued --> processing --> done
  |           |
  |           +--> error
```

`FileStatus` is the union `'queued' | 'processing' | 'done' | 'error'`.

Transitions:

| Trigger | Action | Persistence |
|---------|--------|-------------|
| `addFiles` | Creates entries with status `queued` | Persisted to IDB after thumbnail extraction |
| `updateFileStatus(id, 'processing')` | Called by `useProcessFile` at start of pipeline | Not persisted (transient) |
| `setFileResult(id, result, method)` | Moves to `done`, populates `result`/`cachedResults` | Immediately persisted |
| `updateFileStatus(id, 'error', msg)` | Moves to `error` with message | Immediately persisted |
| `restoreFromDb` | On init, `processing` files are reset to `queued`; `done` files with a valid `resultMeta` keep their status (re-processing is triggered on demand, not at restore) | N/A (read path) |

The `processing` status is explicitly excluded from persistence -- when
`fileToPersistedFile` serializes a file, it maps `'processing'` back to
`'queued'` so a page reload during inference resumes cleanly.

### Per-File Grading (OpenDRT Overrides)

Every `QueuedFile` carries its own grading state:

- **`lookPreset`**: One of `'default' | 'colorful' | 'umbra' | 'base' | 'flat'`
  -- selects the base OpenDRT tonescale configuration (SDR presets defined in
  `web/src/gl/opendrt-params.ts`).
- **`openDrtOverrides`**: A sparse `Partial<OpenDrtConfig>` containing only the
  parameters the user has explicitly adjusted. The full OpenDRT config has 40+
  parameters covering tonescale, saturation, purity, brilliance, exposure, white
  balance, and sharpening.

The rendering path in `OutputCanvas` merges these at render time:

```
configFromPreset(lookPreset) --> configWithOverrides(base, openDrtOverrides, preProcessOverrides) --> GPU uniforms
```

`configWithOverrides` returns a `GradingConfig` (which is
`OpenDrtConfig & PreProcessConfig`), merging the preset, OpenDRT overrides,
and pre-processing overrides into a single object for the GPU uniform buffer.

This merge-on-read pattern means:

- Switching presets preserves user overrides only if the user explicitly set
  them. However, `setFileLookPreset` clears OpenDRT overrides
  (`openDrtOverrides: {}`) to give each preset a clean starting point, while
  preserving `preProcessOverrides`.
- `resetFileOpenDrtOverrides` clears the OpenDRT override map, reverting the
  file to the pure preset look.
- `resetFilePreProcessOverrides` clears the pre-processing override map,
  reverting exposure/WB/sharpening to defaults.
- `configWithOverrides` auto-enables feature groups (e.g., setting `tn_lcon`
  automatically enables `tn_lcon_enable`) to reduce the number of toggles the
  user must manage.

The `GradingPanel` component reads grading state from the selected file with
targeted selectors and dispatches individual overrides via
`setFileOpenDrtOverride`. Exposure and white-balance sliders translate
user-facing values (EV stops, Kelvin, tint) into the internal `wb_temp`,
`wb_tint`, and `exposure` override keys using camera metadata
(`exportData.wbCoeffs`, `exportData.camToXyz`).

### Persistence Integration

The store uses a two-tier persistence strategy:

**IndexedDB** (`web/src/lib/idb-storage.ts`) stores structured metadata:

- `PersistedFile` records (file queue state, grading overrides, result
  metadata). `Float32Array` fields are serialized to `number[]` via
  `serializeResultMeta` before storage.
- `AppSetting` key-value pairs (demosaic method, export format/quality,
  selected file ID).

**OPFS** (Origin Private File System) stores large binary data -- RAW sensor
data and thumbnails. (HWC pixel buffers are no longer stored in OPFS; the
pipeline now re-processes files on demand or hands off GPU buffers directly.)
This separation keeps IDB transactions small and fast.

Three helper functions bridge Zustand to IDB:

- **`fileToPersistedFile(f: QueuedFile): PersistedFile`** -- Converts a
  `QueuedFile` to its IDB-safe representation. Notably maps `'processing'`
  status to `'queued'` and serializes `ProcessingResultMeta` via
  `serializeResultMeta`.
- **`persistFile(f: QueuedFile): void`** -- Immediately writes to IDB.
  Called after `setFileResult` (processing complete), `updateFileStatus` with
  `'error'`, `setFileLensProfile`, `resetFileOpenDrtOverrides`, and
  `resetFilePreProcessOverrides`.
- **`persistFileDebounced(f: QueuedFile): void`** -- Debounced write (300ms per
  file ID). Called by `setFileLookPreset`, `setFileOpenDrtOverride`, and
  `setFilePreProcessOverride` to avoid flooding IDB during rapid slider
  adjustments.

The debounce implementation in `idb-storage.ts` uses a per-file-ID timer map
(`pendingTimers: Map<string, ReturnType<typeof setTimeout>>`), ensuring that
rapid changes to different files don't interfere with each other.

On startup, `useInit` restores persisted state:

1. Reads all `PersistedFile` records and global settings from IDB in parallel
   with WASM/model initialization.
2. Converts each `PersistedFile` to `QueuedFile` via `persistedToQueued`,
   loading thumbnails from OPFS and deserializing result metadata.
3. `done` files with a valid `resultMeta` keep their status; files without
   result metadata are downgraded to `queued` for re-processing.
4. Calls `restoreFromDb` to hydrate the store in a single batch.
5. Matches lenses for restored files that have metadata but no profile yet.
6. Cleans up orphaned OPFS entries (fire-and-forget).

### Global Settings

Global settings (`demosaicMethod`, `exportFormat`, `exportQuality`) follow a
simple pattern: the setter updates Zustand state synchronously, then
fire-and-forget writes to IDB via `putSetting`. These are restored from IDB
during `useInit`.

HDR display state (`displayHdr`, `displayHdrHeadroom`) is probed at startup and
not persisted -- it depends on the current display hardware.

### Component Consumption Patterns

Components use targeted Zustand selectors to minimize re-renders:

- **Primitive selectors**: `useAppStore((s) => s.demosaicMethod)` -- subscribes
  to a single field.
- **Derived selectors**: `useAppStore((s) => s.files.find((f) => f.id === s.selectedFileId))` -- used by `GradingPanel`, `SettingsPanel`, `Histogram` to
  get the selected file object.
- **Per-file grading selectors**: `OutputCanvas` selects `lookPreset` and
  `openDrtOverrides` by file ID rather than subscribing to the full file
  object. This prevents re-renders from unrelated file property changes
  (e.g., progress updates on other files).
- **Imperative reads**: `useAppStore.getState()` is used in async callbacks
  (`useProcessFile`, `useExport`, `OutputCanvas` effect callbacks) where a
  React subscription would be stale or unnecessary.

## Rationale

### Why Zustand Over React Context

- **Selector-based subscriptions**: Context re-renders every consumer when any
  part of the value changes. Zustand's `useStore(selector)` only triggers
  re-renders when the selected slice changes. This matters when grading sliders
  fire dozens of updates per second -- only `OutputCanvas` (to update GPU
  uniforms) and `GradingPanel` (to update slider positions) re-render, not the
  file list or settings panel.
- **Imperative access**: The processing pipeline (`useProcessFile`) runs as a
  long async callback. It reads the latest demosaic method mid-execution via
  `useAppStore.getState()`. Context would require passing values through
  closures or refs.
- **No provider nesting**: A single `create()` call replaces a provider
  component, keeping the component tree shallow.

### Why Per-File Grading State

Each RAW file has different exposure, white balance, and color characteristics.
Grading overrides stored per-file mean:

- Users can switch between files in the queue and return to their adjusted
  look without losing work.
- The `OutputCanvas` component receives grading state as a function of the
  displayed file ID, so switching files naturally loads the correct grading.
- Persistence is straightforward -- `openDrtOverrides` serializes directly to
  IDB alongside other file metadata.

The alternative -- global grading state applied to whichever file is selected --
would discard adjustments on file switch and require a separate "save grading"
workflow.

## Key Files

| File | Role |
|------|------|
| `web/src/store.ts` | Zustand store definition, `QueuedFile`/`FileStatus` types, persistence helpers |
| `web/src/lib/idb-storage.ts` | IndexedDB CRUD for `PersistedFile` and `AppSetting`, debounced write |
| `web/src/gl/opendrt-params.ts` | `OpenDrtConfig` interface, preset definitions, `configWithOverrides` merge |
| `web/src/pipeline/types.ts` | `ProcessingResultMeta`, `SerializableResultMeta`, serialization functions |
| `web/src/hooks/useInit.ts` | Startup restoration: IDB read, OPFS validation, `restoreFromDb` call |
| `web/src/hooks/useProcessFile.ts` | Processing pipeline: reads `demosaicMethod` imperatively, calls `setFileResult` |
| `web/src/components/GradingPanel.tsx` | Grading UI: reads per-file overrides, dispatches `setFileOpenDrtOverride` |
| `web/src/components/OutputCanvas.tsx` | WebGPU rendering: subscribes to per-file `lookPreset`/`openDrtOverrides`, applies to renderer |
| `web/src/components/FileList.tsx` | File queue UI: reads `files`, `selectedFileId`, dispatches `selectFile`/`removeFile` |

## Antipatterns

**Do not subscribe to the entire `files` array for per-file data.** Selecting
`useAppStore((s) => s.files)` means every file status change, progress tick,
or grading slider adjustment on any file triggers a re-render. Instead, select
the specific file by ID or select individual properties:

```ts
// Bad: re-renders on any file change
const files = useAppStore((s) => s.files);
const myFile = files.find((f) => f.id === id);

// Good: re-renders only when this file's overrides change
const overrides = useAppStore((s) =>
  s.files.find((f) => f.id === id)?.openDrtOverrides ?? {}
);
```

**Do not persist `processing` status.** The `fileToPersistedFile` helper
deliberately maps `processing` to `queued`. If you add new persistence
call sites, follow this convention -- a user who reloads during inference
should see the file re-queued, not stuck in a `processing` state with no
active worker.

**Do not call `persistFile` in high-frequency paths.** Grading slider
changes call `persistFileDebounced` (300ms per file). Using the immediate
`persistFile` in a slider handler would flood IndexedDB with dozens of
writes per second. Reserve `persistFile` for discrete state transitions
(result ready, error, override reset).

**Do not store large binary data in the Zustand store.** HWC pixel buffers
(often 50-200 MB) are transient GPU buffers or re-generated on demand. The
store holds only `ProcessingResultMeta` (dimensions, color matrices, camera
metadata). Storing pixel data in Zustand would bloat memory and break
structured-clone serialization to IDB.
