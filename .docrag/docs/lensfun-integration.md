---
title: LensFun Lens Profile Integration
tags: [web, lens-correction, lensfun, exif, matching]
scope: web/src/lib/lensfun.ts, web/scripts/convert-lensfun-db.ts, web/src/store.ts, web/src/hooks/useInit.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

Camera lenses introduce optical distortions -- barrel/pincushion warping,
transverse chromatic aberration (TCA), and vignetting (light falloff toward
edges). Correcting these requires per-lens calibration data that maps focal
length, aperture, and distance to mathematical correction coefficients.

LensFun is an open-source database of such calibration profiles covering
thousands of camera/lens combinations. x-veon integrates LensFun so that when a
user drops a RAW file, the application automatically identifies the lens from
EXIF metadata and attaches the corresponding correction profile. This profile
carries distortion, TCA, and vignetting coefficients that the GPU rendering
pipeline can apply during image processing.

The integration is entirely client-side. At build time, a conversion script
transforms the upstream LensFun XML database into a set of JSON files served as
static assets. At runtime, the browser fetches only the relevant JSON fragments,
matches the EXIF camera and lens strings against the database, and stores the
matched profile on the per-file state in the Zustand store.

## Pattern / Approach

### Database Conversion (Build Time)

The script `web/scripts/convert-lensfun-db.ts` transforms the upstream LensFun
XML database into browser-friendly JSON. It is run manually via
`npx tsx scripts/convert-lensfun-db.ts` and produces static assets under
`web/public/lensfun/`.

The conversion process:

1. **Clone**: Shallow-clones the LensFun GitHub repository with sparse checkout
   limited to `data/db/`, or pulls the latest if the clone already exists.
2. **Parse**: Reads every `.xml` file in the database directory using
   `fast-xml-parser`. The `isArray` callback ensures elements like `camera`,
   `lens`, `mount`, `distortion`, `tca`, and `vignetting` are always parsed as
   arrays, even when the XML contains a single element.
3. **Convert**: Each XML file becomes a JSON file with the same base name. A
   JSON file contains two arrays: `cameras` and `lenses`. Camera records carry
   `make`, `model`, `mount`, and `cropfactor`. Lens records carry `make`,
   `model`, `aliases`, `mounts`, `cropfactor`, and calibration arrays for
   `distortion`, `tca`, and `vignetting`.
4. **Index**: An `index.json` file is written alongside the per-file JSONs. Each
   index entry contains `file` (the JSON filename), `cameras` (an array of
   `"make|model"` strings), and `lenses` (an array of `"make|model"` strings).
   The index allows the runtime to determine which JSON files to fetch without
   downloading the entire database.

Multilingual `<model>` elements in the XML (distinguished by a `lang` attribute)
are resolved by the `extractModel` function, which picks the entry without a
`lang` attribute as the primary model name and collects the rest as aliases.

### Data Model Types

The runtime types in `web/src/lib/lensfun.ts` mirror the JSON schema produced by
the conversion script:

- **`LfCamera`** -- `make`, `model`, `mount`, `cropfactor`, optional `variant`.
  Represents a camera body. The `mount` field is critical for lens matching
  because it constrains which lenses are physically compatible.

- **`LfLens`** -- `make`, `model`, `aliases`, `mounts`, `cropfactor`, optional
  `type`. Contains three calibration arrays:
  - `distortion: LfDistortion[]` -- polynomial coefficients per focal length.
    Supports `poly3` (k1), `poly5` (k1, k2), and `ptlens` (a, b, c) models.
  - `tca: LfTCA[]` -- chromatic aberration coefficients per focal length.
    Supports `poly3` (vr, vb) and `acm` (br, cr, bb, cb) models, with optional
    `kr`/`kb` for higher-order terms.
  - `vignetting: LfVignetting[]` -- light falloff coefficients per focal
    length, aperture, and subject distance. Uses three polynomial terms
    (k1, k2, k3).

- **`LensProfile`** -- the matched result stored on each `QueuedFile`. Contains
  `lensModel`, `mount`, `cropfactor`, and the three calibration arrays copied
  from the matched `LfLens`.

- **`LfDbFile`** -- internal type representing a single parsed JSON file
  (`cameras` + `lenses` arrays).

- **`LfIndexEntry`** -- internal type for an entry in `index.json` (`file`,
  `cameras`, `lenses` string arrays).

### In-Memory Caching

Two module-level caches in `lensfun.ts` prevent redundant network requests:

- `indexCache: LfIndexEntry[] | null` -- holds the parsed `index.json`. Fetched
  once on the first call to `fetchIndex()` and reused for the lifetime of the
  page.
- `dbFileCache: Map<string, LfDbFile>` -- maps JSON filenames to their parsed
  contents. Populated lazily as individual DB files are fetched by
  `fetchDbFile()`.

Because the LensFun database is static and versioned at build time, cache
invalidation is not needed at runtime. A new build produces new JSON files with
potentially different content, and the browser's standard HTTP caching handles
staleness.

### Lens Matching Algorithm

The public entry point is `matchLens(camera: string, lensModel: string)`, which
returns a `Promise<LensProfile | null>`. The algorithm proceeds in four stages:

**Stage 1: Find relevant DB files.**
`findRelevantFiles()` scans the index to select which JSON files to fetch. A
file is relevant if its camera makes overlap with the EXIF camera string (after
normalization). Third-party lens manufacturer files (sigma, tamron, samyang,
zeiss, tokina, voigtlander, misc) are always included, since users frequently
pair third-party lenses with any camera body.

**Stage 2: Match the camera.**
`matchCamera()` identifies the specific `LfCamera` record to extract the lens
mount and crop factor. It tries two strategies:

1. *Exact match* -- normalize both the EXIF camera string and the concatenated
   `"make model"` from the database; compare for equality.
2. *Model substring* -- if the first word of the EXIF string starts with the
   database make and the full EXIF string contains the database model, accept
   the match. This handles cases where EXIF encodes the make differently
   (e.g., "FUJIFILM" vs "Fujifilm Corporation").

If neither strategy matches, lens scoring proceeds without mount filtering.

**Stage 3: Score each lens.**
`scoreLensMatch()` computes a score in [0, 1] for every lens in the loaded DB
files. The score is a weighted combination:

```
finalScore = stringScore * 0.8 + mountScore * 0.2
```

The `stringScore` is the best score across the lens model and all its aliases,
computed by three strategies tried in order:

1. *Exact match* -- normalized EXIF lens string equals normalized candidate.
   Score: 1.0.
2. *Substring containment* -- one string contains the other. Score: ratio of
   shorter length to longer length (rewarding close-length matches).
3. *Jaccard token similarity* -- tokenize both strings on whitespace, `/`, and
   `-`; compute intersection / union. The Jaccard score is scaled by 0.9 to
   keep it below the substring strategy.

String normalization (`normalizeLensStr`) lowercases the input, strips
parentheses, removes the `f/` prefix before aperture values, removes the `mm`
suffix from focal lengths, and collapses whitespace. Make normalization
(`normalizeMake`) additionally strips corporate suffixes like "Corporation",
"Co., Ltd.", "Imaging Corp.", "Optical Co., Ltd.", "Camera AG", and "Digital
Solutions".

The `mountScore` is 1.0 if the lens declares the camera's mount in its `mounts`
array, 0.0 if it does not, and 0.5 if no camera was matched (no mount filter).
This 20% weighting means mount compatibility acts as a tiebreaker rather than a
hard filter -- important because some lenses are used with adapters.

**Stage 4: Select best match.**
Lenses with a final score below 0.5 are discarded. Among the remaining
candidates, the highest-scoring lens wins. Ties are broken by preferring the
lens with more calibration data points (sum of distortion, TCA, and vignetting
array lengths), since a richer profile produces better corrections.

### Integration Flow

The full lifecycle from file import to UI display:

```
RAW file dropped
  -> arrayBuffer read
  -> extractRafQuickMetadata(buf) extracts EXIF: camera, lensModel, focalLength, fNumber
  -> store.addFiles() sets metadata on QueuedFile
  -> if metadata.lensModel exists:
       matchLens(camera, lensModel)
         -> fetchIndex() (cached after first call)
         -> findRelevantFiles() selects DB file list
         -> Promise.all(fetchDbFile(...)) loads JSON fragments
         -> matchCamera() identifies mount + cropfactor
         -> scoreLensMatch() scores every lens in loaded files
         -> best match (score >= 0.5) becomes LensProfile
       -> store.setFileLensProfile(fileId, profile)
         -> updates QueuedFile.lensProfile in Zustand
         -> persists to IndexedDB via putFile()
```

A secondary matching path exists in `setFileResult()`. When processing completes
and the WASM decoder returns richer metadata (`lensModel` from `RawImage`), the
store merges it into `QuickMetadata`. If the file still has no `lensProfile` and
a `lensModel` is now available, `matchLens` runs again via `queueMicrotask()` to
avoid blocking the state update.

### Session Restore Matching

When the application restores files from IndexedDB during initialization (in
`useInit.ts`), the persisted `lensProfile` is restored directly from the
`PersistedFile` record. However, files that have metadata but no saved profile
(for instance, if the profile was not yet matched before the previous session
ended) are re-matched:

```ts
for (const qf of files) {
  if (!qf.lensProfile && qf.metadata?.lensModel) {
    matchLens(qf.metadata.camera, qf.metadata.lensModel)
      .then((profile) => {
        if (profile) useAppStore.getState().setFileLensProfile(qf.id, profile);
      })
      .catch((e) => console.warn('Lens match failed:', e));
  }
}
```

This runs after `setInitialized()`, so it does not block the UI from rendering.
The matching calls are fire-and-forget promises that update state asynchronously.

### Store Integration

The `QueuedFile` interface in `web/src/store.ts` carries a `lensProfile`
field of type `LensProfile | null`. The store exposes a single action for
updating it:

- `setFileLensProfile(fileId, profile)` -- maps over `files`, updates the
  matching entry, and persists the updated record to IndexedDB via `putFile()`.

The `lensProfile` is also included in the `PersistedFile` schema in
`web/src/lib/idb-storage.ts`, ensuring it survives page reloads.

### UI Display

`FileListItem.tsx` displays EXIF lens information from `file.metadata` rather
than from the matched `lensProfile`. It shows `metadata.lensModel`, rounded
`focalLength` with "mm" suffix, and `fNumber` formatted as f/N.N. The matched
profile is not directly surfaced in the file list; it exists to feed the GPU
correction pipeline.

## Rationale

**Why pre-convert XML to JSON?** The LensFun XML database is several megabytes
across dozens of files. XML parsing in the browser is slow and the full database
is unnecessary for any single session. Pre-conversion to JSON enables selective
fetching (only the files relevant to the camera make) and instant parsing via
`response.json()`.

**Why a two-level index + data architecture?** The index is small (tens of KB)
and lets the client determine which data files to fetch. A user with a Fujifilm
camera only downloads the Fujifilm JSON and the third-party lens JSONs, not the
Canon or Nikon files.

**Why Jaccard similarity instead of edit distance?** EXIF lens strings are
notoriously inconsistent -- manufacturers abbreviate, reorder, and omit parts of
the lens name. Token-based Jaccard overlap is robust to reordering and partial
matches. The 0.9 scaling ensures that a substring match (which preserves token
order) always ranks above a purely token-overlap match.

**Why mount compatibility is a soft weight (0.2) not a hard filter?** Users
frequently mount lenses via adapters (e.g., Canon EF lenses on Sony E-mount
bodies). A hard mount filter would silently reject valid matches. The soft
weight favors mount-compatible lenses while still allowing cross-mount matches
when the string similarity is strong.

**Why fire-and-forget matching?** Lens matching involves network I/O (fetching
JSON files) and should not block file import or session restore. The profile is
not needed until the user triggers processing with lens correction enabled, so
async population is acceptable.

**Why re-match during session restore?** The LensFun database may have been
updated between sessions (new build), or a previous session may have ended
before matching completed. Re-matching files that lack a profile ensures no
file is left without correction data if a match is available.

## Key Files

| File | Role |
|------|------|
| `web/scripts/convert-lensfun-db.ts` | Build-time XML-to-JSON conversion script |
| `web/public/lensfun/index.json` | Runtime index mapping camera makes to JSON data files |
| `web/public/lensfun/*.json` | Per-manufacturer camera + lens calibration data |
| `web/src/lib/lensfun.ts` | Runtime matching engine: types, caching, scoring, `matchLens()` |
| `web/src/store.ts` | `QueuedFile.lensProfile` field and `setFileLensProfile` action |
| `web/src/hooks/useInit.ts` | Session-restore lens matching for files without profiles |
| `web/src/components/FileListItem.tsx` | EXIF lens metadata display in the file sidebar |
| `web/src/pipeline/raf-decoder.ts` | WASM decoder that extracts `lensModel` from RAW files |
| `web/src/lib/idb-storage.ts` | `PersistedFile.lensProfile` field for IndexedDB persistence |

## Antipatterns

**Do not fetch the entire database eagerly.** The index exists so that only
relevant files are loaded. Fetching all JSON files on startup would waste
bandwidth and slow down initialization, especially on mobile connections.

**Do not hard-filter by mount.** Adapter usage is common enough that rejecting
lenses based solely on mount incompatibility would produce false negatives. Use
the soft mount weighting in `scoreLensMatch` instead.

**Do not block file import on lens matching.** The `matchLens` call is
intentionally async and non-blocking. Making it synchronous (or awaiting it
inside `addFiles`) would delay the file list from appearing in the UI until
network requests complete.

**Do not bypass the in-memory cache.** The `indexCache` and `dbFileCache` exist
to avoid redundant fetches when processing multiple files from the same camera.
Clearing or ignoring the cache would cause duplicate network requests and
degrade performance in multi-file workflows.

**Do not duplicate normalization logic.** Both `normalizeMake` and
`normalizeLensStr` encode domain knowledge about how manufacturers format EXIF
strings. Any new normalization should be added to these functions rather than
scattered across call sites.

**Do not lower the 0.5 score threshold without evaluation.** The threshold
exists to prevent false-positive matches. Lowering it risks attaching the wrong
lens profile, which would apply incorrect distortion correction and degrade
image quality. If matching recall needs improvement, add normalization rules or
aliases to the conversion script instead.
