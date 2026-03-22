---
title: Checkpoint Registry and Multi-Model Management
tags: [registry, checkpoints, export, onnx, models, browser, inference]
scope: full-stack
generated: 2026-03-22
commit: 347c8dd
---

## Context

X-Veon trains multiple model variants (different CFA sensor types, different base
widths) and ships them to a browser-based inference frontend. A single project
tree can contain dozens of checkpoint directories, each at different stages of
training. The checkpoint registry provides a structured index over all of these
checkpoints so that downstream tooling -- ONNX export and the browser runtime --
can discover, filter, and load models without hard-coded paths.

The system spans three layers:

1. **Python training** (`train.py`) writes `.pt` checkpoint files and registers
   them into `checkpoint_registry.json`.
2. **Python export** (`export_onnx.py`) reads the registry, converts selected
   checkpoints to ONNX, and writes a `models.json` manifest for the web app.
3. **Browser runtime** (`inference.ts`) fetches `models.json`, resolves a model
   key for the active CFA type and user-selected size, and loads the
   corresponding ONNX session on demand.

## Pattern / Approach

### Registry JSON structure

The file `checkpoint_registry.json` at the project root uses a five-level
nesting:

```
cfa_type -> base_width -> variant -> status -> slot -> entry
```

Concrete example:

```json
{
  "xtrans": {
    "16": {
      "base": {
        "stable": {
          "best": {
            "path": "checkpoints/_nowb/xtrans_w16/best.pt",
            "epoch": 200,
            "train_psnr": 38.1234,
            "val_psnr": 37.5678,
            "train_loss": 0.001234,
            "val_loss": 0.001567,
            "history": "checkpoints/_nowb/xtrans_w16/history.json"
          },
          "latest": { "..." : "..." }
        }
      }
    },
    "32": { "..." : "..." },
    "64": { "..." : "..." }
  },
  "bayer": { "..." : "..." }
}
```

**Key dimensions:**

| Level        | Values                         | Purpose                                    |
|------------- |------------------------------- |------------------------------------------- |
| `cfa_type`   | `"xtrans"`, `"bayer"`          | Sensor CFA pattern family                  |
| `base_width` | `"16"`, `"32"`, `"64"` (string)| Network width; maps to S/M/L in browser    |
| `variant`    | `"base"` (currently only)      | Reserved for future variant differentiation|
| `status`     | `"beta"`, `"stable"`           | Training completeness                      |
| `slot`       | `"best"`, `"latest"`           | Best-val-PSNR checkpoint vs periodic save  |

Each slot entry records the checkpoint `path`, `epoch`, four metrics
(`train_psnr`, `val_psnr`, `train_loss`, `val_loss`), and a `history` path
pointing to the full epoch-by-epoch JSON log.

### Core registry functions (`checkpoint_registry.py`)

**`build_registry(project_root)`** -- Full rebuild by scanning all checkpoint
directories under `checkpoints/_nowb/`. For each directory containing both
`config.json` and `history.json`, it reads the CFA type and base width from
config, determines status by comparing the last trained epoch against the
configured total epochs (`"stable"` if `last_epoch >= total_epochs`, otherwise
`"beta"`), identifies the best entry by maximum `val_psnr`, and writes both
`best` and `latest` slots. This is the cold-start / recovery path, run manually
via `python checkpoint_registry.py`.

**`update_registry(registry_path, ...)`** -- Incremental update of a single
slot. Receives all entry fields as keyword arguments. Navigates (or creates) the
nested dict path `cfa_type -> base_width -> variant -> status -> slot` and writes
the entry atomically. The variant is always `"base"`.

**`_save_registry(path, data)`** -- Atomic persistence via
`tempfile.NamedTemporaryFile` followed by `Path.replace()`. This prevents
partial writes if the process crashes mid-save.

**`promote_to_stable(registry_path, ...)`** -- Flips a beta entry to stable.
Moves the entire `"beta"` sub-dict to `"stable"` for a given `cfa_type` and
`base_width`, but only when `"stable"` does not already exist. This prevents
overwriting a previously completed training run with a newer incomplete one.

### How `train.py` registers checkpoints

Training interacts with the registry at three points:

1. **On new best validation PSNR** -- When `val_psnr > best_val_psnr`, the
   training loop saves `best.pt` and calls `update_registry` with
   `status="beta"` and `slot="best"`. This happens every time a new personal
   best is achieved during training.

2. **Every 10 epochs** -- The periodic checkpoint `latest.pt` is saved and
   registered via `update_registry` with `status="beta"` and `slot="latest"`.

3. **On successful completion** -- After the training loop finishes all epochs
   without interruption or fatal error, `promote_to_stable` is called. This
   atomically moves the beta entry to stable. If training is interrupted
   (KeyboardInterrupt) or crashes, the entry remains as beta.

The registry path is always resolved relative to the script location
(`Path(__file__).parent / REGISTRY_FILENAME`), so it stays at the project root
regardless of the output directory used for checkpoints.

### How `export_onnx.py` iterates the registry

The export script supports two modes: legacy single-checkpoint export
(`--checkpoint`) and registry-based batch export (the default).

**`_iter_registry(registry, ...)`** -- Generator that walks the nested registry
dict and yields `(label, checkpoint_path, entry)` tuples. It accepts optional
filters for `cfa_type`, `base_width`, `variant`, `status`, and `slot`. When
`status` is not specified, it prefers `"stable"` over `"beta"` -- it tries
stable first and only falls back to beta if stable is absent. The label is
constructed as `"{sensor}_w{width}_{variant}"` (e.g., `"xtrans_w16_base"`).

**SHA-based skip logic** -- Before exporting, the script computes
`sha256(source_checkpoint)` and compares it against the existing manifest. If
the hash matches and the ONNX file exists on disk, the export is skipped
(unless `--force` is passed). This makes re-running the export idempotent and
fast when only some checkpoints have changed.

**Batch export flow:**

1. Load `checkpoint_registry.json`.
2. Filter entries via `_iter_registry` using CLI flags.
3. For each entry, export to `{output_dir}/{label}.onnx` via `torch.onnx.export`,
   optionally converting to float16 via `onnxconverter_common.float16`.
4. Build a `models.json` manifest mapping each label to its metadata.

### `models.json` manifest structure

Written to `web/public/checkpoints/models.json`, this file is the contract
between the Python export pipeline and the browser runtime:

```json
{
  "xtrans_w16_base": {
    "epoch": 200,
    "base_width": 16,
    "param_count": 123456,
    "size_mb": 0.5,
    "dtype": "float16",
    "file": "xtrans_w16_base.onnx",
    "source_sha256": "abc123...",
    "train_psnr": 38.12,
    "val_psnr": 37.56,
    "train_loss": 0.001234,
    "val_loss": 0.001567
  },
  "xtrans_w32_base": { "..." : "..." },
  "bayer_w16_base": { "..." : "..." }
}
```

Each key is the same label produced by `_iter_registry`. The browser uses
`file` to construct the download URL and the metrics for display. The
`source_sha256` field is used only by the export script for cache invalidation.

### Browser-side model management

**`ModelSize` type** -- Defined in `types.ts` as `'S' | 'M' | 'L'`. These map
to network base widths via a constant in `inference.ts`:

```typescript
const SIZE_TO_WIDTH: Record<ModelSize, number> = { S: 16, M: 32, L: 64 };
```

**`resolveModelKey(cfaType, width)`** -- Given a CFA type and a numeric base
width, returns the manifest key to use. It constructs candidate keys with
the pattern `"{prefix}_w{width}_{suffix}"` and checks for their presence in
the manifest, preferring the `"hl"` variant over `"base"`. Returns `null` if
no matching model is available.

**`getAvailableSizes(cfaType)`** -- Returns a `Set<ModelSize>` containing only
those sizes for which a manifest entry exists for the given CFA type. The
`SettingsPanel` component uses this to disable unavailable size buttons in the
UI.

**`switchModelSize(size)`** -- Loads ONNX sessions for the new size across all
CFA types. For each CFA type, it resolves the manifest key, calls
`getOrLoadSession` (which caches sessions in a `Map`), and updates the
`active` map. Previously loaded sessions remain in cache, so switching back
is instant.

**`initModels(size)`** -- Called once at startup. Fetches `models.json`,
then loads sessions for the requested initial size. Uses a promise guard
(`initPromise`) to prevent duplicate initialization.

**Session caching** -- The `sessions` Map stores `{ session, meta }` entries
keyed by manifest key. The `active` Map tracks which key is currently active
per CFA type. This two-map design allows multiple sizes to be cached while
only one is active at a time.

**`SettingsPanel.tsx` UI** -- Renders three toggle buttons (S, M, L). Each
button is disabled if `getAvailableSizes` does not include that size. On click,
it calls `setModelSize` (Zustand state), then `switchModelSize` (loads the ONNX
session), and finally reprocesses the current file if it was already processed
with the neural-net method.

### Full lifecycle

```
train.py                checkpoint_registry.json        export_onnx.py
--------                ------------------------        --------------
save best.pt  -------->  beta / best entry
save latest.pt ------->  beta / latest entry
all epochs done ------> promote beta -> stable
                                                        _iter_registry (filters)
                                                        export -> .onnx files
                                                        write models.json

models.json             inference.ts                    SettingsPanel.tsx
-----------             ------------                    ----------------
                        fetchManifest()
                        resolveModelKey(cfa, width)
                        createSession(url)              getAvailableSizes()
                        active.set(cfa, key)            switchModelSize(size)
                        runBatch() / runBatchGpu()      reprocess on change
```

1. **Train**: `train.py` saves `best.pt` and `latest.pt`, registering each as
   beta. On completion, the entry is promoted to stable.
2. **Register**: `checkpoint_registry.json` accumulates entries across multiple
   training runs (different CFA types, different widths).
3. **Export**: `export_onnx.py` reads the registry, filters to the desired
   entries (defaulting to stable + best), exports each to ONNX, and writes
   `models.json`.
4. **Manifest**: `models.json` is deployed alongside the ONNX files to
   `web/public/checkpoints/`.
5. **Browser load**: `initModels` fetches the manifest and loads the ONNX
   session for the initial size (default S).
6. **Switch**: The user clicks a size button; `switchModelSize` loads the new
   session (or retrieves it from cache), updates the active model, and
   triggers reprocessing.

## Rationale

**Why a JSON registry instead of directory conventions?** Directory naming alone
cannot capture training status (beta vs stable) or metrics. The registry makes
it possible for the export script to select the best stable checkpoint without
parsing filenames or reading every checkpoint file.

**Why beta/stable instead of just "done"?** In-progress training runs produce
useful checkpoints. Marking them beta lets the export pipeline optionally include
them (via `--status beta`) for testing, while defaulting to stable for
production exports.

**Why atomic writes?** The training loop writes the registry every time a best
checkpoint or periodic save occurs. A crash during write could corrupt the JSON
file. The temp-file-then-rename pattern ensures the registry is always valid.

**Why SHA-based skip in export?** ONNX export is expensive (model load,
tracing, optional fp16 conversion). Skipping unchanged checkpoints makes
repeated exports fast, especially when only one of several models has been
retrained.

**Why cache sessions in the browser?** Loading an ONNX model over the network
and initializing a WebGPU/WASM session is the most expensive operation in the
browser pipeline. Caching means switching from M back to S is instant if both
have been loaded during the session.

## Key Files

| File | Role |
|------|------|
| `checkpoint_registry.py` | Registry CRUD: `build_registry`, `update_registry`, `promote_to_stable` |
| `checkpoint_registry.json` | On-disk registry (generated, not committed) |
| `train.py` | Registers checkpoints during training (beta on save, stable on completion) |
| `export_onnx.py` | Reads registry, exports ONNX, writes `models.json` manifest |
| `web/public/checkpoints/models.json` | Browser-facing manifest (generated by export) |
| `web/src/pipeline/inference.ts` | Session management, `resolveModelKey`, `switchModelSize`, `getAvailableSizes` |
| `web/src/pipeline/types.ts` | `ModelSize` type definition (`'S' \| 'M' \| 'L'`) |
| `web/src/components/SettingsPanel.tsx` | Model size toggle UI |

## Antipatterns

**Hard-coding checkpoint paths in export or browser code.** All paths flow
through the registry and manifest. Adding a new model variant should require
only training it; the registry, export, and browser will pick it up
automatically.

**Promoting to stable manually.** The `promote_to_stable` function guards
against overwriting an existing stable entry. Manually editing the registry
JSON to set status to stable bypasses this safety check and can result in an
incomplete training run being treated as production-ready.

**Skipping the registry and using `--checkpoint` for production exports.** The
legacy single-file mode exists for quick one-off tests. It produces a manifest
with a single entry and a generic key, which will not match the
`resolveModelKey` naming convention and will be invisible to the browser size
selector.

**Adding new model sizes without updating `SIZE_TO_WIDTH`.** The browser maps
`ModelSize` to `base_width` via a hardcoded record. If a new width (e.g., 48)
is trained and exported, it will appear in the manifest but the browser will
never resolve it unless `SIZE_TO_WIDTH` and the `ModelSize` type are extended.

**Calling `switchModelSize` without awaiting it.** The function is async because
it may need to fetch and initialize an ONNX session. Firing and forgetting can
lead to inference running against the old model while the new one loads, causing
confusing quality differences.
