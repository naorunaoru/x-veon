---
title: "Modification Recipes"
tags: [modification-recipes, how-to-add, extension-points, step-by-step]
scope: full-stack
generated: 2026-03-22
commit: 347c8dd
---

# Modification Recipes

Step-by-step instructions for common extension tasks in the x-veon codebase.
Each recipe lists every file to touch, in order, with the specific change required.

---

## 1. Add a New Demosaic Algorithm (WASM Path)

Add a traditional (non-neural-net) demosaic method that runs via the WASM worker pool.
Cross-reference: `demosaic-algorithm-dispatch.md`, `worker-pool-parallelism.md`.

### Steps

1. **`web/src/pipeline/types.ts`** -- Add the new method name (e.g. `'vng'`) to the
   `DemosaicMethod` union type. Place it in the appropriate CFA group comment
   (X-Trans, Bayer, or both).

2. **WASM crate** -- Implement the algorithm in the Rust WASM demosaic crate
   (`web/wasm/demosaic/`). For Bayer algorithms, export it from `demosaic_bayer`; for
   X-Trans, export from `demosaic_image`. The function must accept the algorithm
   name string and dispatch internally.

3. **`web/src/pipeline/demosaic-worker.ts`** -- No change required if the WASM
   `demosaic_image` / `demosaic_bayer` functions already dispatch on the algorithm
   string. The worker passes `e.data.algorithm` directly to WASM.

4. **`web/src/pipeline/demosaic-pool.ts`** -- No change required. The pool forwards
   the `algorithm` string to workers. The `Algorithm` type alias automatically
   includes the new method via `Exclude<DemosaicMethod, 'neural-net'>`.

5. **`web/src/pipeline/demosaic.ts`** -- If the new algorithm has a WebGPU
   compute shader implementation, add a GPU fast-path block in `runDemosaic`
   (following the pattern of the `bilinear` and `dht` branches). Otherwise no
   change needed; the function falls through to the WASM worker pool.

6. **`web/src/components/SettingsPanel.tsx`** -- Add an entry to the
   `DEMOSAIC_OPTIONS` array. Set the `cfa` field to `'bayer'`, `'xtrans'`, or
   omit it if the algorithm works for both CFA types.

7. **Verify processing path** -- `web/src/hooks/useProcessFile.ts` needs no change.
   The `processFile` function reads `demosaicMethod` from the store and passes it
   through; the `method === 'neural-net'` branch is the only special case. Both
   paths end with `gpuPostprocess()` for WB, highlight recovery, CC, and DR gain.

### Gotchas

- The WASM worker receives a `bayerVariant` string (e.g. `'rggb'`) only when
  `isBayer` is true. X-Trans algorithms receive `dy`/`dx` shift values instead.
- The `DemosaicPool` splits images into horizontal strips with overlap. The
  `STRIP_OVERLAP_FACTOR` (3x CFA period) must be sufficient for the new
  algorithm's border requirements.
- Adding a GPU compute path is optional but recommended for performance. Follow
  the pattern in `demosaic-gpu.ts`: write WGSL, create a pipeline in
  `initDemosaicGpu`, export a `runXxxGpu` function, and gate it behind
  `gpuAvailable()` in `demosaic.ts`.

---

## 2. Add a New Grading Control

Add a numeric parameter that flows from the UI slider through Zustand state into
the WebGPU fragment shader.
Cross-reference: `opendrt-tone-mapping.md`, `webgpu-renderer.md`, `zustand-state-management.md`.

### Steps

1. **`web/src/gl/opendrt-params.ts`** -- Determine whether the new parameter is a
   tone-mapping/color control (add to `OpenDrtConfig`) or a pre-processing
   adjustment (add to `PreProcessConfig`). Set its default value in `baseSdr()`
   or `DEFAULT_PREPROCESS` respectively. Propagate to `defaultSdr()`,
   `colorfulSdr()`, `umbraSdr()`, and `flatSdr()` if those presets need
   different values. If the parameter auto-enables a feature group, add logic to
   `configWithOverrides`. The full grading config is typed as
   `GradingConfig = OpenDrtConfig & PreProcessConfig`.

2. **`web/src/gl/renderer.ts`** -- Pack the new value into the uniform buffer.
   Find an unused padding slot in an existing `vec4` (slots marked `0` in the
   offset comments, e.g. `U_ODRT_RS + 3` or `U_ODRT_PT2 + 2`). Write the value
   in `applyOpenDrtUniforms`. If no padding slot is available, add a new vec4
   constant (e.g. `U_NEW_PARAM`), increment `UNIFORM_FLOATS` by 4, and add a
   corresponding `vec4f` field to the WGSL `Uniforms` struct.

3. **`web/src/gl/shaders/opendrt.wgsl`** -- If you added a new `vec4f` to the
   uniform layout, append it to the `Uniforms` struct (order must match the
   TypeScript offset constants). Read the value in the fragment shader function
   where it applies (e.g. `fs_main` or a helper). Implement the processing
   logic.

4. **`web/src/components/GradingPanel.tsx`** -- Add the slider. For a simple
   control, append a tuple to one of the existing param arrays (`TONE_PARAMS` or
   `BRILLIANCE_PARAMS`), or create a new section. Each tuple is
   `[label, configKey, min, max, step]`. The `ParamSlider` component
   automatically reads from `effectiveValue(key)` and writes via
   `handleChange`. For controls needing custom display labels or non-linear
   mapping, write a dedicated `handleXxxChange` callback (see
   `handleExposureChange` or `handleTempChange` as examples).

### Gotchas

- Uniform buffer layout must be 16-byte aligned (vec4 boundaries). Never insert
  a field mid-vec4 without adjusting all subsequent offsets.
- The `UNIFORM_BYTES` constant must equal `UNIFORM_FLOATS * 4`. The GPU buffer
  is allocated with `size: UNIFORM_BYTES` (400 bytes / 100 floats) in the
  `createBuffer` call in `renderer.ts`.
- The WGSL `Uniforms` struct field order must exactly mirror the TypeScript
  `U_*` offset constants. A mismatch silently reads wrong data.
- Per-file overrides are stored as `Partial<OpenDrtConfig>` in
  `QueuedFile.openDrtOverrides` (in `store.ts`). They are persisted to
  IndexedDB, so the key name becomes part of the serialization contract.
- The rendering technology is WebGPU (not WebGL). The shader language is WGSL.

---

## 3. Add a New Export Format

Add a new image output format to the export pipeline.
Cross-reference: `hdr-display-export.md`, `webgpu-renderer.md`.

### Steps

1. **`web/src/pipeline/types.ts`** -- Add the new format name (e.g. `'png'`) to
   the `ExportFormat` union type.

2. **WASM encoder crate** -- Implement encoding in the Rust encoder crate
   (`web/wasm/encoder/`). The `encode_image` function receives a format string and
   must match on it. Add the codec dependency to `Cargo.toml` and handle the new
   format branch.

3. **`web/src/pipeline/encoder-worker.ts`** -- No change required. The worker
   passes the format string directly to the WASM `encode_image` function.

4. **`web/src/pipeline/encoder.ts`** -- Add entries to the `mimeTypes` and
   `extensions` maps inside `encodeImage` (e.g. `png: 'image/png'` and
   `png: 'png'`).

5. **`web/src/hooks/useExport.ts`** -- If the new format requires HDR data, add
   it to the `needsHdr` function. If it needs a different render gamut or
   tonescale configuration, add a branch in `exportFile` (follow the
   `jpeg-hdr` / `avif` / `tiff` pattern).

6. **`web/src/components/ExportDialog.tsx`** -- Add an entry to the
   `FORMAT_OPTIONS` array with `value` and `label`.

7. **`web/src/store.ts`** -- No change required. The `exportFormat` state field
   already accepts any `ExportFormat` value, and `setExportFormat` persists it to
   IndexedDB.

### Gotchas

- The encoder worker runs in a Web Worker. Large allocations (e.g. TIFF
  uncompressed) must fit in the worker's memory. Test with large images
  (6000x4000+).
- Quality slider is disabled for TIFF (`isTiff` check in `ExportDialog.tsx`). If
  the new format also ignores quality, add a similar check.
- The `renderForExport` method on `HdrRenderer` accepts a gamut parameter
  (`'rec709'` or `'rec2020'`). SDR formats should use `'rec709'`; HDR formats
  use `'rec2020'`.

---

## 4. Add a New Loss Component

Add a new loss term to the training objective.
Cross-reference: `loss-functions.md`, `training-loop.md`.

### Steps

1. **`losses.py`** -- Define a new `nn.Module` subclass (e.g.
   `class FrequencyLoss(nn.Module)`) with a `forward(self, pred, target)`
   method that returns a scalar tensor. Place it alongside the existing
   components (`SobelGradientLoss`, `ChromaLoss`, `ZipperLoss`,
   `ColorBiasLoss`, etc.).

2. **`losses.py`** -- Wire it into `DemosaicLoss.__init__`: add a
   `xxx_weight: float` parameter (default `0.0`), store it as `self.xxx_weight`,
   and conditionally instantiate the module (following the pattern
   `self.xxx = FrequencyLoss() if xxx_weight > 0 else None`).

3. **`losses.py`** -- Add the forward pass in `DemosaicLoss.forward`: compute the
   component loss, record it in `components` dict, and accumulate into `total`
   (guarded by `if self.xxx is not None and self.xxx_weight > 0`).

4. **`losses.py`** -- Optionally update the `base` and `finetune` classmethod
   presets if the new component should be active by default in either mode.
   These are convenience constructors; the actual training uses `TrainConfig`
   parameters.

5. **`train.py`** -- Add a field to the `TrainConfig` dataclass (e.g.
   `xxx_weight: float = 0.0`). Update `TrainConfig.base()` and/or
   `TrainConfig.finetune()` presets if the new component should be active by
   default. The `TrainConfig` dataclass is the single source of truth for all
   training parameters.

6. **`train.py`** -- In the loss construction block, add the new weight from
   `TrainConfig` to the `DemosaicLoss` constructor call. `DemosaicLoss`
   conditionally instantiates modules based on whether weights are > 0:
   ```python
   self.xxx = FrequencyLoss() if xxx_weight > 0 else None
   ```

7. **`train.py`** -- Add the new component to the `loss_info` summary string so
   it appears in the training log header.

Note: `DemosaicLoss` also supports `recon_only` mode (computes L1/Huber only on
pixels under reconstruction, with `known_pixel_weight` penalty). If the new loss
component should respect this mask, add the masked computation path following the
pattern in `DemosaicLoss.forward`.

### Gotchas

- All loss modules with learnable buffers (e.g. Gaussian kernels) must use
  `self.register_buffer()` so they move to the correct device with
  `.to(device)`.
- The `DemosaicLoss.forward` return signature is
  `tuple[torch.Tensor, dict[str, torch.Tensor]]`. The dict is logged per-epoch;
  keep key names short.
- Loss presets (`base`/`finetune`) are convenience constructors on both
  `DemosaicLoss` and `TrainConfig`. The actual training uses `TrainConfig`
  parameters which can override any weight.
- Component losses should return a scalar tensor reduced over the batch. Use
  `F.l1_loss` or `.mean()` -- do not return unreduced tensors.
- If the loss needs the data range (e.g. for SSIM constants), accept it as a
  constructor parameter and thread it from `DemosaicLoss.__init__`.

---

## 5. Add a New CFA Pattern

Register a new color filter array pattern for both training and web inference.
Cross-reference: `cfa-pattern-system.md`, `cfa-alignment-detection.md`.

### Steps

1. **`cfa.py`** -- Define the pattern as a NumPy array constant (e.g.
   `QUAD_BAYER_PATTERN = np.array([...], dtype=np.int32)`). Use the channel
   encoding R=0, G=1, B=2. The array must be square (period x period).

2. **`cfa.py`** -- Add the pattern to `CFA_REGISTRY` with a string key (e.g.
   `"quad_bayer": QUAD_BAYER_PATTERN`).

3. **`cfa.py`** -- Update `detect_cfa_from_raw` to try matching the new pattern.
   Add a detection block before the Bayer fallback, ordered from largest period
   to smallest (try the new pattern if `h >= period and w >= period`).

4. **`dataset.py`** -- No change required if `--cfa-type` matches the registry
   key. The `LinearDataset` constructor looks up `CFA_REGISTRY[cfa_type]` and
   asserts `patch_size % patch_alignment(pattern) == 0`.

5. **`train.py`** -- Add the new CFA type name to the `TrainConfig.cfa_type`
   field documentation and any validation logic.

5b. **`model.py`** -- If the new CFA period differs from 2 (Bayer) or 6
   (X-Trans), verify that the `cfa_period` parameter to `XTransUNet` is set
   correctly. When `cfa_period > 2`, the model adds 4 sinusoidal positional
   encoding channels and uses a 7x7 stem kernel. The model takes 5 input
   channels: CFA + R/G/B position masks + clip ratio.

6. **`web/src/pipeline/constants.ts`** -- Add the pattern as a TypeScript constant
   (e.g. `export const QUAD_BAYER_PATTERN: readonly (readonly number[])[] = [...]`).
   Ensure it exactly matches the Python definition.

7. **`web/src/pipeline/preprocessor.ts`** -- Update `findPatternShift` to try
   matching the new pattern. Import the constant from `constants.ts` and add a
   detection block following the existing X-Trans / Bayer pattern.

8. **`web/src/pipeline/types.ts`** -- Add the new CFA type name to the `CfaType`
   union (e.g. `'xtrans' | 'bayer' | 'quad_bayer'`).

9. **`web/src/store.ts`** -- Update the CFA type inference in `addFiles` if the
   new pattern is associated with specific file extensions (the current logic
   maps `.raf` to `'xtrans'` and everything else to `'bayer'`).

10. **`web/src/components/SettingsPanel.tsx`** -- If any demosaic algorithms are
    specific to the new CFA type, tag them with `cfa: 'quad_bayer'` in the
    `DEMOSAIC_OPTIONS` array.

### Gotchas

- The CFA period must divide both the web `PATCH_SIZE` (288) and the training
  `patch_size` (default 96, configurable via `--patch-size`) after lcm with the
  UNet factor (16). The `patch_alignment` function computes `lcm(period, 16)`.
  For example, a period-4 pattern requires `lcm(4, 16) = 16`, which divides both
  288 and 96. A period-5
  pattern would require `lcm(5, 16) = 80`, which also divides. But a period-7
  pattern gives `lcm(7, 16) = 112`, which does NOT divide 288 -- you would need
  to change `PATCH_SIZE`.
- The Python and TypeScript pattern arrays must be identical. A mismatch causes
  silent color channel swaps.
- Only flips (not rotations) are used for augmentation in `LinearDataset` because
  arbitrary rotations break CFA alignment. This constraint applies to all
  patterns.
- The `STRIP_OVERLAP_FACTOR` in `demosaic-pool.ts` uses `3 * period`. For large
  periods, verify that `MIN_STRIP_HEIGHT` (128) is still larger than
  `3 * period`; otherwise the pool will never split into strips.
- Existing demosaic algorithms (AHD, PPG, MHC, etc.) are Bayer-only or
  X-Trans-only. A new CFA type likely needs its own demosaic implementations in
  the WASM crate.

---

## Antipatterns

### Do not bypass the type system

Never cast a raw string to `DemosaicMethod`, `ExportFormat`, or `CfaType` without
first adding the literal to the union. TypeScript will catch usage-site errors only
if the union is the source of truth.

### Do not add uniform fields without updating both sides

The WGSL `Uniforms` struct and the TypeScript `U_*` offset constants in
`renderer.ts` must stay in sync. Adding a field to one without the other causes
silent data corruption (wrong values read by the shader). Always update both files
in the same commit.

### Do not hard-code CFA period assumptions

Functions like `padToAlignment`, `generateTiles`, and `DemosaicPool` derive
alignment from the CFA period. Do not introduce new code that assumes period=2
or period=6 -- always read from the `CfaInfo` or pattern metadata.

### Do not modify loss presets to change training behavior

The `DemosaicLoss.base()` and `DemosaicLoss.finetune()` classmethods (and their
`TrainConfig` counterparts) are convenience constructors for common
configurations. The actual training run uses `TrainConfig` parameters. To change
training behavior, modify `TrainConfig` fields or pass explicit values.

### Do not forget WASM crate changes require a rebuild

Changes to the Rust WASM crates (`web/wasm/demosaic/`, `web/wasm/encoder/`) require
running `wasm-pack build` before the web app can use them. The TypeScript
side imports generated JS/WASM bindings, not Rust source.

### Do not add store fields without persistence

New Zustand state fields that should survive page reloads need corresponding
`putSetting`/`getSetting` calls in the `set*` action and `restoreFromDb`.
Per-file fields go through the `QueuedFile` -> `PersistedFile` -> IDB path.
Forgetting persistence causes values to reset on reload.
