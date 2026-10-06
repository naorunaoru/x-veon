# Checkpoint Policy

This document defines how X-veon checkpoints are versioned, named, promoted, and consumed.

The goal is to make checkpoint compatibility obvious from the version number, keep exported browser models aligned with the intended baseline, and avoid the historical ambiguity of ad-hoc names like `v5.01q` or `v6.1.4q`.

## Scope

This policy applies to:
- training checkpoints (`best.pt`, `latest.pt`)
- checkpoint directories under `checkpoints/`
- ONNX exports derived from those checkpoints
- `checkpoint_registry.json`
- browser manifests such as `shared/public/checkpoints/models.json`

## Current baseline

The current supported checkpoint baseline is:

- **`v7.1.0`**

`v7.1.0` adds the magnitude-spectrum term to the loss (a Minor change) and trains the S model without noise augmentation; `v7.0.0`, the first v7 recipe, loads and exports the same way.

v7 is the packed family (`ARCHITECTURE_TAG = "v7"` in `model.py`). The model takes the app's five-channel input (mosaic, three colour masks, clip ratio), divides the mosaic by its mean over the tile and packs it by space-to-depth: 3×3 for X-Trans, 2×2 for Bayer. A checkpoint records `stages` (2 for the S model) next to `base_width`.

All checkpoint families before major version 7 are **legacy / unsupported**: `train.py`, `export_onnx.py`, `infer_hdr.py` and `ui.py` refuse to load them. `v6.1.4` (mosaic plus white-balance input) stays on the `train-v6` branch.

## Versioning

X-veon checkpoints use semantic-style versioning:

- **`MAJOR.MINOR.PATCH`**

### Major

Increment **Major** when checkpoint compatibility breaks.

Use a new major version when the inference pipeline changes in a way that makes earlier checkpoints unsupported.

Typical triggers:
- model input/output contract changes
- preprocessing changes required by inference
- ONNX input/output schema changes
- tensor layout changes
- conditioning changes (for example, a new required WB input)
- architecture changes that require a new inference path

Rule of thumb:

> If an older checkpoint cannot be loaded and used by the current intended inference pipeline without special compatibility code, this is a Major change.

Examples:
- `v6.x.x` → `v7.0.0`
- adding/removing required model inputs
- changing the meaning of the input tensor

### Minor

Increment **Minor** when inference remains compatible, but the training objective changes.

Typical triggers:
- adding/removing loss components
- materially changing loss weighting philosophy
- changing the best-checkpoint selection metric
- changing the optimization target while keeping the same model/inference contract

Rule of thumb:

> If checkpoints from both versions can run through the same inference pipeline, but were trained under different objective families, this is a Minor change.

Examples:
- `v6.1.x` → `v6.2.0`
- moving from a PSNR-oriented recipe to a more perceptual recipe without changing inference compatibility

### Patch

Increment **Patch** for compatible checkpoint refinements inside the same family.

Typical triggers:
- fine-tuning
- hyperparameter tuning
- LR schedule changes
- batch size / patch size changes
- training length changes
- augmentation tuning
- dataset mix changes that do not change the inference contract
- seed / cache / worker changes

Rule of thumb:

> If the checkpoint is still the same compatible model family and the change is a training/tuning iteration, this is a Patch change.

Examples:
- `v6.1.4` → `v6.1.5`
- rerunning the same compatible recipe with better hyperparameters

## Compatibility is defined by inference, not chronology

Compatibility is determined by the **inference contract**, not just by whether the network looks similar.

That means a change may require a new Major version even if:
- the architecture is mostly the same, or
- most weights could theoretically be reused

If inference must change, the major version must change.

## Width policy

### Default width

- **Base width 16 is the default model width.**
- Default-width checkpoints use **no width suffix**.

Examples:
- `v6.1.4`
- `v6.1.5`
- `v6.2.0`

### Non-default widths

Only add an explicit suffix when the width is not the default.

Recommended format:
- `v6.1.4-w32`
- `v6.1.4-w64`
- `v6.2.0-w8`

This keeps the default path short while leaving room for alternate sizes.

### Historical suffixes

Historical suffixes:
- `q` = quarter-width = base width 16
- `h` = half-width = base width 32

These suffixes are deprecated and should not be used for new checkpoint families.

## Naming rules

### Canonical checkpoint family name

Each compatible family should have a canonical version string:

- `vMAJOR.MINOR.PATCH`
- or `vMAJOR.MINOR.PATCH-wNN` for non-default width variants

Examples:
- `v6.1.4`
- `v6.1.5`
- `v6.2.0`
- `v6.2.0-w32`

### Directory naming

Checkpoint directories should use the canonical version directly and should include CFA type separately.

Recommended pattern:

```text
checkpoints/<track>/<cfa_type>/vMAJOR.MINOR.PATCH/
checkpoints/<track>/<cfa_type>/vMAJOR.MINOR.PATCH-wNN/
```

Examples:

```text
checkpoints/beta/bayer/v6.1.4/
checkpoints/beta/xtrans/v6.1.4/
checkpoints/stable/bayer/v6.1.4/
checkpoints/beta/bayer/v6.2.0-w32/
```

If the repository keeps the current flat historical structure for a while, the version string inside config/registry should still follow this policy even before directory layout is cleaned up.

## Required metadata

Each checkpoint family must record enough metadata to determine compatibility and provenance without guessing.

At minimum, `config.json` should include:
- `cfa_type`
- `base_width`
- `stages`
- `mode`
- `epochs`
- `patch_size`
- `batch_size`
- whether the model/inference path requires any extra conditioning inputs
- `from_checkpoint`
- `resume`
- the canonical checkpoint version
- the declared compatibility major version

Recommended extra fields:
- `checkpoint_version`: e.g. `v7.0.0`
- `checkpoint_major`: e.g. `7`
- `architecture_tag`: short human-readable architecture/inference family label
- `export_compatible`: boolean
- `notes`: optional free-form summary of what changed in this family

## Registry policy

`checkpoint_registry.json` is the machine-readable source of truth for promoted checkpoints.

### Registry expectations

The registry should distinguish at least:
- CFA type (`bayer`, `xtrans`)
- width variant (`16`, `32`, etc.)
- slot (`best`, `latest`)
- track (`beta`, `stable`)
- canonical version string

### Promotion policy

- **beta** = newest validated training result in a compatible family
- **stable** = approved result for export/deployment in that family

Promotion to stable should be explicit. A checkpoint is not stable just because it is newer.

### Export policy

Browser ONNX exports and `shared/public/checkpoints/models.json` should point only to:
- the intended **stable** family for the current deployment, or
- an explicitly chosen beta family during active testing

The web manifest must not silently mix families from incompatible majors.

`export_onnx.py` exports one version per run (`--version` is required), refuses a checkpoint whose `architecture_tag` is not the one `model.py` declares, and writes the app's keys `{cfa}_w{base_width}_base`. It replaces only the manifest entries it exports.

## ONNX export policy

ONNX exports inherit the checkpoint family version.

Rules:
- export metadata must preserve the canonical version string
- ONNX input/output schema changes must trigger a Major version bump
- ONNX exports from unsupported legacy majors should not replace current stable exports without an explicit migration decision

## Baseline and legacy policy

### Baseline

The current baseline is:
- `v7.1.0`

This is the reference compatible family for current work unless a new version is explicitly introduced.

### Legacy checkpoints

Checkpoint lines before major version 7 are legacy.

Legacy checkpoints may still be useful for analysis or comparison, but they are not part of the supported forward-compatible family and should not be treated as drop-in alternatives.

## How to version new work

### Use Patch when:
- fine-tuning `v6.1.4`
- changing LR, schedule, dataset mix, augmentation strength, or run duration
- trying alternative training hyperparameters with the same compatible inference family

Examples:
- `v6.1.5`
- `v6.1.6`

### Use Minor when:
- changing the loss family or training objective materially
- keeping the same inference contract and architecture family

Examples:
- `v6.2.0`
- `v6.3.0`

### Use Major when:
- changing inference compatibility
- requiring a new export path
- changing model inputs/outputs
- breaking support for existing checkpoints

Examples:
- `v7.0.0`
- `v8.0.0`

## Practical examples

### Example 1: compatible fine-tune

- Starting point: `v6.1.4`
- Change: same model family, same inference pipeline, better hyperparameters
- New version: **`v6.1.5`**

### Example 2: new loss recipe

- Starting point: `v6.1.4`
- Change: replace part of the loss stack, keep inference identical
- New version: **`v6.2.0`**

### Example 3: new required conditioning input

- Starting point: `v6.1.4`
- Change: model now requires a new inference-time input that older checkpoints do not support
- New version: **`v7.0.0`**

### Example 4: alternate width experiment

- Starting point: `v6.1.4`
- Change: same family but base width 32
- New version: **`v6.1.4-w32`**

## Recommended migration from historical names

Historical names should be interpreted as follows:
- `v6.1.4q` → `v6.1.4`
- `v6.1.4h` → `v6.1.4-w32`

Apply the same rule to future cleanup of old names:
- default width 16 loses the suffix
- non-default widths become explicit `-wNN`

## Summary

X-veon checkpoint policy is:

- **Major** = inference compatibility break
- **Minor** = compatible training-objective/loss-family change
- **Patch** = compatible tune / fine-tune / hyperparameter iteration
- **base width 16 is the default** and uses no suffix
- **non-default widths use explicit suffixes** like `-w32`
- **`v7.1.0` is the current baseline**
- pre-7 families are legacy unless explicitly revived
