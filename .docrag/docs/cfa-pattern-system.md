---
title: CFA Pattern System
tags: [cfa, masking, mosaicing, bayer, xtrans, architecture]
scope: cfa.py, web/src/pipeline/constants.ts, web/src/pipeline/preprocessor.ts
generated: 2026-03-22
commit: 347c8dd
---

## Context

The x-veon demosaicing pipeline reconstructs full-color RGB images from single-channel CFA (Color Filter Array) sensor data. Every digital camera sensor captures light through a repeating mosaic of colored filters — the CFA determines which color (red, green, or blue) each photosite records. The pipeline must understand the CFA layout to construct the model's 4-channel input tensor, simulate mosaicing during training, detect CFA phase offsets during inference on raw files, and enforce patch-size alignment constraints. All CFA logic is centralized in `cfa.py`, which defines pattern constants, a registry, mask generators, a mosaicing function, period/alignment utilities, and pattern detection routines. The web frontend mirrors the pattern constants in `web/src/pipeline/constants.ts` and implements equivalent mask generation and shift detection in `web/src/pipeline/preprocessor.ts`.

Two CFA types are supported: the Fujifilm X-Trans 6x6 pattern and the standard Bayer 2x2 pattern. The system is designed to be CFA-agnostic — the same model architecture, training loop, and inference code handles both patterns by parameterizing on the pattern array rather than hardcoding geometry.

## Pattern Definitions and Registry

### Color Encoding

A uniform integer encoding is used everywhere: R=0, G=1, B=2. This applies to both Python (`numpy.int32` arrays) and TypeScript (`number[][]` arrays). The libraw/rawpy convention uses a 4-color encoding (R=0, G1=1, B=2, G2=3); the detection routines in `cfa.py` remap G2 to G before matching.

### XTRANS_PATTERN

`XTRANS_PATTERN` is a `numpy.ndarray` of shape `(6, 6)` and dtype `int32`:

```
0 2 1 2 0 1
1 1 0 1 1 2
1 1 2 1 1 0
2 0 1 0 2 1
1 1 2 1 1 0
1 1 0 1 1 2
```

This is the canonical X-Trans layout verified against Fuji X-T10 and X-T2 RAF files. Of the 36 positions, 20 are green (55.6%), 8 red (22.2%), and 8 blue (22.2%). The higher green density compared to Bayer (50%) is an intentional characteristic of the X-Trans design.

### BAYER_PATTERN

`BAYER_PATTERN` is a `numpy.ndarray` of shape `(2, 2)` and dtype `int32`:

```
0 1
1 2
```

This is the canonical RGGB Bayer layout. Other Bayer variants (BGGR, GRBG, GBRG) are phase shifts of this same 2x2 tile. The code handles them at inference time through `find_pattern_shift()` rather than defining separate pattern constants.

### CFA_REGISTRY

`CFA_REGISTRY` is a `dict[str, np.ndarray]` mapping string keys to pattern arrays:

```python
CFA_REGISTRY = {
    "xtrans": XTRANS_PATTERN,
    "bayer":  BAYER_PATTERN,
}
```

All consumers that need to select a CFA type by name look up this registry. The dataset classes (`LinearDataset` in `dataset.py`, `TortureDatasetV2` in `torture_v2.py`) accept a `cfa_type: str` parameter and resolve it via `CFA_REGISTRY[cfa_type]`. The inference script `infer_hdr.py` uses `CFA_REGISTRY[cfa_type]` when the user specifies the CFA explicitly, and falls back to auto-detection via `detect_cfa_from_raw()` otherwise.

### Web-Side Pattern Constants

`web/src/pipeline/constants.ts` defines `XTRANS_PATTERN` and `BAYER_PATTERN` as `readonly (readonly number[])[]` arrays with identical values to the Python definitions. These are imported by `web/src/pipeline/preprocessor.ts` for CFA detection and mask generation in the browser-side pipeline. The web constants file also defines `PATCH_SIZE = 288` and `OVERLAP = 48`, which are the tiling parameters that depend on CFA-period alignment.

## Mask Generation

### make_cfa_mask(h, w, pattern)

Tiles the pattern array to fill an `(h, w)` region. Computes the number of tiles needed in each dimension using ceiling division, calls `numpy.tile()`, then slices to the exact `(h, w)` shape. Returns a `torch.Tensor` of dtype `long` with shape `(h, w)`, where each element is 0 (R), 1 (G), or 2 (B).

The default pattern parameter is `XTRANS_PATTERN` for backward compatibility, but all current callers in `dataset.py` and `torture_v2.py` pass the pattern explicitly from the registry.

### make_channel_masks(h, w, pattern)

Calls `make_cfa_mask()` internally and splits the result into three binary planes. Returns a `torch.Tensor` of shape `(3, h, w)` and dtype `float32`, where:

- `masks[0]` is 1.0 at R positions, 0.0 elsewhere
- `masks[1]` is 1.0 at G positions, 0.0 elsewhere
- `masks[2]` is 1.0 at B positions, 0.0 elsewhere

These three mask planes become channels 1-3 of the model's 4-channel input tensor (channel 0 is the mosaiced CFA value).

### Web-Side Mask Generation

`preprocessor.ts` defines `makeChannelMasks(patchSize, pattern, period)` which returns a `ChannelMasks` object containing three `Float32Array` fields (`r`, `g`, `b`), each of length `patchSize * patchSize`. The logic is equivalent to the Python version: iterate over each pixel, look up the pattern at `(y % period, x % period)`, and set the corresponding mask to 1. The `buildTileInput()` function assembles the 4-channel input by concatenating the CFA patch (channel 0) with the three mask arrays (channels 1-3) into a single `Float32Array` of length `4 * patchSize * patchSize`.

## Mosaicing

### mosaic(rgb, cfa)

Simulates CFA sampling of an RGB image. Accepts either a 3D tensor `(3, H, W)` or a 4D batched tensor `(B, 3, H, W)`. The `cfa` argument is a `(H, W)` long tensor from `make_cfa_mask()`.

The implementation uses `torch.gather()` to index into the channel dimension. For the 3D case, the CFA mask is unsqueezed to `(1, H, W)` and `gather(rgb, 0, cfa_exp)` selects the channel indicated by each CFA position. For the 4D case, the mask is unsqueezed to `(B, 1, H, W)` and `gather(rgb, 1, cfa_exp)` operates along dimension 1 (the channel dimension). The output shape is `(1, H, W)` or `(B, 1, H, W)`.

The `gather`-based implementation is branchless and runs efficiently on GPU. An older loop-based variant exists in `dataset.py` as a local `mosaic()` function that iterates over channels with boolean masking — it predates the centralized `cfa.py` version and produces identical results but does not use `gather`.

### Training Usage

During training, `mosaic()` synthesizes CFA data from ground-truth RGB images. The dataset `__getitem__` methods follow this sequence:

1. Load or generate an RGB image as a `(3, H, W)` tensor
2. Apply augmentations (flips, OLPF blur, exposure shifts)
3. Call `mosaic(rgb, self.cfa)` to produce a `(1, H, W)` CFA image
4. Optionally add sensor noise to the CFA image
5. Concatenate: `torch.cat([cfa_img, self.masks], dim=0)` to form a `(4, H, W)` input tensor
6. Return `(input_tensor, rgb)` as the training pair

## Period and Alignment Utilities

### cfa_period(pattern)

Returns `pattern.shape[0]` — the spatial repetition period of the CFA. This is 6 for X-Trans and 2 for Bayer. Used by `dataset.py` and `infer_hdr.py` to compute padding and tiling parameters.

### patch_alignment(pattern)

Returns `math.lcm(cfa_period(pattern), 16)`. The value 16 is the UNet's spatial downsampling factor (4 levels of 2x pooling = 2^4 = 16). Patch sizes must be divisible by this alignment value so that the CFA tiles cleanly and the encoder/decoder dimensions remain integral. For X-Trans: `lcm(6, 16) = 48`. For Bayer: `lcm(2, 16) = 16`.

`LinearDataset.__init__()` asserts that the configured `patch_size` is divisible by `patch_alignment(self.pattern)`. The standard patch size of 288 satisfies both: 288 / 48 = 6 (X-Trans) and 288 / 16 = 18 (Bayer).

## Pattern Detection

### detect_cfa_from_raw(raw_pattern)

Auto-detects the CFA type from a rawpy `raw_pattern` or `raw_colors_visible` array. First remaps G2 (value 3) to G (value 1). Then tries X-Trans by checking all 36 phase shifts of `XTRANS_PATTERN` against the top-left 6x6 block. If no match, tries Bayer by checking all 4 phase shifts of `BAYER_PATTERN` against the top-left 2x2 block. Returns a `(cfa_name, canonical_pattern)` tuple where `cfa_name` is a key in `CFA_REGISTRY`. Raises `ValueError` if neither pattern matches.

### find_pattern_shift(raw_pattern, reference)

Determines the `(dy, dx)` phase offset of a sensor's CFA relative to the canonical pattern. Extracts the top-left `period x period` block from `raw_pattern`, remaps G2 to G, then brute-force searches all `period^2` rolled versions of the reference pattern. Returns the shift as a `(dy, dx)` tuple.

During inference (`infer_hdr.py`), the shift is used to compute padding: `pad_top = (period - dy) % period`, `pad_left = (period - dx) % period`. This mirror-pads the raw CFA data so that position (0, 0) of the padded image aligns with position (0, 0) of the canonical pattern. The model's precomputed channel masks are always generated for the canonical (unshifted) pattern, so this alignment step ensures the masks match the actual CFA layout.

### Web-Side Detection

`preprocessor.ts` exports `findPatternShift(cfaStr, cfaWidth, crops)` which performs the same logic in TypeScript. It parses a CFA string (characters `R`, `G`, `B`) into a 2D array, applies crop offsets to derive the visible CFA, then calls `matchShift()` to brute-force match against `XTRANS_PATTERN` (if `cfaWidth === 6`) or `BAYER_PATTERN` (if `cfaWidth === 2`). Returns a `CfaInfo` object containing `cfaType`, `pattern`, `period`, `dy`, and `dx`.

## Rationale

### CFA-Agnostic Design

The codebase parameterizes on `pattern: np.ndarray` rather than hardcoding X-Trans-specific constants. This allows the same `XTransUNet` model, the same dataset classes, and the same inference pipeline to handle both X-Trans and Bayer sensors. The model itself is fully CFA-agnostic — it receives a 4-channel input and produces 3-channel output without any internal knowledge of the CFA geometry. All CFA-specific behavior is encoded in the input tensor's mask channels.

Adding a new CFA type (e.g., a quad-Bayer or a hypothetical 4x4 pattern) requires only: (1) defining the pattern array, (2) adding it to `CFA_REGISTRY`, and (3) ensuring the new period divides the patch size (satisfying the `patch_alignment` constraint).

### 4-Channel Input

The model's input tensor has shape `(B, 4, H, W)`:

- **Channel 0**: The mosaiced CFA value. A single-channel image where each pixel holds the intensity recorded by the corresponding color filter. Spatially dense — every pixel has a value.
- **Channels 1-3**: Binary masks for R, G, B positions respectively. Each mask is 1.0 where the corresponding filter exists and 0.0 elsewhere. Exactly one of the three masks is 1.0 at each spatial location.

This representation decouples the signal (channel 0) from the CFA geometry (channels 1-3). The network can learn color interpolation patterns from the masks without the architecture needing to encode CFA structure. The residual skip connection in `XTransUNet.forward()` broadcasts channel 0 to all three output channels as a baseline (`baseline = cfa.expand(-1, 3, -1, -1)`), and the network output is added to this baseline. The network therefore learns color correction deltas rather than absolute values, making it exposure-agnostic.

An alternative approach would be to pack the CFA into 3 sparse channels (R, G, B each with zeros at non-matching positions). The 4-channel design was chosen because the single dense CFA channel preserves the spatial continuity of the sensor signal, and the separate mask channels provide unambiguous position information without the network needing to infer which zeros are "missing data" versus "dark pixels."

## Key Files

| File | Role |
|------|------|
| `cfa.py` | Canonical pattern definitions, `CFA_REGISTRY`, mask generation, mosaicing, period/alignment utilities, pattern detection. All Python code should import from here. |
| `web/src/pipeline/constants.ts` | TypeScript mirror of `XTRANS_PATTERN` and `BAYER_PATTERN`, plus `PATCH_SIZE` and `OVERLAP`. |
| `web/src/pipeline/preprocessor.ts` | Browser-side CFA detection (`findPatternShift`), mask generation (`makeChannelMasks`), tile input assembly (`buildTileInput`), white balance application, and highlight reconstruction — all CFA-aware and parameterized on pattern/period/shift. |
| `dataset.py` | Training dataset; imports from `cfa.py`; uses `CFA_REGISTRY`, `make_cfa_mask`, `make_channel_masks`, `cfa_period`, `patch_alignment`. Assembles the 4-channel input via `torch.cat([cfa_img, self.masks], dim=0)`. |
| `torture_v2.py` | Synthetic pattern dataset; imports from `cfa.py`; uses `CFA_REGISTRY`, `make_cfa_mask`, `make_channel_masks`. |
| `infer_hdr.py` | Production inference script; imports from `cfa.py`; uses `detect_cfa_from_raw`, `find_pattern_shift`, `make_channel_masks`, `cfa_period`, `CFA_REGISTRY`. |
| `model.py` | `XTransUNet` with `in_channels=4`; the forward method extracts channel 0 for the residual CFA skip. |

## Antipatterns

### Do Not Import from xtrans_pattern.py

The file `xtrans_pattern.py` (project root) is a backward-compatibility shim consisting of four re-exports from `cfa.py`:

```python
from cfa import XTRANS_PATTERN, make_cfa_mask, make_channel_masks, mosaic
```

It exists solely because `infer.py` and legacy code under `src/` still import from it. This shim is pending removal. All new code must import directly from `cfa.py`. The shim does not expose `BAYER_PATTERN`, `CFA_REGISTRY`, `cfa_period`, `patch_alignment`, `find_pattern_shift`, or `detect_cfa_from_raw` — using the shim means losing access to the full CFA API.

The older `src/xtrans_pattern.py` (under `src/`) is a completely independent, X-Trans-only implementation that predates `cfa.py`. It hardcodes `XTRANS_PATTERN` directly and its functions do not accept a `pattern` parameter. Do not use it.

### Do Not Duplicate mosaic Logic

`dataset.py` contains a local `mosaic()` function that uses a loop over channels with boolean indexing. `torture_v2.py` contains a local `mosaic_linear()` with similar loop logic. The canonical `gather`-based implementation in `cfa.py` should be used instead. The local copies exist for historical reasons and produce identical output but bypass the centralized API.

### Do Not Hardcode CFA Period or Alignment

Avoid magic numbers like `6`, `48`, or `% 6` for CFA-dependent calculations. Use `cfa_period(pattern)` and `patch_alignment(pattern)` from `cfa.py`. Hardcoded values break when switching between X-Trans (period 6, alignment 48) and Bayer (period 2, alignment 16).

### Do Not Assume a Specific CFA Phase

Raw files from different cameras (or different crop regions of the same sensor) may have different phase offsets relative to the canonical pattern. Always use `find_pattern_shift()` or `detect_cfa_from_raw()` to determine the actual phase, then pad the image to align position (0, 0) with the canonical pattern origin before applying precomputed masks.
