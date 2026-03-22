---
title: Training Dataset Build Pipeline
tags: [dataset, training, raw-processing, build]
scope: build_dataset.py, backfill_metadata.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

The x-veon demosaicing network learns to reconstruct full-color RGB from single-channel CFA (Color Filter Array) sensor data. Training uses a synthetic re-mosaicing strategy: start with a high-quality demosaiced RGB image as ground truth, then simulate the CFA sampling at training time to produce the input. The network never sees real mosaiced sensor data during training; it learns the inverse mapping from synthetic mosaic to known clean RGB.

This approach exists because real raw sensor data is single-channel — there is no pixel-aligned ground truth RGB to supervise against. By demosaicing the raw file with a high-quality classical algorithm (DHT for X-Trans, AHD for Bayer) and treating that result as ground truth, the build pipeline produces supervision targets that are good enough for the network to learn from while remaining free of the artifacts that the network will eventually be trained to avoid.

The build pipeline converts a curated selection of raw camera files into float32 NumPy arrays paired with JSON metadata sidecars. The selection is driven by ranking JSON files produced by upstream sharpness classifiers (`classify_hf_ha.py`, etc.) that score raw files by high-frequency content, ensuring the training set is biased toward images with fine detail, texture, and edge structure rather than blurry or featureless shots.

## Pattern/Approach

### Overview

The build pipeline takes a ranking JSON as input and produces, for each raw file, two output artifacts:

- `{stem}.npy` — a float32 array of shape `(H/4, W/4, 3)` containing linear RGB in sensor space
- `{stem}_meta.json` — a JSON sidecar with camera metadata extracted from the raw header

The entire pipeline runs in `build_dataset.py` with multiprocessing support. The `backfill_metadata.py` script exists as a repair tool for datasets built before metadata extraction was added.

### Step 1: Load Ranking JSON

The `load_raw_map()` function accepts three JSON formats:

1. **Dict-list with paths** — `[{"filename": "DSCF4582", "path": "/mnt/.../DSCF4582.RAF"}, ...]`. This is the current standard format produced by the ranking scripts. Each entry contains the stem and the absolute path to the raw file.

2. **Dict-list with file field** — `[{"file": "DSCF7016.RAF", "path": "/mnt/..."}, ...]`. Similar to above but uses `file` (with extension) instead of `filename` (stem only). The stem is derived by stripping the extension.

3. **Plain string list** (legacy) — `["DSCF7140", ...]`. A flat list of image stems without paths. Requires `--raw-base` to scan a DCIM directory tree for matching files by extension. Supported extensions: RAF, CR2, CR3, NEF, NRW, ARW, SRW, RW2, ORF, PEF, IIQ.

The `--top-n` flag limits processing to the first N entries, preserving the JSON's order (assumed pre-sorted by quality score).

The function returns a `{stem: raw_path}` dictionary. The main function then verifies that all paths exist on disk, logs a warning for missing files, and skips them.

### Step 2: Auto-Detect Sensor Type

Inside `process_raw()`, the raw file is opened with `rawpy.imread()`. Sensor type is detected by examining the CFA pattern shape from `raw.raw_pattern`:

- **X-Trans**: pattern height >= 6 (the 6x6 X-Trans CFA). Demosaiced with `DemosaicAlgorithm.DHT` (Directional Homogeneity Testing), which is optimized for the X-Trans filter array.
- **Bayer**: pattern height < 6 (the 2x2 Bayer CFA). Demosaiced with `DemosaicAlgorithm.AHD` (Adaptive Homogeneity-Directed), a strong general-purpose Bayer demosaicer.

This detection is separate from the CFA pattern system used at training/inference time (see the [CFA Pattern System](cfa-pattern-system.md) doc). The build pipeline detects sensor type only to select the appropriate classical demosaic algorithm. The training dataset code and inference pipeline use `detect_cfa_from_raw()` and `CFA_REGISTRY` from `cfa.py` for CFA-aware operations.

### Step 3: Demosaic with Controlled Settings

The `raw.postprocess()` call uses deliberately restrictive settings to preserve the raw sensor signal:

```
demosaic_algorithm = DHT or AHD (per sensor type)
output_bps         = 16          # 16-bit output
no_auto_bright     = True        # No auto-brightness scaling
no_auto_scale      = True        # CRITICAL: keeps values in raw range, not scaled to 16-bit
gamma              = (1, 1)      # Linear — no gamma curve
output_color       = raw         # No color space conversion, no color matrix
use_camera_wb      = False       # No white balance
use_auto_wb        = False       # No auto white balance
user_wb            = [1,1,1,1]   # Unity WB multipliers
user_flip          = 0           # No EXIF rotation — keep raw sensor orientation
```

The `no_auto_scale=True` flag is the most critical setting. Without it, rawpy scales the output to fill the 16-bit range (0-65535), destroying the relationship between pixel values and the sensor's black/white levels. With this flag, pixel values remain in the raw sensor range and can be accurately normalized in the next step.

White balance is explicitly suppressed (`use_camera_wb=False`, `user_wb=[1,1,1,1]`). WB multipliers are stored in the metadata sidecar and applied optionally during training by `LinearDataset` when `apply_wb=True`. This deferred WB approach allows a single dataset build to support both WB-applied and raw-space training experiments.

### Step 4: Black-Subtract and Normalize Without Clipping

After demosaicing, the 16-bit integer output is normalized to float32:

```python
rgb_f = (rgb_16.astype(np.float32) - black) / (white - black)
```

Where `black = raw.black_level_per_channel[0]` and `white = raw.white_level`. This maps the sensor's usable range to approximately [0, 1], but **no clamping is applied**. Values can exceed 1.0 (highlights above the nominal white level) and can go slightly below 0.0 (noise around the black level).

### Step 5: Downscale 4x via Area Averaging

The normalized float32 image is downscaled by a factor of 4 using manual area averaging:

```python
new_h, new_w = h // DOWNSAMPLE, w // DOWNSAMPLE
h_crop = new_h * DOWNSAMPLE
w_crop = new_w * DOWNSAMPLE
for c in range(3):
    ch = rgb_f[:h_crop, :w_crop, c].reshape(new_h, DOWNSAMPLE, new_w, DOWNSAMPLE)
    downscaled[:, :, c] = ch.mean(axis=(1, 3))
```

Edge pixels that do not form a complete 4x4 block are cropped before reshaping. Each output pixel is the mean of a 4x4 block of input pixels. This is done per-channel without cross-channel mixing.

### Step 6: Save .npy and Metadata JSON

The downscaled array is saved as a float32 `.npy` file. A companion `_meta.json` sidecar is written with:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Absolute path to the original raw file |
| `sensor_type` | string | `"xtrans"` or `"bayer"` |
| `black_level` | float | Black level from the raw header (channel 0 value) |
| `white_level` | float | White level from the raw header |
| `camera_wb` | [float, float, float] | Camera white balance multipliers (R, G, B) |
| `original_size` | [int, int] | Original sensor dimensions [width, height] |
| `downscaled_size` | [int, int] | Output dimensions [width, height] |
| `pattern` | int[][] | Raw CFA pattern array from `raw.raw_pattern` |
| `range_min` | float | Minimum pixel value in the saved array |
| `range_max` | float | Maximum pixel value in the saved array |

The `pattern` field stores the sensor's native CFA pattern as a 2D integer array. For X-Trans sensors this is a 6x6 array; for Bayer sensors it is a 2x2 array. This uses the libraw/rawpy color encoding (R=0, G1=1, B=2, G2=3), which differs from the canonical R=0, G=1, B=2 encoding used in `cfa.py`. The metadata is informational; the training dataset code does not use the per-file pattern to construct CFA masks. Instead, it uses the canonical patterns from `CFA_REGISTRY` (see [CFA Pattern System](cfa-pattern-system.md)).

### Resumability

Both the main loop and `process_raw()` are resumable. The main function scans the output directory for existing `.npy` files and only queues work for stems that are not yet present. Inside `process_raw()`, an existence check at the top short-circuits if the output already exists. This allows the build to be interrupted and restarted without reprocessing completed files.

### Parallelism

Processing uses Python's `multiprocessing.Pool` with a configurable worker count (`-w` flag, default 4). Each worker processes one raw file independently. Memory is bounded by explicit `del` and `gc.collect()` calls after the large intermediate arrays are no longer needed.

### Backfill Metadata

`backfill_metadata.py` is a repair script for datasets built before metadata sidecar generation was added. It scans a dataset directory for `.npy` files that lack a corresponding `_meta.json`, locates the original RAF file by stem in a DCIM directory tree, opens just the raw header (no re-demosaic), and writes the metadata sidecar. It also reads the `.npy` file in memory-mapped mode (`mmap_mode='r'`) to compute `range_min`, `range_max`, and `downscaled_size` without loading the entire array.

The backfill script does not write `sensor_type` (unlike `build_dataset.py`), since it predates the multi-sensor support. It also hardcodes a RAF-specific search path. For new datasets, `build_dataset.py` produces complete metadata at build time.

## Rationale

### Why Downscale 4x

Full-sensor images from modern cameras (6000x4000 for X-Trans, larger for some Bayer sensors) are too large for efficient patch-based training. A 96-pixel training patch at full resolution covers a very small fraction of the image, limiting the number of useful crops. Downscaling 4x reduces a 6000x4000 image to 1500x1000, which yields more structurally diverse patches per image. It also reduces dataset size on disk by 16x (from hundreds of GB to tens of GB for a 2000-image dataset).

The 4x factor is chosen because area averaging over a 4x4 block acts as a low-pass antialiasing filter, producing clean ground truth without aliasing artifacts from the demosaic step. Smaller factors (2x) would preserve more demosaic artifacts in the supervision signal; larger factors (8x) would lose too much fine detail.

### Why No Clipping

The normalization step deliberately does not clamp to [0, 1]. Sensor values above the nominal white level occur in highlights (specular reflections, light sources) and carry real information. Clipping them would teach the network that highlight values are always exactly 1.0, preventing it from learning highlight reconstruction. Values slightly below zero occur from read noise around the black level and are similarly left intact to preserve the noise floor characteristics.

The `range_min` and `range_max` fields in the metadata record the actual value range per image. The training code in `train.py` uses these (along with `camera_wb`) to compute the effective `data_range` parameter for SSIM-based loss functions, which need to know the actual signal range rather than assuming [0, 1].

### Why Store Metadata Separately

Storing metadata in a JSON sidecar rather than embedding it in the `.npy` file has several advantages:

1. **Selective loading**: `LinearDataset` loads WB multipliers from metadata only when `apply_wb=True`. When training without WB (the default), metadata files are never opened, saving I/O.

2. **Inspectability**: JSON sidecars can be examined, grepped, and bulk-analyzed with standard shell tools. This supports dataset quality auditing (e.g., finding images with unusual white levels or extreme value ranges).

3. **Retroactive enrichment**: The `backfill_metadata.py` script demonstrates that metadata can be added or corrected after the dataset is built, without touching the `.npy` files.

4. **NumPy compatibility**: The `.npy` format stores a single array with a fixed dtype and shape header. Embedding variable-length metadata would require switching to `.npz` or a custom format, adding complexity to the hot-path `np.load()` call in `LinearDataset.__getitem__()`.

### Why Unity White Balance at Build Time

White balance is intentionally not baked into the `.npy` files. The `user_wb=[1,1,1,1]` setting means the stored RGB values reflect the raw sensor response (after black subtraction and normalization), where R and B channels are typically dimmer than G because the camera's WB correction has not been applied.

This design supports two training modes from the same dataset:

- **Raw-space training** (`apply_wb=False`): The model learns to demosaic raw sensor values directly. The downstream pipeline applies WB after inference. This is the default.
- **WB-applied training** (`apply_wb=True`): `LinearDataset` reads `camera_wb` from the metadata, normalizes to G=1, and multiplies each channel before mosaicing. The model learns to produce WB-corrected output. This mode also supports WB augmentation (`wb_aug_range`). Exposure augmentation (`exposure_aug_ev`) operates in raw space before WB application.

### Why DHT for X-Trans and AHD for Bayer

DHT (Directional Homogeneity Testing) is one of the few demosaic algorithms in libraw that properly handles the X-Trans 6x6 CFA. Most algorithms (VNG, PPG, bilinear) assume a 2x2 Bayer pattern and produce severe artifacts on X-Trans data. DHT is X-Trans-aware and produces relatively clean results with minimal zipper artifacts.

AHD (Adaptive Homogeneity-Directed) is a well-established Bayer demosaic algorithm that balances quality and speed. It was chosen over simpler methods (bilinear, VNG) for fewer color fringing artifacts, and over slower methods (DCB, AMaZE) as a practical trade-off given that the output serves as training ground truth rather than a final image.

## How the Built Data Is Consumed

### LinearDataset (dataset.py)

The primary consumer. Scans a directory for `.npy` files (excluding `_meta.npy` and `_lum.npy` suffixes). Each `__getitem__` call:

1. Loads the `.npy` file via `np.load()`
2. Extracts a random patch aligned to the CFA period (see [CFA Pattern System](cfa-pattern-system.md) for alignment constraints)
3. Applies exposure augmentation (raw space, before WB)
4. Optionally applies WB from the `_meta.json` sidecar
5. Applies spatial augmentations (flips)
6. Captures the ground truth reference (before OLPF blur)
7. Optionally applies OLPF blur (input path only — ground truth remains sharp)
8. Simulates CFA mosaicing via a local `mosaic()` function
9. Adds sensor noise
10. Returns a `(4-channel input, 3-channel ground truth)` training pair

### train.py

Orchestrates training. Uses `LinearDataset.find_files()` to discover `.npy` files, splits at the image level for train/val, then creates `LinearDataset` or `create_mixed_dataset` instances. The `_compute_data_range()` function reads `_meta.json` files to determine the effective data range (accounting for WB gain) for SSIM loss calibration.

### backfill_metadata.py

Repair tool. Scans for `.npy` files without corresponding `_meta.json` sidecars and regenerates them from raw file headers. Uses memory-mapped `.npy` loading to compute array statistics without full deserialization.

## Key Files

| File | Role |
|------|------|
| `build_dataset.py` | Current build pipeline. Multi-format JSON input, auto-detects sensor type, supports X-Trans and Bayer. Multiprocessing with resumability. |
| `backfill_metadata.py` | Retroactive metadata repair for datasets missing `_meta.json` sidecars. |
| `dataset.py` | Training dataset class (`LinearDataset`). Loads `.npy` files, reads `_meta.json` for WB. Primary consumer of built data. |
| `train.py` | Training script. Uses `_meta.json` for data range computation. |
| `cfa.py` | CFA pattern system. Referenced by `dataset.py` for mask generation and mosaicing. See [CFA Pattern System](cfa-pattern-system.md). |

## Antipatterns

### Do Not Use src/datasets/build_dataset_v4.py

`src/datasets/build_dataset_v4.py` is a **legacy** build script that predates `build_dataset.py`. It is X-Trans-only (hardcodes `DemosaicAlgorithm.DHT`), RAF-only (hardcodes Fuji RAF extension), fetches files from a NAS via SSH/SCP with hardcoded credentials, and does not auto-detect sensor type. It also does not set `user_wb=[1,1,1,1]` or `user_flip=0`, which means its output may include unwanted WB and rotation. The corresponding dataset class `src/datasets/dataset_v4.py` imports from the deprecated `xtrans_pattern.py` shim rather than `cfa.py`.

All new dataset builds must use `build_dataset.py` at the project root. The `src/datasets/` scripts are retained only for reproducibility of older experiments.

### Do Not Clip During Normalization

The normalization formula `(value - black) / (white - black)` must not be followed by `np.clip(0, 1)` or equivalent. Clipping destroys highlight and shadow information that the training pipeline relies on. The `exposure_aug_ev` augmentation in `LinearDataset` explicitly pushes values toward clipping to teach the network about highlight behavior — this only works if the ground truth retains super-white values. If you observe `range_max` values above 1.0 in the metadata, that is correct behavior, not a bug.

### Do Not Bake White Balance into .npy Files

Applying WB at build time would produce a dataset locked to camera-specific color balance. The deferred WB design allows the same `.npy` files to be used for raw-space training, WB-applied training, and WB-augmented training. If you modify the build pipeline, preserve the `user_wb=[1,1,1,1]` setting and store `camera_wb` in the metadata.

### Do Not Skip no_auto_scale

The `no_auto_scale=True` flag in the rawpy `postprocess()` call prevents libraw from scaling output values to fill the 16-bit range. Without this flag, a raw file with white level 4000 would have its values scaled by 65535/4000 = 16.4x, making the subsequent black-subtraction and white-normalization produce incorrect results. If you see pixel values that look far too large or a `range_max` well above 1.5, the likely cause is a missing `no_auto_scale=True`.
