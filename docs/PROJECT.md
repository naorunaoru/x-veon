# X-Trans Demosaicing Project

Neural network demosaicing for Fujifilm X-Trans sensors.

## Model Architecture

**File:** `model.py`

U-Net encoder-decoder with skip connections:
- **Input:** 4 channels (CFA mosaic + R/G/B position masks)
- **Output:** 3 channels (RGB)
- **Levels:** 4 encoder + 4 decoder + bottleneck
- **Channel progression:** 64 → 128 → 256 → 512 → 1024 (bottleneck) → back down
- **Convolutions:** 3×3 throughout with BatchNorm + ReLU
- **Parameters:** 31,038,339

The receptive field easily covers 2-3 X-Trans pattern repeats (12-18 pixels).

## Training: Linear Sensor Space

All training happens in **linear sensor space**:
- No gamma correction
- No white balance applied during training
- No highlight clipping (values can exceed 1.0)
- Peak value normalized to 1.0 for typical exposures

This preserves the full dynamic range of the sensor and allows HDR output.

---

## Datasets

All large datasets are now located on the external drive at `/Volumes/External/xtrans-demosaic/datasets/`

See `/Volumes/External/xtrans-demosaic/README.md` for detailed dataset documentation.

### v4 Dataset (`/Volumes/External/xtrans-demosaic/datasets/xtrans_v4_dataset/`)

**Build script:** `src/datasets/build_dataset_v4.py`

**Pipeline:**
1. Fetch RAF from NAS via SSH/SCP
2. rawpy DHT demosaic with:
   - Linear gamma `(1, 1)`
   - No white balance (`use_camera_wb=False`)
   - Raw color space (`output_color=rawpy.ColorSpace.raw`)
   - No auto brightness/scale
3. Black-subtract, normalize by `(white - black)`, **NO CLIP**
4. 4× downsample via area averaging
5. Save as float32 `.npy`

**Contents:**
- ~2350 images (all RAF files from NAS)
- Shape: ~824×1233×3 (varies by camera)
- float32, linear space, may contain values > 1.0

**Metadata:** Each `{stem}_meta.json` contains:
- Source path, black/white levels, camera WB
- Original and downscaled dimensions
- Actual value range in file

### Torture Dataset v2 (`/Volumes/External/xtrans-demosaic/datasets/torture_dataset/`)

**Build script:** `torture_v2.py`

Synthetic patterns designed to stress demosaicing:
- 5000 samples, ~1.5GB
- 13 pattern types:
  - 0-10: Geometric (stripes, checkers, circles, edges, etc.)
  - 11: Julia set fractals (50% zoomed for detail)
  - 12: Perlin fBm noise
- Smart color selection (HSV sampling, contrast guarantees)
- Stored as `.npz` with CFA input and RGB target

**Purpose:** Test edge cases, NOT for primary training (causes catastrophic forgetting).

### JPEG Fine-tune Data (`data/fuji_jpgs/`)

**Source:** Camera JPEGs organized by folder (103_FUJI through 107_FUJI)

**Usage in v4_ft:** Convert sRGB→linear, simulate CFA by mosaicing.

---

## Experiments & Checkpoints

### v4 (48.2 dB) ✓ SUCCESS

**Checkpoint:** `checkpoints_v4/best.pt`

**Related files:**
- _obsolete/build_dataset_v4.py
- dataset_v4.py
- train_v4.py

**Training:**
- Dataset: v4 (DHT-demosaiced, 4× downsampled RAFs)
- Input: Actual CFA from RAF (raw sensor data)
- Target: DHT demosaic of same RAF (acts as "teacher")
- Loss: L1 + gradient + chroma
- Epochs: ~100+
- **Result:** 48.21 dB PSNR, good general quality

**Key insight:** The model learns to produce output similar to DHT but can potentially exceed it since it's learning the statistical relationship, not copying DHT's algorithm.

### v4_ft ✗ FAILED (6×6 artifacts)

**Checkpoint:** `checkpoints_v4_ft/best.pt` — DO NOT USE

**Training:**
- Dataset: JPEGs converted to linear + synthetic CFA
- Used flip augmentation
- Started from v4 checkpoint

**Related files:**
- _obsolete/dataset_v4_finetune.py
- _obsolete/train_v4_finetune.py
- _obsolete/train_v4_finetune_fast.py
- _obsolete/train_patches.py
- _obsolete/train_patches_ram.py

**Problem:** Output shows periodic 6×6 grid artifacts matching CFA pattern size.

**Root cause:** Under investigation. Initial theory was flip augmentation breaking CFA alignment, but this may be incorrect — flipping RGB before mosaicing should be valid augmentation since we're using the same fixed CFA pattern.

Possible actual causes:
- Different data distribution (synthetic CFA from JPEGs vs real RAFs)
- Training hyperparameters
- Something else in the pipeline

### Torture Fine-tune ✗ FAILED (catastrophic forgetting)

**Checkpoint:** `checkpoints_torture/best.pt` — DO NOT USE

**Training:**
- Dataset: Torture test v2 (synthetic patterns)
- Started from v4 checkpoint
- Low LR (1e-5)

**Problem:** Complete catastrophic forgetting. Model outputs pixel mush on real images.

**Reason:** Synthetic torture patterns have fundamentally different statistics than real photos. Model "forgot" how to process natural images while optimizing for synthetic edge cases.

**Lesson:** Pure synthetic fine-tuning doesn't work. Would need mixed dataset (90% real + 10% synthetic) or strong regularization.

---

## HDR Inference Pipeline

**File:** `infer_hdr.py`

### Why HDR?

The model outputs linear sensor values which can exceed 1.0 for highlights. Standard pipelines clip these to [0,1], losing highlight detail. HDR output preserves the full range.

### Pipeline Steps:

1. **Load RAF:** Extract raw CFA, black/white levels, WB multipliers
2. **Normalize:** `(cfa - black) / (white - black)` — NO CLIPPING
3. **Pattern alignment:** Detect CFA offset, pad to align with training pattern
4. **Tiled inference:**
   - 288×288 patches (divisible by 6 for CFA, 16 for U-Net pooling)
   - 48px overlap with linear blend weights (avoids grid artifacts)
5. **White balance:** Apply camera WB multipliers to output RGB
6. **HLG encoding:** Convert linear to HLG (Hybrid Log-Gamma) for HDR display
7. **AVIF output:** 10-bit with CICP 9/18/9 (BT.2100 HLG)

### Why AVIF + HLG?

- **HEIC + HLG/PQ:** Broken on macOS (requires Apple's proprietary gain map format)
- **AVIF + HLG:** Works correctly, wide compatibility
- **HLG:** Backward-compatible with SDR displays, no metadata required

### Key: No Clipping

The entire pipeline avoids clipping highlights:
- Normalization: no clip
- Model output: not clamped
- HLG encoding: gracefully handles >1.0 values
- Only final U16 quantization clips (at HLG=1.0, which maps to very bright)

---

## File Locations

```
~/projects/xtrans-demosaic/
├── src/                          # Core code
│   ├── model.py
│   ├── losses.py
│   ├── xtrans_pattern.py
│   └── datasets/
│       ├── dataset_v4.py
│       └── build_dataset_v4.py
│
├── scripts/
│   ├── train/
│   │   ├── train_v4.py           # ✓ PRODUCTION
│   │   └── experiments/
│   ├── inference/
│   │   ├── production/
│   │   │   ├── infer_hdr.py     # ✓ HDR AVIF
│   │   │   └── infer_srgb.py    # sRGB TIFF
│   │   └── test/
│   └── utils/
│
├── checkpoints/
│   ├── v4_production/            # ✓ USE THIS
│   │   └── best.pt
│   ├── experiments/
│   │   ├── checkpoints_v4_ft/    # ✗ BROKEN
│   │   ├── checkpoints_torture/  # ✗ BROKEN
│   │   └── checkpoints_v4_ssim/  # Empty
│   └── archive/
│
├── data/                         # Metadata + symlinks to external
│   ├── sharpness_scores.json
│   ├── top200_sharp.json
│   ├── top2000_sharp.json
│   └── [symlinks to /Volumes/External/xtrans-demosaic/datasets/]
│
├── docs/
│   ├── PROJECT.md               # This file
│   ├── EXPERIMENTS.md
│   └── REORGANIZATION_SUMMARY.md
│
└── _obsolete/                   # Archived old scripts

/Volumes/External/xtrans-demosaic/
├── README.md                    # Dataset documentation
└── datasets/
    ├── xtrans_v4_dataset/       (60GB) - ✓ Production training
    ├── fullres_dataset/         (38GB) - Full-res RAF+JPG pairs
    ├── fuji_jpgs/               (32GB) - Camera JPEGs
    ├── torture_dataset/         (1.5GB) - Synthetic patterns
    └── jpeg_patches/            (1.7GB) - Pre-extracted patches
```

---

## Open Questions

1. **v4_ft artifacts:** What actually caused the 6×6 grid pattern? Flip augmentation may not be the culprit.

2. **Texture preservation:** v4 is slightly soft. How to improve without losing stability?
   - SSIM loss?
   - Full-resolution training?
   - Different data augmentation?

3. **Fine-tuning approach:** What's the right way to improve v4 without catastrophic forgetting?

### Sharpness Scoring

**Files:** `data/sharpness_scores.json`, `data/top200_sharp.json`, `data/top2000_sharp.json`

Computed Laplacian variance for all ~3500 RAFs to identify sharp images for training (training on full dataset too slow).

| File | Contents |
|------|----------|
| `sharpness_scores.json` | Full database: `{"folder/filename": score, ...}` (higher = sharper) |
| `top200_sharp.json` | Top 200 sharpest with full paths (`107_FUJI/DSCF7140`, etc.) |
| `top2000_sharp.json` | Top 2000, **filenames only** (no folder prefixes — less useful) |

Used to select training subsets when full dataset is too slow.

---

## Obsolete Scripts Reference

Located in `_obsolete/`. All of these have the **flip augmentation bug** that caused v4_ft 6×6 artifacts.

### Training Scripts

| Script | Data Source | Patch Size | Output Dir | Notes |
|--------|-------------|------------|------------|-------|
| `train_patches.py` | `data/jpeg_patches/` | 96×96 | `checkpoints_v4_ft/` | Loads patches from disk |
| `train_patches_ram.py` | `data/jpeg_patches/` | 96×96 | `checkpoints_v4_ft/` | All patches in RAM (faster) |
| `train_v4_finetune.py` | JPEGs + torture patterns | varies | `checkpoints_v4_ft/` | Uses dataset_v4_finetune.py |
| `train_v4_finetune_fast.py` | JPEGs (on-the-fly) | 96×96 | (configurable) | Caches images, converts sRGB→linear live |
| `train_v3.py` | v3 dataset | - | `checkpoints_v3/` | Original v3 training |

### Dataset Loaders

| Script | Purpose |
|--------|---------|
| `dataset_v4_finetune.py` | XTransFinetuneDataset + TortureFinetuneDataset |
| `dataset.py` | v2/v3 dataset (sRGB-based, obsolete) |

### Other

- `build_dataset_v4.py` — Built the v4 .npy dataset (one-time)
- `infer.py` — v3 inference (non-linear output)
- `compare_checkpoints.py` — A/B comparison tool
- `fit_*.py` — Color correction experiments (unused)
- `raf_*.py` — RAF processing experiments
