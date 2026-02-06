# X-Trans Demosaicing

## Training Versions

### v4 (baseline)
- Linear sensor space, L1 + gradient + chroma loss
- Best: 48.2 dB PSNR
- Issue: some color non-uniformity (red cast near edges)

### v4.1 (in progress)
- Added MS-SSIM to loss (L1 0.65 + MS-SSIM 0.2 + gradient 0.15)
- 5% torture pattern mixing
- 400 images, 100 epochs

### v4.2 (planned)
- Add luminance reference channel to training data
- Weighted monochrome from CFA (0.299R + 0.587G + 0.114B weights per pixel)
- 4x downscaled alongside RGB targets
- Try luminance matching loss (full and/or high-freq only)
- Goal: improve color uniformity without sacrificing detail with Neural Networks

Neural network-based demosaicing for Fujifilm X-Trans sensors using a U-Net architecture. Operates in linear sensor space with HDR output capability.

## Quick Start

### Installation

```bash
pip install torch numpy rawpy pillow pillow-avif opencv-python
```

### Inference (Recommended)

Process a Fujifilm RAF file to HDR AVIF:

```bash
python scripts/inference/production/infer_hdr.py input.RAF output.avif \
    --checkpoint checkpoints/v4_production/best.pt
```

Process to standard sRGB:

```bash
python scripts/inference/production/infer_srgb.py input.RAF output.tiff \
    --checkpoint checkpoints/v4_production/best.pt
```

### Training

Train from scratch on the v4 dataset:

```bash
python scripts/train/train_v4.py \
    --data-dir /Volumes/External/xtrans-demosaic/datasets/xtrans_v4_dataset \
    --output-dir checkpoints/my_experiment \
    --epochs 200 \
    --batch-size 32
```

## Project Structure

```
xtrans-demosaic/
├── src/                          # Core library code
│   ├── model.py                  # U-Net architecture (31M params)
│   ├── xtrans_pattern.py         # X-Trans CFA pattern utilities
│   ├── losses.py                 # Loss functions (L1, gradient, chroma, SSIM)
│   └── datasets/
│       ├── dataset_v4.py         # Dataset loader for training
│       └── build_dataset_v4.py   # Dataset builder (one-time use)
│
├── scripts/
│   ├── train/
│   │   ├── train_v4.py           # ✓ PRODUCTION training script
│   │   └── experiments/          # Experimental training variants
│   │       ├── train_v4_ssim_fixed.py
│   │       ├── train_fullres_ssim.py
│   │       └── train_torture.py  (FAILED - catastrophic forgetting)
│   ├── inference/
│   │   ├── production/
│   │   │   ├── infer_hdr.py     # ✓ HDR AVIF output (recommended)
│   │   │   └── infer_srgb.py    # Standard sRGB TIFF output
│   │   └── test/                # Test/experimental inference scripts
│   └── utils/
│       ├── torture_v2.py        # Synthetic pattern generator
│       └── compare_checkpoints.py
│
├── checkpoints/
│   ├── v4_production/           # ✓ USE THIS (48.2 dB PSNR)
│   │   └── best.pt
│   ├── experiments/
│   │   ├── checkpoints_v4_ft/   # ✗ BROKEN (6×6 artifacts)
│   │   ├── checkpoints_torture/ # ✗ BROKEN (catastrophic forgetting)
│   │   └── checkpoints_v4_ssim/ # Empty (experiment not run)
│   └── archive/                 # Historical checkpoints
│
├── data/                        # Training data metadata + symlinks
│   ├── sharpness_scores.json   # Laplacian variance for all RAFs
│   ├── top200_sharp.json       # Top 200 sharpest images
│   ├── top2000_sharp.json      # Top 2000 sharpest images
│   ├── xtrans_v4_dataset/      # → Symlink to external drive
│   ├── fuji_jpgs/              # → Symlink to external drive
│   ├── jpeg_patches/           # → Symlink to external drive
│   ├── torture_dataset/        # → Symlink to external drive
│   └── fullres_dataset/        # → Symlink to external drive
│
├── docs/
│   ├── PROJECT.md              # Detailed technical documentation
│   └── EXPERIMENTS.md          # Experiment tracking and results
│
└── logs/                       # Training logs

External datasets (on /Volumes/External/xtrans-demosaic/):
  datasets/
    ├── xtrans_v4_dataset/     (60GB) - ✓ Production training data (~2350 images)
    ├── fullres_dataset/       (38GB) - Full-res RAF+JPG pairs (947 images)
    ├── fuji_jpgs/             (32GB) - Camera JPEG outputs (~3500 images)
    ├── torture_dataset/       (1.5GB) - Synthetic test patterns (5000 samples)
    └── jpeg_patches/          (1.7GB) - Pre-extracted patches (16000 files)

  See /Volumes/External/xtrans-demosaic/README.md for detailed dataset documentation.
```

## Model Architecture

**U-Net Encoder-Decoder**
- Input: 4 channels (CFA mosaic + R/G/B position masks)
- Output: 3 channels (RGB)
- Levels: 4 encoder + 4 decoder + bottleneck
- Channel progression: 64 → 128 → 256 → 512 → 1024 → back down
- Parameters: 31,038,339

**Key Features:**
- Operates in **linear sensor space** (no gamma, no WB during training)
- Preserves HDR highlights (values can exceed 1.0)
- Receptive field covers 2-3 X-Trans pattern repeats (12-18 pixels)

## Training Details

### v4 Model (PRODUCTION - 48.2 dB)

**Dataset:** DHT-demosaiced RAFs, 4× downsampled, linear space
**Loss:** L1 + 0.1×gradient + 0.05×chroma
**Result:** 48.21 dB PSNR, good general quality
**Checkpoint:** `checkpoints/v4_production/best.pt`

The model learns to produce output similar to DHT demosaicing but can potentially exceed it by learning statistical relationships.

### Failed Experiments

**v4_ft (fine-tune on JPEGs):** Produced 6×6 grid artifacts. Root cause under investigation.

**Torture fine-tune:** Catastrophic forgetting. Model forgot how to process natural images while optimizing for synthetic patterns.

## HDR Output Pipeline

The production inference script ([infer_hdr.py](scripts/inference/production/infer_hdr.py)) preserves highlight detail:

1. Load RAF → extract raw CFA (no clipping)
2. Normalize to linear sensor space
3. Tiled inference with blending (288×288 patches, 48px overlap)
4. Apply camera white balance
5. HLG encoding (Hybrid Log-Gamma for HDR)
6. Save as 10-bit AVIF with CICP 9/18/9

**Why AVIF + HLG?**
- HEIC + HLG/PQ is broken on macOS (requires Apple's proprietary gain map)
- AVIF + HLG works correctly with wide compatibility
- HLG is backward-compatible with SDR displays

## Common Tasks

### Process a batch of RAFs

```bash
python scripts/inference/production/infer_hdr.py \
    /path/to/raf/folder \
    /path/to/output \
    --batch \
    --checkpoint checkpoints/v4_production/best.pt
```

### Train on a subset (faster iteration)

```bash
# Using direct path to external drive
python scripts/train/train_v4.py \
    --data-dir /Volumes/External/xtrans-demosaic/datasets/xtrans_v4_dataset \
    --filter-file data/top200_sharp.json \
    --epochs 50 \
    --batch-size 32

# Or using symlink (if created)
python scripts/train/train_v4.py \
    --data-dir data/xtrans_v4_dataset \
    --filter-file data/top200_sharp.json \
    --epochs 50 \
    --batch-size 32
```

### Compare two checkpoints

```bash
python scripts/utils/compare_checkpoints.py \
    checkpoints/v4_production/best.pt \
    checkpoints/experiments/my_model.pt \
    test_rafs/DSCF3561.RAF
```

## Key Insights

1. **Linear space is essential:** Training in linear sensor space preserves HDR and avoids gamma-related artifacts
2. **CFA alignment matters:** Augmentations must preserve 6×6 pattern alignment (safe: h/v flips, 180° rotation)
3. **Synthetic data doesn't transfer:** Pure synthetic fine-tuning causes catastrophic forgetting
4. **Teacher network works:** Using DHT as "teacher" produces excellent results

## Open Questions

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for:
- Investigation of v4_ft 6×6 artifacts
- Texture preservation improvements (SSIM loss, full-res training)
- Proper fine-tuning approach without catastrophic forgetting

## License

[Add your license here]

## Citation

[Add citation if applicable]
