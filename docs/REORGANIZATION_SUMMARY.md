# Project Reorganization Summary

**Date:** 2026-02-03

This document summarizes the project reorganization and provides a guide for using the new structure.

## Changes Overview

The project has been reorganized from a flat structure with scattered files into a clean, hierarchical organization.

## New Directory Structure

```
xtrans-demosaic/
├── README.md                    # ✨ NEW: Quick start guide
├── .gitignore                   # ✨ NEW: Git ignore patterns
│
├── src/                         # ✨ NEW: Core library code
│   ├── model.py
│   ├── xtrans_pattern.py
│   ├── losses.py
│   └── datasets/
│       ├── dataset_v4.py
│       └── build_dataset_v4.py  # Moved from _obsolete
│
├── scripts/                     # ✨ NEW: Organized scripts
│   ├── train/
│   │   ├── train_v4.py         # ✓ Production training
│   │   └── experiments/
│   │       ├── train_v4_ssim.py
│   │       ├── train_v4_ssim_fixed.py
│   │       ├── train_fullres_ssim.py
│   │       └── train_torture.py
│   ├── inference/
│   │   ├── production/
│   │   │   ├── infer_hdr.py    # ✓ HDR AVIF (recommended)
│   │   │   └── infer_srgb.py   # Standard sRGB output
│   │   └── test/
│   │       ├── infer_quick.py
│   │       ├── test_hdr_inference.py
│   │       └── hdr_heic.py
│   └── utils/
│       ├── torture_v2.py
│       └── compare_checkpoints.py  # Moved from _obsolete
│
├── checkpoints/                 # Reorganized
│   ├── v4_production/          # Renamed from checkpoints_v4
│   │   ├── best.pt             # ✓ USE THIS
│   │   ├── config.json
│   │   └── history.json
│   ├── experiments/            # Failed/WIP experiments
│   │   ├── checkpoints_v4_ft/
│   │   ├── checkpoints_v4_ssim/
│   │   └── checkpoints_torture/
│   └── archive/                # Historical checkpoints
│       ├── v3_20260202_39.1dB_ep79.pt
│       └── v4_linear_ep105_48.2dB.pt
│
├── docs/                        # Documentation
│   ├── PROJECT.md              # Detailed technical docs
│   ├── EXPERIMENTS.md          # ✨ NEW: Experiment tracking
│   └── REORGANIZATION_SUMMARY.md  # This file
│
├── logs/                        # ✨ NEW: Consolidated logs
│   ├── train_v4.log
│   ├── train_v4_ft.log
│   ├── training.log
│   └── training_ssim.log
│
├── data/                        # Training data metadata
├── output/                      # Inference outputs
├── test_rafs/                   # Test RAF files
│
├── _obsolete/                   # Old scripts (reference only)
└── [old files]                  # Old root-level files (not deleted yet)
    ├── model.py                 # → src/model.py
    ├── train_v4.py             # → scripts/train/train_v4.py
    ├── infer_hdr.py            # → scripts/inference/production/infer_hdr.py
    └── ...
```

## File Mapping (Old → New)

### Core Code
- `model.py` → `src/model.py`
- `xtrans_pattern.py` → `src/xtrans_pattern.py`
- `losses.py` → `src/losses.py`
- `dataset_v4.py` → `src/datasets/dataset_v4.py`
- `_obsolete/build_dataset_v4.py` → `src/datasets/build_dataset_v4.py` ✨

### Training Scripts
- `train_v4.py` → `scripts/train/train_v4.py`
- `train_v4_ssim.py` → `scripts/train/experiments/train_v4_ssim.py`
- `train_v4_ssim_fixed.py` → `scripts/train/experiments/train_v4_ssim_fixed.py`
- `train_fullres_ssim.py` → `scripts/train/experiments/train_fullres_ssim.py`
- `train_torture.py` → `scripts/train/experiments/train_torture.py`

### Inference Scripts
- `infer_hdr.py` → `scripts/inference/production/infer_hdr.py`
- `infer_v4_linear.py` → `scripts/inference/production/infer_srgb.py` (renamed)
- `infer_quick.py` → `scripts/inference/test/infer_quick.py`
- `test_hdr_inference.py` → `scripts/inference/test/test_hdr_inference.py`
- `hdr_heic.py` → `scripts/inference/test/hdr_heic.py`

### Utilities
- `torture_v2.py` → `scripts/utils/torture_v2.py`
- `_obsolete/compare_checkpoints.py` → `scripts/utils/compare_checkpoints.py` ✨

### Checkpoints
- `checkpoints_v4/` → `checkpoints/v4_production/`
- `checkpoints_v4_ft/` → `checkpoints/experiments/checkpoints_v4_ft/`
- `checkpoints_torture/` → `checkpoints/experiments/checkpoints_torture/`
- `checkpoints_v4_ssim/` → `checkpoints/experiments/checkpoints_v4_ssim/`
- `checkpoints_archive/` → `checkpoints/archive/`

### Logs
- `*.log` → `logs/*.log`

## Updated Command Examples

### Before:
```bash
# Training
python train_v4.py --data-dir /Volumes/External/xtrans_v4_dataset --epochs 200

# Inference
python infer_hdr.py input.RAF output.avif --checkpoint checkpoints_v4/best.pt
```

### After:
```bash
# Training
python scripts/train/train_v4.py \
    --data-dir /Volumes/External/xtrans_v4_dataset \
    --output-dir checkpoints/my_experiment \
    --epochs 200

# Inference
python scripts/inference/production/infer_hdr.py \
    input.RAF output.avif \
    --checkpoint checkpoints/v4_production/best.pt
```

## Important Notes

### 1. Old Files Still Present
The old files in the root directory are **still present** as copies. They have **not been deleted** to ensure nothing breaks.

**Next step:** After testing the new structure, you can safely delete:
- Old root-level `.py` files (model.py, train_v4.py, infer_*.py, etc.)
- Old checkpoint directories (checkpoints_v4/, checkpoints_v4_ft/, etc.)
- Old log files (only if you've verified logs/ has them)

### 2. Import Paths Updated
All moved scripts have had their import paths updated to use:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
```

This allows them to import from the `src/` directory regardless of where they're run from.

### 3. Checkpoint Paths
Update your checkpoint paths in any scripts or workflows:
- `checkpoints_v4/best.pt` → `checkpoints/v4_production/best.pt`

### 4. Dataset References
The dataset builder has been moved from `_obsolete/` to `src/datasets/` since it's still useful for rebuilding the dataset if needed.

## Testing the New Structure

### Test 1: Verify Imports
```bash
# From project root
cd /Users/naoru/projects/xtrans-demosaic

# Test that imports work
python -c "import sys; sys.path.insert(0, 'src'); from model import XTransUNet; print('✓ Imports work')"
```

### Test 2: Quick Inference
```bash
python scripts/inference/test/infer_quick.py \
    checkpoints/v4_production/best.pt \
    test_rafs/DSCF3561.RAF \
    test_output.png
```

### Test 3: Training Script Help
```bash
python scripts/train/train_v4.py --help
```

## Cleanup Checklist

After verifying the new structure works:

- [ ] Test inference with new paths
- [ ] Test training script (at least --help or 1 epoch)
- [ ] Verify all needed files are in new locations
- [ ] Delete old root-level .py files
- [ ] Delete old checkpoint directories (keep checkpoints_archive separately if needed)
- [ ] Delete old log files in root
- [ ] Update any external scripts/notebooks that reference old paths

## Benefits of New Structure

1. **Clear Separation:** Core code (src) vs. scripts vs. data
2. **Easy to Find:** Production scripts vs. experiments clearly marked
3. **Import Clarity:** All imports come from src/
4. **Better Organization:** Related files grouped together
5. **Documentation:** README and EXPERIMENTS.md provide guidance
6. **Version Control:** .gitignore properly configured

## Questions?

See:
- [README.md](../README.md) - Quick start and usage
- [PROJECT.md](PROJECT.md) - Technical details
- [EXPERIMENTS.md](EXPERIMENTS.md) - Experiment tracking

---

**Reorganization completed:** 2026-02-03
