# Obsolete Files

These files are superseded by the current v4 training pipeline but kept for reference.

## Training Scripts
- `train_v3.py` - v3 training (MPS), superseded by train_v4.py
- `train_patches.py`, `train_patches_ram.py` - early fine-tuning attempts
- `train_v4_finetune.py`, `train_v4_finetune_fast.py` - fine-tuning wrappers, we use train_v4.py directly

## Datasets
- `dataset.py` - v2 dataset loader (sRGB-based)
- `dataset_v4_finetune.py` - fine-tuning dataset, replaced by pre-converted jpeg_patches/

## Inference
- `infer.py` - v3 inference (non-linear), superseded by infer_v4_linear.py

## Dataset Building
- `build_dataset_v4.py` - one-shot script to build v4 dataset (done)
- `raf_*.py` - various RAF processing experiments

## Color Correction
- `fit_color.py`, `fit_histogram.py`, `fit_jpeg_color.py` - color correction experiments (unused)

## Utilities
- `compare_checkpoints.py` - A/B checkpoint comparison (rarely used)

---
Archived: 2026-02-03
