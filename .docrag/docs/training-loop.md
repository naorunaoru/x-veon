---
title: Training Loop and Fine-Tuning Modes
tags: [training, loop, checkpoints, fine-tuning]
scope: train.py (training orchestration, checkpointing, mode switching)
generated: 2026-03-22
commit: 347c8dd
---

## Context

`train.py` is the single entry point for all model training in x-veon. It orchestrates dataset construction, loss function assembly, the train/evaluate loop, checkpoint management, and training history persistence. The script supports two modes -- `train` (initial training from scratch) and `finetune` (continuing from a pretrained checkpoint) -- which differ in default loss weights and optimizer state handling but share the same loop, data pipeline, and checkpoint format. Both modes are fully configurable via CLI arguments; the mode flag selects a preset starting point, not a hard-coded workflow.

The model being trained is `XTransUNet` described in [unet-model-architecture](unet-model-architecture.md). Loss functions are documented in [loss-functions](loss-functions.md). Data augmentations are documented in [training-augmentations](training-augmentations.md). Synthetic torture patterns mixed into training data are documented in [torture-patterns](torture-patterns.md).

## Pattern / Approach

### TrainConfig Dataclass

Training configuration is centralized in the `TrainConfig` dataclass, which is the single source of truth for all training parameters. CLI arguments map 1:1 to dataclass fields. The config resolution priority is: **preset defaults -> checkpoint config -> CLI overrides**. All CLI argument defaults are `None`; only explicitly-provided arguments override the base config, which prevents unintentional resets when resuming from a checkpoint.

`TrainConfig` provides two preset constructors:

- `TrainConfig.base()` -- returns default-constructed config (L1=1.0, gradient=0.1, chroma=0.05, zipper=0.05, MS-SSIM=0.0, lr=1e-3, 200 epochs).
- `TrainConfig.finetune()` -- returns config with mode=finetune, lr=1e-4, 50 epochs, L1=0.5, MS-SSIM=0.3, gradient=0.2, chroma=0.02, zipper=0.1.

These presets are convenience starting points. All individual weights are overridable via CLI arguments. See [loss-functions](loss-functions.md) for why presets should not be treated as canonical configurations.

`TrainConfig` also provides serialization: `to_dict()` / `save()` write JSON, and `from_json()` reconstructs a config from a checkpoint's `config.json` (skipping transient fields like `data_range`, `device`, `resume`, `from_checkpoint`). The `build_criterion(data_range)` method constructs a `DemosaicLoss` directly from the config's loss fields, replacing the old pattern of constructing a preset then mutating it.

CLI arguments are organized into these groups:

**Data arguments**: `--data-dir` (required, one or more directories; supports `path:N` per-dir limits), `--cfa-type` (xtrans or bayer), `--max-images` (global cap applied after per-dir limits), `--filter-file` (JSON allowlist of image stems).

**Training arguments**: `--mode` (train or finetune), `--epochs`, `--batch-size`, `--patch-size`, `--lr`, `--warmup-epochs` (linear LR warmup), `--val-split` (fraction held out, default 0.1), `--patches-per-image`.

**Loss weight arguments**: `--l1-weight`, `--msssim-weight`, `--gradient-weight`, `--chroma-weight`, `--color-bias-weight`, `--zipper-weight`, `--huber` (use Huber instead of L1), `--huber-delta`, `--per-channel-norm`, `--recon-only` (L1/Huber only on reconstructed non-CFA pixels), `--known-pixel-weight` (weight for known-pixel preservation when recon-only, default 0.1), `--data-range`.

**Augmentation arguments**: `--noise-min`, `--noise-max`, `--shot-noise-max`, `--olpf-sigma-max`, `--wb-aug-range`, `--bright-spot-prob` (probability of synthetic bright spot augmentation), `--bright-spot-intensity-max`, `--bright-spot-sigma-max`, `--downscale-prob` (probability of 2x area-average downscale), `--torture-fraction`, `--torture-patterns`. See [training-augmentations](training-augmentations.md).

**Checkpoint arguments**: `--output-dir` (default `./checkpoints`), `--resume` (path to a `.pt` file), `--from-checkpoint` (load config from checkpoint dir, auto-resumes from best.pt), `--no-resume` (skip auto-resume when using --from-checkpoint).

**Model arguments**: `--base-width` (channel width, default 64).

**Performance arguments**: `--workers` (DataLoader workers), `--seed` (random seed), `--amp` (enable mixed precision), `--cache-patches` (pre-extract patches into RAM), `--cache-gb` (memory budget for patch cache).

### Data Loading and Splitting

**Multi-directory collection.** `--data-dir` accepts one or more directories. Each entry supports an optional `:N` suffix to limit files from that directory (e.g. `--data-dir /data/outdoor:200 /data/indoor:100 /data/studio`). `LinearDataset.find_files()` collects `.npy` files from each directory (excluding `_meta.npy` and `_lum.npy` sidecars). After per-dir limits are applied, `--max-images` applies a global cap on the combined file list.

**Image-level splitting.** The combined file list is shuffled with a seeded RNG (`--seed`, default 42) and partitioned into validation and training sets. The validation set size is `max(1, int(total_images * val_split))`, ensuring at least one validation image. This prevents patch-level leakage where patches from the same source image appear in both train and validation sets.

Each image yields `--patches-per-image` (default 16) random crops per epoch, so the effective dataset size is `num_images * patches_per_image`. Crops are aligned to the CFA grid period (6 for X-Trans, 2 for Bayer) to ensure valid mosaic patterns.

**Torture pattern mixing.** When `--torture-fraction` is greater than zero, `create_mixed_dataset()` wraps the real-image `LinearDataset` in a `ConcatDataset` with a `TortureDataset`. See [torture-patterns](torture-patterns.md) for pattern generation details.

**Patch cache mode.** When `--cache-patches` is set (and torture mixing is not active), a `PatchCacheDataset` pre-extracts all patches into a contiguous RAM array at startup. This eliminates per-batch disk I/O during training. The `--cache-gb` argument optionally caps the memory budget. `PatchCacheDataset` uses a staging buffer architecture: background threads continuously extract fresh patches into staging slots, which are swapped into the active set between epochs (during the gap between `train_epoch()` returning and the next epoch starting). This provides data diversity without blocking the training loop. When cache mode is active, `DataLoader` uses `multiprocessing_context='fork'` so workers share the patch array via copy-on-write pages. Cache mode calls `train_dataset.start_streaming()` before the loop and `train_dataset.cleanup()` after.

**DataLoader configuration.** For non-cached datasets, an `ImageGroupedSampler` groups patches by source image for cache-friendly I/O. For cached datasets, standard shuffle is used instead (patches are already in RAM). Both paths use `drop_last=True` for training. The validation DataLoader uses default `drop_last=False`.

**GPU-resident validation.** On CUDA devices, all validation batches are pre-loaded to GPU memory before training starts. The `evaluate()` function accepts an optional `gpu_batches` parameter to use these pre-materialized tensors, avoiding CPU-GPU transfers during validation.

The validation set always uses `LinearDataset` with augmentation disabled and zero noise. It never includes torture patterns.

### White Balance Mode

When `--apply-wb` is set, per-image white balance multipliers are loaded from sidecar `_meta.json` files and applied to RGB data before mosaicing. This means the model learns to produce white-balanced output. In this mode, pixel values can exceed 1.0 because WB gains amplify R and B channels. The effective data range is auto-computed from metadata (maximum of `range_max * max(wb_r, wb_b) / wb_g` across all images) and passed to the loss function for correct SSIM constant scaling. The auto-computed range can be overridden with `--data-range`.

### AMP Mixed Precision

The `--amp` flag enables PyTorch's automatic mixed precision via `torch.autocast` and `torch.amp.GradScaler`. When enabled:

1. Forward pass and loss computation run inside `torch.autocast(device.type)`, which selects float16 for eligible operations while keeping reductions and accumulations in float32.
2. `GradScaler` scales the loss before backward to prevent float16 gradient underflow, then unscales gradients before the optimizer step.
3. The scaler state is saved in checkpoints and restored on resume.

AMP is supported on CUDA and MPS backends. On CPU, the scaler is a no-op.

### The train_epoch / evaluate Loop

The main loop iterates from `start_epoch` to `cfg.epochs`. Each epoch consists of a `train_epoch()` call followed by an `evaluate()` call, then scheduler step, logging, checkpoint saves, and history persistence.

**`train_epoch(model, loader, optimizer, criterion, device, scaler)`** sets the model to training mode, iterates over all batches, and for each batch:

1. Moves inputs, targets, and clip_levels to device (non-blocking).
2. Zeros gradients.
3. Runs the forward pass under `torch.autocast` (if AMP enabled).
4. When `criterion.recon_only` is set, extracts `channel_masks` from input channels 1-3 (the per-channel CFA masks) and passes them to the criterion.
5. Calls `criterion(outputs, targets, clip_levels=clip_levels, channel_masks=channel_masks)`, which returns `(loss, components_dict)`.
6. Backpropagates via `scaler.scale(loss).backward()` (AMP) or `loss.backward()` (standard).
7. Clips gradients with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
8. Steps the optimizer.
9. Accumulates batch loss and per-component losses on GPU (no `.item()` sync until epoch end). PSNR is computed under `torch.no_grad()`.

Returns average loss, average PSNR, and averaged per-component losses across all batches (single GPU sync point at epoch end).

**`evaluate(model, loader, criterion, device, use_amp, gpu_batches)`** is decorated with `@torch.no_grad()`, sets the model to eval mode, and computes the same metrics without gradient tracking or optimizer steps. When `gpu_batches` is provided (pre-materialized validation data on GPU), it iterates those instead of the loader, skipping CPU-to-GPU transfers.

**PSNR** is computed as `-10 * log10(MSE)` between predicted and target tensors, clamped to 100 dB for near-zero MSE. This is the primary metric for checkpoint selection.

After both functions return, `scheduler.step()` advances the cosine annealing learning rate schedule.

### Training Dashboard

Epoch progress and metrics are displayed via a `TrainingDashboard` (from `dashboard.py`) rather than plain print statements. The dashboard is started before data loading and receives `EpochData` updates after each epoch. It tracks train/val PSNR, per-component losses, learning rate, and epoch timing (separate train and validation wall-clock times). The dashboard also performs NaN/Inf detection: if `has_fatal_error` is set, the training loop breaks early.

After the dashboard stops (at training completion, interruption, or error), a plain-text summary is printed to the console with elapsed time, epoch count, best PSNR, and configuration highlights.

### Optimizer and Scheduler

The optimizer is `AdamW` with configurable learning rate (`--lr`, default `1e-3`) and fixed weight decay of `1e-4`. The base scheduler is `CosineAnnealingLR`. When `--warmup-epochs` is greater than 0, a `LinearLR` warmup (start_factor=1e-3) is prepended via `SequentialLR`, providing a warmup-then-cosine schedule.

The cosine `T_max` depends on whether this is a same-run resume (output dir matches checkpoint dir) or a new run. For same-run resume, `T_max` equals the total epoch count and the scheduler state is restored. For a new run from a checkpoint (different output dir), `T_max` spans only the remaining epochs, giving a full cosine decay cycle for the new phase.

In `finetune` mode, the typical learning rate is `1e-4` (10x lower than base training), providing a conservative step size appropriate for refining an already-converged model.

### Mode Switching: train vs. finetune

The `--mode` flag controls two things:

**Loss preset selection.** The `TrainConfig` preset (base or finetune) sets initial loss weights. `build_criterion()` constructs the `DemosaicLoss` directly from config fields -- there is no post-construction mutation:

- `train` (`TrainConfig.base()`): L1=1.0, gradient=0.1, chroma=0.05, zipper=0.05, MS-SSIM=0.0. Prioritizes PSNR.
- `finetune` (`TrainConfig.finetune()`): L1=0.5, MS-SSIM=0.3, gradient=0.2, chroma=0.02, zipper=0.1. Adds perceptual loss.

These presets are convenience starting points. All individual weights are overridable via CLI arguments. See [loss-functions](loss-functions.md) for the full discussion of why presets should not be treated as canonical configurations.

**Checkpoint resume behavior.** `--resume` specifies a `.pt` file directly. `--from-checkpoint` specifies a checkpoint directory: it loads `config.json` as the base config (all parameters inherited), then auto-resumes from `best.pt` or `latest.pt` in that directory. `--no-resume` skips the auto-resume (config only). CLI arguments always override loaded config.

When resuming:

- `train` mode: restores model weights, optimizer state, and epoch counter. Whether scheduler state and best PSNR are restored depends on whether the output directory matches the checkpoint directory (same-run detection). If writing to the same dir, scheduler and best PSNR carry over (crash recovery). If writing to a new dir, the cosine schedule spans only the remaining epochs and best PSNR tracking starts fresh.
- `finetune` mode: restores model weights only. Creates a fresh optimizer and scheduler. Epoch counter starts at 0. Best PSNR resets to 0. This prevents the fine-tuning phase from inheriting stale momentum.

### Checkpoint Management

The script writes four files to `--output-dir`:

**`best.pt`** -- Saved whenever validation PSNR improves. Contains `model`, `optimizer`, `scheduler`, `epoch`, `best_val_psnr`, `base_width`, `cfa_type`, and optionally `scaler` state. This is the checkpoint used for inference and export.

**`latest.pt`** -- Saved every 10 epochs regardless of validation performance. Same format as `best.pt`. Serves as a crash-recovery point during long training runs.

**`config.json`** -- Written once at the start of training via `TrainConfig.save()`. Contains all `TrainConfig` fields (via `dataclasses.asdict`) plus extra runtime fields: `device`, `data_range`, `from_checkpoint`, and `resume`. This captures the complete configuration that produced the run, making any checkpoint reproducible.

**`history.json`** -- Rewritten after every epoch. Contains a JSON array of per-epoch records, each with: `epoch`, `train_loss`, `train_psnr`, `train_components` (dict of per-component losses), `val_loss`, `val_psnr`, `val_components`, `lr`, `time` (wall-clock seconds for the epoch).

### Checkpoint Registry

Every `best.pt` and `latest.pt` save also calls `update_registry()` from `checkpoint_registry.py`, which writes to `checkpoint_registry.json` at the project root. The registry structure is: `cfa_type -> base_width -> variant -> status -> slot`, where each slot stores the path, epoch, PSNR, loss, and history file path. New checkpoints are registered as `"beta"` status. When training completes all epochs without interruption or NaN/Inf errors, `promote_to_stable()` flips the entry from `"beta"` to `"stable"`. This allows downstream tools (inference, export) to discover the best available checkpoint programmatically.

### Checkpoint Contents

Each `.pt` checkpoint stores:

```
{
    "epoch": int,              # 0-indexed epoch number
    "model": OrderedDict,      # model.state_dict()
    "optimizer": OrderedDict,  # optimizer.state_dict()
    "scheduler": dict,         # scheduler.state_dict()
    "best_val_psnr": float,    # best validation PSNR seen so far
    "base_width": int,         # model architecture width (needed for reconstruction)
    "cfa_type": str,           # "xtrans" or "bayer"
    "scaler": dict,            # (optional) GradScaler state, present only if AMP enabled
}
```

The `base_width` and `cfa_type` fields are embedded in the checkpoint so that inference and export scripts (`infer.py`, `infer_hdr.py`, `export_onnx.py`) can reconstruct the correct model architecture without requiring external configuration.

Model creation now passes `cfa_period` from the CFA registry: `cfa_period(CFA_REGISTRY[cfg.cfa_type])` is resolved at runtime (6 for X-Trans, 2 for Bayer) and passed to `XTransUNet(base_width=..., cfa_period=...)`. This allows the model to adapt its architecture to the CFA grid period.

### Device Selection

`get_device()` selects the compute device by priority: MPS (Apple Silicon) > CUDA > CPU. The selected device string is stored in `config.json` for reference but does not affect checkpoint portability -- checkpoints are loaded with `map_location=device`, so a checkpoint trained on CUDA can be loaded on MPS or CPU.

### Typical Workflow

A standard two-phase training workflow:

```bash
# Phase 1: base training -- high PSNR
python train.py --data-dir /data/npy --epochs 200 --lr 1e-3 \
    --l1-weight 1.0 --gradient-weight 0.1 --chroma-weight 0.05 \
    --noise-max 0.005 --amp --output-dir checkpoints_base

# Phase 2: fine-tuning -- perceptual quality (using --from-checkpoint)
python train.py --from-checkpoint checkpoints_base/ \
    --mode finetune --lr 1e-4 --epochs 50 \
    --msssim-weight 0.3 --zipper-weight 0.1 \
    --torture-fraction 0.05 --amp --output-dir checkpoints_ft
```

Phase 1 drives validation PSNR as high as possible using L1-dominated loss. Phase 2 uses `--from-checkpoint` to inherit the full config (including data dirs), then overrides mode, LR, epochs, and loss weights. It auto-resumes from `best.pt` in the checkpoint directory with a fresh optimizer (finetune mode). Each phase writes its own independent checkpoint directory with its own `config.json`, `history.json`, `best.pt`, and `latest.pt`.

To resume a crashed base training run (same configuration, same output directory):

```bash
python train.py --from-checkpoint checkpoints_base/ --output-dir checkpoints_base
```

Because the output directory matches the checkpoint directory, the scheduler state and best PSNR are restored (same-run detection). The optimizer and epoch counter resume from the saved state.

Multi-directory training with per-dir limits:

```bash
python train.py --data-dir /data/outdoor:200 /data/indoor:100 /data/studio \
    --max-images 500 --epochs 200 --amp --output-dir checkpoints_multi
```

## Rationale

### Why AMP Mixed Precision

X-Trans demosaicing training is memory-bound. The model processes 96x96 (or larger) 3-channel patches through a 5-level UNet with up to 1024 channels at the bottleneck. AMP roughly halves activation memory, allowing larger batch sizes on the same GPU. The float16 forward pass is also faster on GPUs with Tensor Cores. The risk of float16 underflow in gradients is managed by `GradScaler`, which dynamically adjusts the loss scale. Loss computation inside autocast still accumulates reductions in float32, so PSNR and loss tracking remain numerically stable.

### Why Dual Checkpoints (best.pt and latest.pt)

`best.pt` captures the peak-performing model weights according to validation PSNR. But PSNR can plateau for tens of epochs before a new best is found, during which a crash would lose all progress since the last `best.pt` save. `latest.pt` (saved every 10 epochs) provides a recent recovery point. The 10-epoch interval balances I/O cost against recovery granularity -- losing at most 10 epochs of work is acceptable for runs typically lasting 50--200 epochs.

### Why config.json Captures Full Config State

Neural network training is notoriously difficult to reproduce. Small differences in learning rate, loss weights, augmentation parameters, or random seed can produce meaningfully different results. `TrainConfig.save()` serializes all dataclass fields (via `dataclasses.asdict`) plus runtime extras (`device`, `data_range`, `from_checkpoint`, `resume`) to JSON at the start of training. This captures every configurable parameter, including those left at defaults, which is important because defaults can change between code versions. The same `config.json` can be loaded back via `TrainConfig.from_json()` to reproduce or continue a run.

### Why Fresh Optimizer in Finetune Mode

The base training phase uses an L1-dominated loss, which creates an optimizer state (AdamW momentum and variance estimates) tuned to that loss landscape. When fine-tuning switches to a perceptual loss (adding MS-SSIM, increasing gradient weight), the loss surface changes substantially. Reusing the old optimizer state would cause the optimizer to take steps based on stale curvature estimates, leading to instability or slow convergence in the early fine-tuning epochs. Starting fresh forces the optimizer to re-estimate momentum from the new loss landscape. The cosine scheduler also resets, providing a full warmup-to-decay cycle for the fine-tuning phase.

### Why Image-Level Splitting

Splitting at the image level (not the patch level) prevents data leakage. If patches from the same source image appeared in both train and validation sets, the model could memorize image-specific patterns and report artificially high validation PSNR. Since each source image contributes `patches_per_image` patches via random crops, adjacent crops can share significant content. Image-level splitting guarantees that the validation set evaluates generalization to unseen images.

## Key Files

| File | Role |
|------|------|
| `train.py` | Training orchestration: `TrainConfig` dataclass, CLI parsing, data loading, train/eval loop, checkpointing |
| `losses.py` | `DemosaicLoss` composite loss with preset constructors and CLI-overridable weights |
| `dataset.py` | `LinearDataset`, `PatchCacheDataset`, `TortureDataset`, `ImageGroupedSampler`, `create_mixed_dataset` |
| `model.py` | `XTransUNet` and `count_parameters` -- the model being trained |
| `cfa.py` | `CFA_REGISTRY`, `cfa_period()` -- CFA pattern definitions used for model creation and data alignment |
| `checkpoint_registry.py` | `update_registry()`, `promote_to_stable()` -- automatic checkpoint registration |
| `dashboard.py` | `TrainingDashboard`, `EpochData` -- live TUI dashboard for training progress |
| `export_onnx.py` | Reads checkpoint `.pt` files (uses `base_width` and `cfa_type` from checkpoint metadata) |
| `infer.py` | Inference using `best.pt` checkpoints |

## Antipatterns

**Do not treat mode presets as the training configuration.** The `--mode train` and `--mode finetune` flags select `TrainConfig.base()` and `TrainConfig.finetune()` presets respectively, but all production training runs used explicit CLI arguments for every loss weight. To determine what weights a checkpoint was trained with, read its `config.json`, not the preset source code. See the equivalent warning in [loss-functions](loss-functions.md).

**Do not resume finetune mode with `--mode train`.** Using `--mode train --resume checkpoint.pt` restores the optimizer and scheduler state, which is intended for crash recovery of the same training run. If the resumed checkpoint was produced by a different loss configuration, the restored optimizer momentum will be inconsistent with the current loss surface. Use `--mode finetune` when switching loss configurations to get a fresh optimizer.

**Do not rely on `latest.pt` for final results.** `latest.pt` is a periodic snapshot for crash recovery. It may have lower validation PSNR than `best.pt` because it captures whatever epoch fell on a 10-epoch boundary. Always use `best.pt` for inference, export, and evaluation.

**Do not import from `src/` training code.** Files under `src/` (including `src/datasets/`, `src/losses.py`, `src/train_*.py`) are legacy. The active training pipeline is the top-level `train.py` importing from top-level `dataset.py`, `losses.py`, and `model.py`.

**Do not mix validation-set torture patterns.** The validation set is always constructed as a plain `LinearDataset` with augmentation disabled. Adding synthetic patterns to validation would inflate metrics by testing on trivially-reconstructable synthetic data rather than real photographic content.
