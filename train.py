#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Unified training script for X-Trans demosaicing.

Modes:
- train: Initial training from scratch (L1-focused)
- finetune: Fine-tune existing model

Examples:
    # Initial training (single directory)
    python train.py --data-dir /path/to/npy --epochs 200

    # Multiple directories with per-dir file limits
    python train.py --data-dir /data/outdoor:200 /data/indoor:100 /data/studio

    # Continue training from checkpoint (inherits all config, resumes optimizer/epoch)
    python train.py --from-checkpoint checkpoints/

    # Fine-tune from checkpoint with tweaked params
    python train.py --from-checkpoint checkpoints/ \
        --mode finetune --lr 1e-4 --epochs 50 --output-dir checkpoints_v2

    # Fine-tune with torture pattern mixing
    python train.py --data-dir /path/to/npy --resume checkpoints/best.pt \
        --mode finetune --torture-fraction 0.05
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import XTransUNet, count_parameters
from dataset import LinearDataset, PatchCacheDataset, create_mixed_dataset, ImageGroupedSampler
from losses import DemosaicLoss
from checkpoint_registry import update_registry, promote_to_stable, REGISTRY_FILENAME


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio in dB (returns GPU scalar, no sync)."""
    mse = ((pred - target) ** 2).mean()
    # clamp instead of where to avoid shape mismatch (scalar vs [1])
    return -10 * torch.log10(mse.clamp(min=1e-10))


def _compute_data_range(files: list[str]) -> float:
    """Compute max pixel value after WB from metadata."""
    import os
    peak = 1.0
    for npy_path in files:
        stem = os.path.splitext(npy_path)[0]
        meta_path = stem + "_meta.json"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            wb = meta["camera_wb"][:3]
            wb_max = max(wb[0], wb[2]) / wb[1]  # max gain relative to G
            range_max = meta.get("range_max", 1.0)
            peak = max(peak, range_max * wb_max)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
    return peak


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_psnr = torch.tensor(0.0, device=device)
    component_sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    use_amp = scaler is not None

    for batch in loader:
        inputs, targets, clip_levels = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        clip_levels = clip_levels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            channel_masks = inputs[:, 1:4] if criterion.recon_only else None
            loss, components = criterion(outputs, targets, clip_levels=clip_levels,
                                         channel_masks=channel_masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accumulate on GPU — no .item() sync until epoch end
        total_loss = total_loss + loss.detach().squeeze()
        for k, v in components.items():
            v = v.squeeze()
            if k in component_sums:
                component_sums[k] = component_sums[k] + v
            else:
                component_sums[k] = v.clone()
        with torch.no_grad():
            total_psnr = total_psnr + psnr(outputs, targets)
        n_batches += 1

    # Single sync point at epoch end
    avg_components = {k: (v / n_batches).item() for k, v in component_sums.items()}
    return (total_loss / n_batches).item(), (total_psnr / n_batches).item(), avg_components


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False, gpu_batches=None):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_psnr = torch.tensor(0.0, device=device)
    component_sums: dict[str, torch.Tensor] = {}
    n_batches = 0

    source = gpu_batches if gpu_batches is not None else loader
    for batch in source:
        if gpu_batches is not None:
            inputs, targets, clip_levels = batch
        else:
            inputs, targets, clip_levels = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            clip_levels = clip_levels.to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            channel_masks = inputs[:, 1:4] if criterion.recon_only else None
            loss, components = criterion(outputs, targets, clip_levels=clip_levels,
                                         channel_masks=channel_masks)

        total_loss = total_loss + loss.squeeze()
        for k, v in components.items():
            v = v.squeeze()
            if k in component_sums:
                component_sums[k] = component_sums[k] + v
            else:
                component_sums[k] = v.clone()
        total_psnr = total_psnr + psnr(outputs, targets)
        n_batches += 1

    avg_components = {k: (v / n_batches).item() for k, v in component_sums.items()}
    return (total_loss / n_batches).item(), (total_psnr / n_batches).item(), avg_components


def main():
    parser = argparse.ArgumentParser(description="CFA demosaicing training")

    # Data
    parser.add_argument("--data-dir", type=str, nargs="+", default=None,
                        help="Directories with .npy files. Use path:N to limit "
                             "to N files from that directory (e.g. /data/a:100 /data/b:50)")
    parser.add_argument("--cfa-type", type=str, default="xtrans",
                        choices=["xtrans", "bayer"],
                        help="CFA pattern type (default: xtrans)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Global cap on total images (applied after per-dir limits)")
    parser.add_argument("--filter-file", type=str, default=None,
                        help="JSON file with allowed image stems")
    
    # Training
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], default="train",
                        help="Training mode: 'train' for initial, 'finetune' for texture recovery")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs (default: 0)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patches-per-image", type=int, default=16)
    
    # Loss weights (override mode defaults)
    parser.add_argument("--l1-weight", type=float, default=None)
    parser.add_argument("--msssim-weight", type=float, default=None)
    parser.add_argument("--gradient-weight", type=float, default=None)
    parser.add_argument("--chroma-weight", type=float, default=None)
    parser.add_argument("--color-bias-weight", type=float, default=None,
                        help="Weight for mean color bias penalty (penalizes DC color shift)")
    parser.add_argument("--zipper-weight", type=float, default=None,
                        help="Weight for zipper artifact penalty (Laplacian 2nd-order oscillation)")
    parser.add_argument("--huber", action="store_true",
                        help="Use Huber loss instead of L1")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Delta for Huber loss")
    parser.add_argument("--per-channel-norm", action="store_true",
                        help="Normalize L1 loss per channel (addresses G >> R,B sample imbalance)")
    parser.add_argument("--recon-only", action="store_true",
                        help="Compute L1/Huber only on reconstructed (non-CFA) pixels")
    parser.add_argument("--known-pixel-weight", type=float, default=0.1,
                        help="Weight for known-pixel preservation when --recon-only (default: 0.1)")
    parser.add_argument("--data-range", type=float, default=None,
                        help="Max pixel value for SSIM constants (auto-computed from metadata when --apply-wb)")
    
    # White balance
    parser.add_argument("--apply-wb", action="store_true",
                        help="Apply per-image WB to training data (model learns WB'd output)")

    # Augmentation
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.005)
    parser.add_argument("--shot-noise-max", type=float, default=0.0,
                        help="Max shot noise coefficient for Poisson-Gaussian noise model (0 = disabled). "
                             "Noise std at pixel value x: sqrt(shot*x + read^2)")
    parser.add_argument("--olpf-sigma-max", type=float, default=0.0,
                        help="Max Gaussian sigma for OLPF blur simulation (0 = disabled). Applied to RGB before mosaicing.")
    parser.add_argument("--wb-aug-range", type=float, default=0.0,
                        help="WB shift augmentation range in log space (e.g. 0.25 = ~±28%%). Only with --apply-wb")
    parser.add_argument("--bright-spot-prob", type=float, default=0.0,
                        help="Probability of adding synthetic bright spots per sample (0-1).")
    parser.add_argument("--bright-spot-intensity-max", type=float, default=5.0,
                        help="Max intensity factor for bright spots relative to clip level (min is 1.5).")
    parser.add_argument("--bright-spot-sigma-max", type=float, default=20.0,
                        help="Max Gaussian sigma (pixels) for bright spots (min is 2.0).")
    parser.add_argument("--downscale-prob", type=float, default=0.0,
                        help="Probability of cutting 2x patch and area-averaging down (0-1, default 0 = disabled).")
    parser.add_argument("--torture-fraction", type=float, default=0.0,
                        help="Fraction of training data from synthetic torture patterns")
    parser.add_argument("--torture-patterns", type=int, default=500,
                        help="Number of unique torture patterns")
    
    # Checkpoints
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Load training config from checkpoint dir (auto-resumes from best.pt). "
                             "All params are inherited; override any with explicit CLI args.")
    
    # Model
    parser.add_argument("--base-width", type=int, default=64,
                        help="Base channel width (default 64 → 64/128/256/512/1024, use 32 for half-width)")

    # Performance
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 for main process)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (float16)")
    parser.add_argument("--cache-patches", action="store_true",
                        help="Pre-extract patches into RAM (eliminates disk I/O during training)")
    parser.add_argument("--regen-every", type=int, default=5,
                        help="(deprecated, ignored) Kept for checkpoint compat.")
    parser.add_argument("--replace-fraction", type=float, default=0.2,
                        help="(deprecated, ignored) Streaming is now continuous")
    parser.add_argument("--cache-gb", type=float, default=None,
                        help="Memory budget for patch cache in GB (default: sized to n_images * patches_per_image)")
    
    # Two-pass parse: if --from-checkpoint is given, load its config as defaults
    # so that explicit CLI args override, and everything else is inherited.
    _NO_INHERIT = {'device', 'data_range', 'resume'}
    pre_args, _ = parser.parse_known_args()

    if pre_args.from_checkpoint:
        ckpt_dir = Path(pre_args.from_checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        config_path = ckpt_dir / "config.json"
        if not config_path.exists():
            parser.error(f"No config.json found in {ckpt_dir}")
        with open(config_path) as f:
            ckpt_config = json.load(f)
        for key in _NO_INHERIT:
            ckpt_config.pop(key, None)
        parser.set_defaults(**ckpt_config)
        print(f"Loaded config from {config_path}")

    args = parser.parse_args()

    # Auto-set resume from checkpoint dir
    if args.from_checkpoint and args.resume is None:
        ckpt_dir = Path(args.from_checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        for name in ("best.pt", "latest.pt"):
            pt = ckpt_dir / name
            if pt.exists():
                args.resume = str(pt)
                break

    if args.data_dir is None:
        parser.error("--data-dir is required (either explicitly or via --from-checkpoint config)")

    # Normalize data_dir: ensure it's always a list (checkpoint config may store a string)
    if isinstance(args.data_dir, str):
        args.data_dir = [args.data_dir]

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Dataset — split at image level to prevent leakage
    # Collect files from all data directories, each with optional :N limit
    all_files = []
    for entry in args.data_dir:
        if ':' in entry and entry.rsplit(':', 1)[1].isdigit():
            dir_path, limit_str = entry.rsplit(':', 1)
            dir_limit = int(limit_str)
        else:
            dir_path = entry
            dir_limit = None
        dir_files = LinearDataset.find_files(
            dir_path,
            max_images=dir_limit,
            filter_file=args.filter_file,
        )
        print(f"  {dir_path}: {len(dir_files)} files" +
              (f" (limited to {dir_limit})" if dir_limit else ""))
        all_files.extend(dir_files)

    if args.max_images and len(all_files) > args.max_images:
        all_files = all_files[:args.max_images]

    print(f"\nTotal images: {len(all_files)}")
    random.Random(args.seed).shuffle(all_files)

    val_n_images = max(1, int(len(all_files) * args.val_split))
    val_files = all_files[:val_n_images]
    train_files = all_files[val_n_images:]
    print(f"  Images: {len(train_files)} train, {val_n_images} val")

    # Compute effective data range
    if args.data_range is not None:
        data_range = args.data_range
    elif args.apply_wb:
        data_range = _compute_data_range(all_files)
        print(f"  Auto data_range: {data_range:.2f}")
    else:
        data_range = 1.0

    wb_aug = args.wb_aug_range if args.apply_wb else 0.0
    if wb_aug > 0 and not args.apply_wb:
        print("  Warning: --wb-aug-range ignored without --apply-wb")

    shared_kwargs = dict(
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        apply_wb=args.apply_wb,
        cfa_type=args.cfa_type,
    )

    olpf_sigma = (0.0, args.olpf_sigma_max)

    spot_kwargs = dict(
        bright_spot_prob=args.bright_spot_prob,
        bright_spot_intensity=(1.5, args.bright_spot_intensity_max),
        bright_spot_sigma=(2.0, args.bright_spot_sigma_max),
    )

    train_augment_kwargs = dict(
        augment=True,
        noise_sigma=(args.noise_min, args.noise_max),
        shot_noise=(0.0, args.shot_noise_max),
        wb_aug_range=wb_aug,
        olpf_sigma=olpf_sigma,
        downscale_prob=args.downscale_prob,
        **spot_kwargs,
    )

    if args.torture_fraction > 0:
        train_dataset = create_mixed_dataset(
            data_dir=None,
            files=train_files,
            torture_fraction=args.torture_fraction,
            torture_patterns=args.torture_patterns,
            **train_augment_kwargs,
            **shared_kwargs,
        )
    elif args.cache_patches:
        print("  Pre-extracting patches into RAM...")
        train_dataset = PatchCacheDataset(
            files=train_files,
            seed=args.seed,
            cache_gb=args.cache_gb,
            **train_augment_kwargs,
            **shared_kwargs,
        )
        mem_gb = train_dataset._patch_data.nbytes / 1e9
        print(f"  Cached {len(train_dataset)} patches ({mem_gb:.1f} GB RAM)")
    else:
        train_dataset = LinearDataset(
            files=train_files,
            **train_augment_kwargs,
            **shared_kwargs,
        )

    val_dataset = LinearDataset(
        files=val_files,
        augment=False,
        noise_sigma=(0.0, 0.0),
        **shared_kwargs,
    )

    print(f"  Patches: {len(train_dataset)} train, {len(val_dataset)} val")

    use_cache = args.cache_patches and isinstance(train_dataset, PatchCacheDataset)
    pin = device.type == "cuda" and args.workers > 0
    persist = args.workers > 0 and not use_cache
    prefetch = 4 if args.workers > 0 else None

    if use_cache:
        # Patches in RAM — no grouped sampler needed, no persistent workers
        # (workers must respawn to see regenerated patches).
        # Force 'fork' so workers share the patch array via COW pages
        # instead of pickling (forkserver would serialize the entire array).
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, drop_last=True,
            num_workers=args.workers,
            pin_memory=pin, persistent_workers=False,
            prefetch_factor=prefetch,
            multiprocessing_context='fork' if args.workers > 0 else None,
        )
    else:
        train_sampler = ImageGroupedSampler(
            len(train_files), args.patches_per_image, shuffle=True,
        )
        persist = args.workers > 0
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, drop_last=True,
            num_workers=args.workers,
            pin_memory=pin, persistent_workers=persist,
            prefetch_factor=prefetch,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=pin, persistent_workers=persist,
        prefetch_factor=prefetch,
    )

    # Pre-materialize validation batches on GPU to avoid CPU memory contention
    # during evaluation (streaming threads compete for DDR5 bandwidth).
    if device.type == "cuda":
        print("  Pre-loading validation batches to GPU...")
        val_batches = []
        for batch in val_loader:
            inputs, targets, clip_levels = batch
            val_batches.append((
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
                clip_levels.to(device, non_blocking=True),
            ))
        val_vram_mb = sum(
            t.nbytes for b in val_batches for t in b
        ) / 1e6
        print(f"  Validation: {len(val_batches)} batches ({val_vram_mb:.0f} MB VRAM)")
    else:
        val_batches = None

    # Model
    model = XTransUNet(base_width=args.base_width).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Resume
    start_epoch = 0
    best_val_psnr = 0.0
    same_run = False
    ckpt = None
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")
        if not missing and not unexpected:
            print(f"  All weights loaded")

        # Only restore optimizer/scheduler if continuing same training
        if args.mode == "train":
            start_epoch = ckpt.get("epoch", 0) + 1
            # Only carry over best_val_psnr and scheduler when resuming into
            # the same output dir (truly continuing a run). When --from-checkpoint
            # writes to a new dir, start fresh tracking and a fresh LR schedule.
            ckpt_dir = Path(args.resume).parent
            same_run = ckpt_dir.resolve() == Path(args.output_dir).resolve()
            if same_run:
                best_val_psnr = ckpt.get("best_val_psnr", 0.0)
            print(f"  Resuming from epoch {start_epoch}"
                  + (f", best PSNR: {best_val_psnr:.1f}" if same_run else " (fresh best PSNR tracking)"))
        else:
            print(f"  Loaded model weights (fresh optimizer for fine-tuning)")

    # Loss
    if args.mode == "finetune":
        criterion = DemosaicLoss.finetune(data_range=data_range)
    else:
        criterion = DemosaicLoss.base(data_range=data_range)

    # Override with explicit weights if provided
    if args.l1_weight is not None:
        criterion.l1_weight = args.l1_weight
    if args.msssim_weight is not None:
        criterion.msssim_weight = args.msssim_weight
        if criterion.msssim is None and args.msssim_weight > 0:
            from losses import MSSSIM
            criterion.msssim = MSSSIM(data_range=data_range).to(device)
    if args.gradient_weight is not None:
        criterion.gradient_weight = args.gradient_weight
    if args.chroma_weight is not None:
        criterion.chroma_weight = args.chroma_weight
    if args.color_bias_weight is not None:
        criterion.color_bias_weight = args.color_bias_weight
        if criterion.color_bias is None and args.color_bias_weight > 0:
            from losses import ColorBiasLoss
            criterion.color_bias = ColorBiasLoss()
    if args.zipper_weight is not None:
        criterion.zipper_weight = args.zipper_weight
        if criterion.zipper is None and args.zipper_weight > 0:
            from losses import ZipperLoss
            criterion.zipper = ZipperLoss()
    if args.per_channel_norm:
        criterion.per_channel_norm = True
    if args.huber:
        criterion.use_huber = True
        criterion.huber_delta = args.huber_delta
    if args.recon_only:
        criterion.recon_only = True
        criterion.known_pixel_weight = args.known_pixel_weight

    criterion = criterion.to(device)

    loss_name = f"Huber(δ={criterion.huber_delta})" if criterion.use_huber else "L1"
    loss_info = f"Loss: {loss_name}={criterion.l1_weight}"
    if criterion.msssim_weight > 0:
        loss_info += f", MS-SSIM={criterion.msssim_weight}"
    if criterion.gradient_weight > 0:
        loss_info += f", grad={criterion.gradient_weight}"
    if criterion.chroma_weight > 0:
        loss_info += f", chroma={criterion.chroma_weight}"
    if criterion.zipper_weight > 0:
        loss_info += f", zipper={criterion.zipper_weight}"
    if criterion.color_bias_weight > 0:
        loss_info += f", color_bias={criterion.color_bias_weight}"
    if criterion.recon_only:
        loss_info += f" [recon-only, known={criterion.known_pixel_weight}]"
    if criterion.per_channel_norm:
        loss_info += " [per-channel norm]"
    if args.apply_wb:
        loss_info += " [WB training]"
    print(loss_info)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # For a new run from checkpoint, cosine schedule spans the remaining epochs.
    # For same-run resume, use original T_max and restore scheduler state.
    remaining = args.epochs - start_epoch
    t_max = args.epochs if same_run else max(remaining, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs])
    else:
        scheduler = cosine

    # Restore optimizer/scheduler state if continuing same training
    if ckpt is not None and args.mode == "train" and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if same_run:
            scheduler.load_state_dict(ckpt["scheduler"])

    # AMP scaler (no-op on CPU, works on CUDA and MPS)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp) if args.amp else None
    if args.amp:
        if ckpt is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  AMP enabled (float16 mixed precision)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parent / REGISTRY_FILENAME
    history_rel = str(output_dir / "history.json")

    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['data_range'] = data_range
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop — restore history if continuing a run
    history = []
    if same_run:
        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            # Trim entries from epochs we're about to re-run (e.g. crash mid-epoch)
            history = [h for h in history if h["epoch"] < start_epoch]
            print(f"  Restored {len(history)} history entries")
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  CFA: {args.cfa_type}")
    print(f"  Batch: {args.batch_size}, Patch: {args.patch_size}px")
    print(f"  Noise: read=[{args.noise_min}, {args.noise_max}], shot=[0, {args.shot_noise_max}]")
    if args.torture_fraction > 0:
        print(f"  Torture mixing: {args.torture_fraction*100:.1f}%")
    if wb_aug > 0:
        print(f"  WB augmentation: ±{(math.exp(wb_aug)-1)*100:.0f}% (log range {wb_aug:.2f})")
    if args.bright_spot_prob > 0:
        print(f"  Bright spot augmentation: {args.bright_spot_prob*100:.0f}% prob, "
              f"intensity 1.5-{args.bright_spot_intensity_max:.1f}x, "
              f"sigma 2-{args.bright_spot_sigma_max:.0f}px")
    if args.downscale_prob > 0:
        print(f"  Downscale augmentation: {args.downscale_prob*100:.0f}% prob (2x area-average)")
    if use_cache:
        staging_gb = train_dataset._n_staging * train_dataset._extract_size**2 * 3 * 4 / 1e9
        print(f"  Patch cache: ON (staging swap, {train_dataset._n_stream_threads} threads, "
              f"{train_dataset._n_staging} staging slots / {staging_gb:.1f} GB)")
        train_dataset.start_streaming()
    print()

    for epoch in range(start_epoch, args.epochs):
        if use_cache:
            replaced = train_dataset.reset_stats()
            if replaced > 0:
                pct = replaced / len(train_dataset) * 100
                print(f"  Cache: swapped {replaced} patches ({pct:.1f}%)")
        else:
            train_sampler.set_epoch(epoch)
        t0 = time.time()

        train_loss, train_psnr, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        t_train = time.time() - t0

        # Swap staged patches while no DataLoader workers are active
        if use_cache:
            train_dataset.swap_staging()

        t1 = time.time()
        val_loss, val_psnr, val_comp = evaluate(
            model, val_loader, criterion, device, use_amp=args.amp,
            gpu_batches=val_batches,
        )
        t_val = time.time() - t1

        t2 = time.time()
        scheduler.step()

        if device.type == "mps":
            gc.collect()
            torch.mps.synchronize()
            torch.mps.empty_cache()
        t_overhead = time.time() - t2

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # Format components for display
        comp_str = " ".join(f"{k}:{v:.4f}" for k, v in train_comp.items() if k != 'total')

        print(
            f"Ep {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_psnr:.1f}dB | Val: {val_psnr:.1f}dB | "
            f"{comp_str} | LR:{lr_now:.1e} | "
            f"{elapsed:.0f}s (train:{t_train:.1f} val:{t_val:.1f} oh:{t_overhead:.1f})"
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "train_components": train_comp,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_components": val_comp,
            "lr": lr_now,
            "time": elapsed,
        }
        history.append(entry)

        # Save best
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            ckpt_data = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
                "base_width": args.base_width,
                "cfa_type": args.cfa_type,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, output_dir / "best.pt")
            print(f"  -> New best val ({best_val_psnr:.2f} dB)")
            update_registry(
                registry_path, cfa_type=args.cfa_type, base_width=args.base_width,
                status="beta", slot="best",
                path=str(output_dir / "best.pt"), epoch=epoch + 1,
                train_psnr=train_psnr, val_psnr=val_psnr,
                train_loss=train_loss, val_loss=val_loss,
                history=history_rel,
            )

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_data = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
                "base_width": args.base_width,
                "cfa_type": args.cfa_type,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, output_dir / "latest.pt")
            update_registry(
                registry_path, cfa_type=args.cfa_type, base_width=args.base_width,
                status="beta", slot="latest",
                path=str(output_dir / "latest.pt"), epoch=epoch + 1,
                train_psnr=train_psnr, val_psnr=val_psnr,
                train_loss=train_loss, val_loss=val_loss,
                history=history_rel,
            )

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    if use_cache:
        train_dataset.cleanup()

    # Mark as stable if all epochs completed
    promote_to_stable(
        registry_path, cfa_type=args.cfa_type,
        base_width=args.base_width,
    )
    print(f"\nDone. Best val PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
