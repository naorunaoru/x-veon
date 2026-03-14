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
from dataclasses import dataclass, fields, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import XTransUNet, count_parameters
from dataset import LinearDataset, PatchCacheDataset, create_mixed_dataset, ImageGroupedSampler
from losses import DemosaicLoss
from checkpoint_registry import update_registry, promote_to_stable, REGISTRY_FILENAME
from dashboard import TrainingDashboard, EpochData


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Single source of truth for all training parameters.

    Defaults match DemosaicLoss.base() for loss weights.
    Use TrainConfig.finetune() for fine-tuning presets.
    """
    # Data
    data_dir: list[str] | None = None
    cfa_type: str = "xtrans"
    max_images: int | None = None
    filter_file: str | None = None

    # Training
    mode: str = "train"
    epochs: int = 200
    batch_size: int = 32
    patch_size: int = 96
    lr: float = 1e-3
    warmup_epochs: int = 0
    val_split: float = 0.1
    patches_per_image: int = 16

    # Loss (defaults = DemosaicLoss.base() preset)
    l1_weight: float = 1.0
    msssim_weight: float = 0.0
    gradient_weight: float = 0.1
    chroma_weight: float = 0.05
    color_bias_weight: float = 0.0
    zipper_weight: float = 0.05
    huber: bool = False
    huber_delta: float = 1.0
    per_channel_norm: bool = False
    recon_only: bool = False
    known_pixel_weight: float = 0.1
    data_range: float | None = None

    # White balance
    apply_wb: bool = False

    # Augmentation
    noise_min: float = 0.0
    noise_max: float = 0.005
    shot_noise_max: float = 0.0
    olpf_sigma_max: float = 0.0
    wb_aug_range: float = 0.0
    bright_spot_prob: float = 0.0
    bright_spot_intensity_max: float = 5.0
    bright_spot_sigma_max: float = 20.0
    downscale_prob: float = 0.0
    torture_fraction: float = 0.0
    torture_patterns: int = 500

    # Checkpoints
    output_dir: str = "./checkpoints"

    # Model
    base_width: int = 64

    # Performance
    workers: int = 0
    seed: int = 42
    amp: bool = False
    cache_patches: bool = False
    cache_gb: float | None = None

    # --- Presets ---------------------------------------------------------- #

    @classmethod
    def base(cls) -> "TrainConfig":
        """Preset for initial training — matches DemosaicLoss.base()."""
        return cls()

    @classmethod
    def finetune(cls) -> "TrainConfig":
        """Preset for fine-tuning — matches DemosaicLoss.finetune()."""
        return cls(
            mode="finetune",
            lr=1e-4,
            epochs=50,
            l1_weight=0.5,
            msssim_weight=0.3,
            gradient_weight=0.2,
            chroma_weight=0.02,
            zipper_weight=0.1,
        )

    # --- Serialization ---------------------------------------------------- #

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        """Load config from checkpoint config.json."""
        with open(path) as f:
            data = json.load(f)
        valid = {f.name for f in fields(cls)}
        # data_range is recomputed from actual data each run
        skip = {"data_range", "device", "resume", "from_checkpoint"}
        return cls(**{k: v for k, v in data.items() if k in valid and k not in skip})

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path, *, extras: dict | None = None):
        """Save config to JSON, with optional extra runtime fields."""
        data = self.to_dict()
        if extras:
            data.update(extras)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # --- Loss construction ------------------------------------------------ #

    def build_criterion(self, data_range: float = 1.0) -> DemosaicLoss:
        """Build DemosaicLoss from this config's loss parameters."""
        return DemosaicLoss(
            l1_weight=self.l1_weight,
            msssim_weight=self.msssim_weight,
            gradient_weight=self.gradient_weight,
            chroma_weight=self.chroma_weight,
            color_bias_weight=self.color_bias_weight,
            zipper_weight=self.zipper_weight,
            per_channel_norm=self.per_channel_norm,
            use_huber=self.huber,
            huber_delta=self.huber_delta,
            data_range=data_range,
            recon_only=self.recon_only,
            known_pixel_weight=self.known_pixel_weight,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI → Config
# ---------------------------------------------------------------------------

def parse_config() -> tuple[TrainConfig, str | None, str | None]:
    """Parse CLI arguments and build a TrainConfig.

    Priority: preset defaults → checkpoint config → CLI overrides.

    Returns:
        (config, from_checkpoint_path, resume_path)
    """
    parser = argparse.ArgumentParser(description="CFA demosaicing training")

    # --- Routing args (not part of TrainConfig) ---
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Load training config from checkpoint dir (auto-resumes from best.pt). "
                             "All params are inherited; override any with explicit CLI args.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file")

    # --- Config args (all default=None for override detection) ---
    # Data
    parser.add_argument("--data-dir", type=str, nargs="+", default=None,
                        help="Directories with .npy files. Use path:N to limit "
                             "to N files from that directory (e.g. /data/a:100 /data/b:50)")
    parser.add_argument("--cfa-type", type=str, default=None, choices=["xtrans", "bayer"],
                        help="CFA pattern type (default: xtrans)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Global cap on total images (applied after per-dir limits)")
    parser.add_argument("--filter-file", type=str, default=None,
                        help="JSON file with allowed image stems")

    # Training
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], default=None,
                        help="Training mode: 'train' for initial, 'finetune' for texture recovery")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Linear LR warmup epochs")
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--patches-per-image", type=int, default=None)

    # Loss weights
    parser.add_argument("--l1-weight", type=float, default=None)
    parser.add_argument("--msssim-weight", type=float, default=None)
    parser.add_argument("--gradient-weight", type=float, default=None)
    parser.add_argument("--chroma-weight", type=float, default=None)
    parser.add_argument("--color-bias-weight", type=float, default=None,
                        help="Weight for mean color bias penalty")
    parser.add_argument("--zipper-weight", type=float, default=None,
                        help="Weight for zipper artifact penalty")
    parser.add_argument("--huber", action="store_true", default=None,
                        help="Use Huber loss instead of L1")
    parser.add_argument("--huber-delta", type=float, default=None,
                        help="Delta for Huber loss")
    parser.add_argument("--per-channel-norm", action="store_true", default=None,
                        help="Normalize L1 loss per channel (addresses G >> R,B sample imbalance)")
    parser.add_argument("--recon-only", action="store_true", default=None,
                        help="Compute L1/Huber only on reconstructed (non-CFA) pixels")
    parser.add_argument("--known-pixel-weight", type=float, default=None,
                        help="Weight for known-pixel preservation when --recon-only (default: 0.1)")
    parser.add_argument("--data-range", type=float, default=None,
                        help="Max pixel value for SSIM constants (auto-computed from metadata when --apply-wb)")

    # White balance
    parser.add_argument("--apply-wb", action="store_true", default=None,
                        help="Apply per-image WB to training data (model learns WB'd output)")

    # Augmentation
    parser.add_argument("--noise-min", type=float, default=None)
    parser.add_argument("--noise-max", type=float, default=None)
    parser.add_argument("--shot-noise-max", type=float, default=None,
                        help="Max shot noise coefficient (0 = disabled)")
    parser.add_argument("--olpf-sigma-max", type=float, default=None,
                        help="Max Gaussian sigma for OLPF blur simulation (0 = disabled)")
    parser.add_argument("--wb-aug-range", type=float, default=None,
                        help="WB shift augmentation range in log space. Only with --apply-wb")
    parser.add_argument("--bright-spot-prob", type=float, default=None,
                        help="Probability of adding synthetic bright spots (0-1)")
    parser.add_argument("--bright-spot-intensity-max", type=float, default=None,
                        help="Max intensity factor for bright spots (min is 1.5)")
    parser.add_argument("--bright-spot-sigma-max", type=float, default=None,
                        help="Max Gaussian sigma (pixels) for bright spots (min is 2.0)")
    parser.add_argument("--downscale-prob", type=float, default=None,
                        help="Probability of 2x area-average downscale (0-1)")
    parser.add_argument("--torture-fraction", type=float, default=None,
                        help="Fraction of training data from synthetic torture patterns")
    parser.add_argument("--torture-patterns", type=int, default=None,
                        help="Number of unique torture patterns")

    # Checkpoints
    parser.add_argument("--output-dir", type=str, default=None)

    # Model
    parser.add_argument("--base-width", type=int, default=None,
                        help="Base channel width (default 64)")

    # Performance
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader workers (0 for main process)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for train/val split")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable automatic mixed precision (float16)")
    parser.add_argument("--cache-patches", action="store_true", default=None,
                        help="Pre-extract patches into RAM (eliminates disk I/O during training)")
    parser.add_argument("--cache-gb", type=float, default=None,
                        help="Memory budget for patch cache in GB")

    # Deprecated (kept for CLI compat, ignored)
    parser.add_argument("--regen-every", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--replace-fraction", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Separate routing args
    from_checkpoint = args.from_checkpoint
    resume = args.resume

    # Collect explicit CLI overrides (non-None values for config fields only)
    config_field_names = {f.name for f in fields(TrainConfig)}
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k in config_field_names}

    # Step 1: Build base config from preset or checkpoint
    mode = overrides.get("mode", "train")
    if from_checkpoint:
        ckpt_dir = Path(from_checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        config_path = ckpt_dir / "config.json"
        if not config_path.exists():
            parser.error(f"No config.json found in {ckpt_dir}")
        cfg = TrainConfig.from_json(config_path)
    elif mode == "finetune":
        cfg = TrainConfig.finetune()
    else:
        cfg = TrainConfig.base()

    # Step 2: Apply CLI overrides
    for k, v in overrides.items():
        setattr(cfg, k, v)

    # Step 3: Auto-set resume from checkpoint dir
    if from_checkpoint and resume is None:
        ckpt_dir = Path(from_checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        for name in ("best.pt", "latest.pt"):
            pt = ckpt_dir / name
            if pt.exists():
                resume = str(pt)
                break

    if cfg.data_dir is None:
        parser.error("--data-dir is required (either explicitly or via --from-checkpoint config)")

    # Normalize data_dir: checkpoint config may store a string
    if isinstance(cfg.data_dir, str):
        cfg.data_dir = [cfg.data_dir]

    return cfg, from_checkpoint, resume


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg, from_checkpoint, resume = parse_config()

    # Dashboard — start immediately so setup messages appear in the log panel
    dash = TrainingDashboard(total_epochs=cfg.epochs, log_capacity=50)
    dash.start()

    if from_checkpoint:
        ckpt_src = Path(from_checkpoint)
        dash.log(f"Loaded config from {ckpt_src.parent / 'config.json' if ckpt_src.is_file() else ckpt_src / 'config.json'}")

    device = get_device()
    dash.log(f"Device: {device}")
    dash.log(f"Mode: {cfg.mode}")

    # Dataset — split at image level to prevent leakage
    # Collect files from all data directories, each with optional :N limit
    all_files = []
    for entry in cfg.data_dir:
        if ':' in entry and entry.rsplit(':', 1)[1].isdigit():
            dir_path, limit_str = entry.rsplit(':', 1)
            dir_limit = int(limit_str)
        else:
            dir_path = entry
            dir_limit = None
        dir_files = LinearDataset.find_files(
            dir_path,
            max_images=dir_limit,
            filter_file=cfg.filter_file,
        )
        dash.log(f"  {dir_path}: {len(dir_files)} files" +
              (f" (limited to {dir_limit})" if dir_limit else ""))
        all_files.extend(dir_files)

    if cfg.max_images and len(all_files) > cfg.max_images:
        all_files = all_files[:cfg.max_images]

    dash.log(f"Total images: {len(all_files)}")
    random.Random(cfg.seed).shuffle(all_files)

    val_n_images = max(1, int(len(all_files) * cfg.val_split))
    val_files = all_files[:val_n_images]
    train_files = all_files[val_n_images:]
    dash.log(f"  Images: {len(train_files)} train, {val_n_images} val")

    # Compute effective data range
    if cfg.data_range is not None:
        data_range = cfg.data_range
    elif cfg.apply_wb:
        data_range = _compute_data_range(all_files)
        if cfg.wb_aug_range > 0:
            data_range *= math.exp(cfg.wb_aug_range)
        dash.log(f"  Auto data_range: {data_range:.2f}")
    else:
        data_range = 1.0

    wb_aug = cfg.wb_aug_range if cfg.apply_wb else 0.0
    if wb_aug > 0 and not cfg.apply_wb:
        dash.log("  Warning: --wb-aug-range ignored without --apply-wb", "WARN")

    shared_kwargs = dict(
        patch_size=cfg.patch_size,
        patches_per_image=cfg.patches_per_image,
        apply_wb=cfg.apply_wb,
        cfa_type=cfg.cfa_type,
    )

    olpf_sigma = (0.0, cfg.olpf_sigma_max)

    spot_kwargs = dict(
        bright_spot_prob=cfg.bright_spot_prob,
        bright_spot_intensity=(1.5, cfg.bright_spot_intensity_max),
        bright_spot_sigma=(2.0, cfg.bright_spot_sigma_max),
    )

    train_augment_kwargs = dict(
        augment=True,
        noise_sigma=(cfg.noise_min, cfg.noise_max),
        shot_noise=(0.0, cfg.shot_noise_max),
        wb_aug_range=wb_aug,
        olpf_sigma=olpf_sigma,
        downscale_prob=cfg.downscale_prob,
        **spot_kwargs,
    )

    if cfg.torture_fraction > 0:
        train_dataset = create_mixed_dataset(
            data_dir=None,
            files=train_files,
            torture_fraction=cfg.torture_fraction,
            torture_patterns=cfg.torture_patterns,
            **train_augment_kwargs,
            **shared_kwargs,
        )
    elif cfg.cache_patches:
        dash.log("Pre-extracting patches into RAM...")
        train_dataset = PatchCacheDataset(
            files=train_files,
            seed=cfg.seed,
            cache_gb=cfg.cache_gb,
            **train_augment_kwargs,
            **shared_kwargs,
        )
        mem_gb = train_dataset._patch_data.nbytes / 1e9
        dash.log(f"Cached {len(train_dataset)} patches ({mem_gb:.1f} GB RAM)")
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

    dash.log(f"Patches: {len(train_dataset)} train, {len(val_dataset)} val")

    use_cache = cfg.cache_patches and isinstance(train_dataset, PatchCacheDataset)
    pin = device.type == "cuda" and cfg.workers > 0
    persist = cfg.workers > 0 and not use_cache
    prefetch = 4 if cfg.workers > 0 else None

    if use_cache:
        # Patches in RAM — no grouped sampler needed, no persistent workers
        # (workers must respawn to see regenerated patches).
        # Force 'fork' so workers share the patch array via COW pages
        # instead of pickling (forkserver would serialize the entire array).
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True, drop_last=True,
            num_workers=cfg.workers,
            pin_memory=pin, persistent_workers=False,
            prefetch_factor=prefetch,
            multiprocessing_context='fork' if cfg.workers > 0 else None,
        )
    else:
        train_sampler = ImageGroupedSampler(
            len(train_files), cfg.patches_per_image, shuffle=True,
        )
        persist = cfg.workers > 0
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            sampler=train_sampler, drop_last=True,
            num_workers=cfg.workers,
            pin_memory=pin, persistent_workers=persist,
            prefetch_factor=prefetch,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin, persistent_workers=persist,
        prefetch_factor=prefetch,
    )

    # Pre-materialize validation batches on GPU to avoid CPU memory contention
    # during evaluation (streaming threads compete for DDR5 bandwidth).
    if device.type == "cuda":
        dash.log("Pre-loading validation batches to GPU...")
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
        dash.log(f"Validation: {len(val_batches)} batches ({val_vram_mb:.0f} MB VRAM)")
    else:
        val_batches = None

    # Model
    model = XTransUNet(base_width=cfg.base_width).to(device)
    dash.log(f"Model parameters: {count_parameters(model):,}")

    # Resume
    start_epoch = 0
    best_val_psnr = 0.0
    same_run = False
    ckpt = None
    if resume:
        dash.log(f"Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            dash.log(f"  Missing keys: {len(missing)}", "WARN")
        if unexpected:
            dash.log(f"  Unexpected keys (ignored): {len(unexpected)}", "WARN")
        if not missing and not unexpected:
            dash.log(f"  All weights loaded")

        # Only restore optimizer/scheduler if continuing same training
        if cfg.mode == "train":
            start_epoch = ckpt.get("epoch", 0) + 1
            # Only carry over best_val_psnr and scheduler when resuming into
            # the same output dir (truly continuing a run). When --from-checkpoint
            # writes to a new dir, start fresh tracking and a fresh LR schedule.
            ckpt_dir = Path(resume).parent
            same_run = ckpt_dir.resolve() == Path(cfg.output_dir).resolve()
            if same_run:
                best_val_psnr = ckpt.get("best_val_psnr", 0.0)
            dash.log(f"  Resuming from epoch {start_epoch}"
                     + (f", best PSNR: {best_val_psnr:.1f}" if same_run else " (fresh best PSNR tracking)"))
        else:
            dash.log(f"  Loaded model weights (fresh optimizer for fine-tuning)")

    # Loss — built directly from config (single source of truth)
    criterion = cfg.build_criterion(data_range).to(device)

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
    if cfg.apply_wb:
        loss_info += " [WB training]"
    dash.log(loss_info)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    # For a new run from checkpoint, cosine schedule spans the remaining epochs.
    # For same-run resume, use original T_max and restore scheduler state.
    remaining = cfg.epochs - start_epoch
    t_max = cfg.epochs if same_run else max(remaining, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if cfg.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=cfg.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[cfg.warmup_epochs])
    else:
        scheduler = cosine

    # Restore optimizer/scheduler state if continuing same training
    if ckpt is not None and cfg.mode == "train" and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if same_run:
            scheduler.load_state_dict(ckpt["scheduler"])

    # AMP scaler (no-op on CPU, works on CUDA and MPS)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp) if cfg.amp else None
    if cfg.amp:
        if ckpt is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        dash.log(f"AMP enabled (float16 mixed precision)")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parent / REGISTRY_FILENAME
    history_rel = str(output_dir / "history.json")

    # Save config (resolved data_range + device as extras for provenance)
    cfg.save(output_dir / "config.json", extras={
        "device": str(device),
        "data_range": data_range,
        "from_checkpoint": from_checkpoint,
        "resume": resume,
    })

    # Training loop — restore history if continuing a run
    history = []
    if same_run:
        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            # Trim entries from epochs we're about to re-run (e.g. crash mid-epoch)
            history = [h for h in history if h["epoch"] < start_epoch]
            dash.log(f"Restored {len(history)} history entries")
    dash.log(f"Training for {cfg.epochs} epochs...")
    dash.log(f"  CFA: {cfg.cfa_type}")
    dash.log(f"  Batch: {cfg.batch_size}, Patch: {cfg.patch_size}px")
    dash.log(f"  Noise: read=[{cfg.noise_min}, {cfg.noise_max}], shot=[0, {cfg.shot_noise_max}]")
    if cfg.torture_fraction > 0:
        dash.log(f"  Torture mixing: {cfg.torture_fraction*100:.1f}%")
    if wb_aug > 0:
        dash.log(f"  WB augmentation: ±{(math.exp(wb_aug)-1)*100:.0f}% (log range {wb_aug:.2f})")
    if cfg.bright_spot_prob > 0:
        dash.log(f"  Bright spot augmentation: {cfg.bright_spot_prob*100:.0f}% prob, "
                 f"intensity 1.5-{cfg.bright_spot_intensity_max:.1f}x, "
                 f"sigma 2-{cfg.bright_spot_sigma_max:.0f}px")
    if cfg.downscale_prob > 0:
        dash.log(f"  Downscale augmentation: {cfg.downscale_prob*100:.0f}% prob (2x area-average)")
    if use_cache:
        staging_gb = train_dataset._n_staging * train_dataset._extract_size**2 * 3 * 4 / 1e9
        dash.log(f"  Patch cache: ON (staging swap, {train_dataset._n_stream_threads} threads, "
                 f"{train_dataset._n_staging} staging slots / {staging_gb:.1f} GB)")
        train_dataset.start_streaming()

    # Update dashboard with checkpoint info and start training
    dash.start_epoch = start_epoch
    dash.best_val_psnr = best_val_psnr
    if history:
        dash.bulk_load([
            EpochData(
                epoch=h["epoch"],
                train_psnr=h["train_psnr"],
                val_psnr=h["val_psnr"],
                train_components=h.get("train_components", {}),
                val_components=h.get("val_components", {}),
                lr=h["lr"],
                epoch_time=h["time"],
            )
            for h in history
        ])

    for epoch in range(start_epoch, cfg.epochs):
        if use_cache:
            replaced = train_dataset.reset_stats()
            if replaced > 0:
                pct = replaced / len(train_dataset) * 100
                dash.log(f"Cache: swapped {replaced} patches ({pct:.1f}%)")
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
            model, val_loader, criterion, device, use_amp=cfg.amp,
            gpu_batches=val_batches,
        )
        t_val = time.time() - t1

        scheduler.step()

        if device.type == "mps":
            gc.collect()
            torch.mps.synchronize()
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        dash.update(EpochData(
            epoch=epoch + 1,
            train_psnr=train_psnr,
            val_psnr=val_psnr,
            train_components=train_comp,
            val_components=val_comp,
            lr=lr_now,
            epoch_time=elapsed,
            train_time=t_train,
            val_time=t_val,
        ))

        if dash.has_fatal_error:
            dash.log("Stopping training due to NaN/Inf detection.", "ERROR")
            break

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
                "base_width": cfg.base_width,
                "cfa_type": cfg.cfa_type,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, output_dir / "best.pt")
            update_registry(
                registry_path, cfa_type=cfg.cfa_type, base_width=cfg.base_width,
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
                "base_width": cfg.base_width,
                "cfa_type": cfg.cfa_type,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, output_dir / "latest.pt")
            update_registry(
                registry_path, cfa_type=cfg.cfa_type, base_width=cfg.base_width,
                status="beta", slot="latest",
                path=str(output_dir / "latest.pt"), epoch=epoch + 1,
                train_psnr=train_psnr, val_psnr=val_psnr,
                train_loss=train_loss, val_loss=val_loss,
                history=history_rel,
            )

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    dash.stop()

    if use_cache:
        train_dataset.cleanup()

    # Mark as stable if all epochs completed
    if not dash.has_fatal_error:
        promote_to_stable(
            registry_path, cfa_type=cfg.cfa_type,
            base_width=cfg.base_width,
        )
    dash.log(f"Done. Best val PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
