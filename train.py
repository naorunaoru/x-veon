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

"""

import argparse
import gc
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, fields, asdict
from typing import ClassVar
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cfa import CFA_REGISTRY, cfa_period, make_channel_masks, make_model_input
from model import ARCHITECTURE_TAG, XTransUNet, count_parameters
from dataset import LinearDataset, PatchCacheDataset, ImageGroupedSampler
from dataset_record import read_records
from losses import DemosaicLoss, EncodedPSNR
from checkpoint_registry import (
    update_registry, promote_to_stable, REGISTRY_FILENAME, infer_checkpoint_version,
)
from dashboard import TrainingDashboard, EpochData
from state_server import StateServer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Presets — mode-specific overrides layered on top of checkpoint config.
# parse_config() layers: checkpoint config → preset overrides → CLI overrides.
# TrainConfig field defaults are drawn from the "train" preset where applicable.
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict] = {
    "train": {"epochs": 200, "lr": 1e-3},
    "finetune": {"epochs": 50, "lr": 1e-4},
}

_T = _PRESETS["train"]


@dataclass
class TrainConfig:
    """Single source of truth for all training parameters.

    Mode presets are defined in _PRESETS (exposed as TrainConfig.PRESETS).
    """
    PRESETS: ClassVar[dict[str, dict]] = _PRESETS

    # Data
    data_dir: list[str] | None = None
    cfa_type: str = "xtrans"
    max_images: int | None = None
    filter_file: str | None = None

    # Training
    mode: str = "train"
    epochs: int = _T["epochs"]
    batch_size: int = 32
    patch_size: int = 96
    lr: float = _T["lr"]
    warmup_epochs: int = 0
    val_split: float = 0.1
    patches_per_image: int = 16

    # Loss
    l1_weight: float = 1.0
    color_bias_weight: float = 0.0
    huber: bool = False
    huber_delta: float = 1.0
    recon_only: bool = False
    known_pixel_weight: float = 0.1

    # Augmentation
    noise_min: float = 0.0
    noise_max: float = 0.005
    shot_noise_max: float = 0.0
    olpf_sigma_max: float = 0.0
    downscale_prob: float = 0.0
    gain_jitter_stops: float = 3.0

    # Checkpoints
    output_dir: str = "./checkpoints"
    checkpoint_version: str | None = None
    checkpoint_major: int | None = None
    architecture_tag: str | None = None

    # Model
    base_width: int = 16
    stages: int = 2

    # Performance
    workers: int = 0
    seed: int = 42
    amp: bool = False
    cache_patches: bool = False
    cache_gb: float | None = None
    group_images: int = 32

    # --- Serialization ---------------------------------------------------- #

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        """Load config from checkpoint config.json."""
        with open(path) as f:
            data = json.load(f)
        valid = {f.name for f in fields(cls)}
        skip = {"device", "resume", "from_checkpoint", "datasets"}
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

    def build_criterion(self) -> DemosaicLoss:
        """Build DemosaicLoss from this config's loss parameters."""
        return DemosaicLoss(
            l1_weight=self.l1_weight,
            color_bias_weight=self.color_bias_weight,
            use_huber=self.huber,
            huber_delta=self.huber_delta,
            recon_only=self.recon_only,
            known_pixel_weight=self.known_pixel_weight,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_resume_checkpoint(ckpt_dir: Path, same_run: bool) -> Path | None:
    """Which checkpoint in ckpt_dir to resume from.

    Continuing a run (same_run): the one with the higher saved epoch, latest.pt on a tie.
    File dates are not used; copying or restoring files changes them. This is the latest
    *saved* epoch: latest.pt is written every ten epochs, best.pt when validation improves.
    A new run started from finished weights: best.pt.
    """
    best, latest = ckpt_dir / "best.pt", ckpt_dir / "latest.pt"
    if not same_run:
        return best if best.exists() else (latest if latest.exists() else None)
    candidates = [p for p in (latest, best) if p.exists()]      # latest first: max() keeps it on a tie
    if not candidates:
        return None

    def saved_epoch(path: Path) -> int:
        return int(torch.load(path, map_location="cpu", weights_only=True).get("epoch", -1))

    return max(candidates, key=saved_epoch)


def _schedule_total(state: dict) -> int | None:
    """The number of epochs a saved schedule anneals over: the T_max of its cosine part."""
    for part in (state, *state.get("_schedulers", [])):
        if "T_max" in part:
            return int(part["T_max"])
    return None


def build_schedule(optimizer: torch.optim.Optimizer, *, epochs: int, warmup_epochs: int, start_epoch: int,
                   ckpt: dict | None, same_run: bool) -> torch.optim.lr_scheduler.LRScheduler:
    """The run's learning-rate schedule, with the optimizer state of the run it continues.

    A new run, from scratch or from another run's weights, starts at the optimizer's rate and
    anneals over its own epochs: nothing of an earlier optimizer is kept. Continuing the same
    run restores the saved optimizer; with an unchanged total it restores the saved schedule,
    and with a new total it uses the new total's schedule from the saved epoch on, so an
    extended run keeps annealing instead of restarting from zero.
    """
    # For a new run from checkpoint, cosine schedule spans the remaining epochs.
    # For same-run resume, use original T_max and restore scheduler state.
    t_max = epochs if same_run else max(epochs - start_epoch, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler: torch.optim.lr_scheduler.LRScheduler = cosine
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if not (same_run and ckpt is not None and "optimizer" in ckpt):
        return scheduler
    if _schedule_total(ckpt["scheduler"]) == epochs:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        return scheduler
    # Another total: step the new schedule to the saved epoch, then load the optimizer's
    # moments and keep the new schedule's rate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)    # scheduler steps before any optimizer step
        for _ in range(start_epoch):
            scheduler.step()
    rates = [(group["lr"], group["initial_lr"]) for group in optimizer.param_groups]
    optimizer.load_state_dict(ckpt["optimizer"])
    for group, (lr, initial_lr) in zip(optimizer.param_groups, rates):
        group["lr"], group["initial_lr"] = lr, initial_lr
    return scheduler


def load_weights(model: torch.nn.Module, ckpt: dict, source: str) -> None:
    """Load weights strictly. A checkpoint of another layout is refused, never partly loaded."""
    tag = ckpt.get("architecture_tag")
    try:
        if tag not in (None, ARCHITECTURE_TAG):
            raise RuntimeError(f"architecture tag {tag}")
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise SystemExit(
            f"{source} is a checkpoint of another model layout (version "
            f"{ckpt.get('checkpoint_version')}, architecture {tag}); this code builds "
            f"{ARCHITECTURE_TAG}. It cannot be resumed or fine-tuned from."
        ) from e


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, masks, use_amp=False):
    """One pass over the training data. `masks` is (3, H, W) on `device`."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    metric = EncodedPSNR()
    component_sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    channel_masks = masks.unsqueeze(0) if criterion.recon_only else None

    for mosaic, targets in loader:
        mosaic = mosaic.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs = make_model_input(mosaic, masks)

        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(inputs)
        # The loss encodes in float32, outside autocast.
        loss, components = criterion(outputs, targets, channel_masks=channel_masks)

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
            metric.update(outputs, targets)
        n_batches += 1

    # Single sync point at epoch end
    if n_batches == 0:
        # No batches processed — return NaN so dashboard flags it
        avg_components = {k: float("nan") for k in component_sums}
        return float("nan"), float("nan"), avg_components
    avg_components = {k: (v / n_batches).item() for k, v in component_sums.items()}
    return (total_loss / n_batches).item(), metric.value(), avg_components


@torch.no_grad()
def evaluate(model, loader, criterion, device, masks, use_amp=False, gpu_batches=None):
    """Validation pass. The PSNR is pooled over the whole pass, not averaged per batch."""
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    metric = EncodedPSNR()
    component_sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    channel_masks = masks.unsqueeze(0) if criterion.recon_only else None

    source = gpu_batches if gpu_batches is not None else loader
    for mosaic, targets in source:
        if gpu_batches is None:
            mosaic = mosaic.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(make_model_input(mosaic, masks))
        loss, components = criterion(outputs, targets, channel_masks=channel_masks)

        total_loss = total_loss + loss.squeeze()
        for k, v in components.items():
            v = v.squeeze()
            if k in component_sums:
                component_sums[k] = component_sums[k] + v
            else:
                component_sums[k] = v.clone()
        metric.update(outputs, targets)
        n_batches += 1

    avg_components = {k: (v / n_batches).item() for k, v in component_sums.items()}
    return (total_loss / n_batches).item(), metric.value(), avg_components


# ---------------------------------------------------------------------------
# CLI → Config
# ---------------------------------------------------------------------------

@dataclass
class RoutingOptions:
    """Non-config CLI flags that control output routing."""
    detach: bool = False
    socket_path: str | None = None


def parse_config() -> tuple[TrainConfig, str | None, str | None, RoutingOptions]:
    """Parse CLI arguments and build a TrainConfig.

    Priority: preset defaults → checkpoint config → CLI overrides.

    Returns:
        (config, from_checkpoint_path, resume_path, routing)
    """
    parser = argparse.ArgumentParser(description="CFA demosaicing training")

    # --- Routing args (not part of TrainConfig) ---
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Load training config from a checkpoint dir. Continuing a run (same "
                             "output dir) resumes from the latest saved epoch; a new run starts from best.pt. "
                             "All params are inherited; override any with explicit CLI args.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file")
    parser.add_argument("--no-resume", action="store_true",
                        help="Skip auto-resume when using --from-checkpoint (config only)")

    # Observer routing (headless / socket-based state)
    parser.add_argument("--detach", action="store_true", default=False,
                        help="Run headless: no Rich UI, publish state + events on a UNIX socket")
    parser.add_argument("--socket-path", type=str, default=None,
                        help="Override the UNIX socket path used by --detach "
                             "(default: <output_dir>/.train.sock)")

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
                        help="Training mode: 'train' for a run from scratch or its continuation, "
                             "'finetune' to start from finished weights with a fresh optimiser")
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
    parser.add_argument("--color-bias-weight", type=float, default=None,
                        help="Weight for mean color bias penalty")
    parser.add_argument("--huber", action="store_true", default=None,
                        help="Use Huber loss instead of L1")
    parser.add_argument("--huber-delta", type=float, default=None,
                        help="Delta for Huber loss")
    parser.add_argument("--recon-only", action="store_true", default=None,
                        help="Compute L1/Huber only on reconstructed (non-CFA) pixels")
    parser.add_argument("--known-pixel-weight", type=float, default=None,
                        help="Weight for known-pixel preservation when --recon-only (default: 0.1)")

    # Augmentation
    parser.add_argument("--noise-min", type=float, default=None)
    parser.add_argument("--noise-max", type=float, default=None)
    parser.add_argument("--shot-noise-max", type=float, default=None,
                        help="Max shot noise coefficient (0 = disabled)")
    parser.add_argument("--olpf-sigma-max", type=float, default=None,
                        help="Max Gaussian sigma for OLPF blur simulation (0 = disabled)")
    parser.add_argument("--downscale-prob", type=float, default=None,
                        help="Probability of 2x area-average downscale (0-1)")
    parser.add_argument("--gain-jitter-stops", type=float, default=None,
                        help="Random gain after the model's mean normalisation, in stops either way (default 3)")

    # Checkpoints
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-version", type=str, default=None,
                        help="Canonical checkpoint version, e.g. v6.1.4 or v6.1.4-w32")
    parser.add_argument("--architecture-tag", type=str, default=None,
                        help="Optional architecture/inference family label for metadata")

    # Model
    parser.add_argument("--base-width", type=int, default=None,
                        help="Size class (default 16 = S); the packed channel count follows from it")
    parser.add_argument("--stages", type=int, default=None,
                        help="Resolution reductions including the packing (default 2 = S)")

    # Performance
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader workers (0 for main process)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for train/val split")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable automatic mixed precision (bfloat16)")
    parser.add_argument("--cache-patches", action="store_true", default=None,
                        help="Pre-extract patches into RAM (eliminates disk I/O during training)")
    parser.add_argument("--cache-gb", type=float, default=None,
                        help="Memory budget for patch cache in GB")
    parser.add_argument("--group-images", type=int, default=None,
                        help="Images the grouped sampler draws from at a time (default 32)")

    # Deprecated (kept for CLI compat, ignored)
    parser.add_argument("--regen-every", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--replace-fraction", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Separate routing args
    from_checkpoint = args.from_checkpoint
    resume = args.resume
    no_resume = args.no_resume
    routing = RoutingOptions(detach=args.detach, socket_path=args.socket_path)

    # Collect explicit CLI overrides (non-None values for config fields only)
    config_field_names = {f.name for f in fields(TrainConfig)}
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k in config_field_names}

    # Step 1: Build base config from preset or checkpoint
    mode = overrides.get("mode", "train")
    if from_checkpoint:
        # Load checkpoint config, then layer mode preset on top
        ckpt_dir = Path(from_checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        config_path = ckpt_dir / "config.json"
        if not config_path.exists():
            parser.error(f"No config.json found in {ckpt_dir}")
        cfg = TrainConfig.from_json(config_path)
        # Apply mode preset ONLY for keys not already in the checkpoint config
        # AND not explicitly overridden on the CLI.  This prevents the preset
        # from clobbering values that were saved from a previous run.
        with open(config_path) as _f:
            ckpt_keys = set(json.load(_f).keys())
        for k, v in TrainConfig.PRESETS.get(mode, {}).items():
            if k not in ckpt_keys and k not in overrides:
                setattr(cfg, k, v)
    else:
        cfg = TrainConfig(**TrainConfig.PRESETS.get(mode, TrainConfig.PRESETS["train"]))

    # Step 2: Apply explicit CLI overrides (highest priority)
    for k, v in overrides.items():
        setattr(cfg, k, v)

    # Step 3: Auto-set resume from checkpoint dir
    if from_checkpoint and resume is None and not no_resume:
        ckpt_dir = Path(from_checkpoint)
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        same_run = ckpt_dir.resolve() == Path(cfg.output_dir).resolve()
        picked = pick_resume_checkpoint(ckpt_dir, same_run)
        if picked is not None:
            resume = str(picked)

    if cfg.data_dir is None:
        parser.error("--data-dir is required (either explicitly or via --from-checkpoint config)")

    # Normalize data_dir: checkpoint config may store a string
    if isinstance(cfg.data_dir, str):
        cfg.data_dir = [cfg.data_dir]

    # Derive canonical checkpoint version from output dir when not set explicitly.
    if cfg.checkpoint_version is None:
        cfg.checkpoint_version = infer_checkpoint_version(
            {"checkpoint_version": None, "base_width": cfg.base_width},
            Path(cfg.output_dir),
        )
    if cfg.checkpoint_version and cfg.checkpoint_major is None:
        try:
            cfg.checkpoint_major = int(cfg.checkpoint_version.split(".", 1)[0][1:])
        except (ValueError, IndexError):
            pass
    if cfg.architecture_tag is None:
        cfg.architecture_tag = ARCHITECTURE_TAG

    return cfg, from_checkpoint, resume, routing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg, from_checkpoint, resume, routing = parse_config()

    loss_weights = {
        "l1": cfg.l1_weight,
        "color_bias": cfg.color_bias_weight,
    }
    config_summary = cfg.to_dict()

    # Observer — start immediately so setup messages are captured.
    # Foreground: Rich dashboard. Detached: headless UNIX-socket state server.
    output_dir = Path(cfg.output_dir)
    if routing.detach:
        output_dir.mkdir(parents=True, exist_ok=True)
        sock_path = Path(routing.socket_path) if routing.socket_path \
            else output_dir / ".train.sock"
        dash = StateServer(
            sock_path, total_epochs=cfg.epochs,
            best_metric="psnr",
            loss_weights=loss_weights,
            config=config_summary,
            log_capacity=50,
        )
        print(f"[detached] state socket: {sock_path}", flush=True)
    else:
        dash = TrainingDashboard(
            total_epochs=cfg.epochs, log_capacity=50,
            best_metric="psnr",
            loss_weights=loss_weights,
            config=config_summary,
        )
    start_wallclock = time.time()
    dash.start()

    if from_checkpoint:
        ckpt_src = Path(from_checkpoint)
        dash.log(f"Loaded config from {ckpt_src.parent / 'config.json' if ckpt_src.is_file() else ckpt_src / 'config.json'}")

    device = get_device()
    dash.log(f"Device: {device}")
    dash.log(f"Mode: {cfg.mode}")

    # Every data directory must carry the record of the builder that made it.
    datasets = read_records(cfg.data_dir)
    for path, record in datasets.items():
        dash.log(f"  dataset {path}: built by {record['revision'][:10]}"
                 + (" (modified tree)" if record.get("dirty") else ""))

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

    shared_kwargs = dict(
        patch_size=cfg.patch_size,
        patches_per_image=cfg.patches_per_image,
        cfa_type=cfg.cfa_type,
        group_images=cfg.group_images,
    )

    olpf_sigma = (0.0, cfg.olpf_sigma_max)

    train_augment_kwargs = dict(
        augment=True,
        noise_sigma=(cfg.noise_min, cfg.noise_max),
        shot_noise=(0.0, cfg.shot_noise_max),
        olpf_sigma=olpf_sigma,
        downscale_prob=cfg.downscale_prob,
    )

    if cfg.cache_patches:
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
            group_images=cfg.group_images,
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
    val_batches = None  # populated after model + optimizer are loaded

    # Model
    _cfa_pattern = CFA_REGISTRY[cfg.cfa_type]
    model = XTransUNet(base_width=cfg.base_width, cfa_period=cfa_period(_cfa_pattern),
                       stages=cfg.stages, gain_jitter_stops=cfg.gain_jitter_stops).to(device)
    masks = make_channel_masks(cfg.patch_size, cfg.patch_size, _cfa_pattern).to(device)
    dash.log(f"Model parameters: {count_parameters(model):,} ({cfg.stages} stages, architecture {ARCHITECTURE_TAG})")

    # Resume
    start_epoch = 0
    best_val_metric = 0.0
    same_run = False
    ckpt = None
    metric_label = "PSNR"
    if resume:
        dash.log(f"Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        load_weights(model, ckpt, resume)
        dash.log("  All weights loaded")

        # Only restore optimizer/scheduler if continuing same training
        if cfg.mode == "train":
            # Only carry over epoch and best_val_metric when resuming into
            # the same output dir (truly continuing a run). When --from-checkpoint
            # writes to a new dir, start fresh tracking and a fresh LR schedule.
            ckpt_dir = Path(resume).parent
            same_run = ckpt_dir.resolve() == Path(cfg.output_dir).resolve()
            if same_run:
                start_epoch = ckpt.get("epoch", 0) + 1
                # Support loading old checkpoints that used best_val_psnr
                best_val_metric = ckpt.get("best_val_metric", ckpt.get("best_val_psnr", 0.0))
            dash.log(f"  Resuming from epoch {start_epoch}"
                     + (f", best {metric_label}: {best_val_metric:.4f}" if same_run
                        else f" (fresh {metric_label} tracking)"))
        else:
            dash.log(f"  Loaded model weights (fresh optimizer for fine-tuning)")

    # Loss — built directly from config (single source of truth)
    criterion = cfg.build_criterion().to(device)

    loss_name = f"Huber(δ={criterion.huber_delta})" if criterion.use_huber else "L1"
    loss_info = f"Loss: {loss_name}={criterion.l1_weight} on encoded values"
    if criterion.color_bias_weight > 0:
        loss_info += f", color_bias={criterion.color_bias_weight}"
    if criterion.recon_only:
        loss_info += f" [recon-only, known={criterion.known_pixel_weight}]"
    dash.log(loss_info)

    # Optimizer and learning-rate schedule; their state is restored only when continuing the same run
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = build_schedule(optimizer, epochs=cfg.epochs, warmup_epochs=cfg.warmup_epochs,
                               start_epoch=start_epoch, ckpt=ckpt, same_run=same_run)

    if cfg.amp:
        dash.log("AMP enabled (bfloat16 mixed precision)")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parent / REGISTRY_FILENAME
    history_rel = str(output_dir / "history.json")

    # Save config (device and the datasets' build records as extras for provenance)
    cfg.save(output_dir / "config.json", extras={
        "device": str(device),
        "datasets": datasets,
        "from_checkpoint": from_checkpoint,
        "resume": resume,
        "checkpoint_version": cfg.checkpoint_version,
        "checkpoint_major": cfg.checkpoint_major,
        "architecture_tag": cfg.architecture_tag,
    })

    # Pre-materialize validation batches on GPU to avoid CPU memory contention
    # during evaluation. Done after model + optimizer are loaded so we can
    # check actual free VRAM rather than guessing with a fixed percentage.
    # Reserve 2 GB headroom for training activations and batch tensors.
    if device.type == "cuda":
        free_vram, _ = torch.cuda.mem_get_info(device)
        headroom = 2 * 1024**3
        vram_budget = max(0, free_vram - headroom)
        n_val = len(val_dataset)
        ps = cfg.patch_size
        est_bytes = n_val * ps * ps * (1 + 3) * 4  # 1ch mosaic + 3ch target, float32
        if est_bytes < vram_budget:
            dash.log("Pre-loading validation batches to GPU...")
            val_batches = []
            for mosaic, targets in val_loader:
                val_batches.append((
                    mosaic.to(device=device, non_blocking=True),
                    targets.to(device=device, non_blocking=True),
                ))
            val_vram_mb = sum(
                t.nbytes for b in val_batches for t in b
            ) / 1e6
            dash.log(f"Validation: {len(val_batches)} batches ({val_vram_mb:.0f} MB VRAM)")
        else:
            dash.log(f"Validation: streaming from CPU (est. {est_bytes / 1e9:.1f} GB > {vram_budget / 1e9:.1f} GB budget)")

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
    dash.log(f"  Gain jitter: ±{cfg.gain_jitter_stops:g} stops")
    if cfg.downscale_prob > 0:
        dash.log(f"  Downscale augmentation: {cfg.downscale_prob*100:.0f}% prob (2x area-average)")
    if use_cache:
        staging_gb = train_dataset._n_staging * train_dataset._extract_size**2 * 3 * 4 / 1e9
        dash.log(f"  Patch cache: ON (staging swap, {train_dataset._n_stream_threads} threads, "
                 f"{train_dataset._n_staging} staging slots / {staging_gb:.1f} GB)")
        train_dataset.start_streaming()

    # Seed observer with resume state (works for both dashboard and state server).
    history_epochs = [
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
    ]
    if isinstance(dash, TrainingDashboard):
        dash.start_epoch = start_epoch
        if history_epochs:
            dash.bulk_load(history_epochs)
        else:
            dash.best_val_psnr = best_val_metric
    else:
        # Detached state server — seed the snapshot without broadcasting
        # live events. Subscribers that are already connected must not see
        # fake epoch_done messages for history that has already happened;
        # new subscribers will get the seeded state in their initial snapshot.
        dash.seed_resume(
            start_epoch=start_epoch,
            history=history_epochs,
            best=(
                ("psnr", float(best_val_metric), start_epoch)
                if best_val_metric else None
            ),
        )

    interrupted = False
    fatal_exc = None
    last_train_comp: dict = {}
    last_val_comp: dict = {}
    last_train_psnr = float("nan")
    last_val_psnr = float("nan")
    try:
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
                model, train_loader, optimizer, criterion, device, masks, use_amp=cfg.amp
            )
            t_train = time.time() - t0

            # Swap staged patches while no DataLoader workers are active
            if use_cache:
                train_dataset.swap_staging()

            t1 = time.time()
            val_loss, val_psnr, val_comp = evaluate(
                model, val_loader, criterion, device, masks, use_amp=cfg.amp,
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

            last_train_comp = train_comp
            last_val_comp = val_comp
            last_train_psnr = train_psnr
            last_val_psnr = val_psnr

            if dash.has_fatal_error:
                dash.log("Stopping training due to NaN/Inf detection.", "ERROR")
                dash.log(f"  train components: {train_comp}", "ERROR")
                dash.log(f"  val components:   {val_comp}", "ERROR")
                dash.log(f"  train_psnr={train_psnr:.4f}  val_psnr={val_psnr:.4f}", "ERROR")
                break

            entry = {
                "epoch": epoch,
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

            # Save best: validation PSNR on encoded values
            current_metric = val_psnr
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                ckpt_data = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_metric": best_val_metric,
                    "best_metric_name": "psnr",
                    "stages": cfg.stages,
                    "best_val_psnr": val_psnr,  # always store PSNR for reference
                    "base_width": cfg.base_width,
                    "cfa_type": cfg.cfa_type,
                    "checkpoint_version": cfg.checkpoint_version,
                    "checkpoint_major": cfg.checkpoint_major,
                    "architecture_tag": cfg.architecture_tag,
                }
                torch.save(ckpt_data, output_dir / "best.pt")
                update_registry(
                    registry_path, cfa_type=cfg.cfa_type,
                    checkpoint_version=cfg.checkpoint_version or "unversioned",
                    base_width=cfg.base_width,
                    status="beta", slot="best",
                    path=str(output_dir / "best.pt"), epoch=epoch + 1,
                    train_psnr=train_psnr, val_psnr=val_psnr,
                    train_loss=train_loss, val_loss=val_loss,
                    history=history_rel,
                )
                dash.event("new_best", {
                    "metric": "psnr",
                    "value": float(best_val_metric),
                    "epoch": epoch + 1,
                    "checkpoint": str(output_dir / "best.pt"),
                    "val_psnr": float(val_psnr),
                })

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_data = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_metric": best_val_metric,
                    "best_metric_name": "psnr",
                    "stages": cfg.stages,
                    "best_val_psnr": val_psnr,
                    "base_width": cfg.base_width,
                    "cfa_type": cfg.cfa_type,
                    "checkpoint_version": cfg.checkpoint_version,
                    "checkpoint_major": cfg.checkpoint_major,
                    "architecture_tag": cfg.architecture_tag,
                }
                torch.save(ckpt_data, output_dir / "latest.pt")
                update_registry(
                    registry_path, cfa_type=cfg.cfa_type,
                    checkpoint_version=cfg.checkpoint_version or "unversioned",
                    base_width=cfg.base_width,
                    status="beta", slot="latest",
                    path=str(output_dir / "latest.pt"), epoch=epoch + 1,
                    train_psnr=train_psnr, val_psnr=val_psnr,
                    train_loss=train_loss, val_loss=val_loss,
                    history=history_rel,
                )

            # Save history
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as e:
        fatal_exc = e

    # Determine final status before emitting events / stopping.
    # The snapshot contract only allows completed/interrupted/error, so a
    # Python-level crash collapses into "error" in the shared status and is
    # only distinguished by the separate "error" event payload. We keep a
    # more specific local label for the user-facing summary print below.
    had_fatal = dash.has_fatal_error
    if fatal_exc is not None:
        final_status = "error"
        summary_label = "Crashed"
    elif interrupted:
        final_status = "interrupted"
        summary_label = "Interrupted"
    elif had_fatal:
        final_status = "error"
        summary_label = "Error"
    else:
        final_status = "completed"
        summary_label = "Completed"

    n_epochs = len(history)
    total_elapsed = time.time() - start_wallclock

    # Emit explicit lifecycle events before tearing down the observer
    # so subscribers see them on the wire.
    if fatal_exc is not None:
        import traceback as _tb
        dash.event("error", {
            "type": type(fatal_exc).__name__,
            "message": str(fatal_exc),
            "traceback": "".join(_tb.format_exception(
                type(fatal_exc), fatal_exc, fatal_exc.__traceback__,
            )),
            "epochs_completed": n_epochs,
        })
    dash.event("training_done", {
        "status": final_status,
        "epochs_completed": n_epochs,
        "elapsed_seconds": total_elapsed,
        "best": {
            "metric": "psnr",
            "value": float(best_val_metric) if best_val_metric else None,
        },
        "interrupted": interrupted,
        "fatal": had_fatal,
    })

    dash.stop()

    if use_cache:
        train_dataset.cleanup()

    # Mark as stable if all epochs completed without interruption
    if not had_fatal and not interrupted and fatal_exc is None:
        promote_to_stable(
            registry_path, cfa_type=cfg.cfa_type,
            checkpoint_version=cfg.checkpoint_version or "unversioned",
            base_width=cfg.base_width,
        )

    # Print summary to console (visible after dashboard closes)
    avg_epoch = sum(h["time"] for h in history) / n_epochs if n_epochs else 0.0
    data_dirs = ", ".join(cfg.data_dir) if cfg.data_dir else "N/A"

    from dashboard import format_time
    print()
    print("=" * 60)
    print(f"  Training Summary ({summary_label})")
    print("=" * 60)
    print(f"  Elapsed:       {format_time(total_elapsed)} ({n_epochs} epochs)")
    print(f"  Avg epoch:     {avg_epoch:.1f}s")
    print(f"  Sensor:        {cfg.cfa_type}")
    print(f"  Model width:   {cfg.base_width}")
    print(f"  Best {metric_label + ':':10s} {best_val_metric:.4f} dB")
    print(f"  Data:          {data_dirs}")
    print(f"  Output:        {cfg.output_dir}")
    print("=" * 60)

    if fatal_exc is not None:
        raise fatal_exc

    if had_fatal:
        print(f"\n  Training stopped: NaN/Inf detected in loss.")
        # Replay the ERROR-level logs that contain the detailed breakdown.
        # Works for both dashboard (LogEntry) and state server (TrainingLogRecord).
        logs = getattr(dash, "recent_logs", None)
        if logs is None:
            logs = getattr(dash, "logs", None)
        for entry in (logs or []):
            level = getattr(entry, "level", "")
            message = getattr(entry, "message", "")
            if level == "ERROR":
                print(f"  [{level}] {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # If the dashboard is still running, stop it before printing the
        # traceback so the Rich Live display doesn't corrupt the output.
        import traceback
        from dashboard import TrainingDashboard
        TrainingDashboard.force_stop()
        traceback.print_exc()
        raise SystemExit(1)
