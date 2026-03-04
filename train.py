#!/usr/bin/env python3
"""
Unified training script for X-Trans demosaicing.

Modes:
- train: Initial training from scratch (L1-focused)
- finetune: Fine-tune existing model

Examples:
    # Initial training
    python train.py --data-dir /path/to/npy --epochs 200

    # Continue training from checkpoint (inherits all config, resumes optimizer/epoch)
    python train.py --from-checkpoint checkpoints_xtrans_hl_v1h/

    # Fine-tune from checkpoint with tweaked params
    python train.py --from-checkpoint checkpoints_xtrans_hl_v1h/ \
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
from dataset import LinearDataset, create_mixed_dataset
from losses import DemosaicLoss
from checkpoint_registry import update_registry, promote_to_stable, REGISTRY_FILENAME


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10 * torch.log10(torch.tensor(mse)).item()


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
    total_loss = 0.0
    total_psnr = 0.0
    component_sums = {}
    n_batches = 0
    use_amp = scaler is not None

    for batch in loader:
        inputs, targets, clip_levels = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        clip_levels = clip_levels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss, components = criterion(outputs, targets, clip_levels=clip_levels)

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

        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v
        with torch.no_grad():
            total_psnr += psnr(outputs, targets)
        n_batches += 1

    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    return total_loss / n_batches, total_psnr / n_batches, avg_components


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_hl_psnr = 0.0
    n_hl_batches = 0
    component_sums = {}
    n_batches = 0

    for batch in loader:
        inputs, targets, clip_levels = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        clip_levels = clip_levels.to(device)

        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss, components = criterion(outputs, targets, clip_levels=clip_levels)

        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v
        total_psnr += psnr(outputs, targets)

        # Highlight-only PSNR: MSE only where clip_ratio > 0
        clip_ratio = inputs[:, 4:5]  # (B, 1, H, W)
        hl_mask = (clip_ratio > 0).expand_as(outputs)
        n_hl_pixels = hl_mask.sum().item()
        if n_hl_pixels > 0:
            hl_mse = ((outputs - targets) ** 2 * hl_mask).sum().item() / n_hl_pixels
            total_hl_psnr += -10 * math.log10(max(hl_mse, 1e-10))
            n_hl_batches += 1

        n_batches += 1

    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    avg_hl_psnr = total_hl_psnr / n_hl_batches if n_hl_batches > 0 else None
    return total_loss / n_batches, total_psnr / n_batches, avg_components, avg_hl_psnr


def main():
    parser = argparse.ArgumentParser(description="CFA demosaicing training")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with .npy files")
    parser.add_argument("--cfa-type", type=str, default="xtrans",
                        choices=["xtrans", "bayer"],
                        help="CFA pattern type (default: xtrans)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--filter-file", type=str, default=None,
                        help="JSON file with allowed image stems")
    
    # Training
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], default="train",
                        help="Training mode: 'train' for initial, 'finetune' for texture recovery")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--hl-bias-weight", type=float, default=None,
                        help="Weight for highlight DC bias penalty (penalizes color tint in bright regions)")
    parser.add_argument("--hl-rel-weight", type=float, default=None,
                        help="Weight for highlight relative L1 loss (normalizes error by brightness)")
    parser.add_argument("--hl-grad-weight", type=float, default=None,
                        help="Weight for highlight gradient loss (Sobel edge preservation in bright regions)")
    parser.add_argument("--hl-threshold", type=float, default=0.5,
                        help="Brightness threshold for all highlight losses (fraction of clip level). "
                             "Soft ramp from threshold to clip level.")
    parser.add_argument("--hl-fade", type=float, default=0.0,
                        help="Fade out main pixel loss (Huber/L1) in highlight regions (0-1). "
                             "1.0 = fully suppress pixel loss in highlights, letting hl-* losses take over.")
    parser.add_argument("--huber", action="store_true",
                        help="Use Huber loss instead of L1")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Delta for Huber loss")
    parser.add_argument("--per-channel-norm", action="store_true",
                        help="Normalize L1 loss per channel (addresses G >> R,B sample imbalance)")
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
    parser.add_argument("--highlight-aug-prob", type=float, default=0.0,
                        help="Probability of highlight augmentation per sample (0-1). "
                             "When triggered, jointly boosts exposure and lowers clip ceiling.")
    parser.add_argument("--highlight-aug-ev", type=float, default=0.0,
                        help="Max EV for highlight augmentation (e.g. 1.5). "
                             "Boosts exposure by +ev, clips at pre-boost ceiling.")
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
    parser.add_argument("--hl-head", action="store_true",
                        help="Enable highlight reconstruction head (dual decoder)")
    parser.add_argument("--hl-lr", type=float, default=None,
                        help="Separate LR for highlight head (enables differential LR). "
                             "Base model uses --lr, HL head uses --hl-lr.")
    parser.add_argument("--freeze-base", action="store_true",
                        help="Freeze encoder + main decoder, train only highlight head")

    # Performance
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 for main process)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (float16)")
    
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

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Dataset — split at image level to prevent leakage
    print(f"\nLoading dataset from {args.data_dir}...")

    all_files = LinearDataset.find_files(
        args.data_dir,
        max_images=args.max_images,
        filter_file=args.filter_file,
    )
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

    hl_kwargs = dict(
        highlight_aug_prob=args.highlight_aug_prob,
        highlight_aug_ev=args.highlight_aug_ev,
    )

    if args.torture_fraction > 0:
        train_dataset = create_mixed_dataset(
            data_dir=None,
            files=train_files,
            torture_fraction=args.torture_fraction,
            torture_patterns=args.torture_patterns,
            augment=True,
            noise_sigma=(args.noise_min, args.noise_max),
            shot_noise=(0.0, args.shot_noise_max),
            wb_aug_range=wb_aug,
            olpf_sigma=olpf_sigma,
            **hl_kwargs,
            **shared_kwargs,
        )
    else:
        train_dataset = LinearDataset(
            files=train_files,
            augment=True,
            noise_sigma=(args.noise_min, args.noise_max),
            shot_noise=(0.0, args.shot_noise_max),
            wb_aug_range=wb_aug,
            olpf_sigma=olpf_sigma,
            **hl_kwargs,
            **shared_kwargs,
        )

    val_dataset = LinearDataset(
        files=val_files,
        augment=False,
        noise_sigma=(0.0, 0.0),
        **shared_kwargs,
    )

    # Separate validation set with clipping to track HL reconstruction quality
    val_hl_dataset = None
    if args.highlight_aug_prob > 0:
        val_hl_dataset = LinearDataset(
            files=val_files,
            augment=False,
            noise_sigma=(0.0, 0.0),
            highlight_aug_prob=1.0,  # always apply highlight aug
            highlight_aug_ev=args.highlight_aug_ev,
            **shared_kwargs,
        )

    print(f"  Patches: {len(train_dataset)} train, {len(val_dataset)} val"
          + (f", {len(val_hl_dataset)} val_hl" if val_hl_dataset else ""))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    val_hl_loader = None
    if val_hl_dataset is not None:
        val_hl_loader = DataLoader(
            val_hl_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
        )

    # Model
    model = XTransUNet(base_width=args.base_width, hl_head=args.hl_head).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    if args.hl_head:
        hl_params = sum(p.numel() for p in model.highlight_head.parameters())
        print(f"  HighlightHead: {hl_params:,} params")

    # Resume (before freeze/optimizer so we load weights first)
    start_epoch = 0
    best_val_psnr = 0.0
    same_run = False
    ckpt = None
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        # strict=False allows loading base checkpoint into hl_head model
        # (hl_head params will be randomly initialized)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params (randomly initialized): {len(missing)}")
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

    # Freeze base model if requested (only hl_head params will be trainable)
    if args.freeze_base:
        if not args.hl_head:
            print("WARNING: --freeze-base without --hl-head freezes everything!")
        n_frozen = 0
        for name, param in model.named_parameters():
            if not name.startswith("highlight_head."):
                param.requires_grad = False
                n_frozen += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen {n_frozen} params, trainable: {trainable:,}")

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
    # Highlight losses: shared threshold and fade
    hl_t = args.hl_threshold
    criterion.hl_threshold = hl_t
    criterion.hl_fade = args.hl_fade
    if args.hl_bias_weight is not None:
        criterion.hl_bias_weight = args.hl_bias_weight
        if criterion.hl_bias is None and args.hl_bias_weight > 0:
            from losses import HighlightBiasLoss
            criterion.hl_bias = HighlightBiasLoss(threshold=hl_t)
    if args.hl_rel_weight is not None:
        criterion.hl_rel_weight = args.hl_rel_weight
        if criterion.hl_rel is None and args.hl_rel_weight > 0:
            from losses import HighlightRelativeLoss
            criterion.hl_rel = HighlightRelativeLoss(threshold=hl_t)
    if args.hl_grad_weight is not None:
        criterion.hl_grad_weight = args.hl_grad_weight
        if criterion.hl_grad is None and args.hl_grad_weight > 0:
            from losses import HighlightGradientLoss
            criterion.hl_grad = HighlightGradientLoss(threshold=hl_t)
    if args.per_channel_norm:
        criterion.per_channel_norm = True
    if args.huber:
        criterion.use_huber = True
        criterion.huber_delta = args.huber_delta

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
    has_hl = criterion.hl_bias_weight > 0 or criterion.hl_rel_weight > 0 or criterion.hl_grad_weight > 0 or criterion.hl_fade > 0
    if has_hl:
        hl_parts = []
        if criterion.hl_bias_weight > 0:
            hl_parts.append(f"bias={criterion.hl_bias_weight}")
        if criterion.hl_rel_weight > 0:
            hl_parts.append(f"rel={criterion.hl_rel_weight}")
        if criterion.hl_grad_weight > 0:
            hl_parts.append(f"grad={criterion.hl_grad_weight}")
        if criterion.hl_fade > 0:
            hl_parts.append(f"fade={criterion.hl_fade}")
        loss_info += f", hl[t={criterion.hl_threshold}]({', '.join(hl_parts)})"
    if criterion.per_channel_norm:
        loss_info += " [per-channel norm]"
    if args.apply_wb:
        loss_info += " [WB training]"
    print(loss_info)

    # Optimizer (only trainable params — respects --freeze-base)
    if args.hl_lr is not None and args.hl_head and not args.freeze_base:
        # Differential LR: base model at --lr, HL head at --hl-lr
        base_params = [p for n, p in model.named_parameters()
                       if not n.startswith("highlight_head.") and p.requires_grad]
        hl_params = [p for n, p in model.named_parameters()
                     if n.startswith("highlight_head.") and p.requires_grad]
        param_groups = [
            {"params": base_params, "lr": args.lr},
            {"params": hl_params, "lr": args.hl_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        print(f"  Differential LR: base={args.lr:.1e}, hl_head={args.hl_lr:.1e}")
    else:
        train_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    # For a new run from checkpoint, cosine schedule spans the remaining epochs.
    # For same-run resume, use original T_max and restore scheduler state.
    remaining = args.epochs - start_epoch
    t_max = args.epochs if same_run else max(remaining, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # Restore optimizer/scheduler state if continuing same training
    if ckpt is not None and args.mode == "train" and "optimizer" in ckpt and not args.freeze_base:
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
    if args.highlight_aug_prob > 0:
        print(f"  Highlight augmentation: {args.highlight_aug_prob*100:.0f}% prob, "
              f"0 to {args.highlight_aug_ev:.1f} EV")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_loss, val_psnr, val_comp, _ = evaluate(
            model, val_loader, criterion, device, use_amp=args.amp
        )
        val_hl_psnr = None
        val_hl_only_psnr = None
        if val_hl_loader is not None:
            _, val_hl_psnr, _, val_hl_only_psnr = evaluate(
                model, val_hl_loader, criterion, device, use_amp=args.amp
            )
        scheduler.step()

        gc.collect()
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        lr_base = optimizer.param_groups[0]["lr"]
        lr_str = f"LR:{lr_base:.1e}"
        if len(optimizer.param_groups) > 1:
            lr_hl = optimizer.param_groups[1]["lr"]
            lr_str = f"LR:{lr_base:.1e}/{lr_hl:.1e}"

        # Format components for display
        comp_str = " ".join(f"{k}:{v:.4f}" for k, v in train_comp.items() if k != 'total')
        hl_str = ""
        if val_hl_psnr is not None:
            hl_str = f" | HL: {val_hl_psnr:.1f}dB"
            if val_hl_only_psnr is not None:
                hl_str += f" (px:{val_hl_only_psnr:.1f})"

        print(
            f"Ep {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_psnr:.1f}dB | Val: {val_psnr:.1f}dB{hl_str} | "
            f"{comp_str} | {lr_str} | {elapsed:.0f}s"
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "train_components": train_comp,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_components": val_comp,
            "val_hl_psnr": val_hl_psnr,
            "val_hl_only_psnr": val_hl_only_psnr,
            "lr": lr_base,
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
                "hl_head": args.hl_head,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, output_dir / "best.pt")
            print(f"  -> New best ({best_val_psnr:.2f} dB)")
            update_registry(
                registry_path, cfa_type=args.cfa_type, base_width=args.base_width,
                hl_head=args.hl_head, status="beta", slot="best",
                path=str(output_dir / "best.pt"), epoch=epoch + 1,
                train_psnr=train_psnr, val_psnr=val_psnr,
                val_hl_psnr=val_hl_psnr, train_loss=train_loss,
                val_loss=val_loss, history=history_rel,
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
                "hl_head": args.hl_head,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, output_dir / "latest.pt")
            update_registry(
                registry_path, cfa_type=args.cfa_type, base_width=args.base_width,
                hl_head=args.hl_head, status="beta", slot="latest",
                path=str(output_dir / "latest.pt"), epoch=epoch + 1,
                train_psnr=train_psnr, val_psnr=val_psnr,
                val_hl_psnr=val_hl_psnr, train_loss=train_loss,
                val_loss=val_loss, history=history_rel,
            )

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Mark as stable if all epochs completed
    promote_to_stable(
        registry_path, cfa_type=args.cfa_type,
        base_width=args.base_width, hl_head=args.hl_head,
    )
    print(f"\nDone. Best val PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
