#!/usr/bin/env python3
"""
Unified training script for X-Trans demosaicing.

Modes:
- train: Initial training from scratch (L1-focused)
- finetune: Fine-tune existing model

Examples:
    # Initial training
    python train.py --data-dir /path/to/npy --epochs 200

    # Fine-tune
    python train.py --data-dir /path/to/npy --resume checkpoints/best.pt \
        --mode finetune --epochs 50 --lr 1e-4

    # Fine-tune with torture pattern mixing
    python train.py --data-dir /path/to/npy --resume checkpoints/best.pt \
        --mode finetune --torture-fraction 0.05
"""

import argparse
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
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss, components = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    component_sums = {}
    n_batches = 0

    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss, components = criterion(outputs, targets)

        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v
        total_psnr += psnr(outputs, targets)
        n_batches += 1

    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    return total_loss / n_batches, total_psnr / n_batches, avg_components


def main():
    parser = argparse.ArgumentParser(description="CFA demosaicing training")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
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
    parser.add_argument("--olpf-sigma-max", type=float, default=0.0,
                        help="Max Gaussian sigma for OLPF blur simulation (0 = disabled). Applied to RGB before mosaicing.")
    parser.add_argument("--wb-aug-range", type=float, default=0.0,
                        help="WB shift augmentation range in log space (e.g. 0.25 = ~±28%%). Only with --apply-wb")
    parser.add_argument("--exposure-aug-ev", type=float, default=0.0,
                        help="Max upward exposure shift in EV (e.g. 1.5). Clips in raw space before WB.")
    parser.add_argument("--torture-fraction", type=float, default=0.0,
                        help="Fraction of training data from synthetic torture patterns")
    parser.add_argument("--torture-patterns", type=int, default=500,
                        help="Number of unique torture patterns")
    
    # Checkpoints
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
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
    
    args = parser.parse_args()

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

    if args.torture_fraction > 0:
        train_dataset = create_mixed_dataset(
            data_dir=None,
            files=train_files,
            torture_fraction=args.torture_fraction,
            torture_patterns=args.torture_patterns,
            augment=True,
            noise_sigma=(args.noise_min, args.noise_max),
            wb_aug_range=wb_aug,
            exposure_aug_ev=args.exposure_aug_ev,
            olpf_sigma=olpf_sigma,
            **shared_kwargs,
        )
    else:
        train_dataset = LinearDataset(
            files=train_files,
            augment=True,
            noise_sigma=(args.noise_min, args.noise_max),
            wb_aug_range=wb_aug,
            exposure_aug_ev=args.exposure_aug_ev,
            olpf_sigma=olpf_sigma,
            **shared_kwargs,
        )

    val_dataset = LinearDataset(
        files=val_files,
        augment=False,
        noise_sigma=(0.0, 0.0),
        **shared_kwargs,
    )

    print(f"  Patches: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )

    # Model
    model = XTransUNet(base_width=args.base_width).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

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
    if criterion.color_bias_weight > 0:
        loss_info += f", color_bias={criterion.color_bias_weight}"
    if criterion.per_channel_norm:
        loss_info += " [per-channel norm]"
    if args.apply_wb:
        loss_info += " [WB training]"
    print(loss_info)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 0
    best_val_psnr = 0.0
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        
        # Only restore optimizer/scheduler if continuing same training
        if args.mode == "train" and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_psnr = ckpt.get("best_val_psnr", 0.0)
            print(f"  Resuming from epoch {start_epoch}, best PSNR: {best_val_psnr:.1f}")
        else:
            print(f"  Loaded model weights (fresh optimizer for fine-tuning)")

    # AMP scaler (no-op on CPU, works on CUDA and MPS)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp) if args.amp else None
    if args.amp:
        if args.resume and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  AMP enabled (float16 mixed precision)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['data_range'] = data_range
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = []
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  CFA: {args.cfa_type}")
    print(f"  Batch: {args.batch_size}, Patch: {args.patch_size}px")
    print(f"  Noise: [{args.noise_min}, {args.noise_max}]")
    if args.torture_fraction > 0:
        print(f"  Torture mixing: {args.torture_fraction*100:.1f}%")
    if wb_aug > 0:
        print(f"  WB augmentation: ±{(math.exp(wb_aug)-1)*100:.0f}% (log range {wb_aug:.2f})")
    if args.exposure_aug_ev > 0:
        print(f"  Exposure augmentation: 0 to +{args.exposure_aug_ev:.1f} EV (raw-space clipping)")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_loss, val_psnr, val_comp = evaluate(
            model, val_loader, criterion, device, use_amp=args.amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Format components for display
        comp_str = " ".join(f"{k}:{v:.4f}" for k, v in train_comp.items() if k != 'total')

        print(
            f"Ep {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_psnr:.1f}dB | Val: {val_psnr:.1f}dB | "
            f"{comp_str} | LR:{lr:.1e} | {elapsed:.0f}s"
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "train_components": train_comp,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_components": val_comp,
            "lr": lr,
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
            print(f"  -> New best ({best_val_psnr:.2f} dB)")

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

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
