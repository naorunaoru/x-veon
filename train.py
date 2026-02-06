#!/usr/bin/env python3
"""
Unified training script for X-Trans demosaicing.

Modes:
- train: Initial training from scratch (L1-focused)
- finetune: Fine-tune existing model (MS-SSIM + gradient for texture)

Examples:
    # Initial training
    python train.py --data-dir /path/to/npy --epochs 200

    # Fine-tune with MS-SSIM
    python train.py --data-dir /path/to/npy --resume checkpoints/best.pt \
        --mode finetune --epochs 50 --lr 1e-4

    # Fine-tune with torture pattern mixing
    python train.py --data-dir /path/to/npy --resume checkpoints/best.pt \
        --mode finetune --torture-fraction 0.05
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from model import XTransUNet, count_parameters
from dataset import LinearDataset, create_mixed_dataset
from losses import DemosaicLoss


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10 * torch.log10(torch.tensor(mse)).item()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    component_sums = {}
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, components = criterion(outputs, targets)
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    component_sums = {}
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

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
    parser = argparse.ArgumentParser(description="X-Trans demosaicing training")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with .npy files or JPEGs")
    parser.add_argument("--use-jpeg", action="store_true",
                        help="Load JPEGs directly instead of .npy")
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
    
    # Augmentation
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.005)
    parser.add_argument("--torture-fraction", type=float, default=0.0,
                        help="Fraction of training data from synthetic torture patterns")
    parser.add_argument("--torture-patterns", type=int, default=500,
                        help="Number of unique torture patterns")
    
    # Checkpoints
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    # Performance
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 for main process)")
    
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    
    if args.torture_fraction > 0:
        full_dataset = create_mixed_dataset(
            args.data_dir,
            patch_size=args.patch_size,
            torture_fraction=args.torture_fraction,
            torture_patterns=args.torture_patterns,
            augment=True,
            noise_sigma=(args.noise_min, args.noise_max),
            patches_per_image=args.patches_per_image,
            max_images=args.max_images,
            use_jpeg=args.use_jpeg,
        )
    else:
        if args.use_jpeg:
            from dataset import JPEGDataset
            full_dataset = JPEGDataset(
                args.data_dir,
                patch_size=args.patch_size,
                augment=True,
                noise_sigma=(args.noise_min, args.noise_max),
                patches_per_image=args.patches_per_image,
                max_images=args.max_images,
            )
        else:
            full_dataset = LinearDataset(
                args.data_dir,
                patch_size=args.patch_size,
                augment=True,
                noise_sigma=(args.noise_min, args.noise_max),
                patches_per_image=args.patches_per_image,
                max_images=args.max_images,
                filter_file=args.filter_file,
            )

    # Train/val split
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"  Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )

    # Model
    model = XTransUNet().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Loss
    if args.mode == "finetune":
        criterion = DemosaicLoss.finetune()
    else:
        criterion = DemosaicLoss.base()

    # Override with explicit weights if provided
    if args.l1_weight is not None:
        criterion.l1_weight = args.l1_weight
    if args.msssim_weight is not None:
        criterion.msssim_weight = args.msssim_weight
        if criterion.msssim is None and args.msssim_weight > 0:
            from losses import MSSSIM
            criterion.msssim = MSSSIM().to(device)
    if args.gradient_weight is not None:
        criterion.gradient_weight = args.gradient_weight
    if args.chroma_weight is not None:
        criterion.chroma_weight = args.chroma_weight

    criterion = criterion.to(device)
    
    print(f"Loss weights: L1={criterion.l1_weight}, MS-SSIM={criterion.msssim_weight}, "
          f"grad={criterion.gradient_weight}, chroma={criterion.chroma_weight}")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = []
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch: {args.batch_size}, Patch: {args.patch_size}px")
    print(f"  Noise: [{args.noise_min}, {args.noise_max}]")
    if args.torture_fraction > 0:
        print(f"  Torture mixing: {args.torture_fraction*100:.1f}%")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_psnr, val_comp = evaluate(
            model, val_loader, criterion, device
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
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, output_dir / "best.pt")
            print(f"  -> New best ({best_val_psnr:.2f} dB)")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, output_dir / "latest.pt")

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
