"""
Training script v4 for X-Trans demosaicing.

Key changes from v3:
- Linear sensor space (no sRGB gamma, no WB, no clipping)
- Float32 .npy input files from build_dataset_v4.py
- Smaller patches (96x96) due to downscaled images (1233x824)
- Loss computed in linear space

Usage:
    python train_v4.py --data-dir /Volumes/External/xtrans_v4_dataset --epochs 200
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import XTransUNet, count_parameters
from dataset_v4 import XTransLinearDataset, TortureTestLinearDataset
from losses import CombinedLoss


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB (in linear space)."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    # Use 1.0 as peak (nominal white in normalized sensor space)
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
    total_components = {'l1': 0.0, 'gradient': 0.0, 'chroma': 0.0}
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
        for k in total_components:
            total_components[k] += components[k]
        with torch.no_grad():
            total_psnr += psnr(outputs, targets)
        n_batches += 1

    avg_components = {k: v / n_batches for k, v in total_components.items()}
    return total_loss / n_batches, total_psnr / n_batches, avg_components


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_components = {'l1': 0.0, 'gradient': 0.0, 'chroma': 0.0}
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss, components = criterion(outputs, targets)

        total_loss += loss.item()
        for k in total_components:
            total_components[k] += components[k]
        total_psnr += psnr(outputs, targets)
        n_batches += 1

    avg_components = {k: v / n_batches for k, v in total_components.items()}
    return total_loss / n_batches, total_psnr / n_batches, avg_components


@torch.no_grad()
def evaluate_torture(model, criterion, device, size=96, num_patterns=50):
    dataset = TortureTestLinearDataset(size=size, num_patterns=num_patterns)
    loader = DataLoader(dataset, batch_size=8, num_workers=0)

    model.eval()
    total_psnr = 0.0
    total_components = {'l1': 0.0, 'gradient': 0.0, 'chroma': 0.0}
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        _, components = criterion(outputs, targets)
        total_psnr += psnr(outputs, targets)
        for k in total_components:
            total_components[k] += components[k]
        n_batches += 1

    avg_components = {k: v / n_batches for k, v in total_components.items()}
    return total_psnr / n_batches, avg_components


def main():
    parser = argparse.ArgumentParser(description="Train X-Trans demosaicing model v4 (linear)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints_v4")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.005)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--chroma-weight", type=float, default=0.05)
    parser.add_argument("--filter-file", type=str, default=None,
                        help="JSON file with list of filename stems to use")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    full_dataset = XTransLinearDataset(
        args.data_dir,
        patch_size=args.patch_size,
        augment=True,
        noise_sigma=(args.noise_min, args.noise_max),
        patches_per_image=args.patches_per_image,
        max_images=args.max_images,
        filter_file=args.filter_file,
    )
    n_images = len(full_dataset.data_files)
    print(f"  {n_images} images -> {len(full_dataset)} patches/epoch")

    # Train/val split
    val_img_count = max(1, int(n_images * args.val_split))
    val_size = val_img_count * args.patches_per_image
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=use_pin, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_pin,
        persistent_workers=args.workers > 0,
    )

    # Model (same architecture)
    model = XTransUNet().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Loss
    criterion = CombinedLoss(
        gradient_weight=args.gradient_weight,
        chroma_weight=args.chroma_weight,
    ).to(device)
    print(f"Loss: L1 + {args.gradient_weight}*gradient + {args.chroma_weight}*chroma")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 0
    best_val_psnr = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_psnr = ckpt.get("best_val_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['version'] = 'v4-linear'
    config['device'] = str(device)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = []
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Images: {n_images - val_img_count} train, {val_img_count} val")
    print(f"  Patches/epoch: {train_size} train, {val_size} val")
    print(f"  Batch: {args.batch_size}, Patch: {args.patch_size}px")
    print(f"  Noise: sigma in [{args.noise_min}, {args.noise_max}]")
    print(f"  Color space: LINEAR (raw sensor)")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_psnr, val_comp = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_psnr:.1f}dB L1:{train_comp['l1']:.4f} | "
            f"Val: {val_psnr:.1f}dB L1:{val_comp['l1']:.4f} | "
            f"LR:{lr:.1e} | {elapsed:.0f}s"
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_psnr": train_psnr,
            "train_components": train_comp,
            "val_loss": val_loss, "val_psnr": val_psnr,
            "val_components": val_comp,
            "lr": lr, "time": elapsed,
        }
        history.append(entry)

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
                "version": "v4-linear",
            }, output_dir / "best.pt")
            print(f"  -> New best (PSNR: {best_val_psnr:.1f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, output_dir / "latest.pt")

            torture_psnr, torture_comp = evaluate_torture(model, criterion, device)
            print(f"  -> Torture: {torture_psnr:.1f}dB")
            entry["torture_psnr"] = torture_psnr

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val PSNR: {best_val_psnr:.1f}")


if __name__ == "__main__":
    main()
