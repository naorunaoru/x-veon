"""
Fine-tuning script for v4 model on full-resolution JPEGs.

Teaches the model to preserve texture/sharpness that was lost
when training on 4x downsampled linear data.

Key changes from train_v4.py:
- Starts from v4 checkpoint
- Uses full-res JPEGs converted to linear
- Higher gradient weight for sharpness
- Mixes in torture test patterns
- Lower LR, fewer epochs
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split

from model import XTransUNet, count_parameters
from dataset_v4_finetune import XTransFinetuneDataset, TortureFinetuneDataset
from losses import CombinedLoss


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
def evaluate_torture(model, criterion, device, size=96, num_patterns=100):
    dataset = TortureFinetuneDataset(size=size, num_patterns=num_patterns)
    loader = DataLoader(dataset, batch_size=16, num_workers=0)

    model.eval()
    total_psnr = 0.0
    n_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        total_psnr += psnr(outputs, targets)
        n_batches += 1

    return total_psnr / n_batches


def main():
    parser = argparse.ArgumentParser(description="Fine-tune v4 on full-res JPEGs")
    parser.add_argument("--jpeg-dir", type=str, required=True,
                        help="Directory with JPEG files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="v4 checkpoint to fine-tune from")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_v4_ft")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower LR for fine-tuning")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--gradient-weight", type=float, default=0.3,
                        help="Higher gradient weight for sharpness")
    parser.add_argument("--chroma-weight", type=float, default=0.05)
    parser.add_argument("--torture-ratio", type=float, default=0.2,
                        help="Fraction of each epoch from torture patterns")
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.003)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = XTransUNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    orig_psnr = ckpt.get("best_val_psnr", 0)
    orig_epoch = ckpt.get("epoch", 0)
    print(f"  Loaded: epoch {orig_epoch}, PSNR {orig_psnr:.1f} dB")
    print(f"  Parameters: {count_parameters(model):,}")

    # JPEG dataset
    print(f"\nLoading JPEGs from {args.jpeg_dir}...")
    jpeg_dataset = XTransFinetuneDataset(
        args.jpeg_dir,
        patch_size=args.patch_size,
        augment=True,
        noise_sigma=(args.noise_min, args.noise_max),
        patches_per_image=args.patches_per_image,
        max_images=args.max_images,
    )
    n_images = len(jpeg_dataset.jpeg_files)

    # Torture dataset (for mixing in)
    n_torture = int(len(jpeg_dataset) * args.torture_ratio)
    torture_dataset = TortureFinetuneDataset(
        size=args.patch_size,
        num_patterns=n_torture
    )
    print(f"  Torture patterns: {n_torture}")

    # Combined dataset
    combined = ConcatDataset([jpeg_dataset, torture_dataset])
    print(f"  Total samples per epoch: {len(combined)}")

    # Train/val split
    val_size = max(1, int(len(combined) * args.val_split))
    train_size = len(combined) - val_size
    train_dataset, val_dataset = random_split(combined, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )

    # Loss with higher gradient weight
    criterion = CombinedLoss(
        gradient_weight=args.gradient_weight,
        chroma_weight=args.chroma_weight,
    ).to(device)
    print(f"\nLoss: L1 + {args.gradient_weight}*gradient + {args.chroma_weight}*chroma")

    # Optimizer with lower LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['version'] = 'v4-finetune'
    config['base_checkpoint'] = args.checkpoint
    config['base_epoch'] = orig_epoch
    config['base_psnr'] = orig_psnr
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = []
    best_val_psnr = 0.0
    print(f"\nFine-tuning for {args.epochs} epochs...")
    print(f"  Train: {train_size}, Val: {val_size}")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}")
    print()

    for epoch in range(args.epochs):
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
            f"Val: {val_psnr:.1f}dB | "
            f"LR:{lr:.1e} | {elapsed:.0f}s"
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_psnr": train_psnr,
            "val_loss": val_loss, "val_psnr": val_psnr,
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
                "base_checkpoint": args.checkpoint,
                "version": "v4-finetune",
            }, output_dir / "best.pt")
            print(f"  -> New best: {best_val_psnr:.1f} dB")

        # Evaluate on torture test periodically
        if (epoch + 1) % 10 == 0:
            torture_psnr = evaluate_torture(model, criterion, device)
            print(f"  -> Torture: {torture_psnr:.1f} dB")
            entry["torture_psnr"] = torture_psnr
            
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, output_dir / "latest.pt")

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val PSNR: {best_val_psnr:.1f}")


if __name__ == "__main__":
    main()
