#!/usr/bin/env python3
"""
Training script for the highlight reconstruction model (second pass).

Uses HighlightDataset (RGB-space synthetic clipping) and HighlightUNet
(gated convolution U-Net).

Examples:
    # Train from scratch
    python train_hl.py --data-dir /path/to/npy --apply-wb --epochs 200

    # Resume training
    python train_hl.py --data-dir /path/to/npy --apply-wb \
        --resume checkpoints_hl/best.pt
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hl_model import HighlightUNet, count_parameters
from dataset import LinearDataset, HighlightDataset
from losses import SobelGradientLoss


def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              clip_mask: torch.Tensor, hl_weight: float = 1.0) -> torch.Tensor:
    """L1 loss with extra weight on clipped pixels.

    Uniform L1 over the whole patch, plus an additional weighted L1
    on pixels where clip_mask > 0.  This ensures the model gets strong
    gradient signal from the highlight region without ignoring the
    identity-mapping requirement elsewhere.

    Args:
        pred, target: (B, 3, H, W)
        clip_mask: (B, 1, H, W) soft mask 0..1
        hl_weight: multiplier for the masked term
    """
    base = F.l1_loss(pred, target)
    if hl_weight <= 0:
        return base, {"l1": base.item(), "hl_l1": 0.0}

    # Weighted highlight L1: mean of (|error| * mask) / mean(mask)
    mask = clip_mask.expand_as(pred)
    mask_sum = mask.sum()
    if mask_sum < 1.0:
        return base, {"l1": base.item(), "hl_l1": 0.0}

    hl_l1 = ((pred - target).abs() * mask).sum() / mask_sum
    total = base + hl_weight * hl_l1
    return total, {"l1": base.item(), "hl_l1": hl_l1.item()}


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse)


def hl_psnr(pred: torch.Tensor, target: torch.Tensor,
            clip_mask: torch.Tensor) -> float | None:
    """PSNR only on pixels where clip_mask > 0."""
    mask = (clip_mask > 0).expand_as(pred)
    n = mask.sum().item()
    if n == 0:
        return None
    mse = ((pred - target) ** 2 * mask).sum().item() / n
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, device, hl_weight=0.0,
                grad_loss_fn=None, grad_weight=0.0, scaler=None):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_hl_psnr = 0.0
    comp_sums = {}
    n_hl = 0
    n_batches = 0
    use_amp = scaler is not None

    for inputs, targets, clip_levels in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        clip_mask = inputs[:, 3:4]

        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss, comps = masked_l1(outputs, targets, clip_mask, hl_weight)
            if grad_loss_fn is not None and grad_weight > 0:
                grad_l = grad_loss_fn(outputs, targets)
                loss = loss + grad_weight * grad_l
                comps["grad"] = grad_l.item()

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
        for k, v in comps.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v
        with torch.no_grad():
            total_psnr += psnr(outputs, targets)
            hp = hl_psnr(outputs, targets, clip_mask)
            if hp is not None:
                total_hl_psnr += hp
                n_hl += 1
        n_batches += 1

    avg_comps = {k: v / n_batches for k, v in comp_sums.items()}
    return (
        total_loss / n_batches,
        total_psnr / n_batches,
        total_hl_psnr / n_hl if n_hl > 0 else None,
        avg_comps,
    )


@torch.no_grad()
def evaluate(model, loader, device, hl_weight=0.0,
             grad_loss_fn=None, grad_weight=0.0, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_hl_psnr = 0.0
    n_hl = 0
    n_batches = 0

    for inputs, targets, clip_levels in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        clip_mask = inputs[:, 3:4]

        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss, _ = masked_l1(outputs, targets, clip_mask, hl_weight)
            if grad_loss_fn is not None and grad_weight > 0:
                loss = loss + grad_weight * grad_loss_fn(outputs, targets)

        total_loss += loss.item()
        total_psnr += psnr(outputs, targets)
        hp = hl_psnr(outputs, targets, clip_mask)
        if hp is not None:
            total_hl_psnr += hp
            n_hl += 1
        n_batches += 1

    return (
        total_loss / n_batches,
        total_psnr / n_batches,
        total_hl_psnr / n_hl if n_hl > 0 else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Highlight reconstruction training")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--filter-file", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patches-per-image", type=int, default=16)

    # White balance
    parser.add_argument("--apply-wb", action="store_true")
    parser.add_argument("--wb-aug-range", type=float, default=0.0)

    # Highlight augmentation
    parser.add_argument("--highlight-ev-min", type=float, default=0.5,
                        help="Min EV boost for highlight simulation")
    parser.add_argument("--highlight-ev-max", type=float, default=3.0,
                        help="Max EV boost for highlight simulation")
    parser.add_argument("--highlight-prob", type=float, default=1.0,
                        help="Probability of applying highlight clipping (default: always)")
    parser.add_argument("--bright-spot-prob", type=float, default=0.5,
                        help="Probability of adding synthetic bright spots (default 0.5)")
    parser.add_argument("--bright-spot-intensity-max", type=float, default=5.0)
    parser.add_argument("--bright-spot-sigma-max", type=float, default=20.0)

    # Patch rejection
    parser.add_argument("--min-clip-frac", type=float, default=0.05,
                        help="Reject patches with less than this fraction clipped")
    parser.add_argument("--max-clip-frac", type=float, default=0.80,
                        help="Reject patches with more than this fraction clipped")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Max retries for patch rejection")

    # Loss
    parser.add_argument("--hl-weight", type=float, default=5.0,
                        help="Extra weight for L1 on clipped pixels (0 = uniform L1)")
    parser.add_argument("--gradient-weight", type=float, default=0.1,
                        help="Weight for Sobel gradient loss (edge preservation)")

    # Model
    parser.add_argument("--base-width", type=int, default=32,
                        help="Base channel width (default 32 → 32/64/128/256)")

    # Checkpoints
    parser.add_argument("--output-dir", type=str, default="./checkpoints_hl")
    parser.add_argument("--resume", type=str, default=None)

    # Performance
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    device = get_device()
    print(f"Device: {device}")

    # Dataset — split at image level
    print(f"\nLoading dataset from {args.data_dir}...")
    all_files = LinearDataset.find_files(
        args.data_dir,
        max_images=args.max_images,
        filter_file=args.filter_file,
    )
    random.Random(args.seed).shuffle(all_files)

    val_n = max(1, int(len(all_files) * args.val_split))
    val_files = all_files[:val_n]
    train_files = all_files[val_n:]
    print(f"  Images: {len(train_files)} train, {val_n} val")

    shared_kwargs = dict(
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        apply_wb=args.apply_wb,
    )

    hl_kwargs = dict(
        highlight_ev_range=(args.highlight_ev_min, args.highlight_ev_max),
        highlight_prob=args.highlight_prob,
        bright_spot_prob=args.bright_spot_prob,
        bright_spot_intensity=(1.5, args.bright_spot_intensity_max),
        bright_spot_sigma=(2.0, args.bright_spot_sigma_max),
        min_clip_frac=args.min_clip_frac,
        max_clip_frac=args.max_clip_frac,
        max_retries=args.max_retries,
    )

    train_dataset = HighlightDataset(
        files=train_files,
        augment=True,
        wb_aug_range=args.wb_aug_range if args.apply_wb else 0.0,
        **hl_kwargs,
        **shared_kwargs,
    )
    val_dataset = HighlightDataset(
        files=val_files,
        augment=False,
        **hl_kwargs,
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
    model = HighlightUNet(base_width=args.base_width).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Resume
    start_epoch = 0
    best_val_psnr = 0.0
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_psnr = ckpt.get("best_val_psnr", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best PSNR: {best_val_psnr:.1f}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    remaining = args.epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(remaining, 1))

    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    scaler = torch.amp.GradScaler(device.type, enabled=args.amp) if args.amp else None

    # Gradient loss
    grad_loss_fn = None
    if args.gradient_weight > 0:
        grad_loss_fn = SobelGradientLoss().to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    history = []
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch: {args.batch_size}, Patch: {args.patch_size}px")
    print(f"  Highlight EV: {args.highlight_ev_min}-{args.highlight_ev_max}, "
          f"prob: {args.highlight_prob}")
    print(f"  Loss: L1 + hl_weight={args.hl_weight} + grad={args.gradient_weight}")
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_psnr_val, train_hl, train_comps = train_epoch(
            model, train_loader, optimizer, device,
            hl_weight=args.hl_weight, grad_loss_fn=grad_loss_fn,
            grad_weight=args.gradient_weight, scaler=scaler)
        val_loss, val_psnr_val, val_hl = evaluate(
            model, val_loader, device, hl_weight=args.hl_weight,
            grad_loss_fn=grad_loss_fn, grad_weight=args.gradient_weight,
            use_amp=args.amp)
        scheduler.step()

        gc.collect()
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        hl_str = ""
        if val_hl is not None:
            hl_str = f" | HL: {val_hl:.1f}dB"

        comp_str = " ".join(f"{k}:{v:.4f}" for k, v in train_comps.items())
        print(
            f"Ep {epoch + 1:3d}/{args.epochs} | "
            f"Train: {train_psnr_val:.1f}dB | Val: {val_psnr_val:.1f}dB{hl_str} | "
            f"{comp_str} | LR: {lr:.1e} | {elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_psnr": train_psnr_val,
            "train_hl_psnr": train_hl,
            "train_components": train_comps,
            "val_loss": val_loss,
            "val_psnr": val_psnr_val,
            "val_hl_psnr": val_hl,
            "lr": lr,
            "time": elapsed,
        })

        # Save best
        tracking = val_hl if val_hl is not None else val_psnr_val
        if tracking > best_val_psnr:
            best_val_psnr = tracking
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_psnr": best_val_psnr,
                "base_width": args.base_width,
            }, output_dir / "best.pt")
            print(f"  -> New best ({best_val_psnr:.2f} dB)")

        # Periodic save
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_psnr": best_val_psnr,
                "base_width": args.base_width,
            }, output_dir / "latest.pt")

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
