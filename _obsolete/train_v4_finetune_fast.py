#!/usr/bin/env python3
"""
Fast fine-tuning for v4 model on full-resolution JPEGs.

Key optimizations:
- Load JPEGs directly (3MB compressed vs 183MB .npy)
- Convert sRGB to linear on-the-fly (fast math)
- Sequential image access (no random_split across all images)
- Cache recent images in memory
"""

import argparse
import json
import os
import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import XTransUNet, count_parameters
from losses import CombinedLoss
from xtrans_pattern import make_cfa_mask, make_channel_masks


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB (vectorized, fast)."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


class FastJPEGDataset(Dataset):
    """
    Dataset that loads JPEGs directly with on-the-fly linear conversion.
    Uses sequential image access with random patches per image.
    """

    def __init__(
        self,
        jpeg_dir: str,
        patch_size: int = 96,
        augment: bool = True,
        noise_sigma: tuple = (0.0, 0.003),
        patches_per_image: int = 64,  # More patches per image = fewer image loads
        max_images: int | None = None,
        cache_size: int = 10,  # Cache recent images
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image
        self.cache_size = cache_size

        # Find all JPEGs
        self.jpeg_files = sorted(
            glob(os.path.join(jpeg_dir, "**", "*.JPG"), recursive=True) +
            glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
        )
        if max_images:
            self.jpeg_files = self.jpeg_files[:max_images]

        print(f"  Found {len(self.jpeg_files)} JPEG files")

        # CFA masks
        self.cfa_mask = make_cfa_mask(patch_size, patch_size)
        self.channel_masks = make_channel_masks(patch_size, patch_size)

        # Image cache
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        return len(self.jpeg_files) * self.patches_per_image

    def _load_image(self, img_idx: int) -> np.ndarray:
        """Load and cache image, convert to linear."""
        if img_idx in self._cache:
            return self._cache[img_idx]

        # Load JPEG
        img = Image.open(self.jpeg_files[img_idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        img = srgb_to_linear(img).astype(np.float32)

        # Cache management
        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[img_idx] = img
        self._cache_order.append(img_idx)

        return img

    def _mosaic(self, rgb: torch.Tensor) -> torch.Tensor:
        """Apply X-Trans CFA mosaic."""
        cfa = torch.zeros(1, self.patch_size, self.patch_size, dtype=rgb.dtype)
        for ch in range(3):
            mask = (self.cfa_mask == ch)
            cfa[0][mask] = rgb[ch][mask]
        return cfa

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img = self._load_image(img_idx)
        h, w, _ = img.shape

        # Random crop (aligned to 6-pixel grid)
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        top = (random.randint(0, max(0, max_y)) // 6) * 6
        left = (random.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        # To torch (3, H, W)
        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                rgb = rgb.flip(2)
            if random.random() > 0.5:
                rgb = rgb.flip(1)

        # CFA mosaic
        cfa = self._mosaic(rgb)

        # Noise
        if self.noise_sigma[1] > 0:
            sigma = random.uniform(self.noise_sigma[0], self.noise_sigma[1])
            cfa = cfa + torch.randn_like(cfa) * sigma

        # Input: [CFA, R_mask, G_mask, B_mask]
        input_tensor = torch.cat([cfa, self.channel_masks], dim=0)
        return input_tensor, rgb


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10 * np.log10(mse)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss, _ = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            total_psnr += psnr(out, tgt)
        n += 1
    return total_loss / n, total_psnr / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp)
        loss, _ = criterion(out, tgt)
        total_loss += loss.item()
        total_psnr += psnr(out, tgt)
        n += 1
    return total_loss / n, total_psnr / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./checkpoints_v4_ft")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--patches-per-image", type=int, default=64)
    parser.add_argument("--gradient-weight", type=float, default=0.3)
    parser.add_argument("--chroma-weight", type=float, default=0.05)
    parser.add_argument("--cache-size", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading: {args.checkpoint}")
    model = XTransUNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"  Params: {count_parameters(model):,}")

    # Dataset
    print(f"\nLoading JPEGs from {args.jpeg_dir}...")
    full_ds = FastJPEGDataset(
        args.jpeg_dir,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        max_images=args.max_images,
        cache_size=args.cache_size,
    )
    n_images = len(full_ds.jpeg_files)

    # Split: 90% train, 10% val (by image, not by patch)
    val_img_count = max(1, n_images // 10)
    train_img_count = n_images - val_img_count

    train_ds = FastJPEGDataset(
        args.jpeg_dir,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        max_images=train_img_count,
        cache_size=args.cache_size,
    )
    # Val uses remaining images
    val_ds = FastJPEGDataset(
        args.jpeg_dir,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image // 4,
        max_images=args.max_images,
        cache_size=args.cache_size,
    )
    # Hack: shift val to use different images
    val_ds.jpeg_files = val_ds.jpeg_files[train_img_count:train_img_count + val_img_count]

    print(f"  Train: {len(train_ds)} patches from {train_img_count} images")
    print(f"  Val: {len(val_ds)} patches from {val_img_count} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Loss
    criterion = CombinedLoss(
        gradient_weight=args.gradient_weight,
        chroma_weight=args.chroma_weight,
    ).to(device)
    print(f"\nLoss: L1 + {args.gradient_weight}*grad + {args.chroma_weight}*chroma")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    best_psnr = 0.0
    history = []
    print(f"\nTraining {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_psnr = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_psnr:.1f}dB | Val: {val_psnr:.1f}dB | {elapsed:.0f}s")

        history.append({
            "epoch": epoch + 1,
            "train_psnr": train_psnr, "val_psnr": val_psnr,
            "train_loss": train_loss, "val_loss": val_loss,
            "time": elapsed,
        })

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_val_psnr": best_psnr,
                "version": "v4-finetune",
            }, output_dir / "best.pt")
            print(f"  -> New best: {best_psnr:.1f} dB")

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f)

    print(f"\nDone. Best: {best_psnr:.1f} dB")


if __name__ == "__main__":
    main()
