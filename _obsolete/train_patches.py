#!/usr/bin/env python3
"""Simple patch-based fine-tuning for v4 model."""

import os
import random
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model import XTransUNet, count_parameters
from losses import CombinedLoss
from xtrans_pattern import make_cfa_mask, make_channel_masks


class PatchDataset(Dataset):
    def __init__(self, patch_dir, augment=True, noise_sigma=(0, 0.003)):
        self.files = sorted([
            os.path.join(patch_dir, f) for f in os.listdir(patch_dir) 
            if f.endswith(".npy")
        ])
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.cfa_mask = make_cfa_mask(96, 96)
        self.channel_masks = make_channel_masks(96, 96)
        print(f"  Loaded {len(self.files)} patches")
    
    def __len__(self):
        return len(self.files)
    
    def _mosaic(self, rgb):
        cfa = torch.zeros(1, 96, 96, dtype=rgb.dtype)
        for ch in range(3):
            mask = (self.cfa_mask == ch)
            cfa[0][mask] = rgb[ch][mask]
        return cfa
    
    def __getitem__(self, idx):
        patch = np.load(self.files[idx])  # (96, 96, 3)
        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()
        
        if self.augment:
            if random.random() > 0.5:
                rgb = rgb.flip(2)
            if random.random() > 0.5:
                rgb = rgb.flip(1)
        
        cfa = self._mosaic(rgb)
        if self.noise_sigma[1] > 0:
            sigma = random.uniform(*self.noise_sigma)
            cfa = cfa + torch.randn_like(cfa) * sigma
        
        return torch.cat([cfa, self.channel_masks], dim=0), rgb


def psnr(p, t):
    mse = ((p - t) ** 2).mean().item()
    return 100.0 if mse < 1e-10 else -10 * np.log10(mse)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading checkpoint...")
    model = XTransUNet().to(device)
    ckpt = torch.load("checkpoints_v4/best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    # Dataset
    print("Loading patches...")
    full_ds = PatchDataset("data/jpeg_patches")
    val_size = 1600
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Train: {train_size}, Val: {val_size}")

    # Loss and optimizer
    criterion = CombinedLoss(gradient_weight=0.3, chroma_weight=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    out_dir = Path("checkpoints_v4_ft")
    out_dir.mkdir(exist_ok=True)

    print("\nTraining 50 epochs...")
    best_psnr = 0.0
    history = []

    for epoch in range(50):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = train_psnr = 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss, _ = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            with torch.no_grad():
                train_psnr += psnr(out, tgt)
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = val_psnr = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                loss, _ = criterion(out, tgt)
                val_loss += loss.item()
                val_psnr += psnr(out, tgt)
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch+1:2d}/50 | Train: {train_psnr:.1f}dB | Val: {val_psnr:.1f}dB | {elapsed:.0f}s")
        
        history.append({"epoch": epoch+1, "train_psnr": train_psnr, "val_psnr": val_psnr, "time": elapsed})
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_val_psnr": best_psnr}, out_dir / "best.pt")
            print(f"  -> New best: {best_psnr:.1f}dB")
        
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f)

    print(f"\nDone. Best: {best_psnr:.1f}dB")


if __name__ == "__main__":
    main()
