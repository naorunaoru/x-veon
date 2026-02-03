#!/usr/bin/env python3
"""
Fine-tune v4 on full-res JPEGs with SSIM loss.
Fixed version: NO flip augmentation (breaks CFA alignment).
"""

import os
import random
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

from model import XTransUNet
from xtrans_pattern import make_cfa_mask, make_channel_masks


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB."""
    mask = srgb <= 0.04045
    linear = np.where(mask, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


# ============ SSIM Loss ============
def gaussian_kernel(size=11, sigma=1.5, channels=3):
    x = torch.arange(size).float() - size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss.outer(gauss)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer('kernel', gaussian_kernel(window_size, sigma, channels))
        
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        
        mu1 = F.conv2d(pred, self.kernel, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.kernel, padding=pad, groups=self.channels)
        
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.kernel, padding=pad, groups=self.channels) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


# ============ Dataset ============
class XTransFinetuneDataset(Dataset):
    """Load JPEGs, convert to linear, simulate CFA. NO FLIP AUGMENTATION."""

    def __init__(self, jpeg_dir: str, patch_size: int = 288, 
                 noise_sigma: tuple = (0.0, 0.002), patches_per_image: int = 16):
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image

        assert patch_size % 6 == 0 and patch_size % 16 == 0

        self.jpeg_files = sorted(
            glob(os.path.join(jpeg_dir, "**", "*.JPG"), recursive=True) +
            glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
        )
        
        if not self.jpeg_files:
            raise ValueError(f"No JPEGs in {jpeg_dir}")
        print(f"Found {len(self.jpeg_files)} JPEG images")

        self.cfa_mask = make_cfa_mask(patch_size, patch_size)
        self.channel_masks = make_channel_masks(patch_size, patch_size).float()

    def __len__(self):
        return len(self.jpeg_files) * self.patches_per_image

    def _mosaic(self, rgb: torch.Tensor) -> torch.Tensor:
        h, w = self.cfa_mask.shape
        cfa = torch.zeros(1, h, w, dtype=rgb.dtype)
        for ch in range(3):
            mask = (self.cfa_mask == ch)
            cfa[0][mask] = rgb[ch][mask]
        return cfa

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        
        img = Image.open(self.jpeg_files[img_idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        img = srgb_to_linear(img)
        
        h, w = img.shape[:2]
        ps = self.patch_size

        # Pad if needed
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = img.shape[:2]

        # Random crop aligned to 6-pixel grid
        max_y, max_x = h - ps, w - ps
        top = (random.randint(0, max(0, max_y)) // 6) * 6
        left = (random.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+ps, left:left+ps]

        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy())
        
        # NO FLIP AUGMENTATION - this was the bug!
        
        cfa_img = self._mosaic(rgb)

        # Add slight noise
        if self.noise_sigma[1] > 0:
            sigma = random.uniform(self.noise_sigma[0], self.noise_sigma[1])
            cfa_img = cfa_img + torch.randn_like(cfa_img) * sigma

        input_tensor = torch.cat([cfa_img, self.channel_masks], dim=0)
        return input_tensor, rgb


# ============ Training ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/fuji_jpgs')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v4/best.pt')
    parser.add_argument('--output-dir', type=str, default='checkpoints_v4_ssim')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--l1-weight', type=float, default=0.4)
    parser.add_argument('--ssim-weight', type=float, default=0.5)
    parser.add_argument('--grad-weight', type=float, default=0.1)
    parser.add_argument('--patch-size', type=int, default=288)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = XTransUNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        start_psnr = ckpt.get('best_psnr', 0)
    else:
        model.load_state_dict(ckpt)
        start_psnr = 0
    print(f"Loaded: {args.checkpoint} ({start_psnr:.2f} dB)")
    
    dataset = XTransFinetuneDataset(args.data_dir, args.patch_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    ssim_module = SSIM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    Path(args.output_dir).mkdir(exist_ok=True)
    best_val_ssim = 0
    
    print(f"\nLoss weights: L1={args.l1_weight}, SSIM={args.ssim_weight}, Grad={args.grad_weight}")
    print("-" * 70)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        
        for batch_idx, (inp, target) in enumerate(train_loader):
            inp, target = inp.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(inp)
            
            l1 = F.l1_loss(pred, target)
            ssim = 1 - ssim_module(pred, target)
            grad = gradient_loss(pred, target)
            
            loss = args.l1_weight * l1 + args.ssim_weight * ssim + args.grad_weight * grad
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} | L1={l1:.4f} SSIM={1-ssim:.4f}")
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_psnr, val_ssim_total = 0, 0
        with torch.no_grad():
            for inp, target in val_loader:
                inp, target = inp.to(device), target.to(device)
                pred = model(inp)
                
                mse = F.mse_loss(pred, target)
                val_psnr += (10 * torch.log10(1.0 / mse)).item()
                val_ssim_total += ssim_module(pred, target).item()
        
        val_psnr /= len(val_loader)
        val_ssim = val_ssim_total / len(val_loader)
        
        is_best = val_ssim > best_val_ssim
        if is_best:
            best_val_ssim = val_ssim
        
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | PSNR: {val_psnr:.2f} dB | "
              f"SSIM: {val_ssim:.4f} | Best SSIM: {best_val_ssim:.4f} | LR: {lr:.2e} {'*' if is_best else ''}")
        
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_psnr': val_psnr,
            'best_ssim': best_val_ssim,
        }
        torch.save(state, f"{args.output_dir}/latest.pt")
        if is_best:
            torch.save(state, f"{args.output_dir}/best.pt")

if __name__ == '__main__':
    main()
