#!/usr/bin/env python3
"""Fine-tune v4 on full-res RAF/JPEG pairs with SSIM loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import rawpy
from PIL import Image
import argparse
from model import XTransUNet

# SSIM implementation
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
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.kernel, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.kernel, padding=pad, groups=self.channels) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.mean()

def ssim_loss(pred, target, ssim_module):
    return 1 - ssim_module(pred, target)

# Gradient loss
def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

class FullResDataset(Dataset):
    """Load RAF + JPEG pairs, extract random patches."""
    
    def __init__(self, raf_dir, patch_size=288):
        self.patch_size = patch_size
        self.pairs = []
        
        raf_dir = Path(raf_dir)
        for raf_path in sorted(raf_dir.glob("*.RAF")):
            jpg_path = raf_path.with_suffix(".JPG")
            if jpg_path.exists():
                self.pairs.append((raf_path, jpg_path))
        
        print(f"Found {len(self.pairs)} RAF/JPEG pairs")
    
    def __len__(self):
        return len(self.pairs) * 4  # Multiple patches per image
    
    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)
        raf_path, jpg_path = self.pairs[pair_idx]
        
        # Load RAF
        with rawpy.imread(str(raf_path)) as raw:
            cfa = raw.raw_image_visible.astype(np.float32)
            black = raw.black_level_per_channel[0]
            white = raw.white_level
            cfa = (cfa - black) / (white - black)
            cfa = np.clip(cfa, 0, 1)
            pattern = raw.raw_pattern.copy()
        
        # Load JPEG target
        jpg = Image.open(jpg_path)
        jpg = np.array(jpg).astype(np.float32) / 255.0
        
        # JPEG is usually same size as CFA for Fuji
        h, w = cfa.shape
        jh, jw = jpg.shape[:2]
        
        # Resize if needed
        if (jh, jw) != (h, w):
            jpg = np.array(Image.fromarray((jpg * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)) / 255.0
        
        # Random patch (aligned to 6x6 CFA pattern)
        max_y = h - self.patch_size - 6
        max_x = w - self.patch_size - 6
        if max_y < 0 or max_x < 0:
            # Image too small, use what we can
            y, x = 0, 0
            ps = min(h, w, self.patch_size)
            ps = (ps // 6) * 6
        else:
            y = (np.random.randint(0, max_y) // 6) * 6
            x = (np.random.randint(0, max_x) // 6) * 6
            ps = self.patch_size
        
        cfa_patch = cfa[y:y+ps, x:x+ps]
        jpg_patch = jpg[y:y+ps, x:x+ps]
        
        # No flips! CFA pattern alignment is critical
        # Only 90-degree rotations that preserve pattern? Actually no, skip all augmentation
        
        cfa_tensor = torch.from_numpy(cfa_patch).unsqueeze(0)  # [1, H, W]
        jpg_tensor = torch.from_numpy(jpg_patch).permute(2, 0, 1)  # [3, H, W]
        
        return cfa_tensor, jpg_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/Volumes/External/xtrans_dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v4/best.pt')
    parser.add_argument('--output-dir', type=str, default='checkpoints_v4_ssim')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l1-weight', type=float, default=0.5)
    parser.add_argument('--ssim-weight', type=float, default=0.4)
    parser.add_argument('--grad-weight', type=float, default=0.1)
    parser.add_argument('--patch-size', type=int, default=288)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = XTransUNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        best_psnr = ckpt.get('best_psnr', 0)
    else:
        model.load_state_dict(ckpt)
        best_psnr = 0
    print(f"Loaded: {args.checkpoint} ({best_psnr:.2f} dB)")
    
    # Dataset
    dataset = FullResDataset(args.data_dir, args.patch_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Loss and optimizer
    ssim_module = SSIM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    Path(args.output_dir).mkdir(exist_ok=True)
    best_val_psnr = 0
    
    print(f"\nFine-tuning with L1={args.l1_weight}, SSIM={args.ssim_weight}, Grad={args.grad_weight}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        
        for batch_idx, (cfa, target) in enumerate(train_loader):
            cfa, target = cfa.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(cfa)
            
            # Combined loss
            l1 = F.l1_loss(pred, target)
            ssim = ssim_loss(pred, target, ssim_module)
            grad = gradient_loss(pred, target)
            
            loss = args.l1_weight * l1 + args.ssim_weight * ssim + args.grad_weight * grad
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_loader)}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_psnr = 0
        val_ssim = 0
        with torch.no_grad():
            for cfa, target in val_loader:
                cfa, target = cfa.to(device), target.to(device)
                pred = model(cfa)
                
                mse = F.mse_loss(pred, target)
                psnr = 10 * torch.log10(1.0 / mse)
                val_psnr += psnr.item()
                
                ssim_val = ssim_module(pred, target)
                val_ssim += ssim_val.item()
        
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        
        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
        
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Best: {best_val_psnr:.2f} dB {'*' if is_best else ''}")
        
        # Save checkpoints
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_psnr': best_val_psnr,
            'val_ssim': val_ssim,
        }
        torch.save(state, f"{args.output_dir}/latest.pt")
        if is_best:
            torch.save(state, f"{args.output_dir}/best.pt")

if __name__ == '__main__':
    main()
