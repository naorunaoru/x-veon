"""
Dataset v4 for X-Trans demosaicing training.

Loads pre-processed linear float32 .npy files (from build_dataset_v4.py).
Training operates entirely in linear sensor space — no gamma, no WB, no clipping.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from xtrans_pattern import make_cfa_mask, make_channel_masks, XTRANS_PATTERN


def mosaic_linear(rgb: torch.Tensor, cfa_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply X-Trans CFA mosaic to an RGB image.
    
    Args:
        rgb: (3, H, W) float tensor — linear RGB
        cfa_mask: (H, W) long tensor — 0=R, 1=G, 2=B
    
    Returns:
        (1, H, W) float tensor — CFA mosaic
    """
    h, w = cfa_mask.shape
    cfa = torch.zeros(1, h, w, dtype=rgb.dtype)
    for ch in range(3):
        mask = (cfa_mask == ch)
        cfa[0][mask] = rgb[ch][mask]
    return cfa


class XTransLinearDataset(Dataset):
    """
    Dataset that loads linear float32 .npy files and creates CFA training pairs.
    
    Each .npy file is a (H, W, 3) float32 array in linear sensor space
    (black-subtracted, normalized by white-black, no WB, no clip).
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 96,
        augment: bool = True,
        noise_sigma: tuple[float, float] = (0.0, 0.005),
        patches_per_image: int = 16,
        max_images: int | None = None,
        filter_file: str | None = None,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image

        # Must be divisible by 6 (CFA) and 16 (UNet pooling)
        assert patch_size % 6 == 0, f"patch_size must be divisible by 6, got {patch_size}"
        assert patch_size % 16 == 0, f"patch_size must be divisible by 16, got {patch_size}"

        # Collect .npy files
        self.data_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npy') and not f.endswith('_meta.npy')
        ])

        # Filter to specific files if filter provided
        if filter_file is not None:
            import json
            with open(filter_file) as flt:
                allowed_stems = set(json.load(flt))
            self.data_files = [
                f for f in self.data_files
                if os.path.splitext(os.path.basename(f))[0] in allowed_stems
            ]
            print(f"  Filtered to {len(self.data_files)} images from {filter_file}")

        if max_images is not None:
            self.data_files = self.data_files[:max_images]

        if len(self.data_files) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

        print(f"  Found {len(self.data_files)} images")

        # Pre-compute CFA masks for the patch size
        self.cfa = make_cfa_mask(patch_size, patch_size)
        self.masks = make_channel_masks(patch_size, patch_size)

    def __len__(self):
        return len(self.data_files) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = random.Random()

        # Load image (H, W, 3) float32
        img = np.load(self.data_files[img_idx])
        h, w, _ = img.shape

        # Random crop (must align to 6-pixel grid for CFA consistency)
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        top = (rng.randint(0, max(0, max_y)) // 6) * 6
        left = (rng.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        # Convert to torch (3, H, W)
        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        # Augmentation: flips and 90° rotations
        # For X-Trans, we need to be careful — rotations that aren't multiples
        # of 180° would change the CFA pattern alignment.
        # Safe augmentations: horizontal flip, vertical flip, 180° rotation
        if self.augment:
            if rng.random() > 0.5:
                rgb = rgb.flip(2)  # horizontal flip
            if rng.random() > 0.5:
                rgb = rgb.flip(1)  # vertical flip
            # Note: 90° rotations would break CFA alignment, skip them

        # Simulate CFA mosaicing
        cfa_img = mosaic_linear(rgb, self.cfa)  # (1, H, W)

        # Add noise (simulates sensor noise in linear space)
        if self.noise_sigma[1] > 0:
            sigma = rng.uniform(self.noise_sigma[0], self.noise_sigma[1])
            if sigma > 0:
                noise = torch.randn_like(cfa_img) * sigma
                cfa_img = cfa_img + noise
                # Don't clamp — linear space can have small negatives from noise

        # Build 4-channel input: [CFA, R_mask, G_mask, B_mask]
        input_tensor = torch.cat([cfa_img, self.masks], dim=0)  # (4, H, W)

        return input_tensor, rgb


class TortureTestLinearDataset(Dataset):
    """Synthetic torture test patterns in linear space."""

    def __init__(self, size: int = 96, num_patterns: int = 100):
        self.size = size
        self.num_patterns = num_patterns
        self.cfa = make_cfa_mask(size, size)
        self.masks = make_channel_masks(size, size)

    def __len__(self):
        return self.num_patterns

    def __getitem__(self, idx):
        rng = random.Random(idx)
        pattern_type = idx % 5

        h = w = self.size
        rgb = torch.zeros(3, h, w)

        if pattern_type == 0:
            # Diagonal stripes (linear values, typical sensor range)
            freq = rng.uniform(0.05, 0.3)
            angle = rng.uniform(0, 2 * 3.14159)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32), indexing="ij")
            stripe = (torch.sin((xx * torch.cos(torch.tensor(angle))
                      + yy * torch.sin(torch.tensor(angle))) * freq) * 0.5 + 0.5)
            # Linear sensor values are typically 0-0.5 range (before WB)
            c1 = torch.tensor([rng.uniform(0.05, 0.4) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.05, 0.4) for _ in range(3)])
            for c in range(3):
                rgb[c] = c1[c] * stripe + c2[c] * (1 - stripe)

        elif pattern_type == 1:
            # Fine parallel lines
            freq = rng.randint(2, 8)
            horizontal = rng.random() > 0.5
            c1 = torch.tensor([rng.uniform(0.05, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.05, 0.5) for _ in range(3)])
            coords = torch.arange(h if horizontal else w)
            mask = ((coords // freq) % 2 == 0).float()
            for c in range(3):
                if horizontal:
                    rgb[c] = c1[c] * mask.unsqueeze(1) + c2[c] * (1 - mask.unsqueeze(1))
                else:
                    rgb[c] = c1[c] * mask.unsqueeze(0) + c2[c] * (1 - mask.unsqueeze(0))

        elif pattern_type == 2:
            # Checkerboard
            freq = rng.randint(2, 6)
            c1 = torch.tensor([rng.uniform(0.05, 0.4) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.05, 0.4) for _ in range(3)])
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            checker = (((yy // freq) + (xx // freq)) % 2 == 0).float()
            for c in range(3):
                rgb[c] = c1[c] * checker + c2[c] * (1 - checker)

        elif pattern_type == 3:
            # Concentric circles
            freq = rng.uniform(0.1, 0.4)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32), indexing="ij")
            dist = torch.sqrt((xx - w/2)**2 + (yy - h/2)**2)
            ring = torch.sin(dist * freq) * 0.5 + 0.5
            c1 = torch.tensor([rng.uniform(0.05, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.05, 0.5) for _ in range(3)])
            for c in range(3):
                rgb[c] = c1[c] * ring + c2[c] * (1 - ring)

        else:
            # Color gradient
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32) / h,
                torch.arange(w, dtype=torch.float32) / w, indexing="ij")
            t = (yy + xx) / 2
            c1 = torch.tensor([rng.uniform(0.05, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.05, 0.5) for _ in range(3)])
            for c in range(3):
                rgb[c] = c1[c] * (1 - t) + c2[c] * t
            rgb += torch.randn_like(rgb) * 0.02

        cfa_img = mosaic_linear(rgb, self.cfa)
        input_tensor = torch.cat([cfa_img, self.masks], dim=0)
        return input_tensor, rgb
