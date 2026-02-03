"""
Dataset v2 for X-Trans demosaicing training.

Uses real Fujifilm JPEGs as source images, simulates CFA mosaicing.
Includes optional sensor noise simulation for more realistic training.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from xtrans_pattern import make_cfa_mask, make_channel_masks, mosaic


class XTransDataset(Dataset):
    """
    Dataset that loads RGB images and creates synthetic X-Trans CFA pairs.
    
    Uses Fujifilm JPEGs as source — these contain the actual scene content
    and Fuji's color science, giving the model realistic training data.
    """

    def __init__(
        self,
        image_dir: str,
        patch_size: int = 256,
        augment: bool = True,
        max_images: int | None = None,
        noise_sigma: tuple[float, float] = (0.0, 0.01),
        patches_per_image: int = 4,
        filter_keys: list[str] | None = None,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma  # (min, max) range for Gaussian noise std
        self.patches_per_image = patches_per_image

        # Collect image paths
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.image_paths = sorted(
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in exts
        )
        # Filter to specific images if filter_keys provided
        if filter_keys is not None:
            # filter_keys are like "103_FUJI/DSCF3148" (relative, no extension)
            allowed = set(filter_keys)
            image_dir_path = Path(image_dir).resolve()
            self.image_paths = [
                p for p in self.image_paths
                if str(p.resolve().relative_to(image_dir_path)).rsplit(".", 1)[0] in allowed
            ]

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        # X-Trans 6x6 CFA pattern
        self.cfa = make_cfa_mask(patch_size, patch_size)
        self.masks = make_channel_masks(patch_size, patch_size)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def _load_and_crop(self, img, rng):
        """Load image, random crop to patch_size."""
        w, h = img.size

        if h < self.patch_size or w < self.patch_size:
            img = img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.Resampling.BICUBIC,
            )
            w, h = img.size

        top = rng.randint(0, h - self.patch_size)
        left = rng.randint(0, w - self.patch_size)
        return img.crop((left, top, left + self.patch_size, top + self.patch_size))

    def _augment(self, img, rng):
        """Random flips and 90° rotations."""
        if rng.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        k = rng.randint(0, 3)
        if k > 0:
            img = img.rotate(k * 90, expand=False)
        return img

    def _add_noise(self, cfa_img, rng):
        """Add simulated sensor noise to the CFA mosaic."""
        if self.noise_sigma[1] <= 0:
            return cfa_img
        
        sigma = rng.uniform(self.noise_sigma[0], self.noise_sigma[1])
        if sigma <= 0:
            return cfa_img
        
        noise = torch.randn_like(cfa_img) * sigma
        return (cfa_img + noise).clamp(0, 1)

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Use idx as seed for reproducibility within an epoch,
        # but different each epoch due to DataLoader shuffling
        rng = random.Random()
        
        img = Image.open(self.image_paths[img_idx]).convert("RGB")
        patch = self._load_and_crop(img, rng)

        if self.augment:
            patch = self._augment(patch, rng)

        # Convert to tensor: (3, H, W) in [0, 1]
        rgb = self.to_tensor(patch)

        # Simulate mosaicing
        cfa_img = mosaic(rgb, self.cfa)  # (1, H, W)
        
        # Add noise to CFA (simulates sensor read/shot noise)
        cfa_noisy = self._add_noise(cfa_img, rng)

        # Build 4-channel input
        input_tensor = torch.cat([cfa_noisy, self.masks], dim=0)  # (4, H, W)

        return input_tensor, rgb


class TortureTestDataset(Dataset):
    """Synthetic torture test patterns for evaluation."""

    def __init__(self, size: int = 256, num_patterns: int = 100):
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
            # Diagonal stripes
            freq = rng.uniform(0.05, 0.3)
            angle = rng.uniform(0, 2 * 3.14159)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            stripe = (
                torch.sin((xx * torch.cos(torch.tensor(angle))
                          + yy * torch.sin(torch.tensor(angle))) * freq) * 0.5 + 0.5
            )
            c1 = torch.tensor([rng.random(), rng.random(), rng.random()])
            c2 = torch.tensor([rng.random(), rng.random(), rng.random()])
            for c in range(3):
                rgb[c] = c1[c] * stripe + c2[c] * (1 - stripe)

        elif pattern_type == 1:
            # Fine parallel lines
            freq = rng.randint(2, 8)
            horizontal = rng.random() > 0.5
            c1 = torch.tensor([rng.random(), rng.random(), rng.random()])
            c2 = torch.tensor([rng.random(), rng.random(), rng.random()])
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
            c1 = torch.tensor([rng.random(), rng.random(), rng.random()])
            c2 = torch.tensor([rng.random(), rng.random(), rng.random()])
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            checker = (((yy // freq) + (xx // freq)) % 2 == 0).float()
            for c in range(3):
                rgb[c] = c1[c] * checker + c2[c] * (1 - checker)

        elif pattern_type == 3:
            # Concentric circles
            freq = rng.uniform(0.1, 0.4)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            dist = torch.sqrt((xx - w/2)**2 + (yy - h/2)**2)
            ring = torch.sin(dist * freq) * 0.5 + 0.5
            c1 = torch.tensor([rng.random(), rng.random(), rng.random()])
            c2 = torch.tensor([rng.random(), rng.random(), rng.random()])
            for c in range(3):
                rgb[c] = c1[c] * ring + c2[c] * (1 - ring)

        else:
            # Color gradient with noise
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32) / h,
                torch.arange(w, dtype=torch.float32) / w,
                indexing="ij",
            )
            t = (yy + xx) / 2
            c1 = torch.tensor([rng.random(), rng.random(), rng.random()])
            c2 = torch.tensor([rng.random(), rng.random(), rng.random()])
            for c in range(3):
                rgb[c] = c1[c] * (1 - t) + c2[c] * t
            rgb += torch.randn_like(rgb) * 0.05
            rgb.clamp_(0, 1)

        cfa_img = mosaic(rgb, self.cfa)
        input_tensor = torch.cat([cfa_img, self.masks], dim=0)
        return input_tensor, rgb
