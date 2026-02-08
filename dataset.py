"""
Dataset for X-Trans demosaicing training.

Supports:
- Linear .npy files (from build_dataset_v4.py)
- Direct JPEG loading with sRGB→linear conversion
- Optional mixing of synthetic torture patterns
"""

import os
import random
from pathlib import Path
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from xtrans_pattern import make_cfa_mask, make_channel_masks


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def mosaic(rgb: torch.Tensor, cfa_mask: torch.Tensor) -> torch.Tensor:
    """Apply X-Trans CFA mosaic to RGB image. Returns (1, H, W)."""
    h, w = cfa_mask.shape
    cfa = torch.zeros(1, h, w, dtype=rgb.dtype)
    for ch in range(3):
        mask = cfa_mask == ch
        cfa[0][mask] = rgb[ch][mask]
    return cfa


class LinearDataset(Dataset):
    """
    Load pre-computed linear .npy files.
    Fast loading, used for main training.
    Optionally loads luminance reference channel (*_lum.npy).
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
        load_luminance: bool = False,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image
        self.load_luminance = load_luminance
        self.data_dir = data_dir

        assert patch_size % 6 == 0, "patch_size must be divisible by 6 (CFA)"
        assert patch_size % 16 == 0, "patch_size must be divisible by 16 (UNet)"

        # Find .npy files (exclude _lum.npy and _meta.npy)
        self.data_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npy') and not f.endswith('_meta.npy') and not f.endswith('_lum.npy')
        ])

        # Optional filtering
        if filter_file is not None:
            import json
            with open(filter_file) as f:
                allowed = set(json.load(f))
            self.data_files = [
                p for p in self.data_files
                if os.path.splitext(os.path.basename(p))[0] in allowed
            ]

        if max_images:
            self.data_files = self.data_files[:max_images]

        if not self.data_files:
            raise ValueError(f"No .npy files in {data_dir}")

        self.cfa = make_cfa_mask(patch_size, patch_size)
        self.masks = make_channel_masks(patch_size, patch_size)

    def __len__(self):
        return len(self.data_files) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = random.Random()

        img = np.load(self.data_files[img_idx])
        h, w, _ = img.shape

        # Load luminance reference if enabled
        lum = None
        if self.load_luminance:
            lum_path = self.data_files[img_idx].replace('.npy', '_lum.npy')
            if os.path.exists(lum_path):
                lum = np.load(lum_path)

        # Random crop aligned to 6px grid
        max_y, max_x = h - self.patch_size, w - self.patch_size
        top = (rng.randint(0, max(0, max_y)) // 6) * 6
        left = (rng.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        # Crop luminance to match
        lum_patch = None
        if lum is not None:
            lum_patch = lum[top:top+self.patch_size, left:left+self.patch_size]
            lum_patch = torch.from_numpy(lum_patch.copy()).float().unsqueeze(0)

        # Augmentation (only flips - rotations break CFA)
        if self.augment:
            flip_h = rng.random() > 0.5
            flip_v = rng.random() > 0.5
            if flip_h:
                rgb = rgb.flip(2)
                if lum_patch is not None:
                    lum_patch = lum_patch.flip(2)
            if flip_v:
                rgb = rgb.flip(1)
                if lum_patch is not None:
                    lum_patch = lum_patch.flip(1)

        cfa_img = mosaic(rgb, self.cfa)

        # Add noise
        if self.noise_sigma[1] > 0:
            sigma = rng.uniform(*self.noise_sigma)
            if sigma > 0:
                cfa_img = cfa_img + torch.randn_like(cfa_img) * sigma

        input_tensor = torch.cat([cfa_img, self.masks], dim=0)
        
        if self.load_luminance:
            return input_tensor, rgb, lum_patch
        return input_tensor, rgb


class JPEGDataset(Dataset):
    """
    Load JPEGs directly with on-the-fly sRGB→linear conversion.
    Slower but doesn't require pre-processing.
    """

    def __init__(
        self,
        jpeg_dir: str,
        patch_size: int = 96,
        augment: bool = True,
        noise_sigma: tuple[float, float] = (0.0, 0.003),
        patches_per_image: int = 16,
        max_images: int | None = None,
    ):
        from PIL import Image
        self.Image = Image
        
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image

        assert patch_size % 6 == 0 and patch_size % 16 == 0

        self.jpeg_files = sorted(
            glob(os.path.join(jpeg_dir, "**", "*.JPG"), recursive=True) +
            glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True)
        )

        if max_images:
            self.jpeg_files = self.jpeg_files[:max_images]

        if not self.jpeg_files:
            raise ValueError(f"No JPEGs in {jpeg_dir}")

        self.cfa = make_cfa_mask(patch_size, patch_size)
        self.masks = make_channel_masks(patch_size, patch_size)

    def __len__(self):
        return len(self.jpeg_files) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = random.Random()

        img = self.Image.open(self.jpeg_files[img_idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        img = srgb_to_linear(img)
        h, w, _ = img.shape

        max_y, max_x = h - self.patch_size, w - self.patch_size
        top = (rng.randint(0, max(0, max_y)) // 6) * 6
        left = (rng.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        if self.augment:
            if rng.random() > 0.5:
                rgb = rgb.flip(2)
            if rng.random() > 0.5:
                rgb = rgb.flip(1)

        cfa_img = mosaic(rgb, self.cfa)

        if self.noise_sigma[1] > 0:
            sigma = rng.uniform(*self.noise_sigma)
            if sigma > 0:
                cfa_img = cfa_img + torch.randn_like(cfa_img) * sigma

        input_tensor = torch.cat([cfa_img, self.masks], dim=0)
        return input_tensor, rgb


class TortureDataset(Dataset):
    """
    Synthetic torture test patterns.
    Import from torture_v2 for the actual pattern generation.
    """

    def __init__(self, patch_size: int = 96, num_patterns: int = 1000, return_luminance: bool = False):
        from torture_v2 import TortureDatasetV2
        self._inner = TortureDatasetV2(size=patch_size, num_patterns=num_patterns)
        self.return_luminance = return_luminance

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        result = self._inner[idx]
        if self.return_luminance:
            # Return zeros for luminance - torture patterns don't have luminance reference
            # Shape: (1, H, W) to match real luminance patches
            h, w = result[1].shape[1], result[1].shape[2]
            dummy_lum = torch.zeros(1, h, w)
            return result[0], result[1], dummy_lum
        return result


def create_mixed_dataset(
    data_dir: str,
    patch_size: int = 96,
    torture_fraction: float = 0.05,
    torture_patterns: int = 500,
    augment: bool = True,
    noise_sigma: tuple[float, float] = (0.0, 0.005),
    patches_per_image: int = 16,
    max_images: int | None = None,
    use_jpeg: bool = False,
    load_luminance: bool = False,
) -> Dataset:
    """
    Create a dataset mixing real images with synthetic torture patterns.
    
    Args:
        data_dir: Path to .npy files or JPEG directory
        patch_size: Patch size (must be divisible by 6 and 16)
        torture_fraction: Fraction of training data from torture patterns (0-1)
        torture_patterns: Number of unique torture patterns
        augment: Enable augmentation
        noise_sigma: Noise sigma range
        patches_per_image: Patches extracted per image per epoch
        max_images: Limit number of images
        use_jpeg: Use JPEG loading instead of .npy
        load_luminance: Load luminance reference files
    
    Returns:
        Combined dataset with weighted sampling
    """
    # Create main dataset
    if use_jpeg:
        main_dataset = JPEGDataset(
            data_dir, patch_size, augment, noise_sigma, patches_per_image, max_images
        )
    else:
        main_dataset = LinearDataset(
            data_dir, patch_size, augment, noise_sigma, patches_per_image, max_images,
            load_luminance=load_luminance
        )

    if torture_fraction <= 0:
        return main_dataset

    # Calculate torture dataset size to achieve desired fraction
    main_size = len(main_dataset)
    torture_size = int(main_size * torture_fraction / (1 - torture_fraction))
    torture_size = max(1, min(torture_size, torture_patterns * 10))  # Cap at 10x patterns

    torture_dataset = TortureDataset(patch_size, torture_patterns, return_luminance=load_luminance)

    # Repeat torture dataset to match size
    class RepeatedDataset(Dataset):
        def __init__(self, dataset, target_size):
            self.dataset = dataset
            self.target_size = target_size

        def __len__(self):
            return self.target_size

        def __getitem__(self, idx):
            return self.dataset[idx % len(self.dataset)]

    repeated_torture = RepeatedDataset(torture_dataset, torture_size)

    print(f"  Main dataset: {main_size} samples")
    print(f"  Torture dataset: {torture_size} samples ({torture_fraction*100:.1f}% of total)")

    return ConcatDataset([main_dataset, repeated_torture])


# Backwards compatibility
XTransLinearDataset = LinearDataset
