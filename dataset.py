"""
Dataset for X-Trans demosaicing training.

Supports:
- Linear .npy files (from build_dataset_v4.py)
- Direct JPEG loading with sRGB→linear conversion
- Optional mixing of synthetic torture patterns
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from xtrans_pattern import make_cfa_mask, make_channel_masks


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
    """

    def __init__(
        self,
        data_dir: str | None = None,
        patch_size: int = 96,
        augment: bool = True,
        noise_sigma: tuple[float, float] = (0.0, 0.005),
        patches_per_image: int = 16,
        max_images: int | None = None,
        filter_file: str | None = None,
        apply_wb: bool = False,
        files: list[str] | None = None,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image
        self.apply_wb = apply_wb
        self.data_dir = data_dir

        assert patch_size % 6 == 0, "patch_size must be divisible by 6 (CFA)"
        assert patch_size % 16 == 0, "patch_size must be divisible by 16 (UNet)"

        if files is not None:
            self.data_files = list(files)
        else:
            if data_dir is None:
                raise ValueError("Either data_dir or files must be provided")
            # Find .npy files (exclude _lum.npy and _meta.npy)
            self.data_files = sorted([
                os.path.join(data_dir, f) for f in os.listdir(data_dir)
                if f.endswith('.npy') and not f.endswith('_meta.npy') and not f.endswith('_lum.npy')
            ])

            # Optional filtering
            if filter_file is not None:
                with open(filter_file) as f:
                    allowed = set(json.load(f))
                self.data_files = [
                    p for p in self.data_files
                    if os.path.splitext(os.path.basename(p))[0] in allowed
                ]

            if max_images:
                self.data_files = self.data_files[:max_images]

        if not self.data_files:
            raise ValueError(f"No .npy files found")

        # Load per-image WB multipliers from metadata
        self.wb_multipliers = None
        if apply_wb:
            self.wb_multipliers = []
            n_missing = 0
            for npy_path in self.data_files:
                stem = os.path.splitext(npy_path)[0]
                meta_path = stem + "_meta.json"
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    wb = np.array(meta["camera_wb"][:3], dtype=np.float32)
                    wb = wb / wb[1]  # Normalize to G=1
                    self.wb_multipliers.append(wb)
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    self.wb_multipliers.append(np.array([1.0, 1.0, 1.0], dtype=np.float32))
                    n_missing += 1
            if n_missing:
                print(f"  WB: {n_missing}/{len(self.data_files)} images missing metadata, using identity WB")

        self.cfa = make_cfa_mask(patch_size, patch_size)
        self.masks = make_channel_masks(patch_size, patch_size)

    @staticmethod
    def find_files(
        data_dir: str,
        max_images: int | None = None,
        filter_file: str | None = None,
    ) -> list[str]:
        """Scan data_dir for .npy files, with optional filtering."""
        files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npy') and not f.endswith('_meta.npy') and not f.endswith('_lum.npy')
        ])
        if filter_file is not None:
            with open(filter_file) as jf:
                allowed = set(json.load(jf))
            files = [
                p for p in files
                if os.path.splitext(os.path.basename(p))[0] in allowed
            ]
        if max_images:
            files = files[:max_images]
        return files

    def __len__(self):
        return len(self.data_files) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = random.Random()

        img = np.load(self.data_files[img_idx])
        h, w, _ = img.shape

        # Random crop aligned to 6px grid
        max_y, max_x = h - self.patch_size, w - self.patch_size
        top = (rng.randint(0, max(0, max_y)) // 6) * 6
        left = (rng.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        # Apply white balance before mosaicing (model learns WB'd data)
        if self.wb_multipliers is not None:
            wb = torch.from_numpy(self.wb_multipliers[img_idx]).float()
            rgb = rgb * wb.view(3, 1, 1)

        # Augmentation (only flips - rotations break CFA)
        if self.augment:
            if rng.random() > 0.5:
                rgb = rgb.flip(2)
            if rng.random() > 0.5:
                rgb = rgb.flip(1)

        cfa_img = mosaic(rgb, self.cfa)

        # Add noise
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

    def __init__(self, patch_size: int = 96, num_patterns: int = 1000):
        from torture_v2 import TortureDatasetV2
        self._inner = TortureDatasetV2(size=patch_size, num_patterns=num_patterns)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


def create_mixed_dataset(
    data_dir: str | None = None,
    patch_size: int = 96,
    torture_fraction: float = 0.05,
    torture_patterns: int = 500,
    augment: bool = True,
    noise_sigma: tuple[float, float] = (0.0, 0.005),
    patches_per_image: int = 16,
    max_images: int | None = None,
    apply_wb: bool = False,
    files: list[str] | None = None,
) -> Dataset:
    """
    Create a dataset mixing real images with synthetic torture patterns.
    """
    main_dataset = LinearDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        augment=augment,
        noise_sigma=noise_sigma,
        patches_per_image=patches_per_image,
        max_images=max_images,
        apply_wb=apply_wb,
        files=files,
    )

    if torture_fraction <= 0:
        return main_dataset

    # Calculate torture dataset size to achieve desired fraction
    main_size = len(main_dataset)
    torture_size = int(main_size * torture_fraction / (1 - torture_fraction))
    torture_size = max(1, min(torture_size, torture_patterns * 10))  # Cap at 10x patterns

    torture_dataset = TortureDataset(patch_size, torture_patterns)

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
