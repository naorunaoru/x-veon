"""
Dataset for fine-tuning v4 on full-resolution JPEGs converted to linear space.

The v4 model was trained on 4x downsampled linear data and learned smooth outputs.
Fine-tuning on full-res JPEGs (which have camera sharpening) teaches texture preservation.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from xtrans_pattern import make_cfa_mask, make_channel_masks


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB."""
    # sRGB transfer function inverse
    mask = srgb <= 0.04045
    linear = np.where(mask, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB [0,1]."""
    linear = np.clip(linear, 0, 1)
    mask = linear <= 0.0031308
    srgb = np.where(mask, linear * 12.92, 1.055 * (linear ** (1/2.4)) - 0.055)
    return srgb


class XTransFinetuneDataset(Dataset):
    """
    Dataset that loads full-res JPEGs, converts to linear, and simulates CFA.
    
    Target: linear RGB (camera-processed, includes sharpening/texture)
    Input: simulated CFA from linear RGB + channel masks
    """

    def __init__(
        self,
        jpeg_dir: str,
        patch_size: int = 96,
        augment: bool = True,
        noise_sigma: tuple[float, float] = (0.0, 0.003),
        patches_per_image: int = 32,
        max_images: int | None = None,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.patches_per_image = patches_per_image

        # Must be divisible by 6 (CFA) and 16 (UNet pooling)
        assert patch_size % 6 == 0
        assert patch_size % 16 == 0

        # Collect JPEG files (recursive)
        from glob import glob
        self.jpeg_files = sorted(
            glob(os.path.join(jpeg_dir, "**", "*.JPG"), recursive=True) +
            glob(os.path.join(jpeg_dir, "**", "*.jpg"), recursive=True) +
            glob(os.path.join(jpeg_dir, "**", "*.jpeg"), recursive=True)
        )

        if max_images is not None:
            self.jpeg_files = self.jpeg_files[:max_images]

        if len(self.jpeg_files) == 0:
            raise ValueError(f"No JPEG files found in {jpeg_dir}")

        print(f"  Found {len(self.jpeg_files)} JPEG images")

        # Pre-compute CFA masks
        self.cfa_mask = make_cfa_mask(patch_size, patch_size)
        self.channel_masks = make_channel_masks(patch_size, patch_size)

    def __len__(self):
        return len(self.jpeg_files) * self.patches_per_image

    def _mosaic(self, rgb: torch.Tensor) -> torch.Tensor:
        """Apply X-Trans CFA mosaic to RGB tensor."""
        h, w = self.cfa_mask.shape
        cfa = torch.zeros(1, h, w, dtype=rgb.dtype)
        for ch in range(3):
            mask = (self.cfa_mask == ch)
            cfa[0][mask] = rgb[ch][mask]
        return cfa

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = random.Random()

        # Load JPEG
        img = Image.open(self.jpeg_files[img_idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1] sRGB
        
        # Convert to linear
        img = srgb_to_linear(img)  # Now linear RGB
        
        h, w, _ = img.shape

        # Random crop (align to 6-pixel grid)
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        
        if max_y < 0 or max_x < 0:
            # Image too small, pad it
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = img.shape[:2]
            max_y = h - self.patch_size
            max_x = w - self.patch_size
        
        top = (rng.randint(0, max(0, max_y)) // 6) * 6
        left = (rng.randint(0, max(0, max_x)) // 6) * 6
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        # Convert to torch (3, H, W)
        rgb = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        # Augmentation: flips only (90° would break CFA alignment)
        if self.augment:
            if rng.random() > 0.5:
                rgb = rgb.flip(2)  # horizontal flip
            if rng.random() > 0.5:
                rgb = rgb.flip(1)  # vertical flip

        # Simulate CFA mosaicing
        cfa_img = self._mosaic(rgb)  # (1, H, W)

        # Add noise (sensor noise simulation)
        if self.noise_sigma[1] > 0:
            sigma = rng.uniform(self.noise_sigma[0], self.noise_sigma[1])
            if sigma > 0:
                noise = torch.randn_like(cfa_img) * sigma
                cfa_img = cfa_img + noise

        # Build input: [CFA, R_mask, G_mask, B_mask]
        input_tensor = torch.cat([cfa_img, self.channel_masks], dim=0)  # (4, H, W)

        return input_tensor, rgb


class TortureFinetuneDataset(Dataset):
    """
    Synthetic high-frequency patterns for fine-tuning.
    Teaches the model to preserve sharp edges and textures.
    """

    def __init__(self, size: int = 96, num_patterns: int = 500):
        self.size = size
        self.num_patterns = num_patterns
        self.cfa_mask = make_cfa_mask(size, size)
        self.channel_masks = make_channel_masks(size, size)

    def __len__(self):
        return self.num_patterns

    def _mosaic(self, rgb: torch.Tensor) -> torch.Tensor:
        h, w = self.cfa_mask.shape
        cfa = torch.zeros(1, h, w, dtype=rgb.dtype)
        for ch in range(3):
            mask = (self.cfa_mask == ch)
            cfa[0][mask] = rgb[ch][mask]
        return cfa

    def __getitem__(self, idx):
        rng = random.Random(idx)
        pattern_type = idx % 8  # More pattern variety

        h = w = self.size
        rgb = torch.zeros(3, h, w)

        if pattern_type == 0:
            # Fine diagonal stripes (high frequency)
            freq = rng.uniform(0.3, 0.8)
            angle = rng.uniform(0, 2 * 3.14159)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32), indexing="ij")
            stripe = (torch.sin((xx * torch.cos(torch.tensor(angle))
                      + yy * torch.sin(torch.tensor(angle))) * freq) * 0.5 + 0.5)
            c1 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            for c in range(3):
                rgb[c] = c1[c] * stripe + c2[c] * (1 - stripe)

        elif pattern_type == 1:
            # Very fine parallel lines (1-2 pixel)
            freq = rng.randint(1, 3)
            horizontal = rng.random() > 0.5
            c1 = torch.tensor([rng.uniform(0.1, 0.6) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.1, 0.6) for _ in range(3)])
            coords = torch.arange(h if horizontal else w)
            mask = ((coords // freq) % 2 == 0).float()
            for c in range(3):
                if horizontal:
                    rgb[c] = c1[c] * mask.unsqueeze(1) + c2[c] * (1 - mask.unsqueeze(1))
                else:
                    rgb[c] = c1[c] * mask.unsqueeze(0) + c2[c] * (1 - mask.unsqueeze(0))

        elif pattern_type == 2:
            # Fine checkerboard
            freq = rng.randint(1, 4)
            c1 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            checker = (((yy // freq) + (xx // freq)) % 2 == 0).float()
            for c in range(3):
                rgb[c] = c1[c] * checker + c2[c] * (1 - checker)

        elif pattern_type == 3:
            # Concentric circles (high freq)
            freq = rng.uniform(0.3, 0.6)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32), indexing="ij")
            dist = torch.sqrt((xx - w/2)**2 + (yy - h/2)**2)
            ring = torch.sin(dist * freq) * 0.5 + 0.5
            c1 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            for c in range(3):
                rgb[c] = c1[c] * ring + c2[c] * (1 - ring)

        elif pattern_type == 4:
            # Sharp edges (step function)
            edge_x = rng.randint(h//4, 3*h//4)
            edge_y = rng.randint(w//4, 3*w//4)
            c1 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.1, 0.5) for _ in range(3)])
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            if rng.random() > 0.5:
                mask = (xx > edge_x).float()
            else:
                mask = (yy > edge_y).float()
            for c in range(3):
                rgb[c] = c1[c] * mask + c2[c] * (1 - mask)

        elif pattern_type == 5:
            # Fabric-like texture (two frequencies)
            freq1 = rng.uniform(0.2, 0.4)
            freq2 = rng.uniform(0.4, 0.8)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32), indexing="ij")
            tex = (torch.sin(xx * freq1) * torch.sin(yy * freq2) * 0.5 + 0.5)
            c1 = torch.tensor([rng.uniform(0.2, 0.6) for _ in range(3)])
            c2 = torch.tensor([rng.uniform(0.1, 0.3) for _ in range(3)])
            for c in range(3):
                rgb[c] = c1[c] * tex + c2[c] * (1 - tex)

        elif pattern_type == 6:
            # Hair-like strands
            num_strands = rng.randint(10, 30)
            base = torch.tensor([rng.uniform(0.3, 0.5) for _ in range(3)])
            for c in range(3):
                rgb[c] = base[c]
            strand_color = torch.tensor([rng.uniform(0.05, 0.15) for _ in range(3)])
            for _ in range(num_strands):
                x0 = rng.randint(0, w)
                angle = rng.uniform(-0.3, 0.3)
                width = rng.uniform(0.5, 2)
                for y in range(h):
                    x = int(x0 + y * angle)
                    if 0 <= x < w:
                        for dx in range(int(-width), int(width)+1):
                            if 0 <= x+dx < w:
                                for c in range(3):
                                    rgb[c, y, x+dx] = strand_color[c]

        else:
            # Random noise texture
            base = torch.tensor([rng.uniform(0.2, 0.5) for _ in range(3)])
            noise_amp = rng.uniform(0.05, 0.2)
            for c in range(3):
                rgb[c] = base[c] + torch.randn(h, w) * noise_amp
            rgb = torch.clamp(rgb, 0, 1)

        # Ensure in valid range
        rgb = torch.clamp(rgb, 0, 1)
        
        cfa_img = self._mosaic(rgb)
        input_tensor = torch.cat([cfa_img, self.channel_masks], dim=0)
        return input_tensor, rgb
