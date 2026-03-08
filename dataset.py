"""
Dataset for X-Trans demosaicing training.

Supports:
- Linear .npy files (from build_dataset_v4.py)
- Direct JPEG loading with sRGB→linear conversion
- Optional mixing of synthetic torture patterns
"""

import colorsys
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

from cfa import make_cfa_mask, make_channel_masks, CFA_REGISTRY, cfa_period, patch_alignment
from losses import _gaussian_kernel_2d


# Bright spot color palette: (h_center, h_range, s_min, s_max, weight)
_SPOT_PALETTE = [
    (0.08, 0.04, 0.10, 0.25, 2),  # warm white (tungsten/sodium)
    (0.55, 0.05, 0.08, 0.20, 2),  # cool white (LED/fluorescent)
    (0.00, 0.03, 0.85, 1.00, 1),  # red (brake lights)
    (0.11, 0.02, 0.80, 1.00, 1),  # amber (turn signals, sodium vapor)
    (0.63, 0.04, 0.75, 1.00, 1),  # blue (LEDs, neon)
    (0.50, 0.03, 0.75, 1.00, 1),  # cyan/green (neon)
    (0.85, 0.05, 0.75, 1.00, 1),  # magenta (neon pink)
]
_SPOT_WEIGHTS = [e[4] for e in _SPOT_PALETTE]


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
        shot_noise: tuple[float, float] = (0.0, 0.0),
        patches_per_image: int = 16,
        max_images: int | None = None,
        filter_file: str | None = None,
        apply_wb: bool = False,
        wb_aug_range: float = 0.0,
        files: list[str] | None = None,
        cfa_type: str = "xtrans",
        olpf_sigma: tuple[float, float] = (0.0, 0.0),
        highlight_aug_prob: float = 0.0,
        highlight_aug_ev: float = 0.0,
        bright_spot_prob: float = 0.0,
        bright_spot_intensity: tuple[float, float] = (1.5, 5.0),
        bright_spot_sigma: tuple[float, float] = (2.0, 20.0),
        bright_spot_count: tuple[int, int] = (1, 5),
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.shot_noise = shot_noise
        self.olpf_sigma = olpf_sigma
        self.patches_per_image = patches_per_image
        self.apply_wb = apply_wb
        self.wb_aug_range = wb_aug_range
        self.highlight_aug_prob = highlight_aug_prob
        self.highlight_aug_ev = highlight_aug_ev
        self.bright_spot_prob = bright_spot_prob
        self.bright_spot_intensity = bright_spot_intensity
        self.bright_spot_sigma = bright_spot_sigma
        self.bright_spot_count = bright_spot_count
        self.data_dir = data_dir

        self.pattern = CFA_REGISTRY[cfa_type]
        self.period = cfa_period(self.pattern)
        alignment = patch_alignment(self.pattern)
        assert patch_size % alignment == 0, (
            f"patch_size must be divisible by {alignment} "
            f"(lcm of CFA period {self.period} and UNet factor 16)"
        )

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

        self.cfa = make_cfa_mask(patch_size, patch_size, self.pattern)
        self.masks = make_channel_masks(patch_size, patch_size, self.pattern)

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

    def _add_bright_spots(
        self,
        rgb: torch.Tensor,
        wb: torch.Tensor,
        clip_scale: float,
        rng: random.Random,
    ) -> torch.Tensor:
        """Add synthetic bright spots simulating point light sources."""
        _, H, W = rgb.shape
        n_spots = rng.randint(*self.bright_spot_count)

        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        result = rgb
        for _ in range(n_spots):
            # Position (allow slightly off-patch for edge feathering)
            cx = rng.uniform(-0.1 * W, 1.1 * W)
            cy = rng.uniform(-0.1 * H, 1.1 * H)

            # Anisotropic gaussian: independent sigma per axis + rotation
            su = rng.uniform(*self.bright_spot_sigma)
            sv = rng.uniform(*self.bright_spot_sigma)
            theta = rng.uniform(0, math.pi)
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            dx = (xx - cx) * cos_t + (yy - cy) * sin_t
            dy = -(xx - cx) * sin_t + (yy - cy) * cos_t
            blob = torch.exp(-0.5 * ((dx / su) ** 2 + (dy / sv) ** 2))

            # Sample color from palette
            entry = rng.choices(_SPOT_PALETTE, weights=_SPOT_WEIGHTS, k=1)[0]
            h_c, h_r, s_lo, s_hi, _ = entry
            h = (h_c + rng.uniform(-h_r, h_r)) % 1.0
            s = rng.uniform(s_lo, s_hi)
            r, g, b = colorsys.hsv_to_rgb(h, s, 1.0)
            color = torch.tensor([r ** 2.2, g ** 2.2, b ** 2.2])

            # Normalize so peak channel = 1, scale to clip_level * intensity
            color = color / (color.max() + 1e-8)
            intensity = rng.uniform(*self.bright_spot_intensity)
            amplitude = color * wb * clip_scale * intensity

            result = result + amplitude.view(3, 1, 1) * blob.unsqueeze(0)

        return result

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        # Seed RNG deterministically for validation (augment=False) so that
        # the same crops and highlight EV boosts are used every epoch,
        # giving comparable metrics.  Training uses unseeded RNG for variety.
        rng = random.Random(idx) if not self.augment else random.Random()

        img = np.load(self.data_files[img_idx], mmap_mode='r')
        h, w, _ = img.shape

        # Random crop aligned to CFA grid
        max_y, max_x = h - self.patch_size, w - self.patch_size
        top = (rng.randint(0, max(0, max_y)) // self.period) * self.period
        left = (rng.randint(0, max(0, max_x)) // self.period) * self.period
        patch = img[top:top+self.patch_size, left:left+self.patch_size]

        rgb = torch.from_numpy(np.array(patch.transpose(2, 0, 1))).float()

        # Apply white balance before mosaicing (model learns WB'd data)
        wb = torch.ones(3)
        if self.wb_multipliers is not None:
            wb = torch.from_numpy(self.wb_multipliers[img_idx]).float()
            # WB shift augmentation: perturb R and B gains in log space
            if self.augment and self.wb_aug_range > 0:
                r_shift = math.exp(rng.uniform(-self.wb_aug_range, self.wb_aug_range))
                b_shift = math.exp(rng.uniform(-self.wb_aug_range, self.wb_aug_range))
                wb = wb * torch.tensor([r_shift, 1.0, b_shift])
            rgb = rgb * wb.view(3, 1, 1)

        # Highlight augmentation: boost exposure by +ev EV, then clip the
        # CFA at the original (pre-boost) ceiling.  The model must recover
        # the ev stops of headroom that were clipped away.
        # Decoupled from self.augment so val_hl can use augment=False.
        do_highlight = (self.highlight_aug_prob > 0
                        and rng.random() < self.highlight_aug_prob)
        clip_scale = 1.0
        if do_highlight and self.highlight_aug_ev > 0:
            ev = rng.uniform(0, self.highlight_aug_ev)
            rgb = rgb * (2.0 ** ev)
            # Clip at pre-boost level: undo the exposure gain for the ceiling
            clip_scale = 2.0 ** (-ev)

        # Bright spot augmentation: add synthetic point light sources
        do_bright_spots = (self.bright_spot_prob > 0
                           and rng.random() < self.bright_spot_prob)
        if do_bright_spots:
            rgb = self._add_bright_spots(rgb, wb, clip_scale, rng)

        # Geometric augmentation: flips + 90° rotations (applied before
        # mosaicing, so CFA is applied fresh to the transformed image)
        if self.augment:
            if rng.random() > 0.5:
                rgb = rgb.flip(2)
            if rng.random() > 0.5:
                rgb = rgb.flip(1)
            k = rng.randint(0, 3)
            if k > 0:
                rgb = torch.rot90(rgb, k, [1, 2])

        # OLPF simulation: blur RGB before mosaicing (optical domain)
        # Clip ref at per-channel ceiling: model shouldn't be penalized for
        # not recovering values above sensor saturation (unrecoverable).
        ref = rgb
        if do_bright_spots:
            ref = ref.clamp(max=(wb * clip_scale).view(3, 1, 1))
        if self.augment and self.olpf_sigma[1] > 0:
            sigma = rng.uniform(*self.olpf_sigma)
            if sigma > 0:
                ks = max(3, int(sigma * 6) | 1)
                pad = ks // 2
                kernel = _gaussian_kernel_2d(ks, sigma, 3)
                rgb = F.conv2d(rgb.unsqueeze(0), kernel, padding=pad, groups=3).squeeze(0)

        cfa_img = mosaic(rgb, self.cfa)

        # Sensor saturation: raw photosites clip at white level (1.0 in
        # normalized raw space). In WB'd space the clip level per channel
        # is wb[ch], since raw_clip=1.0 × wb[ch].
        # clip_scale < 1 from highlight augmentation lowers the ceiling.
        clip_levels = wb[self.cfa.long()].unsqueeze(0) * clip_scale  # (1, H, W)

        if do_highlight or do_bright_spots:
            cfa_img = cfa_img.clamp(max=clip_levels)

        # Clip proximity: 0 below 50% of clip level, ramps 0→1 from 50% to 100%.
        # Only encodes proximity to clipping, not scene luminance.
        raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
        clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)  # (1, H, W)

        # Poisson-Gaussian noise: noise_std(x) = sqrt(shot * x + read^2)
        read_sigma = rng.uniform(*self.noise_sigma)
        shot_coeff = rng.uniform(*self.shot_noise)
        if read_sigma > 0 or shot_coeff > 0:
            noise_var = shot_coeff * cfa_img.clamp(min=0) + read_sigma ** 2
            cfa_img = cfa_img + torch.randn_like(cfa_img) * noise_var.sqrt()

        input_tensor = torch.cat([cfa_img, self.masks, clip_ratio], dim=0)  # (5, H, W)
        clip_ch = wb * clip_scale  # (3,) per-channel clip levels for loss
        return input_tensor, ref, clip_ch


class TortureDataset(Dataset):
    """
    Synthetic torture test patterns.
    Import from torture_v2 for the actual pattern generation.
    """

    def __init__(self, patch_size: int = 96, num_patterns: int = 1000, cfa_type: str = "xtrans"):
        from torture_v2 import TortureDatasetV2
        self._inner = TortureDatasetV2(size=patch_size, num_patterns=num_patterns, cfa_type=cfa_type)

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
    shot_noise: tuple[float, float] = (0.0, 0.0),
    patches_per_image: int = 16,
    max_images: int | None = None,
    apply_wb: bool = False,
    wb_aug_range: float = 0.0,
    files: list[str] | None = None,
    cfa_type: str = "xtrans",
    olpf_sigma: tuple[float, float] = (0.0, 0.0),
    highlight_aug_prob: float = 0.0,
    highlight_aug_ev: float = 0.0,
    bright_spot_prob: float = 0.0,
    bright_spot_intensity: tuple[float, float] = (1.5, 5.0),
    bright_spot_sigma: tuple[float, float] = (2.0, 20.0),
    bright_spot_count: tuple[int, int] = (1, 5),
) -> Dataset:
    """
    Create a dataset mixing real images with synthetic torture patterns.
    """
    main_dataset = LinearDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        augment=augment,
        noise_sigma=noise_sigma,
        shot_noise=shot_noise,
        patches_per_image=patches_per_image,
        max_images=max_images,
        apply_wb=apply_wb,
        wb_aug_range=wb_aug_range,
        files=files,
        cfa_type=cfa_type,
        olpf_sigma=olpf_sigma,
        highlight_aug_prob=highlight_aug_prob,
        highlight_aug_ev=highlight_aug_ev,
        bright_spot_prob=bright_spot_prob,
        bright_spot_intensity=bright_spot_intensity,
        bright_spot_sigma=bright_spot_sigma,
        bright_spot_count=bright_spot_count,
    )

    if torture_fraction <= 0:
        return main_dataset

    # Calculate torture dataset size to achieve desired fraction
    main_size = len(main_dataset)
    torture_size = int(main_size * torture_fraction / (1 - torture_fraction))
    torture_size = max(1, min(torture_size, torture_patterns * 10))  # Cap at 10x patterns

    torture_dataset = TortureDataset(patch_size, torture_patterns, cfa_type=cfa_type)

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
