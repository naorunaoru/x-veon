# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
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
from torch.utils.data import Dataset, ConcatDataset, Sampler

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


def mosaic(rgb: torch.Tensor, channel_masks: torch.Tensor) -> torch.Tensor:
    """Apply CFA mosaic to RGB image using pre-computed channel masks.

    Args:
        rgb: (3, H, W) image
        channel_masks: (3, H, W) binary masks for R, G, B positions
    Returns:
        (1, H, W) mosaiced image
    """
    return (rgb * channel_masks).sum(dim=0, keepdim=True)


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
        bright_spot_prob: float = 0.0,
        bright_spot_intensity: tuple[float, float] = (1.5, 5.0),
        bright_spot_sigma: tuple[float, float] = (2.0, 20.0),
        bright_spot_count: tuple[int, int] = (1, 5),
        downscale_prob: float = 0.0,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.shot_noise = shot_noise
        self.olpf_sigma = olpf_sigma
        self.patches_per_image = patches_per_image
        self.apply_wb = apply_wb
        self.wb_aug_range = wb_aug_range
        self.bright_spot_prob = bright_spot_prob
        self.bright_spot_intensity = bright_spot_intensity
        self.bright_spot_sigma = bright_spot_sigma
        self.bright_spot_count = bright_spot_count
        self.downscale_prob = downscale_prob
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

            # Normalize so peak channel = 1, scale to clip_level * intensity.
            # Use wb only (not clip_scale) so highlight aug EV boost doesn't
            # shrink bright spots — they represent real light sources.
            color = color / (color.max() + 1e-8)
            intensity = rng.uniform(*self.bright_spot_intensity)
            amplitude = color * wb * intensity

            result = result + amplitude.view(3, 1, 1) * blob.unsqueeze(0)

        return result

    def _load_image(self, img_idx: int) -> np.ndarray:
        """Load image with per-worker cache (size 1) to avoid redundant file opens.

        Keeps the mmap object cached rather than copying the full image.
        The grouped sampler ensures only one image is active per worker,
        so at most one mmap fd is held open at a time.
        """
        if getattr(self, '_cached_idx', -1) == img_idx:
            return self._cached_img
        self._cached_img = np.load(self.data_files[img_idx], mmap_mode='r')
        self._cached_idx = img_idx
        return self._cached_img

    def _get_rng(self, idx: int) -> random.Random:
        """Return a per-call RNG, reusing a single instance per worker."""
        rng = getattr(self, '_worker_rng', None)
        if rng is None:
            self._worker_rng = rng = random.Random()
        if not self.augment:
            # Deterministic seed for validation reproducibility
            rng.seed(idx)
        return rng

    def _process_patch(self, rgb, img_idx, rng):
        """Apply WB, augmentation, mosaicing, noise to an RGB patch.

        Args:
            rgb: (3, H, W) float32 tensor — raw linear patch
            img_idx: index into self.data_files / self.wb_multipliers
            rng: random.Random instance for this sample
        Returns:
            (input_tensor, ref, clip_ch) — same as __getitem__
        """
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

        clip_scale = 1.0

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
        if self.augment and self.olpf_sigma[1] > 0:
            sigma = rng.uniform(*self.olpf_sigma)
            if sigma > 0:
                ks = max(3, int(sigma * 6) | 1)
                pad = ks // 2
                kernel = _gaussian_kernel_2d(ks, sigma, 3)
                rgb = F.conv2d(rgb.unsqueeze(0), kernel, padding=pad, groups=3).squeeze(0)

        cfa_img = mosaic(rgb, self.masks)

        # Sensor saturation: raw photosites clip at white level (1.0 in
        # normalized raw space). In WB'd space the clip level per channel
        # is wb[ch], since raw_clip=1.0 × wb[ch].
        clip_levels = wb[self.cfa.long()].unsqueeze(0) * clip_scale  # (1, H, W)

        if do_bright_spots:
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

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = self._get_rng(idx)

        img = self._load_image(img_idx)
        h, w, _ = img.shape

        # Decide whether to cut a 2x patch and area-average down
        do_downscale = (self.augment and self.downscale_prob > 0
                        and rng.random() < self.downscale_prob)
        crop_size = self.patch_size * 2 if do_downscale else self.patch_size

        # Fall back to 1x if image is too small for the 2x crop
        if crop_size > h or crop_size > w:
            crop_size = self.patch_size
            do_downscale = False

        # Random crop aligned to CFA grid
        max_y, max_x = h - crop_size, w - crop_size
        top = (rng.randint(0, max(0, max_y)) // self.period) * self.period
        left = (rng.randint(0, max(0, max_x)) // self.period) * self.period
        patch = img[top:top+crop_size, left:left+crop_size]

        # Read contiguously from mmap (sequential I/O), then HWC→CHW in RAM
        rgb = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).contiguous()

        # Area-average 2x downscale
        if do_downscale:
            rgb = rgb.view(3, crop_size // 2, 2, crop_size // 2, 2).mean(dim=(2, 4))

        return self._process_patch(rgb, img_idx, rng)


class PatchCacheDataset(LinearDataset):
    """Streaming patch cache with staging buffer and pointer-swap updates.

    Architecture:
    - Physical buffer: N active + S staging slots
    - slot_map[i]: maps logical slot i → physical slot index
    - Streaming threads write ONLY to staging area (zero memory contention
      with DataLoader workers during training)
    - swap_staging(): updates slot_map pointers in microseconds
    - Call swap_staging() between train and val when no DataLoader workers
      are active
    """

    def __init__(self, *, seed: int = 0,
                 cache_gb: float | None = None,
                 staging_fraction: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        import threading
        import queue
        import mmap as _mmap

        self._extract_size = self.patch_size * 2 if self.downscale_prob > 0 else self.patch_size
        es = self._extract_size

        # Compute slot count from memory budget or fall back to n_images * ppi
        bytes_per_slot = es * es * 3 * 4  # float32
        if cache_gb is not None:
            n_slots = int(cache_gb * 1e9 / bytes_per_slot)
        else:
            n_slots = len(self.data_files) * self.patches_per_image
        self._n_slots = n_slots

        # Staging buffer for background streaming
        n_staging = max(256, int(n_slots * staging_fraction))
        self._n_staging = n_staging
        n_physical = n_slots + n_staging

        # Allocate via anonymous shared mmap so forked DataLoader workers
        # see the same physical pages without pickling.
        def _shared_array(shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            buf = _mmap.mmap(-1, nbytes, _mmap.MAP_SHARED | _mmap.MAP_ANONYMOUS)
            return np.ndarray(shape, dtype=dtype, buffer=buf), buf

        self._patch_data, self._mmap_patches = _shared_array(
            (n_physical, es, es, 3), np.float32)
        self._patch_img_idx, self._mmap_idx = _shared_array(
            (n_physical,), np.int32)
        # Logical → physical slot indirection (shared with forked workers)
        self._slot_map, self._mmap_map = _shared_array(
            (n_slots,), np.int32)
        self._slot_map[:] = np.arange(n_slots, dtype=np.int32)

        self._replaced_count = 0
        self._stats_lock = threading.Lock()

        # Initial fill (writes to physical slots 0..n_slots-1)
        self._fill_buffer(seed)

        # Staging queues: free physical slots available for writing,
        # and completed (physical, logical) pairs ready to be swapped in.
        self._free_slots = queue.Queue()
        for i in range(n_slots, n_physical):
            self._free_slots.put(i)
        self._ready_swaps = queue.Queue()

        # Background streaming state
        self._n_stream_threads = 4
        self._stream_threads: list = []
        self._stream_stop = threading.Event()

    def _fill_buffer(self, seed: int = 0):
        """Initial fill: extract patches from all images into the buffer."""
        from concurrent.futures import ThreadPoolExecutor

        n_images = len(self.data_files)
        ppi = self.patches_per_image
        es = self._extract_size
        ps = self.patch_size
        period = self.period

        rng = random.Random(seed)
        crops = []
        for img_i in range(n_images):
            img = np.load(self.data_files[img_i], mmap_mode='r')
            h, w, _ = img.shape
            crop_size = es if (es <= h and es <= w) else ps
            max_y = max(0, h - crop_size)
            max_x = max(0, w - crop_size)
            for _ in range(ppi):
                top = (rng.randint(0, max_y) // period) * period
                left = (rng.randint(0, max_x) // period) * period
                crops.append((img_i, top, left, crop_size))

        def _extract_image(img_i):
            img = np.load(self.data_files[img_i], mmap_mode='r')
            base = img_i * ppi
            for j in range(ppi):
                _, top, left, crop_size = crops[base + j]
                slot = base + j
                patch = img[top:top + crop_size, left:left + crop_size]
                if crop_size < es:
                    self._patch_data[slot] = 0
                    self._patch_data[slot, :crop_size, :crop_size] = patch
                else:
                    self._patch_data[slot] = patch
                self._patch_img_idx[slot] = img_i

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_extract_image, range(n_images)))

    def _stream_worker(self):
        """Worker thread: load patches into staging slots."""
        import queue as _queue

        n_images = len(self.data_files)
        es = self._extract_size
        ps = self.patch_size
        period = self.period
        ppi = self.patches_per_image
        n_slots = self._n_slots
        rng = random.Random()

        img = None
        local_count = 0
        crop_size = max_y = max_x = 0

        while not self._stream_stop.is_set():
            # Get a free staging slot (blocks until one is available)
            try:
                physical = self._free_slots.get(timeout=1.0)
            except _queue.Empty:
                continue

            if self._stream_stop.is_set():
                self._free_slots.put(physical)
                return

            # Load a new image periodically
            if img is None or local_count % ppi == 0:
                img_i = rng.randint(0, n_images - 1)
                try:
                    img = np.load(self.data_files[img_i], mmap_mode='r')
                except Exception:
                    self._free_slots.put(physical)
                    img = None
                    continue
                img_h, img_w, _ = img.shape
                crop_size = es if (es <= img_h and es <= img_w) else ps
                max_y = max(0, img_h - crop_size)
                max_x = max(0, img_w - crop_size)

            top = (rng.randint(0, max_y) // period) * period
            left = (rng.randint(0, max_x) // period) * period

            patch = img[top:top + crop_size, left:left + crop_size]
            if crop_size < es:
                self._patch_data[physical] = 0
                self._patch_data[physical, :crop_size, :crop_size] = patch
            else:
                self._patch_data[physical] = patch
            self._patch_img_idx[physical] = img_i

            # Queue for swap into a random active slot
            logical = rng.randint(0, n_slots - 1)
            self._ready_swaps.put((physical, logical))
            local_count += 1

    def start_streaming(self):
        """Start background streaming threads."""
        if self._stream_threads:
            return
        import threading
        self._stream_stop.clear()
        for _ in range(self._n_stream_threads):
            t = threading.Thread(target=self._stream_worker, daemon=True)
            self._stream_threads.append(t)
        for t in self._stream_threads:
            t.start()

    def stop_streaming(self):
        """Stop background streaming threads."""
        self._stream_stop.set()
        for t in self._stream_threads:
            t.join(timeout=5)
        self._stream_threads.clear()

    def swap_staging(self) -> int:
        """Swap staged patches into the active cache by updating slot_map.

        Call when no DataLoader workers are reading (between train and val).
        Returns number of patches swapped in.
        """
        import queue as _queue
        swaps = []
        while True:
            try:
                swaps.append(self._ready_swaps.get_nowait())
            except _queue.Empty:
                break

        for physical, logical in swaps:
            old_physical = int(self._slot_map[logical])
            self._slot_map[logical] = physical
            self._free_slots.put(old_physical)

        with self._stats_lock:
            self._replaced_count += len(swaps)
        return len(swaps)

    def cleanup(self):
        """Release shared memory mappings."""
        self.stop_streaming()
        self._patch_data = None
        self._patch_img_idx = None
        self._slot_map = None
        for buf in (self._mmap_patches, self._mmap_idx, self._mmap_map):
            buf.close()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def reset_stats(self) -> int:
        """Reset epoch stats, returning patches_replaced since last reset."""
        with self._stats_lock:
            replaced = self._replaced_count
            self._replaced_count = 0
            return replaced

    def __len__(self):
        return self._n_slots

    def __getitem__(self, idx):
        rng = self._get_rng(idx)
        es = self._extract_size
        ps = self.patch_size

        physical = int(self._slot_map[idx])
        img_idx = int(self._patch_img_idx[physical])

        # Single-copy HWC→CHW: transpose view + ascontiguousarray does the
        # layout conversion in one memcpy.  Safe without defensive copy because
        # staging swap guarantees active slots are never written during training.
        if es > ps:
            do_downscale = self.augment and rng.random() < self.downscale_prob
            if do_downscale:
                rgb = torch.from_numpy(np.ascontiguousarray(
                    self._patch_data[physical].transpose(2, 0, 1)))
                rgb = rgb.view(3, ps, 2, ps, 2).mean(dim=(2, 4))
            else:
                max_off = es - ps
                top = (rng.randint(0, max_off) // self.period) * self.period
                left = (rng.randint(0, max_off) // self.period) * self.period
                rgb = torch.from_numpy(np.ascontiguousarray(
                    self._patch_data[physical, top:top+ps, left:left+ps].transpose(2, 0, 1)))
        else:
            rgb = torch.from_numpy(np.ascontiguousarray(
                self._patch_data[physical].transpose(2, 0, 1)))

        return self._process_patch(rgb, img_idx, rng)


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
    bright_spot_prob: float = 0.0,
    bright_spot_intensity: tuple[float, float] = (1.5, 5.0),
    bright_spot_sigma: tuple[float, float] = (2.0, 20.0),
    bright_spot_count: tuple[int, int] = (1, 5),
    downscale_prob: float = 0.0,
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
        bright_spot_prob=bright_spot_prob,
        bright_spot_intensity=bright_spot_intensity,
        bright_spot_sigma=bright_spot_sigma,
        bright_spot_count=bright_spot_count,
        downscale_prob=downscale_prob,
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


class ImageGroupedSampler(Sampler):
    """Yields indices grouped by source image for cache-friendly data loading.

    Instead of fully shuffling all patch indices (which scatters patches from
    the same image across workers), this shuffles at the image level and emits
    all patches for each image consecutively.  Combined with a per-worker
    image cache, this reduces file opens from N*patches_per_image to N.
    """

    def __init__(self, num_images: int, patches_per_image: int, shuffle: bool = True):
        self.num_images = num_images
        self.patches_per_image = patches_per_image
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        image_order = list(range(self.num_images))
        if self.shuffle:
            g = random.Random(self.epoch)
            g.shuffle(image_order)
        for img_idx in image_order:
            base = img_idx * self.patches_per_image
            patches = list(range(base, base + self.patches_per_image))
            if self.shuffle:
                random.shuffle(patches)
            yield from patches

    def __len__(self):
        return self.num_images * self.patches_per_image

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# Backwards compatibility
XTransLinearDataset = LinearDataset
