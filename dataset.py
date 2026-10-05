# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Dataset for X-Trans demosaicing training.

Supports:
- Linear .npy files (from build_dataset_v4.py)
- Direct JPEG loading with sRGB→linear conversion
"""

import collections
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from cfa import make_cfa_mask, CFA_REGISTRY, cfa_period, patch_alignment
from losses import _gaussian_kernel_2d


def mosaic(rgb: torch.Tensor, cfa: torch.Tensor) -> torch.Tensor:
    """Apply CFA mosaic to RGB image.

    Args:
        rgb: (3, H, W) image
        cfa: (H, W) long tensor with values 0 (R), 1 (G), 2 (B)
    Returns:
        (1, H, W) mosaiced image
    """
    return torch.gather(rgb, 0, cfa.unsqueeze(0).expand(1, -1, -1))


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
        files: list[str] | None = None,
        cfa_type: str = "xtrans",
        olpf_sigma: tuple[float, float] = (0.0, 0.0),
        downscale_prob: float = 0.0,
        group_images: int = 32,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.shot_noise = shot_noise
        self.olpf_sigma = olpf_sigma
        self.patches_per_image = patches_per_image
        self.downscale_prob = downscale_prob
        self.group_images = group_images
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

        self.cfa = make_cfa_mask(patch_size, patch_size, self.pattern)

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

    def _load_image(self, img_idx: int) -> np.ndarray:
        """Open an image as a memory map, keeping the maps of the last `group_images` images.

        The grouped sampler draws from `group_images` images at a time, so a worker
        revisits the same few files until the group is used up.
        """
        cache = getattr(self, '_open_images', None)
        if cache is None:
            cache = self._open_images = collections.OrderedDict()
        img = cache.get(img_idx)
        if img is None:
            img = np.load(self.data_files[img_idx], mmap_mode='r')
            cache[img_idx] = img
            while len(cache) > self.group_images:
                cache.popitem(last=False)
        else:
            cache.move_to_end(img_idx)
        return img

    def _get_rng(self, idx: int) -> random.Random:
        """Return a per-call RNG, reusing a single instance per worker."""
        rng = getattr(self, '_worker_rng', None)
        if rng is None:
            self._worker_rng = rng = random.Random()
        if not self.augment:
            # Deterministic seed for validation reproducibility
            rng.seed(idx)
        return rng

    def _process_patch(self, rgb, rng):
        """Augment, mosaic and add noise to an RGB patch.

        Args:
            rgb: (3, H, W) float32 tensor, raw linear, 1.0 = the sensor's clip level
            rng: random.Random instance for this sample
        Returns:
            (mosaic, target): (1, H, W) and (3, H, W). Neither exceeds 1.0.
        """
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

        # Nothing above saturation reaches training.
        ref = rgb.clamp(max=1.0)

        # OLPF simulation: blur RGB before mosaicing (optical domain); the target stays unblurred
        if self.augment and self.olpf_sigma[1] > 0:
            sigma = rng.uniform(*self.olpf_sigma)
            if sigma > 0:
                ks = max(3, int(sigma * 6) | 1)
                pad = ks // 2
                kernel = _gaussian_kernel_2d(ks, sigma, 3)
                rgb = F.conv2d(rgb.unsqueeze(0), kernel, padding=pad, groups=3).squeeze(0)

        cfa_img = mosaic(rgb, self.cfa)

        # Poisson-Gaussian noise: noise_std(x) = sqrt(shot * x + read^2)
        read_sigma = rng.uniform(*self.noise_sigma)
        shot_coeff = rng.uniform(*self.shot_noise)
        if read_sigma > 0 or shot_coeff > 0:
            noise_var = shot_coeff * cfa_img.clamp(min=0) + read_sigma ** 2
            cfa_img = cfa_img + torch.randn_like(cfa_img) * noise_var.sqrt()

        # A photosite saturates after the noise: clipped areas are flat, as on a sensor.
        cfa_img = cfa_img.clamp(max=1.0)

        return cfa_img, ref  # (1, H, W), (3, H, W)

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        rng = self._get_rng(idx)

        img = self._load_image(img_idx)
        h, w, _ = img.shape
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"{self.data_files[img_idx]} is {w}x{h}, smaller than the {self.patch_size} px patch")

        # Decide whether to cut a 2x patch and area-average down
        do_downscale = (self.augment and self.downscale_prob > 0
                        and rng.random() < self.downscale_prob)
        crop_size = self.patch_size * 2 if do_downscale else self.patch_size

        # Fall back to 1x if image is too small for the 2x crop
        if crop_size > h or crop_size > w:
            crop_size = self.patch_size
            do_downscale = False

        # Random crop at any offset: the target has no CFA phase, the mosaic is applied afterwards
        max_y, max_x = h - crop_size, w - crop_size
        top = rng.randint(0, max(0, max_y))
        left = rng.randint(0, max(0, max_x))
        patch = img[top:top+crop_size, left:left+crop_size]

        # Read contiguously from mmap (sequential I/O), uint16→float32, HWC→CHW
        rgb = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32) / 65535.0).permute(2, 0, 1).contiguous()

        # Area-average 2x downscale
        if do_downscale:
            rgb = rgb.view(3, crop_size // 2, 2, crop_size // 2, 2).mean(dim=(2, 4))

        return self._process_patch(rgb, rng)


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
        """Initial fill: extract patches into exactly the n_slots active slots.

        Slots are spread evenly over the images (contiguous per image, for I/O locality),
        so every slot is filled whether the cache is smaller or larger than the dataset.
        """
        from concurrent.futures import ThreadPoolExecutor

        n_images = len(self.data_files)
        n_slots = self._n_slots
        es = self._extract_size
        ps = self.patch_size
        if n_images == 0 or n_slots == 0:
            return

        counts = [n_slots // n_images + (1 if i < n_slots % n_images else 0) for i in range(n_images)]
        starts = [0] * n_images
        for i in range(1, n_images):
            starts[i] = starts[i - 1] + counts[i - 1]

        rng = random.Random(seed)
        crops = []
        for img_i in range(n_images):
            img = np.load(self.data_files[img_i], mmap_mode='r')
            h, w, _ = img.shape
            if h < ps or w < ps:
                raise ValueError(f"{self.data_files[img_i]} is {w}x{h}, smaller than the {ps} px patch")
            crop_size = es if (es <= h and es <= w) else ps
            max_y = max(0, h - crop_size)
            max_x = max(0, w - crop_size)
            for _ in range(counts[img_i]):
                crops.append((img_i, rng.randint(0, max_y), rng.randint(0, max_x), crop_size))
        assert len(crops) == n_slots

        def _extract_image(img_i):
            if counts[img_i] == 0:
                return
            img = np.load(self.data_files[img_i], mmap_mode='r')
            for j in range(counts[img_i]):
                slot = starts[img_i] + j
                _, top, left, crop_size = crops[slot]
                patch = img[top:top + crop_size, left:left + crop_size]
                norm_patch = np.asarray(patch, dtype=np.float32) / 65535.0
                if crop_size < es:
                    self._patch_data[slot] = 0
                    self._patch_data[slot, :crop_size, :crop_size] = norm_patch
                else:
                    self._patch_data[slot] = norm_patch
                self._patch_img_idx[slot] = img_i

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_extract_image, range(n_images)))

    def _stream_worker(self):
        """Worker thread: load patches into staging slots."""
        import queue as _queue

        n_images = len(self.data_files)
        es = self._extract_size
        ps = self.patch_size
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

            top = rng.randint(0, max_y)
            left = rng.randint(0, max_x)

            patch = img[top:top + crop_size, left:left + crop_size]
            norm_patch = np.asarray(patch, dtype=np.float32) / 65535.0
            if crop_size < es:
                self._patch_data[physical] = 0
                self._patch_data[physical, :crop_size, :crop_size] = norm_patch
            else:
                self._patch_data[physical] = norm_patch
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
                top = rng.randint(0, max_off)
                left = rng.randint(0, max_off)
                rgb = torch.from_numpy(np.ascontiguousarray(
                    self._patch_data[physical, top:top+ps, left:left+ps].transpose(2, 0, 1)))
        else:
            rgb = torch.from_numpy(np.ascontiguousarray(
                self._patch_data[physical].transpose(2, 0, 1)))

        return self._process_patch(rgb, rng)


class ImageGroupedSampler(Sampler):
    """Yields patch indices drawn from `group_images` images at a time.

    The image order is shuffled each epoch and cut into groups; each group's patch
    indices are yielded in shuffled order. A batch therefore mixes up to `group_images`
    photos, while a worker only has that many files open (see LinearDataset._load_image).
    """

    def __init__(self, num_images: int, patches_per_image: int, shuffle: bool = True,
                 group_images: int = 32):
        self.num_images = num_images
        self.patches_per_image = patches_per_image
        self.shuffle = shuffle
        self.group_images = max(1, group_images)
        self.epoch = 0

    def __iter__(self):
        image_order = list(range(self.num_images))
        g = random.Random(self.epoch)
        if self.shuffle:
            g.shuffle(image_order)
        for start in range(0, self.num_images, self.group_images):
            patches = [img_idx * self.patches_per_image + j
                       for img_idx in image_order[start:start + self.group_images]
                       for j in range(self.patches_per_image)]
            if self.shuffle:
                g.shuffle(patches)
            yield from patches

    def __len__(self):
        return self.num_images * self.patches_per_image

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# Backwards compatibility
XTransLinearDataset = LinearDataset
