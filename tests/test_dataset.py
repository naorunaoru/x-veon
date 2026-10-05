#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dataset: (mosaic, target) at known intensity through every loading path, clamps, crops, sampler."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import ImageGroupedSampler, LinearDataset, PatchCacheDataset  # noqa: E402

VALUES = [1000, 2000, 3000, 4000]
PATCH = 96
BYTES_PER_SLOT = PATCH * PATCH * 3 * 4


def _write(folder: str, arrays: list[np.ndarray]) -> list[str]:
    files = []
    for k, a in enumerate(arrays):
        path = str(Path(folder) / f"img{k}.npy")
        np.save(path, a)
        files.append(path)
    return files


def _flat(value: int, h: int = 200, w: int = 232) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint16)


class LinearDatasetTest(unittest.TestCase):
    def test_mosaic_and_target_at_known_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ds = LinearDataset(files=_write(d, [_flat(v) for v in VALUES]), patch_size=PATCH, cfa_type="bayer",
                               augment=False, noise_sigma=(0.0, 0.0), patches_per_image=2)
            self.assertEqual(len(ds), 8)
            for idx in range(len(ds)):
                mosaic, target = ds[idx]
                self.assertEqual(tuple(mosaic.shape), (1, PATCH, PATCH))
                self.assertEqual(tuple(target.shape), (3, PATCH, PATCH))
                expected = VALUES[idx // 2] / 65535.0
                self.assertTrue(torch.allclose(target, torch.full_like(target, expected), atol=1e-7))
                self.assertTrue(torch.allclose(mosaic, torch.full_like(mosaic, expected), atol=1e-7))

    def test_nothing_exceeds_the_clip_level_under_noise(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ds = LinearDataset(files=_write(d, [_flat(65535)]), patch_size=PATCH, cfa_type="xtrans", augment=True,
                               noise_sigma=(0.005, 0.005), shot_noise=(0.001, 0.001), patches_per_image=8)
            for idx in range(len(ds)):
                mosaic, target = ds[idx]
                self.assertLessEqual(float(mosaic.max()), 1.0)
                self.assertLessEqual(float(target.max()), 1.0)
                self.assertLess(float(mosaic.min()), 1.0)            # the noise is there, below the clip level

    def test_crop_offsets_are_not_tied_to_the_cfa_period(self) -> None:
        y, x = np.mgrid[0:200, 0:232]
        coded = np.repeat((y * 256 + x)[..., None], 3, axis=2).astype(np.uint16)     # value encodes position
        with tempfile.TemporaryDirectory() as d:
            ds = LinearDataset(files=_write(d, [coded]), patch_size=PATCH, cfa_type="xtrans", augment=False,
                               noise_sigma=(0.0, 0.0), patches_per_image=40)
            offsets = [divmod(int(round(float(ds[i][1][0, 0, 0]) * 65535)), 256) for i in range(len(ds))]
        self.assertTrue(any(top % 6 or left % 6 for top, left in offsets), offsets[:5])

    def test_an_image_smaller_than_the_patch_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            files = _write(d, [_flat(100, 50, 50)])
            ds = LinearDataset(files=files, patch_size=PATCH, cfa_type="bayer", augment=False, noise_sigma=(0.0, 0.0))
            with self.assertRaises(ValueError) as ctx:
                ds[0]
            self.assertIn("img0.npy", str(ctx.exception))
            with self.assertRaises(ValueError) as ctx:
                PatchCacheDataset(files=files, patch_size=PATCH, cfa_type="bayer", augment=False, noise_sigma=(0.0, 0.0))
            self.assertIn("img0.npy", str(ctx.exception))

    def test_open_images_are_bounded_by_the_group_size(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ds = LinearDataset(files=_write(d, [_flat(v) for v in VALUES]), patch_size=PATCH, cfa_type="bayer",
                               augment=False, noise_sigma=(0.0, 0.0), patches_per_image=1, group_images=2)
            for idx in range(4):
                ds[idx]
            self.assertEqual(list(ds._open_images), [2, 3])


class PatchCacheTest(unittest.TestCase):
    def _check_every_slot(self, ds: PatchCacheDataset) -> None:
        for i in range(len(ds)):
            mosaic, target = ds[i]
            value = VALUES[int(ds._patch_img_idx[int(ds._slot_map[i])])]
            self.assertTrue(torch.allclose(target, torch.full_like(target, value / 65535.0), atol=1e-7),
                            f"slot {i}: {float(target.min())}..{float(target.max())}, expected {value / 65535.0}")
            self.assertTrue(torch.allclose(mosaic, torch.full_like(mosaic, value / 65535.0), atol=1e-7))

    def test_first_fill_and_streamed_patches_carry_the_same_intensity_as_uncached_loading(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            files = _write(d, [_flat(v) for v in VALUES])
            for n_slots in (8, 40):                              # smaller and larger than 4 images x 4 patches
                with self.subTest(n_slots=n_slots):
                    ds = PatchCacheDataset(files=files, seed=0, cache_gb=(n_slots + 0.5) * BYTES_PER_SLOT / 1e9,
                                           patch_size=PATCH, cfa_type="bayer", augment=False,
                                           noise_sigma=(0.0, 0.0), patches_per_image=4)
                    try:
                        self.assertEqual(len(ds), n_slots)
                        self._check_every_slot(ds)               # no empty slot, values divided by 65535
                        ds.start_streaming()
                        deadline = time.time() + 10
                        while ds._ready_swaps.qsize() == 0 and time.time() < deadline:
                            time.sleep(0.05)
                        self.assertGreater(ds.swap_staging(), 0)
                        self._check_every_slot(ds)
                    finally:
                        ds.cleanup()


class SamplerTest(unittest.TestCase):
    def test_every_index_once_and_batches_mix_images(self) -> None:
        sampler = ImageGroupedSampler(100, 16, group_images=32)
        indices = list(sampler)
        self.assertEqual(len(sampler), 1600)
        self.assertEqual(sorted(indices), list(range(1600)))
        self.assertGreater(len({i // 16 for i in indices[:32]}), 2)
        sampler.set_epoch(1)
        self.assertNotEqual(list(sampler), indices)

    def test_a_group_of_one_is_the_old_behaviour(self) -> None:
        indices = list(ImageGroupedSampler(10, 16, group_images=1))
        self.assertEqual(len({i // 16 for i in indices[:16]}), 1)


if __name__ == "__main__":
    unittest.main()
