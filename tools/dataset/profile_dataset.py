#!/usr/bin/env python3
"""Profile LinearDataset.__getitem__ to find the CPU bottleneck."""

import time
import argparse
import random
import math

import numpy as np
import torch
import torch.nn.functional as F

from dataset import LinearDataset, ImageGroupedSampler, mosaic
from losses import _gaussian_kernel_2d


def profile_getitem(dataset, indices, warmup=10):
    """Time each section of __getitem__ manually."""
    timings = {}

    def timed_call(idx):
        t = {}

        t0 = time.perf_counter()
        img_idx = idx // dataset.patches_per_image
        rng = dataset._get_rng(idx)
        t["rng"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        img = dataset._load_image(img_idx)
        h, w, _ = img.shape
        t["load_image"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        max_y, max_x = h - dataset.patch_size, w - dataset.patch_size
        top = rng.randint(0, max(0, max_y))
        left = rng.randint(0, max(0, max_x))
        patch = img[top:top + dataset.patch_size, left:left + dataset.patch_size]
        t["crop"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rgb = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32) / 65535.0).permute(2, 0, 1).contiguous()
        t["to_tensor"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if dataset.augment:
            if rng.random() > 0.5:
                rgb = rgb.flip(2)
            if rng.random() > 0.5:
                rgb = rgb.flip(1)
            k = rng.randint(0, 3)
            if k > 0:
                rgb = torch.rot90(rgb, k, [1, 2])
        t["augment_geo"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = rgb.clamp(max=1.0)
        if dataset.augment and dataset.olpf_sigma[1] > 0:
            sigma = rng.uniform(*dataset.olpf_sigma)
            if sigma > 0:
                ks = max(3, int(sigma * 6) | 1)
                pad = ks // 2
                kernel = _gaussian_kernel_2d(ks, sigma, 3)
                rgb = F.conv2d(rgb.unsqueeze(0), kernel, padding=pad, groups=3).squeeze(0)
        t["olpf"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cfa_img = mosaic(rgb, dataset.cfa)
        t["mosaic"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        read_sigma = rng.uniform(*dataset.noise_sigma)
        shot_coeff = rng.uniform(*dataset.shot_noise)
        if read_sigma > 0 or shot_coeff > 0:
            noise_var = shot_coeff * cfa_img.clamp(min=0) + read_sigma ** 2
            cfa_img = cfa_img + torch.randn_like(cfa_img) * noise_var.sqrt()
        cfa_img = cfa_img.clamp(max=1.0)
        t["noise"] = time.perf_counter() - t0

        for k, v in t.items():
            timings.setdefault(k, []).append(v)

    # Warmup
    for i in indices[:warmup]:
        timed_call(i)
    timings.clear()

    # Profile
    for i in indices:
        timed_call(i)

    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cfa-type", default="xtrans")
    parser.add_argument("--n-items", type=int, default=500)
    args = parser.parse_args()

    files = LinearDataset.find_files(args.data_dir)
    random.Random(42).shuffle(files)
    train_files = files[5:]

    dataset = LinearDataset(
        files=train_files,
        augment=True,
        noise_sigma=(0.0, 0.005),
        shot_noise=(0.0, 0.0001),
        cfa_type=args.cfa_type,
    )

    sampler = ImageGroupedSampler(len(train_files), dataset.patches_per_image)
    sampler.set_epoch(0)
    indices = list(sampler)[:args.n_items]

    print(f"Profiling {args.n_items} __getitem__ calls...")
    timings = profile_getitem(dataset, indices)

    print(f"\n{'Section':<16} {'Mean (μs)':>10} {'Median (μs)':>12} {'P95 (μs)':>10} {'Total %':>8}")
    print("-" * 60)
    total_mean = sum(np.mean(v) for v in timings.values())
    for key in timings:
        vals = np.array(timings[key]) * 1e6
        pct = np.mean(timings[key]) / total_mean * 100
        print(f"{key:<16} {np.mean(vals):>10.0f} {np.median(vals):>12.0f} {np.percentile(vals, 95):>10.0f} {pct:>7.1f}%")
    print(f"{'TOTAL':<16} {total_mean*1e6:>10.0f}")


if __name__ == "__main__":
    main()
