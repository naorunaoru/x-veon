#!/usr/bin/env python3
"""Compare C (darktable reference) and TS highlight reconstruction outputs.

Usage:
    python compare.py fixture.bin c_output.bin ts_output.bin

Prints per-pixel statistics and generates a visual diff image.
"""

import sys
import struct
import numpy as np


def load_fixture_header(path):
    with open(path, "rb") as f:
        hdr = f.read(256)
    w, h, period, dy, dx = struct.unpack_from("<5I", hdr, 0)
    pattern = np.frombuffer(hdr[20:20 + period * period], dtype=np.uint8).reshape(period, period)
    return w, h, period, dy, dx, pattern


def load_output(path, npix):
    data = np.fromfile(path, dtype=np.float32)
    if data.size != npix:
        print(f"WARNING: expected {npix} pixels, got {data.size}")
    return data[:npix]


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <fixture.bin> <c_output.bin> <ts_output.bin>")
        sys.exit(1)

    fixture_path, c_path, ts_path = sys.argv[1], sys.argv[2], sys.argv[3]

    w, h, period, dy, dx, pattern = load_fixture_header(fixture_path)
    npix = w * h
    print(f"Image: {w}x{h}, period={period}, shift=({dy},{dx})")

    # Load original input for reference
    with open(fixture_path, "rb") as f:
        f.seek(256)
        original = np.fromfile(f, dtype=np.float32, count=npix)

    c_out = load_output(c_path, npix)
    ts_out = load_output(ts_path, npix)

    # Find clipped pixels (where reconstruction happened)
    clipped = original >= 1.0
    n_clipped = np.sum(clipped)
    print(f"Clipped pixels: {n_clipped} ({100 * n_clipped / npix:.2f}%)")

    # Overall stats
    diff = ts_out - c_out
    abs_diff = np.abs(diff)

    print(f"\n--- All pixels ---")
    print(f"  MAE:  {np.mean(abs_diff):.6f}")
    print(f"  Max:  {np.max(abs_diff):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(diff ** 2)):.6f}")

    if n_clipped > 0:
        diff_clip = diff[clipped]
        abs_clip = abs_diff[clipped]
        print(f"\n--- Clipped pixels only ---")
        print(f"  MAE:  {np.mean(abs_clip):.6f}")
        print(f"  Max:  {np.max(abs_clip):.6f}")
        print(f"  RMSE: {np.sqrt(np.mean(diff_clip ** 2)):.6f}")
        print(f"  Mean C:  {np.mean(c_out[clipped]):.4f}")
        print(f"  Mean TS: {np.mean(ts_out[clipped]):.4f}")

        # Percentiles of absolute error on clipped pixels
        percs = [50, 90, 95, 99]
        pvals = np.percentile(abs_clip, percs)
        print(f"  Percentiles: " + ", ".join(f"p{p}={v:.6f}" for p, v in zip(percs, pvals)))

    # Unclipped pixels should be identical
    if npix - n_clipped > 0:
        unclipped_diff = abs_diff[~clipped]
        print(f"\n--- Unclipped pixels (should be identical) ---")
        print(f"  Max diff: {np.max(unclipped_diff):.9f}")

    # Save visual diff as raw float32 (can be viewed with Python/imagemagick)
    diff_path = ts_path.replace(".bin", "_diff.bin")
    abs_diff.astype(np.float32).tofile(diff_path)
    print(f"\nDiff map written to {diff_path}")

    # Also save a PNG heatmap if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        diff_2d = abs_diff.reshape(h, w)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # C output
        axes[0].imshow(c_out.reshape(h, w), cmap="gray", vmin=0, vmax=2)
        axes[0].set_title("C (darktable reference)")
        axes[0].axis("off")

        # TS output
        axes[1].imshow(ts_out.reshape(h, w), cmap="gray", vmin=0, vmax=2)
        axes[1].set_title("TS implementation")
        axes[1].axis("off")

        # Diff heatmap
        vmax = max(0.01, np.percentile(abs_diff, 99))
        im = axes[2].imshow(diff_2d, cmap="hot", vmin=0, vmax=vmax)
        axes[2].set_title(f"|C - TS| (p99={vmax:.4f})")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046)

        png_path = ts_path.replace(".bin", "_diff.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Diff heatmap written to {png_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
