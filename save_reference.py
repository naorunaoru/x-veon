"""Save rawpy decoded output as ground truth for testing the Fuji decompressor."""
import numpy as np
import rawpy

RAF_PATH = "/Volumes/4T/naoru/DCIM/107_FUJI/DSCF7001.RAF"

with rawpy.imread(RAF_PATH) as raw:
    img = raw.raw_image.copy()  # uint16, shape (H, W)
    print(f"Shape: {img.shape}, dtype: {img.dtype}")
    print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")
    print(f"Top-left 12x6 values:")
    print(img[:6, :12])

    # Save a small patch for quick comparison
    patch = img[:96, :96].copy()
    np.save("/Users/naoru/projects/xtrans-demosaic/reference_patch_96x96.npy", patch)
    print(f"\nSaved 96x96 patch")

    # Save full image for thorough comparison
    np.save("/Users/naoru/projects/xtrans-demosaic/reference_full.npy", img)
    print(f"Saved full {img.shape} reference")
