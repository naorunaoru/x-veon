"""Check what CFA pattern rawpy reports for the compressed RAF file."""
import rawpy
import numpy as np

path = "/Volumes/4T/naoru/DCIM/107_FUJI/DSCF7001.RAF"
with rawpy.imread(path) as raw:
    print(f"raw_pattern shape: {raw.raw_pattern.shape}")
    print(f"raw_pattern:\n{raw.raw_pattern}")
    print(f"\nColor description: {raw.color_desc}")
    print(f"num_colors: {raw.num_colors}")
    print(f"raw_colors shape: {raw.raw_colors.shape}")
    print(f"raw_colors 6x6 top-left:\n{raw.raw_colors[:6, :6]}")
    print(f"\nSizes: {raw.sizes}")
    print(f"raw_image shape: {raw.raw_image.shape}")

    # Check which rows/cols have optical black
    img = raw.raw_image
    print(f"\nRow sums (first 10): {[img[r,:].sum() for r in range(10)]}")
    print(f"Col sums (first 10): {[img[:,c].sum() for c in range(10)]}")
