#!/usr/bin/env python3
"""
Demosaic RAF with rawpy (linear, camera RGB) and downscale 4x.
Memory-efficient version.
"""
import numpy as np
import rawpy
from PIL import Image
import subprocess
import gc

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.clip(x, 0.0031308, None), 1.0/2.4) - 0.055)

def get_dr_gain(filepath):
    for exiftool in ['exiftool', '/opt/homebrew/bin/exiftool']:
        try:
            result = subprocess.run([exiftool, '-DevelopmentDynamicRange', '-s3', str(filepath)],
                                    capture_output=True, text=True, timeout=5)
            return int(result.stdout.strip()) / 100.0
        except Exception:
            continue
    return 1.0

raf_path = "test_rafs/DSCF3561.RAF"
dr_gain = get_dr_gain(raf_path)

# Process with rawpy — linear, camera WB, no color matrix
# Use half_size=False for full res, but output_bps=16 for dynamic range
raw = rawpy.imread(raf_path)

# Check available X-Trans demosaic algorithms
print("Testing demosaic algorithms for X-Trans...")
for algo_name in ['AHD', 'VNG', 'PPG', 'DCB', 'DHT', 'AAHD']:
    try:
        algo = getattr(rawpy.DemosaicAlgorithm, algo_name, None)
        if algo is None:
            print(f"  {algo_name}: not available in this build")
            continue
        raw_test = rawpy.imread(raf_path)
        # Use half_size to test quickly
        out = raw_test.postprocess(
            demosaic_algorithm=algo,
            half_size=True,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            output_bps=16,
            no_auto_bright=True,
        )
        print(f"  {algo_name}: OK ({out.shape})")
        del out, raw_test
        gc.collect()
    except Exception as e:
        print(f"  {algo_name}: {e}")

# Now do the full-res version with the best available algorithm
# Use output_bps=16, convert to float32 for processing
print(f"\nFull-res demosaic...")
raw = rawpy.imread(raf_path)
rgb_16 = raw.postprocess(
    use_camera_wb=True,
    output_color=rawpy.ColorSpace.raw,
    gamma=(1, 1),
    output_bps=16,
    no_auto_bright=True,
)
print(f"Shape: {rgb_16.shape}, dtype: {rgb_16.dtype}")
print(f"Range: [{rgb_16.min()}, {rgb_16.max()}]")

# Convert to float32 and normalize
h, w = rgb_16.shape[:2]
rgb_f = rgb_16.astype(np.float32) / 65535.0
del rgb_16
gc.collect()

# Apply DR gain
rgb_f *= dr_gain
print(f"After DR gain ({dr_gain}x): [{rgb_f.min():.4f}, {rgb_f.max():.4f}]")
print(f"Per-ch mean: R={rgb_f[:,:,0].mean():.4f} G={rgb_f[:,:,1].mean():.4f} B={rgb_f[:,:,2].mean():.4f}")

# Downscale 4x using area averaging
new_h, new_w = h // 4, w // 4
h4, w4 = new_h * 4, new_w * 4
print(f"Downscaling {w}x{h} -> {new_w}x{new_h}")

# Do it channel by channel to save memory
downscaled = np.zeros((new_h, new_w, 3), dtype=np.float32)
for c in range(3):
    ch = rgb_f[:h4, :w4, c].reshape(new_h, 4, new_w, 4)
    downscaled[:, :, c] = ch.mean(axis=(1, 3))
del rgb_f
gc.collect()

print(f"Downscaled range: [{downscaled.min():.4f}, {downscaled.max():.4f}]")

# Save linear numpy
np.save("output_6x6/DSCF3561_demosaic_4x_linear.npy", downscaled)

# Save sRGB preview
clipped = np.clip(downscaled, 0.0, 1.0)
srgb = linear_to_srgb(clipped)
out_img = (srgb * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(out_img).save("output_6x6/DSCF3561_demosaic_4x_srgb.png")
print(f"Saved: DSCF3561_demosaic_4x_srgb.png ({new_w}x{new_h})")

print("Done!")
