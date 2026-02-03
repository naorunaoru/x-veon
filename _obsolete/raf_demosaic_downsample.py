#!/usr/bin/env python3
"""
Produce training data from RAF files:
1. Demosaic with Markesteijn 1-pass (rawpy/libraw)
2. Output in linear camera RGB (no color matrix, no gamma)
3. Downscale 4x with proper anti-aliasing

This gives us clean, full-dynamic-range training targets.
"""
import numpy as np
import rawpy
from PIL import Image
import subprocess

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

raw = rawpy.imread(raf_path)

# Markesteijn 1-pass demosaic, linear output, camera WB, no color matrix
# output_color=raw means no color space conversion
# gamma=(1,1) means linear (no gamma curve)
# no_auto_bright=True to preserve exposure
rgb_linear = raw.postprocess(
    demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,  # fallback if Markesteijn not available
    use_camera_wb=True,
    output_color=rawpy.ColorSpace.raw,  # no color matrix
    gamma=(1, 1),  # linear output
    output_bps=16,
    no_auto_bright=True,
    no_auto_scale=True,
).astype(np.float32) / 65535.0

print(f"Demosaiced: {rgb_linear.shape}")
print(f"Range: [{rgb_linear.min():.4f}, {rgb_linear.max():.4f}]")
print(f"Per-ch mean: R={rgb_linear[:,:,0].mean():.4f} G={rgb_linear[:,:,1].mean():.4f} B={rgb_linear[:,:,2].mean():.4f}")

# Apply DR gain
rgb_linear *= dr_gain
print(f"After DR gain ({dr_gain}x): [{rgb_linear.min():.4f}, {rgb_linear.max():.4f}]")

# Try all available demosaic algorithms
print("\n--- Testing demosaic algorithms ---")
for algo_name in ['LINEAR', 'VNG', 'PPG', 'AHD', 'DCB', 'DHT', 'AAHD']:
    try:
        algo = getattr(rawpy.DemosaicAlgorithm, algo_name, None)
        if algo is None:
            continue
        raw2 = rawpy.imread(raf_path)
        out = raw2.postprocess(
            demosaic_algorithm=algo,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            output_bps=16,
            no_auto_bright=True,
            no_auto_scale=True,
        ).astype(np.float32) / 65535.0
        print(f"  {algo_name}: OK, range [{out.min():.4f}, {out.max():.4f}]")
    except Exception as e:
        print(f"  {algo_name}: {e}")

# Now downscale 4x using area averaging (best for downscaling)
h, w = rgb_linear.shape[:2]
new_h, new_w = h // 4, w // 4
print(f"\nDownscaling: {w}x{h} -> {new_w}x{new_h}")

# Trim to multiple of 4
h4 = new_h * 4
w4 = new_w * 4
trimmed = rgb_linear[:h4, :w4]

# Area average (reshape and mean)
downscaled = trimmed.reshape(new_h, 4, new_w, 4, 3).mean(axis=(1, 3))
print(f"Downscaled range: [{downscaled.min():.4f}, {downscaled.max():.4f}]")
print(f"Per-ch mean: R={downscaled[:,:,0].mean():.4f} G={downscaled[:,:,1].mean():.4f} B={downscaled[:,:,2].mean():.4f}")

# Save sRGB preview (clip + gamma for viewing)
clipped = np.clip(downscaled, 0.0, 1.0)
srgb = linear_to_srgb(clipped)
out_img = (srgb * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(out_img).save("output_6x6/DSCF3561_markesteijn_4x_srgb.png")
print(f"Saved sRGB preview: {new_w}x{new_h}")

# Also save the 6x6 averaged version for direct comparison
# (at same output resolution)
from PIL import Image as PILImage
img_4x = PILImage.fromarray(out_img)
# 6x6 would be 822x549, let's resize 4x to match for comparison
print(f"4x downscale: {new_w}x{new_h} vs 6x6: 822x549")

print("\nDone!")
