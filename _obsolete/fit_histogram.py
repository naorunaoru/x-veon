#!/usr/bin/env python3
"""
Build per-channel LUTs by matching model output histograms to camera JPEG histograms.
This captures the full non-linear color pipeline (color matrix + tone curve + film sim)
without requiring pixel-level alignment.
"""
import numpy as np
from PIL import Image
import os
import json

jpeg_base = "data/fuji_jpgs"
raf_to_jpeg = {}
for folder in os.listdir(jpeg_base):
    folder_path = os.path.join(jpeg_base, folder)
    if not os.path.isdir(folder_path):
        continue
    for f in os.listdir(folder_path):
        if f.upper().endswith('.JPG'):
            stem = os.path.splitext(f)[0]
            raf_to_jpeg[stem] = os.path.join(folder_path, f)

# Accumulate histograms from all matching pairs
model_hists = [np.zeros(256, dtype=np.int64) for _ in range(3)]
jpeg_hists = [np.zeros(256, dtype=np.int64) for _ in range(3)]

count = 0
for f in sorted(os.listdir("output_nocorr")):
    if not f.endswith('_demosaic.png'):
        continue
    stem = f.replace('_demosaic.png', '')
    if stem not in raf_to_jpeg:
        continue
    
    model_img = np.array(Image.open(os.path.join("output_nocorr", f)))
    jpeg_img = np.array(Image.open(raf_to_jpeg[stem]))
    
    # Resize JPEG to match model dimensions
    if model_img.shape[:2] != jpeg_img.shape[:2]:
        jpeg_img = np.array(Image.open(raf_to_jpeg[stem]).resize(
            (model_img.shape[1], model_img.shape[0]), Image.LANCZOS))
    
    for c in range(3):
        mh, _ = np.histogram(model_img[:,:,c], bins=256, range=(0, 256))
        jh, _ = np.histogram(jpeg_img[:,:,c], bins=256, range=(0, 256))
        model_hists[c] += mh
        jpeg_hists[c] += jh
    
    count += 1
    print(f"  {stem}: added")

print(f"\nTotal images: {count}")

# Build per-channel LUT via CDF matching
luts = []
for c, name in enumerate("RGB"):
    # Compute CDFs
    model_cdf = np.cumsum(model_hists[c]).astype(np.float64)
    model_cdf /= model_cdf[-1]
    
    jpeg_cdf = np.cumsum(jpeg_hists[c]).astype(np.float64)
    jpeg_cdf /= jpeg_cdf[-1]
    
    # For each model value, find the JPEG value with the closest CDF
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find where model_cdf[i] falls in jpeg_cdf
        idx = np.searchsorted(jpeg_cdf, model_cdf[i])
        lut[i] = min(idx, 255)
    
    luts.append(lut)
    
    # Print some key mappings
    print(f"\n{name} channel LUT (input -> output):")
    for v in [0, 32, 64, 96, 128, 160, 192, 224, 255]:
        print(f"  {v:3d} -> {lut[v]:3d}")

# Save LUT
lut_data = {
    "R": luts[0].tolist(),
    "G": luts[1].tolist(),
    "B": luts[2].tolist(),
    "n_images": count,
}
with open("color_lut.json", "w") as f:
    json.dump(lut_data, f)
print(f"\nSaved color_lut.json")

# Apply LUT to one test image and save preview
test_img = np.array(Image.open("output_nocorr/DSCF5212_demosaic.png"))
corrected = np.stack([luts[c][test_img[:,:,c]] for c in range(3)], axis=2)
Image.fromarray(corrected).save("output/DSCF5212_lut_corrected.png")

# Also save preview
w, h = corrected.shape[1], corrected.shape[0]
preview = Image.fromarray(corrected).resize((800, int(h*800/w)), Image.LANCZOS)
preview.save("output/DSCF5212_lut_preview.jpg", quality=85)

# And the camera JPEG for comparison
jpeg_ref = Image.open(raf_to_jpeg["DSCF5212"])
jpeg_ref = jpeg_ref.resize((800, int(jpeg_ref.height*800/jpeg_ref.width)), Image.LANCZOS)
jpeg_ref.save("output/DSCF5212_jpeg_ref.jpg", quality=85)

print("Saved test output + reference JPEG preview")
