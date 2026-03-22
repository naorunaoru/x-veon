"""Quick comparison of decoder output vs reference to diagnose CFA mapping issues."""
import numpy as np

ref = np.load("/Users/naoru/projects/xtrans-demosaic/reference_full.npy")
# Run decoder and save output first, or load from saved
try:
    out = np.load("/Users/naoru/projects/xtrans-demosaic/decoder_output.npy")
except:
    print("Run fuji_decompress.py first to save decoder_output.npy")
    exit(1)

print(f"Shapes: ref={ref.shape}, out={out.shape}")
print(f"Ranges: ref=[{ref.min()},{ref.max()}], out=[{out.min()},{out.max()}]")

# Compare first few rows
print(f"\nRef rows 0-7, cols 0-12:")
print(ref[:8, :12])
print(f"\nOut rows 0-7, cols 0-12:")
print(out[:8, :12])

# Check histograms match
ref_hist = np.histogram(ref, bins=100, range=(0, 8000))[0]
out_hist = np.histogram(out, bins=100, range=(0, 8000))[0]
hist_match = np.allclose(ref_hist, out_hist, atol=100)
print(f"\nHistogram roughly match: {hist_match}")
print(f"Ref histogram sum: {ref_hist.sum()}, Out histogram sum: {out_hist.sum()}")

# Check if values are the same but in different positions
ref_sorted = np.sort(ref.ravel())
out_sorted = np.sort(out.ravel())
sorted_match = np.array_equal(ref_sorted, out_sorted)
print(f"Sorted values exact match: {sorted_match}")
if not sorted_match:
    diffs = ref_sorted.astype(np.int32) - out_sorted.astype(np.int32)
    print(f"  Sorted diff range: [{diffs.min()}, {diffs.max()}]")
    print(f"  Sorted diff mean: {np.abs(diffs).mean():.4f}")

# Check specific strips
for strip_x in [0, 768, 768*4]:
    region_ref = ref[:12, strip_x:strip_x+12]
    region_out = out[:12, strip_x:strip_x+12]
    match = np.array_equal(region_ref, region_out)
    print(f"\nStrip at x={strip_x}, rows 0-11, cols 0-11: match={match}")
    if not match:
        print(f"  Ref:\n{region_ref}")
        print(f"  Out:\n{region_out}")
