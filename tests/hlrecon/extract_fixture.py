#!/usr/bin/env python3
"""Extract a test fixture from a RAW file for highlight reconstruction testing.

Writes a binary file with:
  - Header: width(u32), height(u32), period(u32), dy(u32), dx(u32),
            pattern (period*period u8), padding to 256 bytes
  - Body:   width*height float32 (normalized CFA, black-subtracted, 0-1 range)

Usage:
    python extract_fixture.py input.RAF output.bin
"""

import sys
import struct
import numpy as np

sys.path.insert(0, "/Users/naoru/projects/xtrans-demosaic")
import rawpy
from cfa import detect_cfa_from_raw, find_pattern_shift, CFA_REGISTRY


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.raw> <output.bin>")
        sys.exit(1)

    raw_path = sys.argv[1]
    out_path = sys.argv[2]

    raw = rawpy.imread(raw_path)
    cfa_type, _ = detect_cfa_from_raw(raw.raw_pattern)
    print(f"Detected CFA: {cfa_type}")

    pattern = np.array(CFA_REGISTRY[cfa_type])
    period = pattern.shape[0]

    # Get visible area
    top, left = raw.sizes.top_margin, raw.sizes.left_margin
    vis_h, vis_w = raw.sizes.iheight, raw.sizes.iwidth

    # Extract raw image
    raw_image = raw.raw_image_visible.astype(np.float32)
    black = float(raw.black_level_per_channel[0])
    white = float(raw.white_level)

    # Normalize to 0-1
    cfa = (raw_image - black) / (white - black)
    cfa = np.clip(cfa, 0.0, None)  # don't clip top — highlights go above 1.0

    # Find pattern shift
    dy, dx = find_pattern_shift(raw.raw_pattern, pattern)
    print(f"Dimensions: {vis_w}x{vis_h}, period={period}, shift=dy{dy} dx{dx}")

    # Write binary fixture
    header = bytearray(256)
    struct.pack_into("<5I", header, 0, vis_w, vis_h, period, dy, dx)
    # Write pattern bytes starting at offset 20
    flat_pattern = pattern.flatten().astype(np.uint8)
    header[20:20 + len(flat_pattern)] = flat_pattern.tobytes()

    with open(out_path, "wb") as f:
        f.write(header)
        cfa.astype(np.float32).tofile(f)

    print(f"Written {out_path}: {256 + vis_w * vis_h * 4} bytes")
    raw.close()


if __name__ == "__main__":
    main()
