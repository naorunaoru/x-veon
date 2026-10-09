#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dataset builder: the white level is the sensor's in every image."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# A frame whose brightest raw value is about 80% of the white level: LibRaw's default
# scaling would stretch it by 1.26.
SAMPLE = Path(os.path.expanduser("~/Downloads/xtrans-samples/X-T3/AFXT2720.RAF"))


@unittest.skipUnless(SAMPLE.exists(), f"sample RAF not present: {SAMPLE}")
class WhiteLevelTest(unittest.TestCase):
    def test_65535_is_the_sensors_white_level_even_without_clipped_highlights(self) -> None:
        import rawpy

        from build_dataset import demosaic_linear

        with rawpy.imread(str(SAMPLE)) as raw:
            black, white = float(raw.black_level_per_channel[0]), float(raw.white_level)
            peak = float(np.max(raw.raw_image_visible))
            self.assertTrue(0.75 * white < peak < white, "the sample must sit in LibRaw's adjustment range")
            built, sensor_type = demosaic_linear(raw)
        self.assertEqual(sensor_type, "xtrans")
        self.assertEqual(built.dtype, np.uint16)
        with rawpy.imread(str(SAMPLE)) as raw:
            # The same demosaic without LibRaw's scaling: black-subtracted raw units.
            unscaled = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT, output_bps=16, no_auto_bright=True,
                no_auto_scale=True, gamma=(1, 1), output_color=rawpy.ColorSpace.raw, use_camera_wb=False,
                use_auto_wb=False, user_wb=[1, 1, 1, 1], highlight_mode=rawpy.HighlightMode.Ignore,
                half_size=False, user_flip=0,
            ).astype(np.float64)
        exact = unscaled * 65535.0 / (white - black)
        mid = (exact > 0.02 * 65535) & (exact < 0.5 * 65535)
        ratio = float(np.median(built[mid] / exact[mid]))
        self.assertAlmostEqual(ratio, 1.0, delta=0.001)


if __name__ == "__main__":
    unittest.main()
