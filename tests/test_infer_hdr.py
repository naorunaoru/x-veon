#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""infer_hdr.py: padding keeps the CFA phase, tiles stay on the CFA period, and a whole RAF goes through."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_hdr import pad_same_phase, process_raw  # noqa: E402
from model import XTransUNet  # noqa: E402

SAMPLE = Path.home() / "Downloads/xtrans-samples/X-T3/AFXT2720.RAF"


class PaddingTest(unittest.TestCase):
    def test_padded_rows_and_columns_come_from_the_same_cfa_phase(self):
        period = 6
        a = np.arange(12 * 18, dtype=np.float32).reshape(12, 18)
        for pad_top, pad_left in ((0, 0), (2, 5), (5, 1)):
            p = pad_same_phase(a, pad_top, pad_left, period)
            self.assertEqual(p.shape, (12 + pad_top, 18 + pad_left))
            np.testing.assert_array_equal(p[pad_top:, pad_left:], a)
            for y in range(pad_top):
                # a padded row repeats the image row that has the same phase as its position
                np.testing.assert_array_equal(p[y, pad_left:], a[(y - pad_top) % period])
            for x in range(pad_left):
                np.testing.assert_array_equal(p[pad_top:, x], a[:, (x - pad_left) % period])


@unittest.skipUnless(SAMPLE.exists(), f"sample RAF not present: {SAMPLE}")
class WholeFileTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = XTransUNet(base_width=16, cfa_period=6, stages=2).eval()   # untrained: the plumbing is under test

    def test_a_raf_goes_through_in_both_highlight_modes_with_no_dark_border(self):
        for hlrecon in ("rgb", "cfa"):
            with self.subTest(hlrecon=hlrecon):
                rgb, meta = process_raw(str(SAMPLE), self.model, "cpu", 288, 48, hlrecon=hlrecon)
                self.assertEqual(rgb.shape[2], 3)
                self.assertTrue(np.isfinite(rgb).all())
                # A blend ramp starting at 0 left the first row and column unweighted, i.e. black.
                self.assertGreater(rgb[0].mean(), 0.5 * rgb[100].mean())
                self.assertGreater(rgb[:, 0].mean(), 0.5 * rgb[:, 100].mean())

    def test_a_stride_off_the_cfa_period_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            process_raw(str(SAMPLE), self.model, "cpu", 288, 50)
        self.assertIn("CFA period 6", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
