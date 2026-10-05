#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluator: the parts whose mistakes would silently change a gate."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval_truth as E  # noqa: E402


def _source(rgb: np.ndarray, name: str) -> E.Source:
    g = float(rgb[..., 1].mean())
    one = np.ones(3, dtype=np.float32)
    return E.Source(name, rgb.astype(np.float32), one, 0.18 / g, one, 0.0, 1.0)


class RippleTest(unittest.TestCase):
    def test_a_four_pixel_wave_is_measured_in_full(self) -> None:
        levels = E.GREY * 0.1
        for period in (4, 6):
            x = np.arange(48)
            wave = 1 + 0.02 * np.cos(2 * np.pi * x / period)
            values = levels.reshape(3, 1, 1) * np.broadcast_to(wave, (3, 48, 48))
            self.assertAlmostEqual(E.ripple_percent(values, levels), 2.0 / math.sqrt(2), places=3)

    def test_a_uniform_output_has_no_ripple(self) -> None:
        levels = E.GREY * 0.003
        self.assertLess(E.ripple_percent(np.broadcast_to(levels.reshape(3, 1, 1), (3, 24, 24)), levels), 1e-4)


class TilingTest(unittest.TestCase):
    def test_blend_weights_are_the_apps(self) -> None:
        w = E.blend_weights()
        self.assertAlmostEqual(float(w[0]), 1 / 25)
        self.assertAlmostEqual(float(w[23]), 24 / 25)
        self.assertEqual(float(w[24]), 1.0)
        for i in range(E.OVERLAP):                      # opposing ramps of neighbouring tiles sum to 1
            self.assertAlmostEqual(float(w[E.TILE - E.OVERLAP + i] + w[i]), 1.0, places=6)

    def test_seam_tiles_share_cfa_and_packing_phase_and_hold_the_band_inside(self) -> None:
        for origin in (E.SEAM_A, E.SEAM_B, E.SEAM_NATURAL_REF, E.SEAM_SHADOW_REF):
            self.assertEqual(origin % 12, 0)
        b0, b1 = E.SEAM_BAND
        for ref in (E.SEAM_NATURAL_REF, E.SEAM_SHADOW_REF):
            self.assertGreaterEqual(b0 - ref, E.INTERIOR)
            self.assertGreaterEqual(ref + E.TILE - b1, E.INTERIOR)
        self.assertGreaterEqual(E.SEAM_SHADOW_REF, E.SHADOW_FROM)     # the reference tile is all shadow
        self.assertGreaterEqual(b0 - E.SHADOW_FROM, 40)


class ExposureTest(unittest.TestCase):
    def test_floor_classification(self) -> None:
        self.assertEqual(E.eligible(np.array([2e-4, 1e-4, 5e-5])).tolist(), [True, True, False])

    def test_each_exposure_is_compared_on_the_same_tiles(self) -> None:
        rng = np.random.default_rng(0)
        textured = rng.random((288, 576, 3)) * 0.2                 # bright, hard to predict
        smooth = np.full((288, 576, 3), 0.003) * (1 + 0.01 * rng.random((288, 576, 3)))   # dark, easy
        ev = E.Evaluation([_source(textured, "bright"), _source(smooth, "dark")], "bayer")
        self.assertEqual(len(ev.tiles), 4)

        def predict(truth: np.ndarray, wb: np.ndarray) -> np.ndarray:   # exactly exposure-equivariant
            blurred: np.ndarray = 0.5 * (truth + np.roll(truth, 1, axis=3))
            return blurred

        as_shot = predict(ev.truth, ev.camera_wb)
        rows = {r["ev"]: r for r in E.measure_exposure(ev, predict, as_shot)}
        self.assertEqual(rows[-2]["eligible_tiles"], 4)
        self.assertEqual(rows[-6]["eligible_tiles"], 2)             # the dark source fell below the floor
        self.assertEqual(rows[-6]["below_floor_tiles"], 2)
        for r in rows.values():                                     # an equivariant model passes at every exposure
            self.assertAlmostEqual(r["eligible_db"], r["eligible_as_shot_db"], places=3)
        # ...which it would not if as-shot were scored on all four tiles
        all_tiles = E.accuracy_db(ev.display(as_shot)[..., E.IN, E.IN], ev.display(ev.truth)[..., E.IN, E.IN])
        self.assertGreater(abs(all_tiles - rows[-6]["eligible_as_shot_db"]), 0.5)


class SourcesTest(unittest.TestCase):
    def test_a_file_that_is_not_xtrans_is_skipped_and_the_rest_load(self) -> None:
        good = _source(np.full((288, 288, 3), 0.1), "good.RAF")

        def loader(path: str) -> E.Source:
            if path == "bayer.RAF":
                raise ValueError("Could not match CFA pattern to reference (period=6)")
            return good

        self.assertEqual(E.load_sources(["bayer.RAF", "good.RAF"], loader), [good])


class AccuracyTest(unittest.TestCase):
    def test_accuracy_db(self) -> None:
        a = np.zeros((2, 3, 4, 4))
        self.assertAlmostEqual(E.accuracy_db(a + 0.01, a), 40.0, places=6)


if __name__ == "__main__":
    unittest.main()
