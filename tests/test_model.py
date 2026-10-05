#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Packed model: layout, exposure behaviour on both sides of the floor, jitter, precision."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfa import CFA_REGISTRY, cfa_period, make_channel_masks, make_model_input  # noqa: E402
from model import ARCHITECTURE_TAG, MEAN_FLOOR, XTransUNet, count_parameters  # noqa: E402

SIZE = 144


def _input(cfa_type: str, level: float = 0.1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mosaic = torch.rand(2, 1, SIZE, SIZE, generator=g) * 2.0 * level
    return make_model_input(mosaic, make_channel_masks(SIZE, SIZE, CFA_REGISTRY[cfa_type]))


def _model(cfa_type: str, **kw: float) -> XTransUNet:
    torch.manual_seed(0)
    return XTransUNet(base_width=16, cfa_period=cfa_period(CFA_REGISTRY[cfa_type]), **kw).eval()  # type: ignore[arg-type]


class LayoutTest(unittest.TestCase):
    def test_output_shape_for_both_cfa_types_and_stage_counts(self) -> None:
        for cfa_type in ("xtrans", "bayer"):
            for stages in (2, 3):
                with self.subTest(cfa_type=cfa_type, stages=stages):
                    y = _model(cfa_type, stages=stages)(_input(cfa_type))
                    self.assertEqual(tuple(y.shape), (2, 3, SIZE, SIZE))

    def test_one_stage_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            XTransUNet(stages=1)

    def test_s_model_sizes_and_tag(self) -> None:
        self.assertEqual(ARCHITECTURE_TAG, "v7")
        self.assertEqual(count_parameters(_model("xtrans")), 560_835)
        self.assertEqual(count_parameters(_model("bayer")), 201_752)

    def test_packing_then_unpacking_returns_the_input(self) -> None:
        x = torch.rand(1, 5, 24, 24)
        for f in (2, 3):
            self.assertTrue(torch.equal(F.pixel_shuffle(F.pixel_unshuffle(x, f), f), x))

    def test_zero_correction_returns_the_measured_samples_in_their_own_channels(self) -> None:
        for cfa_type in ("xtrans", "bayer"):
            m = _model(cfa_type)
            assert m.head.bias is not None
            torch.nn.init.zeros_(m.head.weight)
            torch.nn.init.zeros_(m.head.bias)
            x = _input(cfa_type)
            self.assertTrue(torch.allclose(m(x), x[:, 0:1] * x[:, 1:4], rtol=1e-5, atol=1e-8))


class ExposureTest(unittest.TestCase):
    def test_above_the_floor_output_scales_exactly_with_the_mosaic(self) -> None:
        for cfa_type in ("xtrans", "bayer"):
            m = _model(cfa_type)
            x = _input(cfa_type)
            with torch.no_grad():
                base = m(x)
                for k in (1 / 64, 1 / 4, 4.0):
                    scaled = torch.cat([x[:, 0:1] * k, x[:, 1:]], dim=1)   # clip channel held fixed
                    self.assertGreaterEqual(float(scaled[:, 0].mean()), MEAN_FLOOR)
                    err = (m(scaled) - k * base).abs().max() / (k * base).abs().max()
                    self.assertLess(float(err), 1e-4, f"{cfa_type} k={k}")

    def test_below_the_floor_output_is_finite(self) -> None:
        m = _model("xtrans")
        masks = make_channel_masks(SIZE, SIZE, CFA_REGISTRY["xtrans"])
        with torch.no_grad():
            self.assertTrue(torch.isfinite(m(make_model_input(torch.zeros(1, 1, SIZE, SIZE), masks))).all())
            negative_mean = None
            for seed in range(50):
                noise = torch.randn(1, 1, SIZE, SIZE, generator=torch.Generator().manual_seed(seed)) * 0.005
                if float(noise.mean()) < 0:
                    negative_mean = noise
                    break
            assert negative_mean is not None
            self.assertTrue(torch.isfinite(m(make_model_input(negative_mean, masks))).all())
            self.assertTrue(torch.isfinite(m(make_model_input(-negative_mean, masks))).all())

    def test_output_is_continuous_across_the_floor(self) -> None:
        m = _model("xtrans")
        x = _input("xtrans")
        unit = x[:, 0:1] / x[:, 0:1].mean(dim=(2, 3), keepdim=True)
        with torch.no_grad():
            above = m(torch.cat([unit * MEAN_FLOOR * 1.01, x[:, 1:]], dim=1))
            below = m(torch.cat([unit * MEAN_FLOOR * 0.99, x[:, 1:]], dim=1))
        self.assertLess(float((above - below).abs().max() / above.abs().max()), 0.1)


class JitterAndPrecisionTest(unittest.TestCase):
    def test_jitter_acts_in_training_only(self) -> None:
        m = _model("bayer", gain_jitter_stops=3.0)
        x = _input("bayer")
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), m(x)))
            m.train()
            self.assertFalse(torch.equal(m(x), m(x)))

    def test_output_stays_float32_under_autocast(self) -> None:
        m = _model("bayer")
        x = _input("bayer")
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            y = m(x)
        self.assertEqual(y.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
