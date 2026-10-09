#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Encoded loss and pooled metric."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses import GAMMA, MEAN_FLOOR, TOE, DemosaicLoss, EncodedPSNR, FFTMagnitudeLoss, encode, encode_pair  # noqa: E402


def _pair(seed: int = 0, level: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(4, 3, 24, 24, generator=g) * 2 * level
    pred = target + torch.randn(4, 3, 24, 24, generator=g) * 0.1 * level
    return pred, target


class EncodeTest(unittest.TestCase):
    def test_curve_values(self) -> None:
        one = torch.ones(1)
        self.assertAlmostEqual(float(encode(torch.zeros(1), one)), 0.0, places=7)
        self.assertAlmostEqual(float(encode(one, one)), (1 + TOE) ** GAMMA - TOE ** GAMMA, places=6)
        slope = GAMMA * TOE ** (GAMMA - 1)
        self.assertAlmostEqual(float(encode(torch.tensor([-0.5]), one)), -0.5 * slope, places=5)

    def test_gradient_is_finite_at_zero_and_below(self) -> None:
        u = torch.tensor([-0.5, -1e-3, 0.0, 0.3], requires_grad=True)
        encode(u, torch.ones(1)).sum().backward()
        assert u.grad is not None
        self.assertTrue(torch.isfinite(u.grad).all())
        slope = GAMMA * TOE ** (GAMMA - 1)
        self.assertTrue(torch.allclose(u.grad[:3], torch.full((3,), slope), rtol=1e-4))


class LossTest(unittest.TestCase):
    def test_brightness_does_not_change_the_loss_above_the_floor(self) -> None:
        crit = DemosaicLoss()
        pred, target = _pair()
        base, _ = crit(pred, target)
        for k in (1 / 64, 1 / 8, 4.0):
            self.assertGreaterEqual(float((target * k).mean(dim=(1, 2, 3)).min()), MEAN_FLOOR)
            scaled, _ = crit(pred * k, target * k)
            self.assertAlmostEqual(float(scaled), float(base), delta=1e-5 * float(base))

    def test_finite_value_and_gradient_in_the_awkward_cases(self) -> None:
        crit = DemosaicLoss()
        _, target = _pair()
        for pred, tgt in ((torch.zeros_like(target), target),
                          (-target, target),
                          (torch.randn_like(target) * 0.005, torch.zeros_like(target))):
            pred = pred.clone().requires_grad_(True)
            loss, _ = crit(pred, tgt)
            loss.backward()
            assert pred.grad is not None
            self.assertTrue(math.isfinite(float(loss.detach())))
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_components_and_recon_only(self) -> None:
        pred, target = _pair()
        _, comps = DemosaicLoss()(pred, target)
        self.assertEqual(set(comps), {"l1", "total"})
        masks = torch.zeros(1, 3, 24, 24)
        masks[:, 1] = 1.0
        loss, comps = DemosaicLoss(recon_only=True)(pred, target, channel_masks=masks)
        self.assertEqual(set(comps), {"l1", "l1_recon", "l1_known", "total"})
        self.assertAlmostEqual(float(loss), float(comps["l1_recon"] + 0.1 * comps["l1_known"]), places=6)
        _, comps = DemosaicLoss(use_huber=True, color_bias_weight=0.1)(pred, target)
        self.assertEqual(set(comps), {"huber", "color_bias", "total"})


class MetricTest(unittest.TestCase):
    def test_score_does_not_depend_on_batch_grouping(self) -> None:
        pred, target = _pair()
        pred2, target2 = _pair(seed=1, level=0.01)       # a much darker, differently scored batch
        whole, parts = EncodedPSNR(), EncodedPSNR()
        whole.update(torch.cat([pred, pred2]), torch.cat([target, target2]))
        for p, t in ((pred[:1], target[:1]), (pred[1:], target[1:]), (pred2, target2)):
            parts.update(p, t)
        self.assertAlmostEqual(whole.value(), parts.value(), places=3)
        enc_p, enc_t = encode_pair(torch.cat([pred, pred2]), torch.cat([target, target2]))
        expected = -10 * math.log10(float(((enc_p - enc_t) ** 2).mean()))
        self.assertAlmostEqual(whole.value(), expected, places=3)

    def test_identical_inputs_and_empty_metric(self) -> None:
        _, target = _pair()
        m = EncodedPSNR()
        self.assertTrue(math.isnan(m.value()))
        m.update(target, target)
        self.assertTrue(math.isfinite(m.value()))
        self.assertAlmostEqual(m.value(), 100.0, places=3)



def _texture(seed: int = 0, size: int = 144) -> torch.Tensor:
    """A patch with fine texture at several scales, values around 0.1."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(2, 3, size // 4, size // 4, generator=g)
    coarse = torch.nn.functional.interpolate(base, size=(size, size), mode="bilinear", align_corners=False)
    return 0.05 + 0.05 * coarse + 0.03 * torch.rand(2, 3, size, size, generator=g)


def _blur(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([1.0, 2.0, 1.0]) / 4
    k2 = (k[:, None] * k[None, :]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), k2, groups=3)


class FFTMagnitudeTest(unittest.TestCase):
    """The magnitude-spectrum term on encoded values."""

    def test_identical_inputs_cost_nothing_and_brightness_changes_nothing(self) -> None:
        target = _texture()
        term = FFTMagnitudeLoss()
        self.assertAlmostEqual(float(term(*encode_pair(target, target))), 0.0, places=6)
        base = float(term(*encode_pair(_blur(target), target)))
        self.assertGreater(base, 0.0)
        for k in (1 / 64, 16.0):                                    # 6 stops down, 4 up
            self.assertAlmostEqual(float(term(*encode_pair(_blur(target) * k, target * k))), base, places=5)

    def test_missing_texture_costs_but_its_position_does_not(self) -> None:
        target = _texture()
        term = FFTMagnitudeLoss()
        moved = float(term(*encode_pair(torch.roll(target, shifts=1, dims=3), target)))   # one pixel over
        blurred = float(term(*encode_pair(_blur(target), target)))
        self.assertLess(moved, 1e-5)
        self.assertGreater(blurred, 100 * max(moved, 1e-7))

    def test_finite_value_and_gradient_on_flat_and_black_patches(self) -> None:
        for target in (torch.full((2, 3, 144, 144), 0.2), torch.zeros(2, 3, 144, 144)):
            pred = target.clone().requires_grad_(True)                # a flat spectrum: the worst case for |F|
            value = FFTMagnitudeLoss()(*encode_pair(pred, target))
            value.backward()
            assert pred.grad is not None
            self.assertTrue(torch.isfinite(value) and torch.isfinite(pred.grad).all())

    def test_demosaic_loss_adds_the_term_with_its_weight(self) -> None:
        target = _texture()
        pred = _blur(target)
        plain, components = DemosaicLoss()(pred, target)
        self.assertNotIn("fft", components)
        total, components = DemosaicLoss(fft_weight=0.5)(pred, target)
        self.assertAlmostEqual(float(total), float(plain) + 0.5 * float(components["fft"]), places=6)
        # The term sees the same encoded values as the L1 term, not the raw ones.
        on_encoded = float(FFTMagnitudeLoss()(*encode_pair(pred, target)))
        on_raw = float(FFTMagnitudeLoss()(pred, target))
        self.assertAlmostEqual(float(components["fft"]), on_encoded, places=6)
        self.assertNotAlmostEqual(on_encoded, on_raw, places=3)


if __name__ == "__main__":
    unittest.main()
