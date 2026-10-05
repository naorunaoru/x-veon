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

from losses import GAMMA, MEAN_FLOOR, TOE, DemosaicLoss, EncodedPSNR, encode, encode_pair  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
