#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The five-channel model input is assembled in one place."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfa import CFA_REGISTRY, make_channel_masks, make_model_input  # noqa: E402


class MakeModelInputTest(unittest.TestCase):
    def test_layout_and_masks(self) -> None:
        masks = make_channel_masks(12, 12, CFA_REGISTRY["xtrans"])
        mosaic = torch.rand(3, 1, 12, 12)
        x = make_model_input(mosaic, masks)
        self.assertEqual(tuple(x.shape), (3, 5, 12, 12))
        self.assertTrue(torch.equal(x[:, 0:1], mosaic))
        self.assertTrue(torch.equal(x[1, 1:4], masks))
        self.assertTrue(torch.equal(make_model_input(mosaic, masks.unsqueeze(0)), x))

    def test_clip_ratio_matches_the_app_formula_at_clip_level_one(self) -> None:
        values = torch.tensor([-0.2, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.7])
        mosaic = values.view(1, 1, 1, -1).expand(1, 1, 2, -1).contiguous()
        masks = make_channel_masks(2, 8, CFA_REGISTRY["bayer"])
        clip = make_model_input(mosaic, masks)[0, 4, 0]
        app = torch.clamp(torch.clamp(values / 1.0, max=1.0) * 2.0 - 1.0, min=0.0)  # tile-blend-gpu.ts
        self.assertTrue(torch.allclose(clip, app))
        self.assertTrue(torch.allclose(clip, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.8, 1.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
