#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The committed S configurations load as written and survive the command line that uses them."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import TrainConfig, parse_config  # noqa: E402

EXPECTED = {"mode": "train", "epochs": 400, "batch_size": 96, "patch_size": 144, "lr": 0.001, "warmup_epochs": 0,
            "val_split": 0.3, "patches_per_image": 16, "seed": 42, "l1_weight": 1.0, "recon_only": False,
            "noise_max": 0.005, "shot_noise_max": 0.0005, "olpf_sigma_max": 0.0, "downscale_prob": 0.75,
            "gain_jitter_stops": 3.0, "base_width": 16, "stages": 2, "group_images": 32, "amp": False,
            "checkpoint_version": "v7.0.0", "architecture_tag": "v7"}


class SConfigTest(unittest.TestCase):
    def test_files_hold_only_known_keys_and_the_agreed_values(self) -> None:
        known = {f.name for f in fields(TrainConfig)}
        for cfa_type in ("xtrans", "bayer"):
            path = REPO_ROOT / "configs" / f"s_{cfa_type}" / "config.json"
            data = json.loads(path.read_text())
            self.assertEqual(set(data) - known, set(), f"unknown keys in {path}")
            self.assertEqual(data["cfa_type"], cfa_type)
            for key, value in EXPECTED.items():
                self.assertEqual(data[key], value, f"{path}: {key}")
            for removed in ("data_dir", "output_dir", "workers", "cache_patches", "cache_gb"):
                self.assertNotIn(removed, data)
            self.assertEqual(data["patch_size"] % 12, 0)

    def test_the_documented_command_keeps_every_value(self) -> None:
        argv = ["train.py", "--from-checkpoint", str(REPO_ROOT / "configs" / "s_xtrans"), "--no-resume",
                "--data-dir", "/data/raise:1500", "/data/hf_ha:1500",
                "--output-dir", "checkpoints/xtrans/v7.0.0", "--cache-patches", "--cache-gb", "40", "--workers", "4"]
        with mock.patch.object(sys, "argv", argv):
            cfg, _, resume, _ = parse_config()
        self.assertIsNone(resume)
        for key, value in EXPECTED.items():
            self.assertEqual(getattr(cfg, key), value, key)      # the mode preset must not overwrite the file
        self.assertEqual(cfg.data_dir, ["/data/raise:1500", "/data/hf_ha:1500"])
        self.assertEqual((cfg.cache_patches, cfg.cache_gb, cfg.workers), (True, 40.0, 4))


if __name__ == "__main__":
    unittest.main()
