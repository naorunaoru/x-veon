#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exporter: one named version to the app's keys, manifest merged, other layouts refused."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfa import CFA_REGISTRY, cfa_period  # noqa: E402
from export_onnx import app_key, export_checkpoint, export_selected, select  # noqa: E402
from model import XTransUNet  # noqa: E402

PATCH = 48      # small tiles keep the test quick; a multiple of 12 as the packing needs


def _checkpoint(folder: Path, cfa_type: str, *, tag: str = "v7", version: str = "v7.0.0") -> str:
    torch.manual_seed(0)
    model = XTransUNet(base_width=16, cfa_period=cfa_period(CFA_REGISTRY[cfa_type]))
    path = folder / f"{cfa_type}-{version}.pt"
    torch.save({"epoch": 3, "model": model.state_dict(), "base_width": 16, "stages": 2, "cfa_type": cfa_type,
                "architecture_tag": tag, "checkpoint_version": version, "checkpoint_major": int(version[1])}, path)
    return str(path)


def _registry(folder: Path) -> dict:
    def entry(path: str) -> dict:
        return {"base_width": 16, "stable": {"best": {"path": path, "epoch": 4, "val_psnr": 30.0, "train_psnr": 29.0,
                                                       "val_loss": 0.03, "train_loss": 0.03}}}
    return {
        "xtrans": {"v7.0.0": entry(_checkpoint(folder, "xtrans")),
                   "v6.1.4": entry(_checkpoint(folder, "xtrans", tag="v6", version="v6.1.4"))},
        "bayer": {"v7.0.0": entry(_checkpoint(folder, "bayer"))},
    }


class SelectTest(unittest.TestCase):
    def test_a_version_must_be_named_and_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            registry = _registry(Path(d))
            with self.assertRaises(SystemExit) as ctx:
                select(registry, version=None)
            self.assertIn("--version is required", str(ctx.exception))
            with self.assertRaises(SystemExit):
                select(registry, version="v9.9.9")

    def test_only_the_named_version_is_selected_one_per_cfa_type(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            picked = select(_registry(Path(d)), version="v7.0.0")
            self.assertEqual(sorted(k for k, _, _ in picked), ["bayer_w16_base", "xtrans_w16_base"])
            self.assertTrue(all("v7.0.0" in path for _, path, _ in picked))
            self.assertEqual(app_key("xtrans", 16), "xtrans_w16_base")


class ExportTest(unittest.TestCase):
    def test_exporting_one_cfa_type_keeps_the_other_entries(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as d:
            registry = _registry(Path(d))
            out = Path(d) / "out"
            out.mkdir()
            other = {"file": "xtrans_w16_base.onnx", "source_sha256": "old", "base_width": 16}
            (out / "models.json").write_text(json.dumps({"xtrans_w16_base": other, "unrelated": {"x": 1}}))

            manifest = export_selected(registry, out, version="v7.0.0", cfa_type="bayer",
                                       patch_size=PATCH, verify_export=True)
            self.assertEqual(manifest["xtrans_w16_base"], other)          # untouched
            self.assertEqual(manifest["unrelated"], {"x": 1})
            entry = manifest["bayer_w16_base"]
            self.assertEqual((entry["file"], entry["checkpoint_version"], entry["stages"], entry["dtype"]),
                             ("bayer_w16_base.onnx", "v7.0.0", 2, "float32"))
            self.assertEqual(json.loads((out / "models.json").read_text()), manifest)

            session = ort.InferenceSession(str(out / "bayer_w16_base.onnx"))
            self.assertEqual(len(session.get_inputs()), 1)
            self.assertEqual(list(session.get_inputs()[0].shape)[1:], [5, PATCH, PATCH])
            zeros = np.zeros((2, 5, PATCH, PATCH), np.float32)             # an all-zero tile, batch of 2
            result = session.run(None, {"input": zeros})[0]
            self.assertEqual(result.shape, (2, 3, PATCH, PATCH))
            self.assertTrue(np.isfinite(result).all())

            manifest = export_selected(registry, out, version="v7.0.0", cfa_type="xtrans", patch_size=PATCH)
            self.assertEqual(manifest["xtrans_w16_base"]["checkpoint_version"], "v7.0.0")
            self.assertEqual(manifest["bayer_w16_base"], entry)

    def test_a_single_checkpoint_file_is_exported_under_its_app_key_and_merged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            out.mkdir()
            (out / "models.json").write_text(json.dumps({"xtrans_w16_base": {"file": "kept.onnx"}}))
            manifest = export_checkpoint(_checkpoint(Path(d), "bayer"), out, patch_size=PATCH)
            self.assertEqual(manifest["xtrans_w16_base"], {"file": "kept.onnx"})
            self.assertTrue((out / manifest["bayer_w16_base"]["file"]).exists())

    def test_a_checkpoint_of_another_architecture_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with self.assertRaises(SystemExit) as ctx:
                export_selected(_registry(Path(d)), out, version="v6.1.4", patch_size=PATCH)
            self.assertIn("architecture tag v6", str(ctx.exception))
            self.assertFalse((out / "xtrans_w16_base.onnx").exists())


if __name__ == "__main__":
    unittest.main()
