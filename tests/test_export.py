#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exporter: one named version to the app's keys, manifest merged, other layouts refused."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import export_onnx  # noqa: E402
from cfa import CFA_REGISTRY, cfa_period, make_channel_masks, make_model_input  # noqa: E402
from export_onnx import app_key, export_checkpoint, export_selected, select, verify_tiles  # noqa: E402
from model import XTransUNet  # noqa: E402

PATCH = 48      # small tiles keep the test quick; a multiple of 12 as the packing needs


def _checkpoint(folder: Path, cfa_type: str, *, tag: str = "v7", version: str = "v7.0.0", seed: int = 0) -> str:
    torch.manual_seed(seed)
    model = XTransUNet(base_width=16, cfa_period=cfa_period(CFA_REGISTRY[cfa_type]))
    path = folder / f"{cfa_type}-{version}{f'-{seed}' if seed else ''}.pt"
    torch.save({"epoch": 3, "model": model.state_dict(), "base_width": 16, "stages": 2, "cfa_type": cfa_type,
                "architecture_tag": tag, "checkpoint_version": version, "checkpoint_major": int(version[1])}, path)
    return str(path)


def _digests(folder: Path) -> dict[str, str]:
    """File name -> sha256 of every file in folder; compares fast and diffs short when it fails."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}


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
            patcher = mock.patch.object(export_onnx, "verify", wraps=export_onnx.verify)
            verify = patcher.start()
            self.addCleanup(patcher.stop)

            manifest = export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=PATCH)
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
            # Each CFA type's export went through verify (spec 11.1).
            self.assertEqual(sorted(Path(c.args[0]).name for c in verify.call_args_list),
                             ["bayer-v7.0.0.pt", "xtrans-v7.0.0.pt"])

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


class VerifyTilesTest(unittest.TestCase):
    def test_the_verify_batch_holds_what_the_app_sends(self) -> None:
        tiles = verify_tiles(PATCH)
        names = list(tiles)
        mosaic = torch.cat(list(tiles.values()))
        self.assertGreaterEqual(mosaic.shape[0], 2)                     # the app sends tiles in batches
        x = make_model_input(mosaic, make_channel_masks(PATCH, PATCH, CFA_REGISTRY["xtrans"]))
        clipped = x[names.index("tile with clipped photosites")]
        self.assertEqual(float(clipped[0].max()), 1.0)                  # photosites at the clip level
        self.assertGreater(int(((clipped[0] > 0.95) & (clipped[0] < 1.0)).sum()), 0)   # and just below it
        self.assertGreater(float(clipped[4].min(dim=1).values.max()), 0.9)               # whole rows near clipping
        zero = x[names.index("all-zero tile")]
        self.assertEqual((float(zero[0].abs().max()), float(zero[4].abs().max())), (0.0, 0.0))
        self.assertLess(float(x[names.index("tile")][0].max()), 0.5)     # an ordinary tile, clip channel empty


class SafeExportTest(unittest.TestCase):
    """A wrong export must not land, and must not hide behind an unchanged checkpoint."""

    def _first_export(self, folder: Path) -> tuple[dict, Path]:
        registry = _registry(folder)
        out = folder / "out"
        export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=PATCH)
        return registry, out

    def test_a_failed_verify_leaves_the_previous_file_and_manifest_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            registry, out = self._first_export(Path(d))
            before = _digests(out)
            # A retrained checkpoint whose export comes out wrong: the file written holds other weights.
            registry["bayer"]["v7.0.0"]["stable"]["best"]["path"] = _checkpoint(Path(d), "bayer", seed=1)
            wrong = _checkpoint(Path(d), "bayer", seed=2)
            real_export = export_onnx.export
            with mock.patch.object(export_onnx, "export",
                                   side_effect=lambda ckpt, path, *a, **k: real_export(wrong, path, *a, **k)):
                with self.assertRaises(SystemExit) as ctx:
                    export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=PATCH)
            self.assertIn("disagree", str(ctx.exception))
            # Nothing replaced and nothing left behind: same files, byte for byte.
            self.assertEqual(_digests(out), before)

    def test_an_up_to_date_entry_is_verified_before_it_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            registry, out = self._first_export(Path(d))
            onnx_file = out / "bayer_w16_base.onnx"
            ckpt = registry["bayer"]["v7.0.0"]["stable"]["best"]["path"]
            with mock.patch.object(export_onnx, "verify", wraps=export_onnx.verify) as verify, \
                    mock.patch.object(export_onnx, "export", wraps=export_onnx.export) as export:
                export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=PATCH)
            export.assert_not_called()
            verify.assert_called_once_with(ckpt, str(onnx_file), PATCH)
            # The file on disk no longer matches its unchanged checkpoint: the run stops and names it.
            export_onnx.export(_checkpoint(Path(d), "bayer", seed=1), str(onnx_file), PATCH)
            with self.assertRaises(SystemExit) as ctx:
                export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=PATCH)
            self.assertIn(str(onnx_file), str(ctx.exception))

    def test_an_up_to_date_file_of_another_patch_size_says_how_to_replace_it(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as d:
            registry, out = self._first_export(Path(d))              # exported at PATCH
            before = _digests(out)
            with self.assertRaises(SystemExit) as ctx:
                export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=2 * PATCH)
            message = str(ctx.exception)
            for part in (str(out / "bayer_w16_base.onnx"), "different patch size", "--force"):
                self.assertIn(part, message)
            self.assertEqual(_digests(out), before)
            export_selected(registry, out, version="v7.0.0", cfa_type="bayer", patch_size=2 * PATCH, force=True)
            shape = ort.InferenceSession(str(out / "bayer_w16_base.onnx")).get_inputs()[0].shape
            self.assertEqual(list(shape)[1:], [5, 2 * PATCH, 2 * PATCH])

    def test_the_command_line_always_verifies_and_still_accepts_verify(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ckpt = _checkpoint(Path(d), "xtrans")
            for extra in ([], ["--verify"]):
                with self.subTest(extra=extra):
                    argv = ["export_onnx.py", "--checkpoint", ckpt, "--output-dir", str(Path(d) / "out"),
                            "--patch-size", str(PATCH), *extra]
                    with mock.patch.object(sys, "argv", argv), \
                            mock.patch.object(export_onnx, "verify", wraps=export_onnx.verify) as verify:
                        export_onnx.main()
                    verify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
