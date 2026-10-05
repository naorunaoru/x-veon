#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Training script: resume choice, strict loading, dataset record, and a two-epoch run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_record import claim_destination, new_record  # noqa: E402
from model import XTransUNet  # noqa: E402
from train import load_weights, pick_resume_checkpoint  # noqa: E402

MODULES = ["train.py", "model.py", "dataset.py", "dataset_record.py", "losses.py", "cfa.py",
           "checkpoint_registry.py", "dashboard.py", "observer.py", "state_server.py", "state_client.py"]


def _save(path: Path, epoch: int, mtime: float) -> None:
    torch.save({"epoch": epoch}, path)
    os.utime(path, (mtime, mtime))


class ResumeChoiceTest(unittest.TestCase):
    def test_continuing_a_run_takes_the_higher_saved_epoch_whatever_the_file_dates(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            _save(folder / "best.pt", epoch=97, mtime=1_000)        # older file, later epoch
            _save(folder / "latest.pt", epoch=89, mtime=2_000)
            self.assertEqual(pick_resume_checkpoint(folder, same_run=True), folder / "best.pt")
            _save(folder / "latest.pt", epoch=99, mtime=500)
            self.assertEqual(pick_resume_checkpoint(folder, same_run=True), folder / "latest.pt")
            _save(folder / "best.pt", epoch=99, mtime=3_000)        # a tie goes to latest.pt
            self.assertEqual(pick_resume_checkpoint(folder, same_run=True), folder / "latest.pt")

    def test_a_new_run_starts_from_best(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self.assertIsNone(pick_resume_checkpoint(folder, same_run=False))
            _save(folder / "latest.pt", epoch=99, mtime=2_000)
            self.assertEqual(pick_resume_checkpoint(folder, same_run=False), folder / "latest.pt")
            _save(folder / "best.pt", epoch=50, mtime=1_000)
            self.assertEqual(pick_resume_checkpoint(folder, same_run=False), folder / "best.pt")


class LoadWeightsTest(unittest.TestCase):
    def test_a_checkpoint_of_this_layout_loads(self) -> None:
        model = XTransUNet(base_width=16, cfa_period=2)
        load_weights(model, {"model": XTransUNet(base_width=16, cfa_period=2).state_dict(),
                             "architecture_tag": "v7"}, "x.pt")

    def test_another_layout_is_refused_with_a_plain_message(self) -> None:
        model = XTransUNet(base_width=16, cfa_period=2)
        v6_like = {"model": {"enc1.block.0.weight": torch.zeros(16, 9, 7, 7)}, "architecture_tag": "v6",
                   "checkpoint_version": "v6.1.4"}
        three_stage = {"model": XTransUNet(base_width=16, cfa_period=2, stages=3).state_dict()}
        for ckpt in (v6_like, three_stage, {"model": model.state_dict(), "architecture_tag": "v6"}):
            with self.assertRaises(SystemExit) as ctx:
                load_weights(model, ckpt, "old/best.pt")
            self.assertIn("old/best.pt is a checkpoint of another model layout", str(ctx.exception))


class TrainingRunTest(unittest.TestCase):
    """Runs train.py in a copy of the code, so the checkpoint registry it writes stays out of the repo."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.code = Path(self.tmp) / "code"
        self.code.mkdir()
        for name in MODULES:
            shutil.copy(REPO_ROOT / name, self.code / name)
        self.data = Path(self.tmp) / "data"
        self.data.mkdir()
        self.sock = f"/tmp/xv-train-{os.getpid()}.sock"            # AF_UNIX paths are short on macOS

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        if os.path.exists(self.sock):
            os.unlink(self.sock)

    def _images(self, n: int = 6) -> None:
        rng = np.random.default_rng(0)
        for k in range(n):
            np.save(self.data / f"img{k}.npy", rng.integers(500, 20000, (200, 232, 3), dtype=np.uint16))

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "train.py", *args, "--detach", "--socket-path", self.sock],
                              cwd=self.code, capture_output=True, text=True, timeout=600)

    def test_a_directory_without_a_build_record_stops_the_run(self) -> None:
        self._images(2)
        out = Path(self.tmp) / "out"
        r = self._run("--data-dir", str(self.data), "--cfa-type", "bayer", "--epochs", "1", "--output-dir", str(out))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(str(self.data), r.stdout + r.stderr)
        self.assertIn("build_info.json", r.stdout + r.stderr)
        self.assertFalse((out / "best.pt").exists())

    def test_two_epochs_then_continue(self) -> None:
        claim_destination(str(self.data), new_record({"source": "synthetic"}, str(REPO_ROOT)))   # as the builder does, before any data
        self._images()
        for cfa_type in ("bayer", "xtrans"):
            with self.subTest(cfa_type=cfa_type):
                out = Path(self.tmp) / f"out_{cfa_type}" / "v7.0.0"
                r = self._run("--data-dir", str(self.data), "--cfa-type", cfa_type, "--epochs", "2",
                              "--batch-size", "4", "--patch-size", "96", "--patches-per-image", "4",
                              "--val-split", "0.34", "--base-width", "16", "--stages", "2",
                              "--output-dir", str(out))
                self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
                ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual((ckpt["stages"], ckpt["architecture_tag"], ckpt["checkpoint_version"]),
                                 (2, "v7", "v7.0.0"))
                history = json.loads((out / "history.json").read_text())
                self.assertEqual([h["epoch"] for h in history], [0, 1])
                self.assertTrue(all(np.isfinite(h["val_psnr"]) and np.isfinite(h["train_psnr"]) for h in history))
                config = json.loads((out / "config.json").read_text())
                self.assertIn(str(self.data), config["datasets"])
                self.assertIsNone(config["resume"])
                saved_epoch = int(ckpt["epoch"])            # best.pt is the only checkpoint after two epochs
                # Continue the same run for one more epoch: it starts after the last saved epoch.
                r = self._run("--from-checkpoint", str(out), "--epochs", "3")
                self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
                self.assertEqual(json.loads((out / "config.json").read_text())["resume"], str(out / "best.pt"))
                continued = json.loads((out / "history.json").read_text())
                self.assertEqual([h["epoch"] for h in continued], [0, 1, 2])
                # The epochs up to the saved one are carried over as they were, not trained again.
                self.assertEqual(continued[:saved_epoch + 1], history[:saved_epoch + 1])


if __name__ == "__main__":
    unittest.main()
