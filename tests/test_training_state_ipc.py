#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CPU-only smoke tests for the training state store + UNIX socket IPC.

Covers:
- TrainingStateStore reducer behavior (log truncation, update, new_best, error)
- StateServer ping/get_state/subscribe over an AF_UNIX socket
- state_client.py CLI subprocess paths for ping/get-state/subscribe
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, ClassVar, cast

# Make the repo importable when tests are run via `python -m unittest`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import EpochData  # noqa: E402
from observer import TrainingStateStore, sanitize_for_json  # noqa: E402
from state_server import StateServer  # noqa: E402


def _make_epoch(n: int, *, train_psnr=30.0, val_psnr=29.0) -> EpochData:
    return EpochData(
        epoch=n,
        train_psnr=train_psnr,
        val_psnr=val_psnr,
        train_components={"l1": 0.01, "total": 0.02},
        val_components={"l1": 0.012, "total": 0.023, "msssim": 0.995},
        lr=1e-3,
        epoch_time=1.5,
        train_time=1.2,
        val_time=0.3,
    )


class TrainingStateStoreTests(unittest.TestCase):
    def test_initial_snapshot_shape(self):
        s = TrainingStateStore(
            total_epochs=10, start_epoch=0, best_metric="psnr",
            loss_weights={"l1": 1.0}, config={"mode": "train"}, log_capacity=5,
        )
        snap = s.snapshot_dict()
        self.assertEqual(snap["status"], "starting")
        self.assertEqual(snap["total_epochs"], 10)
        self.assertEqual(snap["current_epoch"], 0)
        self.assertEqual(snap["best"]["metric"], "psnr")
        self.assertIsNone(snap["best"]["value"])
        self.assertIsNone(snap["latest_epoch"])
        self.assertEqual(snap["loss_weights"], {"l1": 1.0})
        self.assertEqual(snap["config"], {"mode": "train"})
        self.assertEqual(snap["recent_logs"], [])
        self.assertIsNone(snap["last_error"])

    def test_log_appends_and_truncates(self):
        s = TrainingStateStore(total_epochs=1, log_capacity=3)
        for i in range(5):
            s.log(f"m{i}")
        snap = s.snapshot_dict()
        msgs = [r["message"] for r in snap["recent_logs"]]
        self.assertEqual(msgs, ["m2", "m3", "m4"])
        levels = {r["level"] for r in snap["recent_logs"]}
        self.assertEqual(levels, {"INFO"})

    def test_update_sets_latest_and_current_epoch(self):
        s = TrainingStateStore(total_epochs=3)
        s.mark_started()
        s.update(_make_epoch(1))
        s.update(_make_epoch(2, train_psnr=31.5))
        snap = s.snapshot_dict()
        self.assertEqual(snap["current_epoch"], 2)
        self.assertEqual(snap["status"], "running")
        self.assertEqual(snap["latest_epoch"]["epoch"], 2)
        self.assertAlmostEqual(snap["latest_epoch"]["train_psnr"], 31.5)

    def test_new_best_event_updates_best(self):
        s = TrainingStateStore(total_epochs=3, best_metric="psnr")
        s.event("new_best", {"metric": "psnr", "value": 32.1, "epoch": 7})
        snap = s.snapshot_dict()
        self.assertEqual(snap["best"]["metric"], "psnr")
        self.assertAlmostEqual(snap["best"]["value"], 32.1)
        self.assertEqual(snap["best"]["epoch"], 7)

    def test_error_event_records_last_error_and_status(self):
        s = TrainingStateStore(total_epochs=3)
        s.mark_started()
        s.event("error", {
            "type": "RuntimeError", "message": "boom", "traceback": "tb",
        })
        snap = s.snapshot_dict()
        self.assertEqual(snap["status"], "error")
        self.assertIsNotNone(snap["last_error"])
        self.assertEqual(snap["last_error"]["type"], "RuntimeError")
        self.assertEqual(snap["last_error"]["message"], "boom")
        self.assertTrue(s.has_fatal_error)

    def test_training_done_defaults_to_completed(self):
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        s.event("training_done", {})
        self.assertEqual(s.snapshot_dict()["status"], "completed")

    def test_training_done_normalizes_unknown_status(self):
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        s.event("training_done", {"status": "running"})
        self.assertEqual(s.snapshot_dict()["status"], "completed")
        s2 = TrainingStateStore(total_epochs=1)
        s2.mark_started()
        s2.event("training_done", {"status": "bogus"})
        self.assertEqual(s2.snapshot_dict()["status"], "completed")

    def test_training_done_accepts_terminal_statuses(self):
        for status in ("completed", "interrupted", "error"):
            with self.subTest(status=status):
                s = TrainingStateStore(total_epochs=1)
                s.mark_started()
                s.event("training_done", {"status": status})
                self.assertEqual(s.snapshot_dict()["status"], status)

    def test_training_done_preserves_prior_interruption(self):
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        s.mark_interrupted()
        s.event("training_done", {"status": "completed"})
        self.assertEqual(s.snapshot_dict()["status"], "interrupted")

    def test_fatal_error_overrides_training_done_status(self):
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        s.event("error", {"type": "RuntimeError", "message": "boom"})
        # Even if training_done requests completed, _fatal forces error.
        s.event("training_done", {"status": "completed"})
        self.assertEqual(s.snapshot_dict()["status"], "error")

    def test_mark_completed_preserves_error_status(self):
        """mark_completed (called from StateServer.stop) must not overwrite
        a terminal status set by a prior training_done event."""
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        s.event("training_done", {"status": "error"})
        # _fatal was never raised, but status is already terminal.
        s.mark_completed()
        self.assertEqual(s.snapshot_dict()["status"], "error")

    def test_mark_completed_preserves_interrupted_status(self):
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        s.event("training_done", {"status": "interrupted"})
        s.mark_completed()
        self.assertEqual(s.snapshot_dict()["status"], "interrupted")

    def test_mark_completed_from_running_finalizes_completed(self):
        s = TrainingStateStore(total_epochs=1)
        s.mark_started()
        # No training_done emitted — mark_completed is the safety net.
        s.mark_completed()
        self.assertEqual(s.snapshot_dict()["status"], "completed")

    def test_new_best_rejects_nan_and_non_numeric(self):
        s = TrainingStateStore(total_epochs=1, best_metric="psnr")
        s.event("new_best", {"metric": "psnr", "value": float("nan"), "epoch": 1})
        self.assertIsNone(s.snapshot_dict()["best"]["value"])
        s.event("new_best", {"metric": "psnr", "value": "not-a-number", "epoch": 1})
        self.assertIsNone(s.snapshot_dict()["best"]["value"])
        s.event("new_best", {"metric": "psnr", "value": 31.0, "epoch": 2})
        self.assertAlmostEqual(s.snapshot_dict()["best"]["value"], 31.0)

    def test_update_with_nan_marks_fatal_and_sanitizes(self):
        s = TrainingStateStore(total_epochs=2)
        s.mark_started()
        bad = EpochData(
            epoch=1, train_psnr=float("nan"), val_psnr=20.0,
            train_components={"l1": float("inf")},
            val_components={"l1": 0.1}, lr=1e-3, epoch_time=0.1,
        )
        s.update(bad)
        self.assertTrue(s.has_fatal_error)
        snap = s.snapshot_dict()
        self.assertEqual(snap["status"], "error")
        self.assertIsNone(snap["latest_epoch"]["train_psnr"])  # NaN sanitized
        self.assertIsNone(snap["latest_epoch"]["train_components"]["l1"])

    def test_sanitize_for_json_handles_mixed(self):
        out = sanitize_for_json({
            "a": float("nan"),
            "b": [1, 2.0, float("inf")],
            "c": {"d": 3, "e": "s"},
            "bytes": b"hi",
        })
        # Verify JSON-serializable (strict=True rejects NaN/Inf)
        json.dumps(out, allow_nan=False)
        self.assertIsNone(out["a"])
        self.assertIsNone(out["b"][2])
        self.assertEqual(out["c"], {"d": 3, "e": "s"})
        self.assertEqual(out["bytes"], "hi")


# ── StateServer IPC smoke tests ──────────────────────────────────────────────


class _SockClient:
    """Minimal in-process client to exercise the state server."""

    def __init__(self, path: Path, timeout: float = 3.0):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(str(path))
        self._buf = bytearray()

    def send(self, obj):
        self.s.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    def recv_line(self, timeout: float = 3.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("recv_line timed out")
            self.s.settimeout(remaining)
            chunk = self.s.recv(4096)
            if not chunk:
                raise ConnectionError("server closed")
            self._buf.extend(chunk)
        nl = self._buf.find(b"\n")
        line = bytes(self._buf[:nl])
        del self._buf[: nl + 1]
        return cast(dict[str, Any], json.loads(line.decode("utf-8")))

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


class StateServerIPCTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = Path(self.tmp.name) / "t.sock"
        self.server = StateServer(
            self.sock_path,
            total_epochs=5, start_epoch=0, best_metric="psnr",
            loss_weights={"l1": 1.0}, config={"mode": "train"},
            log_capacity=10,
        )
        self.server.start()

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def test_socket_file_mode_is_user_only(self):
        mode = os.stat(self.sock_path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_ping(self):
        c = _SockClient(self.sock_path)
        try:
            c.send({"cmd": "ping"})
            resp = c.recv_line()
            self.assertEqual(resp, {"type": "pong"})
        finally:
            c.close()

    def test_get_state(self):
        self.server.log("starting up")
        c = _SockClient(self.sock_path)
        try:
            c.send({"cmd": "get_state"})
            resp = c.recv_line()
        finally:
            c.close()
        self.assertEqual(resp["type"], "state")
        state = resp["state"]
        self.assertIn("status", state)
        self.assertEqual(state["total_epochs"], 5)
        self.assertEqual(state["loss_weights"], {"l1": 1.0})
        self.assertEqual(
            [r["message"] for r in state["recent_logs"]],
            ["starting up"],
        )

    def test_subscribe_initial_snapshot_then_events(self):
        c = _SockClient(self.sock_path)
        try:
            c.send({"cmd": "subscribe"})
            # 1) initial snapshot
            first = c.recv_line()
            self.assertEqual(first["type"], "state")

            # 2) inject an epoch_done via update()
            self.server.update(_make_epoch(1, train_psnr=28.5, val_psnr=27.3))
            ev = c.recv_line(timeout=3.0)
            self.assertEqual(ev["type"], "event")
            self.assertEqual(ev["kind"], "epoch_done")
            self.assertEqual(ev["data"]["epoch"], 1)
            self.assertAlmostEqual(ev["data"]["entry"]["train_psnr"], 28.5)

            # 3) inject an explicit new_best
            self.server.event("new_best", {
                "metric": "psnr", "value": 30.0, "epoch": 1,
                "checkpoint": "/tmp/best.pt",
            })
            ev2 = c.recv_line(timeout=3.0)
            self.assertEqual(ev2["type"], "event")
            self.assertEqual(ev2["kind"], "new_best")
            self.assertAlmostEqual(ev2["data"]["value"], 30.0)
        finally:
            c.close()

    def test_subscribe_filter_only_new_best(self):
        c = _SockClient(self.sock_path)
        try:
            c.send({"cmd": "subscribe", "events": ["new_best"]})
            first = c.recv_line()
            self.assertEqual(first["type"], "state")

            # Inject an epoch_done — must NOT arrive.
            self.server.update(_make_epoch(1))
            # Inject a new_best — must arrive next.
            self.server.event("new_best", {
                "metric": "psnr", "value": 42.0, "epoch": 1,
            })
            ev = c.recv_line(timeout=3.0)
            self.assertEqual(ev["kind"], "new_best")
        finally:
            c.close()

    def test_subscribe_receives_training_done_event(self):
        c = _SockClient(self.sock_path)
        try:
            c.send({"cmd": "subscribe"})
            first = c.recv_line()
            self.assertEqual(first["type"], "state")
            self.server.event("training_done", {
                "status": "completed", "epochs_completed": 2,
            })
            ev = c.recv_line(timeout=3.0)
            self.assertEqual(ev["type"], "event")
            self.assertEqual(ev["kind"], "training_done")
            self.assertEqual(ev["data"]["status"], "completed")
            self.assertEqual(ev["data"]["epochs_completed"], 2)
        finally:
            c.close()
        # Snapshot should reflect terminal status after the event.
        self.assertEqual(self.server.snapshot_dict()["status"], "completed")

    def test_subscribe_receives_error_event(self):
        c = _SockClient(self.sock_path)
        try:
            c.send({"cmd": "subscribe"})
            first = c.recv_line()
            self.assertEqual(first["type"], "state")
            self.server.event("error", {
                "type": "RuntimeError", "message": "boom", "traceback": "tb",
            })
            ev = c.recv_line(timeout=3.0)
            self.assertEqual(ev["kind"], "error")
            self.assertEqual(ev["data"]["type"], "RuntimeError")
            self.assertEqual(ev["data"]["message"], "boom")
        finally:
            c.close()
        snap = self.server.snapshot_dict()
        self.assertEqual(snap["status"], "error")
        self.assertIsNotNone(snap["last_error"])
        self.assertTrue(self.server.has_fatal_error)

    def test_training_done_delivered_when_stop_follows_immediately(self):
        """Regression: stop() must flush queued lifecycle events to subscribers
        before tearing down their connections."""
        path = Path(self.tmp.name) / "flush.sock"
        srv = StateServer(path, total_epochs=1)
        srv.start()
        try:
            c = _SockClient(path)
            c.send({"cmd": "subscribe"})
            initial = c.recv_line()
            self.assertEqual(initial["type"], "state")

            # Emit training_done then stop the server with no delay. The
            # subscriber must still receive the training_done event.
            srv.event("training_done", {
                "status": "completed", "epochs_completed": 0,
            })
            srv.stop()

            ev = c.recv_line(timeout=3.0)
            self.assertEqual(ev["type"], "event")
            self.assertEqual(ev["kind"], "training_done")
            self.assertEqual(ev["data"]["status"], "completed")
            c.close()
        finally:
            # Ensure socket file is cleaned up even on test failure.
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def test_error_delivered_when_stop_follows_immediately(self):
        path = Path(self.tmp.name) / "flush_err.sock"
        srv = StateServer(path, total_epochs=1)
        srv.start()
        try:
            c = _SockClient(path)
            c.send({"cmd": "subscribe"})
            self.assertEqual(c.recv_line()["type"], "state")

            srv.event("error", {"type": "RuntimeError", "message": "boom"})
            srv.event("training_done", {
                "status": "error", "epochs_completed": 0,
            })
            srv.stop()

            kinds = []
            for _ in range(2):
                kinds.append(c.recv_line(timeout=3.0)["kind"])
            self.assertEqual(kinds, ["error", "training_done"])
            c.close()
        finally:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def test_stop_removes_socket_file(self):
        # Bring up + tear down a separate server so we can assert cleanup
        # without disturbing self.server.
        path = Path(self.tmp.name) / "t2.sock"
        srv = StateServer(path, total_epochs=1)
        srv.start()
        self.assertTrue(path.exists())
        srv.stop()
        # give the accept thread a beat to unwind
        for _ in range(20):
            if not path.exists():
                break
            time.sleep(0.05)
        self.assertFalse(path.exists(), "socket file should be unlinked on stop()")


# ── state_client.py CLI smoke tests (subprocess) ─────────────────────────────


class StateClientCLITests(unittest.TestCase):
    """Exercise the public CLI to catch regressions in the protocol."""

    tmp: ClassVar[tempfile.TemporaryDirectory[str]]
    sock_path: ClassVar[Path]
    server: ClassVar[StateServer]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sock_path = Path(cls.tmp.name) / "cli.sock"
        cls.server = StateServer(
            cls.sock_path,
            total_epochs=3, best_metric="psnr",
            loss_weights={"l1": 1.0}, config={"mode": "train"},
        )
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.tmp.cleanup()

    def _run(self, *args, timeout=5) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "state_client.py"),
             "--socket", str(self.sock_path), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )

    def test_cli_ping(self):
        r = self._run("ping")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(json.loads(r.stdout.strip()), {"type": "pong"})

    def test_cli_get_state(self):
        self.server.log("hello")
        r = self._run("get-state")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        payload = json.loads(r.stdout.strip())
        self.assertEqual(payload["type"], "state")
        self.assertEqual(payload["state"]["total_epochs"], 3)

    def test_cli_subscribe_receives_event(self):
        """Start subscribe, then inject a new_best, then close server-side."""
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "state_client.py"),
             "--socket", str(self.sock_path), "subscribe",
             "--events", "new_best"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert proc.stdout is not None
            # Wait for the initial snapshot line to be flushed.
            initial = proc.stdout.readline()
            self.assertTrue(initial, "no initial snapshot line")
            init = json.loads(initial)
            self.assertEqual(init["type"], "state")

            # Inject a new_best event; subscriber must receive it.
            self.server.event("new_best", {
                "metric": "psnr", "value": 33.3, "epoch": 2,
            })

            line = proc.stdout.readline()
            self.assertTrue(line, "no event line received")
            payload = json.loads(line)
            self.assertEqual(payload["type"], "event")
            self.assertEqual(payload["kind"], "new_best")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            for f in (proc.stdout, proc.stderr):
                if f is not None:
                    try:
                        f.close()
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()
