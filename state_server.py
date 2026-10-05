#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
UNIX-socket training state server.

Implements a ``TrainingObserver`` that accepts client connections on an
``AF_UNIX`` stream socket and speaks a newline-delimited JSON protocol:

Requests (client → server):
    {"cmd":"ping"}
    {"cmd":"get_state"}
    {"cmd":"subscribe"}
    {"cmd":"subscribe","events":["epoch_done","new_best"]}

Responses (server → client):
    {"type":"pong"}
    {"type":"state","state":{...}}
    {"type":"event","kind":"epoch_done","ts":...,"data":{...}}
    {"type":"error","message":"..."}
"""

from __future__ import annotations

import errno
import json
import os
import queue
import socket
import stat
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from dashboard import EpochData
from observer import (
    EVENT_KINDS,
    TrainingLogRecord,
    TrainingStateStore,
    epoch_data_to_dict,
    sanitize_for_json,
)


# Max bytes for a single request line — more than enough for all commands.
_MAX_REQUEST_BYTES = 64 * 1024

# Upper bound on queued events per subscriber. Slow clients get dropped
# silently rather than blocking training.
_SUBSCRIBER_QUEUE_SIZE = 256


class _Subscriber:
    """A single subscriber connection with its event filter and queue."""

    __slots__ = ("conn", "addr", "events", "queue", "closed")

    def __init__(self, conn: socket.socket, events: set[str] | None):
        self.conn = conn
        self.addr = ""
        # events=None means "all"
        self.events: set[str] | None = events
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=_SUBSCRIBER_QUEUE_SIZE,
        )
        self.closed = False


def _write_line(conn: socket.socket, obj: dict[str, Any]) -> None:
    """Send a single JSON object followed by newline. Raises on socket error."""
    line = json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n"
    conn.sendall(line.encode("utf-8"))


class StateServer:
    """Headless observer backed by a UNIX-socket broadcast server.

    Usage:
        server = StateServer("/tmp/train.sock", total_epochs=100)
        server.start()
        server.log("hello")
        server.update(EpochData(...))
        server.event("new_best", {"metric":"psnr","value":32.1,"epoch":3})
        server.stop()
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        total_epochs: int,
        start_epoch: int = 0,
        best_metric: str = "psnr",
        loss_weights: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        log_capacity: int = 50,
        backlog: int = 8,
    ):
        self.socket_path = Path(socket_path)
        self._backlog = backlog

        self._store = TrainingStateStore(
            total_epochs=total_epochs,
            start_epoch=start_epoch,
            best_metric=best_metric,
            loss_weights=loss_weights,
            config=config,
            log_capacity=log_capacity,
        )

        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._sub_lock = threading.Lock()
        self._subscribers: list[_Subscriber] = []

    # ── Observer lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        self._store.mark_started()
        self._bind_socket()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="state-server-accept", daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._store.mark_completed()

        # Close listening socket — this unblocks accept().
        if self._server_sock is not None:
            try:
                self._server_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        # Signal all subscribers to flush + close. The sentinel is placed
        # *after* any prior events (e.g. training_done / error broadcast
        # just before stop()) so subscribers drain those before exiting.
        with self._sub_lock:
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub.queue.put_nowait(None)
            except queue.Full:
                pass

        # Give subscribers a brief grace period to flush pending events
        # and exit cleanly via the sentinel. Then force-close any that
        # are stuck (e.g. blocked on a slow client's kernel buffer).
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._sub_lock:
                if not self._subscribers:
                    break
            time.sleep(0.02)
        with self._sub_lock:
            stragglers = list(self._subscribers)
        for sub in stragglers:
            try:
                sub.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None

        # Remove the socket file.
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass

    # ── Observer log/update/event ─────────────────────────────────────────

    def log(self, message: str, level: str = "INFO") -> None:
        self._store.log(message, level)

    def update(self, data: EpochData) -> None:
        self._store.update(data)
        self._broadcast("epoch_done", {
            "epoch": data.epoch,
            "entry": epoch_data_to_dict(data),
        })

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self._store.event(kind, payload)
        self._broadcast(kind, sanitize_for_json(payload) if payload else {})

    @property
    def has_fatal_error(self) -> bool:
        return self._store.has_fatal_error

    # ── Public snapshot accessor (useful for tests) ───────────────────────

    def snapshot_dict(self) -> dict[str, Any]:
        return self._store.snapshot_dict()

    @property
    def recent_logs(self) -> list[TrainingLogRecord]:
        """Snapshot of the recent log ring. Safe for external readers."""
        return self._store.recent_logs()

    # ── Resume seeding (non-broadcasting) ─────────────────────────────────

    def seed_resume(
        self,
        *,
        start_epoch: int,
        history: Iterable[EpochData] | None = None,
        best: tuple[str, float, int | None] | None = None,
    ) -> None:
        """Seed snapshot state from a resumed run WITHOUT broadcasting.

        Applies ``start_epoch``, replays any prior epoch data into the
        snapshot via the store's ``update()`` (which does not broadcast),
        and optionally records a best-metric record. Any subscriber that
        connects after this call will receive the updated snapshot, but
        subscribers already connected do NOT receive fake live events
        for history that has already happened.
        """
        self._store.set_start_epoch(start_epoch)
        if history:
            for entry in history:
                self._store.update(entry)
        if best is not None:
            metric, value, epoch = best
            self._store.set_best(metric, value, epoch)

    # ── Socket setup ──────────────────────────────────────────────────────

    def _bind_socket(self) -> None:
        path = self.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unlink stale socket file safely: only if it's a socket (never a real file).
        if path.exists() or path.is_symlink():
            try:
                st = path.lstat()
                if stat.S_ISSOCK(st.st_mode) or stat.S_ISLNK(st.st_mode):
                    path.unlink()
                else:
                    raise OSError(
                        f"refusing to unlink non-socket path {path}"
                    )
            except FileNotFoundError:
                pass

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(path))
        except OSError:
            sock.close()
            raise
        try:
            os.chmod(str(path), 0o600)
        except OSError:
            pass
        sock.listen(self._backlog)
        sock.settimeout(0.5)  # periodic wakeup so accept loop can exit
        self._server_sock = sock

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop_event.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError as e:
                if e.errno in (errno.EBADF, errno.EINVAL):
                    return
                if self._stop_event.is_set():
                    return
                traceback.print_exc()
                continue

            conn.settimeout(2.0)
            t = threading.Thread(
                target=self._handle_client,
                args=(conn,),
                name="state-server-client",
                daemon=True,
            )
            t.start()

    # ── Client handling ───────────────────────────────────────────────────

    def _read_line(self, conn: socket.socket) -> str | None:
        """Read a single newline-delimited UTF-8 line, or None on EOF."""
        buf = bytearray()
        while True:
            if len(buf) > _MAX_REQUEST_BYTES:
                return None
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:
                return buf.decode("utf-8", errors="replace") if buf else None
            buf.extend(chunk)
            nl = buf.find(b"\n")
            if nl != -1:
                line = buf[:nl].decode("utf-8", errors="replace")
                return line

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            raw = self._read_line(conn)
            if raw is None or not raw.strip():
                self._safe_write(conn, {"type": "error", "message": "empty request"})
                conn.close()
                return
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as e:
                self._safe_write(conn, {"type": "error", "message": f"bad json: {e}"})
                conn.close()
                return

            cmd = req.get("cmd") if isinstance(req, dict) else None
            if cmd == "ping":
                self._safe_write(conn, {"type": "pong"})
                conn.close()
                return
            if cmd == "get_state":
                self._safe_write(conn, {
                    "type": "state",
                    "state": self._store.snapshot_dict(),
                })
                conn.close()
                return
            if cmd == "subscribe":
                events_filter = req.get("events") if isinstance(req, dict) else None
                filt: set[str] | None
                if events_filter is None:
                    filt = None
                elif isinstance(events_filter, list):
                    filt = {str(x) for x in events_filter}
                else:
                    self._safe_write(conn, {
                        "type": "error",
                        "message": "events must be a list",
                    })
                    conn.close()
                    return
                self._run_subscriber(conn, filt)
                return

            self._safe_write(conn, {
                "type": "error", "message": f"unknown cmd: {cmd!r}",
            })
            conn.close()
        except Exception:
            traceback.print_exc()
            try:
                conn.close()
            except OSError:
                pass

    def _run_subscriber(self, conn: socket.socket, events: set[str] | None) -> None:
        sub = _Subscriber(conn, events)
        with self._sub_lock:
            self._subscribers.append(sub)

        try:
            # Send initial snapshot immediately.
            self._safe_write(conn, {
                "type": "state",
                "state": self._store.snapshot_dict(),
            })
            # Remove the per-connection read timeout for the sending side.
            conn.settimeout(None)

            # Drain the queue until the None sentinel arrives (set by stop())
            # or the queue is empty after stop_event is set. This guarantees
            # that lifecycle events broadcast just before stop() (training_done,
            # error) are flushed to the subscriber.
            while True:
                try:
                    item = sub.queue.get(timeout=0.5)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                if item is None:
                    break
                try:
                    _write_line(conn, item)
                except OSError:
                    break
        finally:
            sub.closed = True
            with self._sub_lock:
                try:
                    self._subscribers.remove(sub)
                except ValueError:
                    pass
            try:
                conn.close()
            except OSError:
                pass

    def _safe_write(self, conn: socket.socket, obj: dict[str, Any]) -> None:
        try:
            _write_line(conn, obj)
        except OSError:
            pass

    # ── Broadcast ────────────────────────────────────────────────────────

    def _broadcast(self, kind: str, data: dict[str, Any]) -> None:
        if kind not in EVENT_KINDS:
            # Still broadcast unknown kinds — lets callers extend without
            # editing this list — but skip if empty string.
            if not kind:
                return
        msg = {
            "type": "event",
            "kind": kind,
            "ts": time.time(),
            "data": data,
        }
        with self._sub_lock:
            subs = list(self._subscribers)
        for sub in subs:
            if sub.events is not None and kind not in sub.events:
                continue
            try:
                sub.queue.put_nowait(msg)
            except queue.Full:
                # Drop on overflow — slow clients must not block training.
                pass
