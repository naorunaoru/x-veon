#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Shared observer + state reducer contract for training output sinks.

The training loop talks to a single ``TrainingObserver``; concrete sinks
(dashboard, socket server, multicast) implement this interface. A
``TrainingStateStore`` owns a bounded, JSON-serializable snapshot derived
from log/update/event calls — useful for both foreground and detached modes.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, cast, runtime_checkable

from dashboard import EpochData

# Events emitted by the training loop (documented contract).
EVENT_KINDS: tuple[str, ...] = (
    "epoch_done",
    "new_best",
    "training_done",
    "error",
)

STATUSES: tuple[str, ...] = (
    "starting",
    "running",
    "completed",
    "interrupted",
    "error",
)

# Only these statuses are valid as the *final* outcome reported by
# a training_done event. Anything else is coerced to "completed".
_FINAL_STATUSES: frozenset[str] = frozenset({"completed", "interrupted", "error"})


# ── JSON sanitization ────────────────────────────────────────────────────────

def _sanitize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_sanitize_value(v) for v in value]
    return str(value)


def sanitize_for_json(value: Any) -> Any:
    """Return a JSON-safe copy of ``value`` (NaN/Inf → None, non-primitives stringified)."""
    return _sanitize_value(value)


# ── Snapshot data types ──────────────────────────────────────────────────────

@dataclass
class TrainingLogRecord:
    ts: float
    level: str
    message: str


@dataclass
class BestMetricRecord:
    metric: str
    value: float | None
    epoch: int | None


@dataclass
class TrainingStateSnapshot:
    status: str = "starting"
    total_epochs: int = 0
    start_epoch: int = 0
    current_epoch: int = 0
    start_time: float | None = None
    elapsed_seconds: float = 0.0
    best: BestMetricRecord = field(
        default_factory=lambda: BestMetricRecord(metric="psnr", value=None, epoch=None)
    )
    latest_epoch: dict[str, Any] | None = None
    loss_weights: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    recent_logs: list[TrainingLogRecord] = field(default_factory=list)
    last_error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = cast(dict[str, Any], sanitize_for_json(asdict(self)))
        if self.start_time is not None:
            data["elapsed_seconds"] = max(0.0, time.time() - self.start_time)
        return data


# ── Observer protocol ────────────────────────────────────────────────────────

@runtime_checkable
class TrainingObserver(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def log(self, message: str, level: str = "INFO") -> None: ...
    def update(self, data: EpochData) -> None: ...
    def event(self, kind: str, payload: dict[str, Any]) -> None: ...

    @property
    def has_fatal_error(self) -> bool: ...


# ── State reducer ────────────────────────────────────────────────────────────

def epoch_data_to_dict(data: EpochData) -> dict[str, Any]:
    """Convert an EpochData to a JSON-safe dict snapshot entry."""
    return cast(dict[str, Any], sanitize_for_json({
        "epoch": data.epoch,
        "train_psnr": data.train_psnr,
        "val_psnr": data.val_psnr,
        "train_components": data.train_components,
        "val_components": data.val_components,
        "lr": data.lr,
        "epoch_time": data.epoch_time,
        "train_time": data.train_time,
        "val_time": data.val_time,
    }))


class TrainingStateStore:
    """Thread-safe state reducer driven by log/update/event calls.

    The store owns a ``TrainingStateSnapshot`` and a bounded ring of recent
    log records. ``snapshot_dict()`` returns a JSON-safe copy for network
    serialization; NaN/Inf are converted to None.
    """

    def __init__(
        self,
        *,
        total_epochs: int = 0,
        start_epoch: int = 0,
        best_metric: str = "psnr",
        loss_weights: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        log_capacity: int = 50,
    ):
        self._lock = threading.Lock()
        self._log_capacity = log_capacity
        self._logs: deque[TrainingLogRecord] = deque(maxlen=log_capacity)
        self._fatal = False

        self.snapshot = TrainingStateSnapshot(
            status="starting",
            total_epochs=total_epochs,
            start_epoch=start_epoch,
            current_epoch=start_epoch,
            best=BestMetricRecord(metric=best_metric, value=None, epoch=None),
            loss_weights=dict(loss_weights or {}),
            config=sanitize_for_json(dict(config or {})),
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def mark_started(self) -> None:
        with self._lock:
            self.snapshot.start_time = time.time()
            self.snapshot.status = "running"

    def mark_completed(self, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self._fatal:
                self.snapshot.status = "error"
            elif self.snapshot.status not in _FINAL_STATUSES:
                self.snapshot.status = "completed"

    def mark_interrupted(self) -> None:
        with self._lock:
            self.snapshot.status = "interrupted"

    @property
    def has_fatal_error(self) -> bool:
        with self._lock:
            return self._fatal

    # ── Log/update/event ──────────────────────────────────────────────────

    def log(self, message: str, level: str = "INFO") -> None:
        record = TrainingLogRecord(
            ts=time.time(), level=level.upper(), message=str(message),
        )
        with self._lock:
            self._logs.append(record)
            self.snapshot.recent_logs = list(self._logs)

    def update(self, data: EpochData) -> None:
        entry = epoch_data_to_dict(data)
        with self._lock:
            if self.snapshot.status == "starting":
                self.snapshot.status = "running"
            self.snapshot.current_epoch = data.epoch
            self.snapshot.latest_epoch = entry

            bad = False
            for v in list(data.train_components.values()) + list(data.val_components.values()) + \
                    [data.train_psnr, data.val_psnr]:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    bad = True
                    break
            if bad:
                self._fatal = True
                self.snapshot.status = "error"

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        safe = sanitize_for_json(payload) if isinstance(payload, dict) else {}
        with self._lock:
            if kind == "new_best":
                metric = safe.get("metric") or self.snapshot.best.metric
                value = safe.get("value")
                epoch = safe.get("epoch")
                if isinstance(value, (int, float)) and not (
                    isinstance(value, float) and math.isnan(value)
                ):
                    self.snapshot.best = BestMetricRecord(
                        metric=str(metric),
                        value=float(value),
                        epoch=int(epoch) if epoch is not None else None,
                    )
            elif kind == "training_done":
                if self._fatal:
                    self.snapshot.status = "error"
                elif self.snapshot.status != "interrupted":
                    requested = safe.get("status", "completed")
                    self.snapshot.status = (
                        requested if requested in _FINAL_STATUSES else "completed"
                    )
            elif kind == "error":
                self._fatal = True
                self.snapshot.status = "error"
                err = {
                    "type": safe.get("type", "Error"),
                    "message": safe.get("message", ""),
                    "traceback": safe.get("traceback", ""),
                    "ts": time.time(),
                }
                self.snapshot.last_error = err

    # ── Snapshot access ───────────────────────────────────────────────────

    def snapshot_dict(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot.to_dict()

    def recent_logs(self) -> list[TrainingLogRecord]:
        """Return a snapshot copy of the recent-log ring."""
        with self._lock:
            return list(self._logs)

    # ── Resume seeding (non-broadcasting) ─────────────────────────────────
    #
    # These mutate snapshot state in-place for a resumed run. They are
    # intentionally separate from update()/event() so that a wrapping
    # observer (e.g. StateServer) knows *not* to fan out fake live events
    # for history that has already happened.

    def set_start_epoch(self, epoch: int) -> None:
        with self._lock:
            self.snapshot.start_epoch = int(epoch)
            self.snapshot.current_epoch = int(epoch)

    def set_best(
        self, metric: str, value: float, epoch: int | None = None,
    ) -> None:
        if not isinstance(value, (int, float)):
            return
        if isinstance(value, float) and math.isnan(value):
            return
        with self._lock:
            self.snapshot.best = BestMetricRecord(
                metric=str(metric),
                value=float(value),
                epoch=int(epoch) if epoch is not None else None,
            )


# ── Multicast observer ───────────────────────────────────────────────────────

class MulticastObserver:
    """Forward all observer calls to a list of wrapped observers.

    Any single observer raising from a hook is logged to stderr but does
    not prevent the others from receiving the call.
    """

    def __init__(self, observers: Iterable[TrainingObserver]):
        self._observers: list[TrainingObserver] = list(observers)

    def _each(self, name: str, *args, **kwargs) -> None:
        for obs in self._observers:
            fn = getattr(obs, name, None)
            if fn is None:
                continue
            try:
                fn(*args, **kwargs)
            except Exception:
                traceback.print_exc()

    def start(self) -> None: self._each("start")
    def stop(self) -> None: self._each("stop")
    def log(self, message: str, level: str = "INFO") -> None:
        self._each("log", message, level)
    def update(self, data: EpochData) -> None:
        self._each("update", data)
    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self._each("event", kind, payload)

    @property
    def has_fatal_error(self) -> bool:
        return any(getattr(o, "has_fatal_error", False) for o in self._observers)
