# Detached Training Routing + State Snapshot Implementation Plan

> **For Hermes:** Use subagent-driven-development principles when executing, but for this run implement in one bounded worker pass. Do **not** commit automatically. Keep the foreground dashboard behavior intact while adding a detached/headless mode with UNIX-socket state queries and event subscriptions.

**Goal:** Add a detached training mode to `train.py` that publishes complete state snapshots and subscribable lifecycle events over a UNIX socket, while preserving the current Rich dashboard path.

**Architecture:** Introduce a small routing layer between the training loop and output sinks. The training loop emits logs, epoch updates, and explicit lifecycle events into a shared observer/router API. Foreground mode uses the existing `TrainingDashboard`; detached mode uses a new UNIX-socket state server backed by a snapshot reducer; an optional multicast router can support both simultaneously later without changing the training loop again.

**Tech Stack:** Python 3.11 stdlib (`socket`, `threading`, `json`, `argparse`, `dataclasses`, `pathlib`, `collections`), existing `dashboard.py`, existing `train.py`, `unittest` for smoke tests.

---

## Constraints and acceptance criteria

- Preserve current foreground behavior by default: `python train.py ...` should still launch the Rich dashboard.
- Add a detached/headless flag, preferably `--detach`, that does **not** launch Rich Live UI.
- Detached mode must expose a UNIX socket, defaulting to `<output_dir>/.train.sock` unless overridden.
- Clients must be able to:
  - fetch the latest full state snapshot
  - subscribe to a filtered or unfiltered event stream
- Initial required events:
  - `epoch_done`
  - `new_best`
  - `training_done`
  - `error`
- State snapshots must include enough data for a future dashboard attach flow:
  - current status
  - elapsed time/start time
  - current epoch / total epochs / start epoch
  - best metric name/value/epoch
  - latest epoch metrics and timings
  - recent logs
  - loss weights
  - training config summary
- Do **not** implement full daemonization. Detached means headless + socket server; the caller can background the process externally.
- Do **not** change training math, checkpoint contents, or registry semantics except where needed to emit events consistently.
- Do **not** commit in this run.

---

## Task 1: Add the routing/state abstraction

**Objective:** Create a small shared interface that lets `train.py` talk to either the dashboard or the detached socket server without knowing which sink it is using.

**Files:**
- Create: `observer.py`

**Step 1: Create `observer.py` with the shared types**

Implement these core pieces:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Iterable, Protocol
import math
import threading
import time

from dashboard import EpochData


def _sanitize_value(value: Any) -> Any:
    ...


def sanitize_for_json(value: Any) -> Any:
    ...


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
    best: BestMetricRecord = field(default_factory=lambda: BestMetricRecord(metric="psnr", value=None, epoch=None))
    latest_epoch: dict[str, Any] | None = None
    loss_weights: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    recent_logs: list[TrainingLogRecord] = field(default_factory=list)
    last_error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = sanitize_for_json(asdict(self))
        if self.start_time is not None:
            data["elapsed_seconds"] = max(0.0, time.time() - self.start_time)
        return data


class TrainingObserver(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def log(self, message: str, level: str = "INFO") -> None: ...
    def update(self, data: EpochData) -> None: ...
    def event(self, kind: str, payload: dict[str, Any]) -> None: ...
    @property
    def has_fatal_error(self) -> bool: ...
```

Add `MulticastObserver` that forwards all methods to a list of observers.

**Step 2: Add a reusable state reducer**

In the same file, add `StateReducerMixin` or `TrainingStateStore` that owns a `TrainingStateSnapshot` and updates it from:
- `log(...)`
- `update(EpochData)`
- `event(kind, payload)`

Requirements:
- Keep only a bounded list of recent logs, e.g. 50.
- Update `best` based on explicit `new_best` events.
- Derive `epoch_done` state from `update(EpochData)`.
- Set `status` to one of `starting`, `running`, `completed`, `interrupted`, `error`.
- Expose `get_snapshot()` returning a deep-ish JSON-safe dict via `to_dict()`.

**Step 3: Verification**

Run:
```bash
python -m py_compile observer.py
```
Expected: no output, exit 0.

---

## Task 2: Extend the dashboard to participate in the shared contract

**Objective:** Make `TrainingDashboard` compatible with the shared observer API and able to export a complete snapshot for future attach flows.

**Files:**
- Modify: `dashboard.py`

**Step 1: Add snapshot support**

Without changing the current rendering behavior, add methods/properties so the dashboard can produce a `TrainingStateSnapshot`-compatible dict.

Minimum additions:
- store `config` summary and `loss_weights` already passed in
- `get_snapshot()` method returning:
  - status
  - epoch counters
  - start time / elapsed
  - best metric data
  - latest epoch data
  - recent logs
  - loss weights
  - config summary if provided
- optionally inherit/reuse the state reducer from `observer.py`; if that creates circular imports, keep dashboard-local snapshot assembly but preserve the same output shape

**Step 2: Add `event(kind, payload)`**

Implement:
- `new_best` → optionally log a styled line or silently update snapshot state
- `training_done` → update final status / summary
- `error` → record last error, set fatal state if appropriate, add log line
- unknown events → ignore

Do **not** break the existing `update()` display behavior.

**Step 3: Preserve compatibility**

Keep these existing public entry points working:
- `start()`
- `stop()`
- `log()`
- `update()`
- `bulk_load()`
- `has_fatal_error`
- `force_stop()`

**Step 4: Verification**

Run:
```bash
python -m py_compile dashboard.py
```
Expected: no output, exit 0.

---

## Task 3: Add a UNIX-socket state server and a tiny client

**Objective:** Implement detached-mode IPC with one-shot state queries and streaming subscriptions.

**Files:**
- Create: `state_server.py`
- Create: `state_client.py`

**Step 1: Implement `StateServer` in `state_server.py`**

Implement a class roughly like:

```python
class StateServer(TrainingObserver):
    def __init__(self, socket_path: str | Path, *, total_epochs: int, start_epoch: int = 0,
                 best_metric: str = "psnr", loss_weights: dict[str, float] | None = None,
                 config: dict[str, Any] | None = None, log_capacity: int = 50):
        ...
```

Requirements:
- owns a `TrainingStateSnapshot` reducer/store
- binds an `AF_UNIX` stream socket
- unlinks stale socket files safely before bind
- chmod socket file to `0600`
- starts background accept loop thread(s)
- supports bounded subscriber queues so slow clients do not block training
- cleans up socket path on `stop()`

**Step 2: Implement the line-delimited JSON protocol**

Client requests:

```json
{"cmd":"ping"}
{"cmd":"get_state"}
{"cmd":"subscribe"}
{"cmd":"subscribe","events":["epoch_done","new_best"]}
```

Server responses:

```json
{"type":"pong"}
{"type":"state","state":{...}}
{"type":"event","kind":"epoch_done","ts":...,"data":{...}}
{"type":"event","kind":"new_best","ts":...,"data":{...}}
{"type":"event","kind":"training_done","ts":...,"data":{...}}
{"type":"event","kind":"error","ts":...,"data":{...}}
{"type":"error","message":"..."}
```

Protocol rules:
- newline-delimited UTF-8 JSON
- send an immediate snapshot after `subscribe`
- sanitize NaN/Inf to `null`
- `new_best` must only be emitted after checkpoint save succeeds

**Step 3: Implement a tiny CLI in `state_client.py`**

Add a simple CLI with subcommands:

```bash
python state_client.py --socket /path/to/.train.sock ping
python state_client.py --socket /path/to/.train.sock get-state
python state_client.py --socket /path/to/.train.sock subscribe
python state_client.py --socket /path/to/.train.sock subscribe --events epoch_done,new_best
```

The client only needs to print received JSON objects, one per line.

**Step 4: Verification**

Run:
```bash
python -m py_compile state_server.py state_client.py
python state_client.py --help
```
Expected:
- compile succeeds
- CLI help prints without crashing

---

## Task 4: Wire detached mode into `train.py`

**Objective:** Replace direct dashboard coupling with the shared observer contract and emit explicit lifecycle events.

**Files:**
- Modify: `train.py`

**Step 1: Extend CLI/config parsing**

Add non-config routing args to `parse_config()`:
- `--detach`
- `--socket-path` (optional override)

Return enough routing info from `parse_config()` so `main()` can choose observer mode without stuffing transport details into `TrainConfig` unless needed.

**Step 2: Replace `dash` with `observer`**

In `main()`:
- construct `TrainingDashboard(...)` for default foreground mode
- construct `StateServer(...)` for detached mode
- optionally leave room for future multicast, but do not over-engineer it
- call `observer.start()` and `observer.stop()`
- replace `dash.log(...)` with `observer.log(...)`
- replace `dash.update(...)` with `observer.update(...)`
- preserve `has_fatal_error` checks through the shared interface

**Step 3: Emit explicit lifecycle events**

At the natural existing points, emit:

- `new_best` after `best.pt` save and registry update succeed:

```python
observer.event("new_best", {
    "metric": cfg.best_metric,
    "value": best_val_metric,
    "epoch": epoch + 1,
    "checkpoint": str(output_dir / "best.pt"),
    "val_psnr": val_psnr,
})
```

- `training_done` after successful completion / interruption / NaN stop, with a final summary payload
- `error` in `except Exception as e`, including exception type, message, and traceback string

Keep `epoch_done` synthesized from `observer.update(EpochData(...))`; do not add a second redundant event call for each epoch unless necessary.

**Step 4: Preserve final console summary behavior**

Detached mode may still print the final plain-text summary after `observer.stop()`, but must not launch Rich Live UI.

**Step 5: Verification**

Run:
```bash
python train.py --help
```
Expected:
- new flags appear
- help exits 0

Also run:
```bash
python -m py_compile train.py
```

---

## Task 5: Add smoke tests for state snapshots and IPC

**Objective:** Add fast, CPU-only tests that verify the new routing/state behavior without needing model training.

**Files:**
- Create: `tests/test_training_state_ipc.py`

**Step 1: Add snapshot reducer tests**

Write tests covering:
- initial snapshot shape
- `log()` appends and truncates correctly
- `update(EpochData)` updates latest epoch and current epoch
- `new_best` event updates best metric fields
- `error` event records `last_error` and status

Use `unittest.TestCase` to avoid adding dependencies.

**Step 2: Add UNIX socket smoke test**

Create a temporary socket path with `tempfile.TemporaryDirectory()`.
Start `StateServer`, then use either direct sockets or `subprocess.run([sys.executable, 'state_client.py', ...])` to verify:
- `ping`
- `get_state`
- `subscribe` returns an initial snapshot and then receives an injected `new_best` or `epoch_done`

Keep the test deterministic and short.

**Step 3: Verification**

Run:
```bash
python -m unittest tests.test_training_state_ipc -v
```
Expected: all tests pass.

---

## Task 6: Final verification and operator notes

**Objective:** Verify the implementation end-to-end enough for handoff and document how to use it.

**Files:**
- Modify if needed: `README.md` only if the new detached mode is obvious enough to document briefly; otherwise skip docs in this implementation pass

**Step 1: Run final checks**

Run exactly:

```bash
python -m py_compile observer.py state_server.py state_client.py dashboard.py train.py
python -m unittest tests.test_training_state_ipc -v
python train.py --help
python state_client.py --help
```

**Step 2: Optional manual smoke test if lightweight**

If it can be done without starting a real training job, add a tiny in-test/manual harness that:
- starts `StateServer`
- injects `log`, `update`, `event('new_best', ...)`
- confirms `state_client.py get-state` prints a full snapshot

**Step 3: Report clearly**

In the implementation summary, include:
- files created/changed
- protocol shape
- how detached mode is invoked
- how to query the socket
- any limitations or follow-up items

**Do not commit.**

---

## Suggested execution prompt for Claude Code

Use this plan file as the source of truth and implement it directly:

```text
Read docs/plans/2026-04-22-detached-training-routing.md and implement it carefully. Keep the default foreground dashboard behavior unchanged. Add detached/headless mode with UNIX-socket state snapshots and event subscriptions, plus CPU-only smoke tests. Do not commit. Run the verification commands from the plan and summarize results plus any deviations.
```

---

## Done criteria

The implementation is done when all of these are true:

- `train.py` supports `--detach`
- detached mode starts without Rich Live UI
- a UNIX socket is created and cleaned up properly
- `state_client.py get-state` returns a full snapshot shape
- `state_client.py subscribe` receives the initial snapshot and subsequent events
- `new_best`, `training_done`, and `error` are explicitly emitted
- `epoch_done` is observable through the event stream
- the foreground dashboard still works by default
- the new smoke tests pass
- no commit was created
