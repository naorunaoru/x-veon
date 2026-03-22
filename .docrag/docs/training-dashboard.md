---
title: Training Dashboard (Rich TUI)
tags: [training, dashboard, tui, monitoring, rich]
scope: dashboard.py, train.py
generated: 2026-03-22
commit: 347c8dd
---

## Context

Training runs for X-Trans demosaicing can last hours or days. Operators need
real-time visibility into PSNR progression, per-component loss trends, learning
rate schedules, system resource utilisation, and timing estimates -- all without
leaving the terminal. The training dashboard provides this as a full-screen Rich
TUI that updates live during training and can also replay saved history files
after the fact.

The dashboard is implemented entirely in `dashboard.py` and is consumed by
`train.py` through three classes/dataclasses: `TrainingDashboard`, `EpochData`,
and supporting types `LogEntry` and `SystemSnapshot`.

## Pattern / Approach

### TUI Layout

The dashboard renders a four-panel layout that adapts to terminal width:

```
+------------------+------------------------------------------+
|  General         |  PSNR sparklines (full width)             |
|  (progress,      +--------------------+---------------------+
|   timing, hist,  |  Train Losses      |  Val Losses          |
|   LR, system)    |                    |                      |
+------------------+--------------------+---------------------+
+-------------------------------------------------------------+
|  Log                                                        |
+-------------------------------------------------------------+
```

- **General panel** -- epoch progress bar, elapsed time, ETA, average/stddev
  epoch timing, epoch-time histogram (after 5 epochs), current learning rate,
  and system resource sparklines (CPU, RAM, GPU, VRAM, disk I/O).
- **PSNR panel** -- braille sparklines for train and val PSNR over all epochs,
  current values colour-coded by threshold, epoch-over-epoch delta, rolling
  average delta with plateau detection.
- **Train Losses / Val Losses panels** -- table of every loss component with
  current value, instant delta from previous epoch, and rolling delta over
  `rolling_window` epochs. NaN/Inf values are highlighted in red.
- **Log panel** -- timestamped, level-coloured log entries. Height dynamically
  fills whatever terminal lines remain below the top panels.

At terminals wider than 140 columns the two loss panels sit side-by-side;
otherwise they stack vertically.

### EpochData Dataclass

`EpochData` is the single unit of information passed from the training loop to
the dashboard on every epoch. Defined in `dashboard.py`:

```python
@dataclass
class EpochData:
    epoch: int                          # 1-indexed epoch number
    train_psnr: float                   # mean training PSNR (dB)
    val_psnr: float                     # mean validation PSNR (dB)
    train_components: dict[str, float]  # per-component train losses (e.g. l1, gradient, chroma)
    val_components: dict[str, float]    # per-component val losses
    lr: float                           # current learning rate
    epoch_time: float                   # wall-clock seconds for the full epoch
    train_time: float = 0.0             # wall-clock seconds for train pass only
    val_time: float = 0.0               # wall-clock seconds for val pass only
```

Fields:
- `epoch` -- 1-based. `train.py` passes `epoch + 1` (the loop variable is 0-based).
- `train_psnr` / `val_psnr` -- average PSNR in dB computed by
  `train_epoch()` / `evaluate()` in `train.py`.
- `train_components` / `val_components` -- dictionaries keyed by loss component
  name (e.g. `"l1"`, `"gradient"`, `"chroma"`, `"msssim"`, `"zipper"`). The
  `"total"` key is skipped during rendering.
- `lr` -- read from `optimizer.param_groups[0]["lr"]` after `scheduler.step()`.
- `epoch_time` -- total wall time including both train and val passes.
- `train_time` / `val_time` -- optional breakdown for the "Last" timing line in
  the General panel.

### TrainingDashboard Class

#### Constructor

```python
TrainingDashboard(
    total_epochs: int,
    start_epoch: int = 0,
    rolling_window: int = 10,
    log_capacity: int = 10,
    best_val_psnr: float = 0.0,
)
```

- `total_epochs` -- drives the progress bar denominator.
- `start_epoch` -- offset for resumed runs (set by `train.py` after
  construction via `dash.start_epoch = start_epoch`).
- `rolling_window` -- number of epochs for the rolling delta averages shown in
  PSNR and loss panels. Default 10.
- `log_capacity` -- max log entries retained in the circular buffer (deque).
  `train.py` overrides this to 50.
- `best_val_psnr` -- initial best PSNR, updated by `train.py` via
  `dash.best_val_psnr = best_val_psnr` when resuming a run.

Internal state includes `self.history` (list of all `EpochData` objects),
`self.logs` (deque of `LogEntry`), `self._fatal` (bool), and system monitoring
state.

#### Lifecycle: start, update, stop

The dashboard follows a strict `start -> update* -> stop` lifecycle.

**`start()`** -- Records the wall-clock start time. Takes an initial system
snapshot. Creates a `rich.live.Live` context in full-screen mode with 2 Hz
refresh. Registers `stop()` via `atexit` as a safety net. Launches the
background system sampler thread.

**`update(data: EpochData)`** -- Called once per epoch from the training loop.
Scans all metric values (train components, val components, train PSNR, val PSNR)
for NaN or Inf. If any are found, logs an ERROR and sets `self._fatal = True`.
Appends the data to `self.history`. If `val_psnr` exceeds the current best,
updates `best_val_psnr` and `best_val_epoch` and logs the new best. Triggers a
display refresh.

**`stop()`** -- Signals the system sampler thread to stop via
`self._sys_stop.set()`, joins it with a 2-second timeout, then stops the
`Live` display. Idempotent -- safe to call multiple times.

The class also supports context-manager usage (`with dashboard:`), which calls
`start()` on enter and `stop()` on exit.

**`log(message, level="INFO")`** -- Appends a timestamped `LogEntry` to the log
deque and refreshes the display. Used extensively by `train.py` for setup
messages, cache statistics, and warnings.

**`bulk_load(epochs: list[EpochData])`** -- Loads a batch of historical epochs
without per-epoch rendering. Used when resuming a training run to restore the
dashboard sparklines from saved `history.json`. Performs the same NaN/Inf scan
as `update()` and tracks best val PSNR across all loaded entries.

**`print_static()`** -- Renders the dashboard once to the console without a
Live context. Used by the replay mode for instant (non-animated) rendering.

#### NaN/Inf Detection (`has_fatal_error`)

The `update()` method inspects every numeric value in the epoch data:

```python
all_values = (
    list(data.train_components.values())
    + list(data.val_components.values())
    + [data.train_psnr, data.val_psnr]
)
for v in all_values:
    if math.isnan(v) or math.isinf(v):
        self._fatal = True
```

The `has_fatal_error` property exposes `self._fatal`. In `train.py`, this is
checked immediately after every `dash.update()` call:

```python
dash.update(EpochData(...))

if dash.has_fatal_error:
    dash.log("Stopping training due to NaN/Inf detection.", "ERROR")
    break
```

This creates a clean separation: the dashboard detects the problem, and the
training loop decides to break. The dashboard never calls `sys.exit()` or
raises an exception -- it only sets a flag.

### Supporting Data Types

**`LogEntry`** -- `timestamp: datetime`, `level: str`, `message: str`. The
level controls colour via `LEVEL_STYLES` (DEBUG dim, INFO white, WARN yellow,
ERROR red).

**`SystemSnapshot`** -- CPU percent, CPU temp, RAM used/total, GPU utilisation,
VRAM used/total, GPU temp, disk busy percent, disk read/write MB/s. Populated
by `sample_system()` using optional `psutil` and `pynvml` dependencies.

### Threading and Async Considerations

The dashboard uses one background daemon thread for system monitoring:

1. **System sampler thread** (`_sys_sampler`) -- Started in `start()`, stopped
   in `stop()`. Wakes every 1 second to refresh the display. Every 2 seconds
   it also calls `sample_system()` to update CPU/RAM/GPU/disk metrics and
   appends a `SystemSnapshot` to the history deque. The thread is a daemon, so
   it will not prevent process exit.

2. **`rich.live.Live`** -- The Live display runs its own internal refresh timer
   at 2 Hz (`refresh_per_second=2`). The dashboard also triggers manual
   refreshes via `self._live.update(self._render())` from `_refresh()`,
   called by `update()`, `log()`, and the sampler thread.

3. **Thread safety** -- The sampler thread writes to `self._sys` and
   `self._sys_history` while the main thread reads them during `_render()`.
   Both are simple attribute assignments (atomic in CPython due to the GIL).
   The `deque(maxlen=...)` used for system history is also GIL-safe for
   append/iteration. No explicit locks are used.

4. **atexit registration** -- `start()` registers `stop()` with `atexit` to
   ensure the terminal is restored even if the caller forgets to call `stop()`
   or an unhandled exception occurs.

### How train.py Integrates the Dashboard

The integration in `train.py` follows this sequence:

1. **Immediate start** -- The dashboard is created and started before any
   dataset or model setup, so all setup log messages appear in the TUI:
   ```python
   dash = TrainingDashboard(total_epochs=cfg.epochs, log_capacity=50)
   dash.start()
   ```

2. **Setup logging** -- Throughout dataset creation, model loading, checkpoint
   restoration, and loss configuration, `train.py` calls `dash.log()` to
   report progress.

3. **History restoration** -- When resuming a same-directory run, `train.py`
   loads `history.json`, sets `dash.start_epoch` and `dash.best_val_psnr`,
   then calls `dash.bulk_load()` with the historical `EpochData` entries.

4. **Per-epoch update** -- Inside the training loop, after `train_epoch()`,
   `evaluate()`, and `scheduler.step()`:
   ```python
   dash.update(EpochData(
       epoch=epoch + 1,
       train_psnr=train_psnr,
       val_psnr=val_psnr,
       train_components=train_comp,
       val_components=val_comp,
       lr=lr_now,
       epoch_time=elapsed,
       train_time=t_train,
       val_time=t_val,
   ))
   ```

5. **Fatal error check** -- Immediately after `update()`:
   ```python
   if dash.has_fatal_error:
       dash.log("Stopping training due to NaN/Inf detection.", "ERROR")
       break
   ```

6. **Explicit stop** -- After the training loop (whether completed, interrupted
   via KeyboardInterrupt, or broken due to NaN), `train.py` calls
   `dash.stop()` explicitly. This restores the terminal before the post-training
   summary is printed.

### Post-Training Summary

After `dash.stop()`, `train.py` prints a plain-text summary to stdout that is
visible after the TUI closes. It reports:

- Status: Completed, Interrupted, Error, or Crashed
- Total elapsed time and epoch count
- Average epoch time
- CFA type and model width
- Best validation PSNR
- Data directories and output directory

The dashboard also supports standalone replay for post-hoc analysis:
```
python dashboard.py --replay checkpoints/history.json
python dashboard.py --replay checkpoints/history.json --animate
```

### Visualisation Helpers

- **`sparkline(values, width)`** -- Braille-based sparkline that packs 2 data
  points per character for 2x density. Uses a colour gradient from red (low)
  through yellow to green (high). NaN values are replaced with 0.
- **`sys_sparkline(values, width)`** -- Similar but with a neutral grey
  gradient and a fixed 0-100 scale for resource utilisation percentages.
- **`delta_text(current, previous, invert, precision)`** -- Renders a delta
  value with directional arrow. The `invert` flag controls whether decrease is
  good (True for losses) or bad (True for PSNR is False since higher is better).
- **`format_time(seconds)`** -- Human-readable duration (e.g. "23s", "5m 12s",
  "1h 30m 05s").
- **PSNR thresholds** -- Values above 30 dB are bright green, 25-30 green,
  20-25 yellow, below 20 red.

## Rationale

- **Full-screen TUI over scrolling logs** -- Training runs produce hundreds of
  epochs of output. A scrolling log makes it impossible to see trends. The
  Rich Live display provides a fixed layout where sparklines and deltas show
  trajectory at a glance.

- **Braille sparklines** -- Each braille character encodes a 2x4 dot grid,
  allowing 2 data points per character width. This doubles the effective
  resolution compared to standard block characters.

- **NaN detection in the dashboard, not the training loop** -- Centralising
  NaN/Inf checks in `update()` means every code path that feeds data to the
  dashboard gets protection automatically. The training loop just checks
  `has_fatal_error` rather than implementing its own NaN scanning.

- **Background system sampling** -- Sampling CPU/GPU/disk metrics can block
  briefly (especially disk I/O counters). Running it on a daemon thread with a
  2-second interval keeps the main training loop unaffected.

- **`atexit` registration** -- Rich's Live display takes over the terminal
  (alternate screen mode). If the process exits without calling `stop()`, the
  terminal would be left in a broken state. The atexit handler prevents this.

- **`bulk_load()` for history replay** -- Calling `update()` hundreds of times
  during resume would trigger hundreds of unnecessary renders. `bulk_load()`
  loads all entries and renders once.

- **Context manager support** -- The `with dashboard:` pattern guarantees
  `stop()` runs even if exceptions propagate, aligning with Python resource
  management conventions.

## Key Files

| File | Role |
|------|------|
| `dashboard.py` | `TrainingDashboard`, `EpochData`, `LogEntry`, `SystemSnapshot`, sparkline helpers, replay/mock modes |
| `train.py` | Creates and drives the dashboard through the training lifecycle |

## Antipatterns

- **Calling `update()` without checking `has_fatal_error`** -- The dashboard
  sets `_fatal` but does not halt training. If the caller ignores the flag,
  training continues with corrupt metrics, wasting compute and potentially
  saving a broken checkpoint.

- **Forgetting to call `stop()`** -- While `atexit` provides a safety net, it
  runs late in shutdown. Always call `stop()` explicitly before printing
  post-training output, otherwise print statements will be swallowed by the
  alternate screen buffer.

- **Passing 0-indexed epochs to `EpochData`** -- The dashboard uses `epoch` as
  the progress numerator. `train.py` correctly passes `epoch + 1`. Passing the
  raw 0-based loop variable would make the progress bar off by one and the
  "best epoch" display incorrect.

- **Modifying `history` from outside the dashboard** -- The dashboard maintains
  its own `self.history` list derived from `update()` and `bulk_load()` calls.
  Directly mutating this list would desynchronise sparkline data from internal
  tracking state like `best_val_psnr`.

- **Running `sample_system()` on the main thread** -- The function calls
  `psutil.cpu_percent()`, `pynvml` queries, and disk I/O counter reads that can
  block. Doing this in the training loop would add jitter to epoch timing
  measurements.

- **Using `update()` for bulk history restore** -- Each `update()` call
  triggers a full `_render()` and `Live.update()`. For hundreds of historical
  epochs, use `bulk_load()` instead to avoid hundreds of wasted render cycles.
