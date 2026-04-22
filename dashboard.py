#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Training dashboard with rich TUI display.

Layout:
    ┌─────────────────┬──────────────────────────────────────┐
    │  General         │  PSNR sparklines (full width)        │
    │  (progress,      ├─────────────────┬────────────────────┤
    │   timing, hist,  │  Train Losses   │  Val Losses        │
    │   LR, system)    │                 │                    │
    └─────────────────┴─────────────────┴────────────────────┘
    ┌────────────────────────────────────────────────────────┐
    │  Log                                                   │
    └────────────────────────────────────────────────────────┘

Usage:
    python dashboard.py [--fast] [--epochs N]
    python dashboard.py --replay checkpoints/history.json
    python dashboard.py --replay checkpoints/history.json --animate

    from dashboard import TrainingDashboard, EpochData
"""

import atexit
import json
import math
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Optional system monitoring ───────────────────────────────────────────────

try:
    import psutil

    # Prime the CPU percent counter (first call always returns 0)
    psutil.cpu_percent(interval=None)
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_NAME = pynvml.nvmlDeviceGetName(_NVML_HANDLE)
    if isinstance(_GPU_NAME, bytes):
        _GPU_NAME = _GPU_NAME.decode()
    _HAS_NVML = True
except Exception:
    _NVML_HANDLE = None
    _GPU_NAME = ""
    _HAS_NVML = False

# ── Constants ────────────────────────────────────────────────────────────────

SPARK_CHARS = " ▁▂▃▄▅▆▇"
SPARK_WIDTH = 50

# Braille dot encoding: each char is 2×4 dots.
# Left column bits (top→bottom): dot1=0x01, dot2=0x02, dot3=0x04, dot7=0x40
# Right column bits:             dot4=0x08, dot5=0x10, dot6=0x20, dot8=0x80
_BRAILLE_BASE = 0x2800
_BRAILLE_LEFT  = [0, 0x40, 0x44, 0x46, 0x47]   # 0–4 dots filled bottom-up
_BRAILLE_RIGHT = [0, 0x80, 0xA0, 0xB0, 0xB8]

SPARK_GRADIENT = [
    "bright_red", "red", "yellow", "yellow",
    "green", "green", "bright_green", "bright_green",
]

# Neutral gradient for resource utilization (informational, not good/bad)
SYS_SPARK_GRADIENT = [
    "bright_black", "bright_black", "grey50", "grey50",
    "grey70", "grey70", "grey82", "grey82",
]
SYS_SPARK_WIDTH = 20

LEVEL_STYLES = {
    "DEBUG": "dim",
    "INFO": "bright_white",
    "WARN": "bold yellow",
    "WARNING": "bold yellow",
    "ERROR": "bold red",
}

HIGHER_IS_BETTER = {"msssim"}

BORDER_GENERAL = "bright_blue"
BORDER_PSNR = "bright_green"
BORDER_TRAIN = "bright_yellow"
BORDER_VAL = "bright_cyan"
BORDER_LOG = "bright_magenta"

PSNR_THRESHOLDS = [(30, "bright_green"), (25, "green"), (20, "yellow"), (0, "red")]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _psnr_style(psnr: float) -> str:
    for threshold, style in PSNR_THRESHOLDS:
        if psnr >= threshold:
            return style
    return "dim"


def sparkline(values: list[float], width: int = SPARK_WIDTH) -> Text:
    """Braille sparkline — packs 2 data points per character for 2× density."""
    if not values:
        return Text(chr(_BRAILLE_BASE) * width, style="dim")
    # Each char holds 2 values, so we can show width*2 data points
    vis = values[-(width * 2):]
    # Left-pad so sparkline always occupies full width
    n_slots = width * 2
    if len(vis) < n_slots:
        vis = [vis[0]] * (n_slots - len(vis)) + vis
    vis = [v if v == v else 0.0 for v in vis]  # replace NaN with 0
    vmin, vmax = min(vis), max(vis)
    span = vmax - vmin
    text = Text()
    for i in range(0, len(vis), 2):
        lv, rv = vis[i], vis[i + 1]
        if span < 1e-10:
            li, ri = 2, 2
        else:
            li = max(0, min(4, int((lv - vmin) / span * 4)))
            ri = max(0, min(4, int((rv - vmin) / span * 4)))
        ch = chr(_BRAILLE_BASE + _BRAILLE_LEFT[li] + _BRAILLE_RIGHT[ri])
        # Color by the higher of the two values (maps to 8-level gradient)
        avg = (lv + rv) / 2
        cidx = max(0, min(7, int((avg - vmin) / span * 7))) if span >= 1e-10 else 4
        text.append(ch, style=SPARK_GRADIENT[cidx])
    return text


def delta_text(
    current: float, previous: float, invert: bool = False, precision: int = 4
) -> Text:
    """Delta with directional arrow. invert=True means decrease is good."""
    if math.isnan(current) or math.isnan(previous):
        return Text("NaN!", style="bold red")
    delta = current - previous
    if abs(delta) < 10 ** -(precision + 1):
        return Text(f" {abs(delta):.{precision}f}", style="dim")
    if delta > 0:
        style = "red" if invert else "green"
        arrow = "▲"
    else:
        style = "green" if invert else "red"
        arrow = "▼"
    return Text(f"{arrow}{abs(delta):.{precision}f}", style=style)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"


def sys_sparkline(values: list[float], width: int = SYS_SPARK_WIDTH) -> Text:
    """Braille sparkline for resource utilization (fixed 0-100 scale)."""
    if not values:
        return Text(chr(_BRAILLE_BASE) * width, style="dim")
    vis = values[-(width * 2):]
    n_slots = width * 2
    if len(vis) < n_slots:
        vis = [0.0] * (n_slots - len(vis)) + vis
    text = Text()
    for i in range(0, len(vis), 2):
        lv, rv = vis[i], vis[i + 1]
        li = max(0, min(4, int(lv / 100 * 4)))
        ri = max(0, min(4, int(rv / 100 * 4)))
        ch = chr(_BRAILLE_BASE + _BRAILLE_LEFT[li] + _BRAILLE_RIGHT[ri])
        cidx = max(0, min(7, int((lv + rv) / 2 / 100 * 7.99)))
        text.append(ch, style=SYS_SPARK_GRADIENT[cidx])
    return text


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str


@dataclass
class EpochData:
    epoch: int
    train_psnr: float
    val_psnr: float
    train_components: dict[str, float]
    val_components: dict[str, float]
    lr: float
    epoch_time: float
    train_time: float = 0.0
    val_time: float = 0.0


@dataclass
class SystemSnapshot:
    cpu_percent: float = 0.0
    cpu_temp: Optional[int] = None
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_util: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    gpu_temp: Optional[int] = None
    disk_busy_pct: Optional[float] = None
    disk_read_mbs: Optional[float] = None
    disk_write_mbs: Optional[float] = None


# Previous disk I/O counters + timestamp for delta calculation
_prev_disk_io: Optional[tuple] = None  # (timestamp, counters)


def sample_system() -> SystemSnapshot:
    """Sample current CPU/RAM/GPU metrics. Graceful fallback if unavailable."""
    snap = SystemSnapshot()

    if _HAS_PSUTIL:
        snap.cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        snap.ram_used_gb = mem.used / (1 << 30)
        snap.ram_total_gb = mem.total / (1 << 30)
        try:
            temps = psutil.sensors_temperatures()
            for key in ("k10temp", "coretemp", "cpu_thermal", "zenpower"):
                if key in temps and temps[key]:
                    snap.cpu_temp = int(temps[key][0].current)
                    break
        except Exception:
            pass

    if _HAS_NVML and _NVML_HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            snap.gpu_util = util.gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            snap.vram_used_gb = mem_info.used / (1 << 30)
            snap.vram_total_gb = mem_info.total / (1 << 30)
            snap.gpu_temp = pynvml.nvmlDeviceGetTemperature(
                _NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

    if _HAS_PSUTIL:
        global _prev_disk_io
        try:
            now = time.monotonic()
            counters = psutil.disk_io_counters()
            if counters is not None and _prev_disk_io is not None:
                prev_t, prev_c = _prev_disk_io
                dt = now - prev_t
                if dt > 0:
                    snap.disk_read_mbs = (counters.read_bytes - prev_c.read_bytes) / dt / (1 << 20)
                    snap.disk_write_mbs = (counters.write_bytes - prev_c.write_bytes) / dt / (1 << 20)
                    busy_ms = getattr(counters, 'busy_time', None)
                    prev_busy = getattr(prev_c, 'busy_time', None)
                    if busy_ms is not None and prev_busy is not None:
                        snap.disk_busy_pct = min(100.0, (busy_ms - prev_busy) / (dt * 1000) * 100)
            if counters is not None:
                _prev_disk_io = (now, counters)
        except Exception:
            pass

    return snap


# ── Dashboard ────────────────────────────────────────────────────────────────

class TrainingDashboard:
    """Real-time training dashboard using rich Live display.

    Usage::

        dashboard = TrainingDashboard(total_epochs=600)
        with dashboard:
            for epoch in range(600):
                dashboard.update(EpochData(...))
                dashboard.log("Cache: swapped 11200 patches")
                if dashboard.has_fatal_error:
                    break
    """

    def __init__(
        self,
        total_epochs: int,
        start_epoch: int = 0,
        rolling_window: int = 10,
        log_capacity: int = 10,
        best_val_psnr: float = 0.0,
        best_metric: str = "psnr",
        loss_weights: dict[str, float] | None = None,
    ):
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.rolling_window = rolling_window
        self.best_val_psnr = best_val_psnr
        self.best_val_epoch = 0
        self.best_metric = best_metric  # "psnr" or "msssim"
        self.loss_weights = loss_weights or {}

        self.history: list[EpochData] = []
        self.logs: deque[LogEntry] = deque(maxlen=log_capacity)
        self.start_time: Optional[float] = None
        self._fatal = False
        self._sys: SystemSnapshot = SystemSnapshot()
        self._sys_history: deque[SystemSnapshot] = deque(maxlen=SYS_SPARK_WIDTH * 2)
        self._metric_label = "PSNR" if best_metric == "psnr" else "MS-SSIM"

        term_size = shutil.get_terminal_size((160, 40))
        self.console = Console(
            width=max(term_size.columns, 120),
            force_terminal=True,
        )
        self._live: Optional[Live] = None
        self._sys_stop: Optional[threading.Event] = None
        self._sys_thread: Optional[threading.Thread] = None

    _active_instance: "Optional[TrainingDashboard]" = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    @classmethod
    def force_stop(cls):
        """Stop the active dashboard (if any) so tracebacks print cleanly."""
        if cls._active_instance is not None:
            cls._active_instance.stop()
            cls._active_instance = None

    def start(self):
        self.start_time = time.time()
        self._sys = sample_system()
        self._sys_history.append(self._sys)
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=2,
            screen=True,
        )
        self._live.start()
        TrainingDashboard._active_instance = self
        atexit.register(self.stop)
        # Background system sampling every 5 seconds
        self._sys_stop = threading.Event()
        self._sys_thread = threading.Thread(target=self._sys_sampler, daemon=True)
        self._sys_thread.start()

    def stop(self):
        TrainingDashboard._active_instance = None
        if self._sys_stop:
            self._sys_stop.set()
        if self._sys_thread:
            self._sys_thread.join(timeout=2)
            self._sys_thread = None
        if self._live:
            self._live.stop()
            self._live = None

    def _sys_sampler(self):
        """Background thread: refresh display every 1s, sample system every 2s."""
        tick = 0
        while not self._sys_stop.wait(1.0):
            tick += 1
            if tick % 2 == 0:
                self._sys = sample_system()
                self._sys_history.append(self._sys)
            self._refresh()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── Helpers ────────────────────────────────────────────────────────

    def _best_metric_value(self, data: EpochData) -> float:
        """Extract the value used for best-checkpoint comparison."""
        if self.best_metric == "msssim":
            return data.val_components.get("msssim", float("nan"))
        return data.val_psnr

    # ── Public API ───────────────────────────────────────────────────────

    def log(self, message: str, level: str = "INFO"):
        self.logs.append(LogEntry(
            timestamp=datetime.now(), level=level.upper(), message=message,
        ))
        self._refresh()

    def update(self, data: EpochData):
        """Record epoch results and refresh display."""
        bad_keys: list[str] = []
        for k, v in data.train_components.items():
            if math.isnan(v) or math.isinf(v):
                bad_keys.append(f"train/{k}={v}")
        for k, v in data.val_components.items():
            if math.isnan(v) or math.isinf(v):
                bad_keys.append(f"val/{k}={v}")
        if math.isnan(data.train_psnr) or math.isinf(data.train_psnr):
            bad_keys.append(f"train_psnr={data.train_psnr}")
        if math.isnan(data.val_psnr) or math.isinf(data.val_psnr):
            bad_keys.append(f"val_psnr={data.val_psnr}")
        if bad_keys:
            self.log(
                f"NaN/Inf in epoch {data.epoch}: {', '.join(bad_keys)}. "
                "Training should be stopped.", "ERROR"
            )
            self._fatal = True

        self.history.append(data)

        metric_val = self._best_metric_value(data)
        if not math.isnan(metric_val) and metric_val > self.best_val_psnr:
            self.best_val_psnr = metric_val
            self.best_val_epoch = data.epoch
            if self.best_metric == "msssim":
                self.log(f"New best val MS-SSIM ({metric_val:.6f})")
            else:
                self.log(f"New best val ({metric_val:.2f} dB)")

        self._refresh()

    def bulk_load(self, epochs: list[EpochData]):
        """Load many epochs without per-epoch rendering (instant replay)."""
        for data in epochs:
            all_values = (
                list(data.train_components.values())
                + list(data.val_components.values())
                + [data.train_psnr, data.val_psnr]
            )
            for v in all_values:
                if math.isnan(v) or math.isinf(v):
                    self._fatal = True
                    break
            self.history.append(data)
            metric_val = self._best_metric_value(data)
            if not math.isnan(metric_val) and metric_val > self.best_val_psnr:
                self.best_val_psnr = metric_val
                self.best_val_epoch = data.epoch

        if self.history:
            first, last = self.history[0].epoch, self.history[-1].epoch
            self.logs.append(LogEntry(
                datetime.now(), "INFO",
                f"Loaded epochs {first}-{last} ({len(self.history)} total)",
            ))
            if self.best_val_epoch > 0:
                if self.best_metric == "msssim":
                    best_str = f"{self.best_val_psnr:.6f}"
                else:
                    best_str = f"{self.best_val_psnr:.2f} dB"
                self.logs.append(LogEntry(
                    datetime.now(), "INFO",
                    f"Best val {self._metric_label}: {best_str} (ep {self.best_val_epoch})",
                ))
            if self._fatal:
                self.logs.append(LogEntry(
                    datetime.now(), "ERROR", "NaN/Inf detected in loaded history",
                ))

        self._refresh()

    @property
    def has_fatal_error(self) -> bool:
        return self._fatal

    def print_static(self):
        """Render once without Live context."""
        self.console.print(self._render())

    # ── Internal ─────────────────────────────────────────────────────────

    def _refresh(self):
        if self._live:
            self._live.update(self._render())

    def _rolling_delta(self, values: list[float], window: int = None) -> Optional[float]:
        w = window or self.rolling_window
        if len(values) < 2:
            return None
        recent = values[-w:]
        if len(recent) < 2:
            return None
        return (recent[-1] - recent[0]) / (len(recent) - 1)

    # ── Rendering ────────────────────────────────────────────────────────

    def _render(self) -> Group:
        """
        ┌ General ──┬ PSNR (full right width) ─┐
        │           ├ Train Loss ┬ Val Loss ────┤
        └───────────┴────────────┴──────────────┘
        ┌ Log ──────────────────────────────────┐
        └───────────────────────────────────────┘
        """
        # Pick up terminal resize
        term = shutil.get_terminal_size((160, 40))
        self.console.width = max(term.columns, 120)

        # Right subgrid: PSNR on top, losses below
        train_loss = self._render_loss_panel("Train", is_train=True)
        val_loss = self._render_loss_panel("Val", is_train=False)

        wide = self.console.width >= 140
        if wide:
            loss_row = Table(show_header=False, box=None, padding=(0, 0), expand=True)
            loss_row.add_column(ratio=1)
            loss_row.add_column(ratio=1)
            loss_row.add_row(train_loss, val_loss)
        else:
            loss_row = Group(train_loss, val_loss)

        right = Group(self._render_psnr(), loss_row)

        # Top: General | Right subgrid
        top = Table(show_header=False, box=None, padding=(0, 0), expand=True)
        top.add_column(ratio=2)
        top.add_column(ratio=3)
        top.add_row(self._render_general(), right)

        # Measure actual rendered height of the top section, then give
        # the log panel exactly the remaining terminal lines.
        top_height = sum(
            1 for _ in self.console.render_lines(top, pad=False)
        )
        term_h = shutil.get_terminal_size((160, 40)).lines
        log_height = max(5, term_h - top_height)
        return Group(top, self._render_log(log_height))

    # ── General panel ────────────────────────────────────────────────────

    def _render_general(self) -> Panel:
        parts: list = []
        current = self.history[-1] if self.history else None
        n_done = max(current.epoch, self.start_epoch) if current else self.start_epoch

        # Progress bar
        pct = n_done / self.total_epochs if self.total_epochs > 0 else 0
        bar_w = 20
        filled = int(pct * bar_w)
        t = Text()
        t.append("Epoch ", style="bold")
        t.append(f"{n_done}/{self.total_epochs} ", style="bold bright_cyan")
        t.append("[")
        t.append("⣿" * filled, style="bright_blue")
        t.append("⣿" * (bar_w - filled), style="bright_black")
        t.append("] ")
        t.append(f"{pct * 100:.0f}%", style="bold bright_white")
        parts.append(t)

        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            t = Text()
            t.append("Elapsed: ", style="dim")
            t.append(format_time(elapsed), style="bright_white")
            parts.append(t)

            if self.history:
                avg_t = sum(d.epoch_time for d in self.history) / len(self.history)
                std_t = (
                    sum((d.epoch_time - avg_t) ** 2 for d in self.history)
                    / len(self.history)
                ) ** 0.5
                remaining = self.total_epochs - n_done
                eta = avg_t * remaining

                t = Text()
                t.append("ETA: ", style="dim")
                t.append(format_time(eta), style="bright_white")
                parts.append(t)

                t = Text()
                t.append("Avg epoch: ", style="dim")
                t.append(f"{avg_t:.1f}s", style="bright_white")
                t.append(f" ±{std_t:.1f}s", style="dim")
                parts.append(t)

                if current and current.train_time > 0:
                    t = Text()
                    t.append("Last: ", style="dim")
                    t.append(f"{current.train_time:.1f}s", style="bright_yellow")
                    t.append(" train  ", style="dim")
                    t.append(f"{current.val_time:.1f}s", style="bright_cyan")
                    t.append(" val", style="dim")
                    parts.append(t)

        # Epoch time histogram
        if len(self.history) >= 5:
            parts.append(Text(""))
            parts.append(Text("Epoch time:", style="bold"))
            parts.append(self._render_histogram())

        # Learning rate
        if current:
            t = Text()
            t.append("LR: ", style="dim")
            t.append(f"{current.lr:.2e}", style="bright_white")
            parts.append(t)

        # System resources
        parts.append(Text(""))
        parts.append(Text("System:", style="bold"))
        parts.extend(self._render_system())

        return Panel(
            Group(*parts),
            title=f"[bold {BORDER_GENERAL}]General[/bold {BORDER_GENERAL}]",
            border_style=BORDER_GENERAL,
        )

    def _render_histogram(self) -> Text:
        times = [d.epoch_time for d in self.history]
        t_min, t_max = min(times), max(times)
        span = t_max - t_min
        bucket_size = max(1, round(span / 5)) if span >= 1 else 1

        start = int(t_min // bucket_size) * bucket_size
        end = int(t_max // bucket_size + 1) * bucket_size

        buckets: dict[int, int] = {}
        for b in range(start, end, bucket_size):
            buckets[b] = 0
        for t in times:
            b = int(t // bucket_size) * bucket_size
            b = max(start, min(b, end - bucket_size))
            buckets[b] = buckets.get(b, 0) + 1

        max_count = max(buckets.values()) if buckets else 1
        bar_max = 16

        sorted_buckets = sorted(buckets.keys())
        total = sum(buckets.values())
        cumulative = 0
        median_bucket = sorted_buckets[0]
        for b in sorted_buckets:
            cumulative += buckets[b]
            if cumulative >= total / 2:
                median_bucket = b
                break

        text = Text()
        for b in sorted_buckets:
            count = buckets[b]
            if count == 0:
                continue
            bar_len = int(count / max_count * bar_max) if max_count > 0 else 0
            dist = abs(b - median_bucket) / max(bucket_size, 1)
            if dist <= 1:
                bar_style = "bright_green"
            elif dist <= 2:
                bar_style = "yellow"
            else:
                bar_style = "bright_red"
            text.append(f" {b:3d}-{b + bucket_size:<3d}s ", style="dim")
            text.append("⣿" * bar_len, style=bar_style)
            text.append(f" {count}\n", style="dim")
        return text

    def _render_system(self) -> list:
        """Render CPU/RAM/GPU with sparklines showing history."""
        s = self._sys
        hist = list(self._sys_history)

        table = Table(
            show_header=False, box=None, padding=(0, 1), expand=False,
        )
        table.add_column(no_wrap=True, style="dim", min_width=4)   # label
        table.add_column(no_wrap=True, justify="right", min_width=4)  # value
        table.add_column(no_wrap=True)                              # sparkline
        table.add_column(no_wrap=True, style="dim")                 # detail

        if _HAS_PSUTIL:
            cpu_vals = [h.cpu_percent for h in hist]
            cpu_detail = ""
            if s.cpu_temp is not None:
                temp_style = "bright_red" if s.cpu_temp >= 85 else (
                    "bright_yellow" if s.cpu_temp >= 75 else "bright_white"
                )
                cpu_detail = Text(f"{s.cpu_temp}°C", style=temp_style)
            table.add_row(
                "CPU",
                Text(f"{s.cpu_percent:3.0f}%", style="bright_white"),
                sys_sparkline(cpu_vals),
                cpu_detail,
            )
            ram_pct = s.ram_used_gb / s.ram_total_gb * 100 if s.ram_total_gb > 0 else 0
            ram_vals = [
                h.ram_used_gb / h.ram_total_gb * 100
                for h in hist if h.ram_total_gb > 0
            ]
            table.add_row(
                "RAM",
                Text(f"{ram_pct:3.0f}%", style="bright_white"),
                sys_sparkline(ram_vals),
                f"{s.ram_used_gb:.0f}/{s.ram_total_gb:.0f} GB",
            )

        if _HAS_NVML and s.gpu_util is not None:
            gpu_vals = [h.gpu_util for h in hist if h.gpu_util is not None]
            temp_text = ""
            if s.gpu_temp is not None:
                temp_style = "bright_red" if s.gpu_temp >= 85 else (
                    "bright_yellow" if s.gpu_temp >= 75 else "bright_white"
                )
                temp_text = Text(f"{s.gpu_temp}°C", style=temp_style)
            table.add_row(
                "GPU",
                Text(f"{s.gpu_util:3.0f}%", style="bright_white"),
                sys_sparkline(gpu_vals),
                temp_text,
            )
            if s.vram_used_gb is not None and s.vram_total_gb is not None:
                vram_pct = s.vram_used_gb / s.vram_total_gb * 100
                vram_vals = [
                    h.vram_used_gb / h.vram_total_gb * 100
                    for h in hist
                    if h.vram_used_gb is not None and h.vram_total_gb
                ]
                table.add_row(
                    "VRAM",
                    Text(f"{vram_pct:3.0f}%", style="bright_white"),
                    sys_sparkline(vram_vals),
                    f"{s.vram_used_gb:.1f}/{s.vram_total_gb:.0f} GB",
                )

        if _HAS_PSUTIL and s.disk_busy_pct is not None:
            disk_vals = [h.disk_busy_pct for h in hist if h.disk_busy_pct is not None]
            io_parts = []
            if s.disk_read_mbs is not None and s.disk_read_mbs > 0.5:
                io_parts.append(f"R {s.disk_read_mbs:.0f}")
            if s.disk_write_mbs is not None and s.disk_write_mbs > 0.5:
                io_parts.append(f"W {s.disk_write_mbs:.0f}")
            io_text = " ".join(io_parts) + " MB/s" if io_parts else ""
            table.add_row(
                "Disk",
                Text(f"{s.disk_busy_pct:3.0f}%", style="bright_white"),
                sys_sparkline(disk_vals),
                io_text,
            )

        if not _HAS_PSUTIL and not _HAS_NVML:
            return [Text(" (no monitoring available)", style="dim")]

        return [table]

    # ── PSNR panel ───────────────────────────────────────────────────────

    def _render_psnr(self) -> Panel:
        parts: list = []
        current = self.history[-1] if self.history else None

        # Train PSNR
        t = Text()
        t.append("Train  ", style="bold bright_yellow")
        train_psnrs = [d.train_psnr for d in self.history]
        t.append_text(sparkline(train_psnrs))
        if current:
            t.append(f" {current.train_psnr:.2f}", style=f"bold {_psnr_style(current.train_psnr)}")
            t.append(" dB", style="dim")
        parts.append(t)

        if len(train_psnrs) >= 2:
            parts.append(self._psnr_delta_line(train_psnrs))

        # Val PSNR
        parts.append(Text(""))
        t = Text()
        t.append("Val    ", style="bold bright_cyan")
        val_psnrs = [d.val_psnr for d in self.history]
        t.append_text(sparkline(val_psnrs))
        if current:
            t.append(f" {current.val_psnr:.2f}", style=f"bold {_psnr_style(current.val_psnr)}")
            t.append(" dB", style="dim")
        parts.append(t)

        if len(val_psnrs) >= 2:
            parts.append(self._psnr_delta_line(val_psnrs))

        # Best (metric-aware)
        best = Text()
        best.append(f"  Best {self._metric_label}: ", style="dim")
        if self.best_metric == "msssim":
            best.append(f"{self.best_val_psnr:.6f}", style="bold bright_green")
        else:
            best.append(f"{self.best_val_psnr:.2f} dB", style="bold bright_green")
        best.append(f" (ep {self.best_val_epoch})", style="dim")
        parts.append(best)

        return Panel(
            Group(*parts),
            title=f"[bold {BORDER_PSNR}]PSNR[/bold {BORDER_PSNR}]",
            border_style=BORDER_PSNR,
        )

    def _psnr_delta_line(self, values: list[float]) -> Text:
        line = Text("       Δ ")
        line.append_text(delta_text(values[-1], values[-2], invert=False, precision=2))
        rd = self._rolling_delta(values)
        if rd is not None:
            line.append(f"  │ {self.rolling_window}ep avg Δ ", style="dim")
            if abs(rd) < 0.01:
                line.append(f"{rd:+.3f}", style="bold yellow")
                line.append(" plateau?", style="bold yellow")
            elif rd > 0:
                line.append(f"{rd:+.3f}", style="bold green")
            else:
                line.append(f"{rd:+.3f}", style="bold red")
        return line

    # ── Loss panels ──────────────────────────────────────────────────────

    def _render_loss_panel(self, title: str, is_train: bool) -> Panel:
        border = BORDER_TRAIN if is_train else BORDER_VAL
        title_style = "bright_yellow" if is_train else "bright_cyan"

        table = Table(
            show_header=True, header_style="bold dim", box=None,
            padding=(0, 1), expand=True,
        )
        table.add_column("", style="bold", no_wrap=True)
        table.add_column("Value", justify="right", no_wrap=True)
        table.add_column("%", justify="right", no_wrap=True)
        table.add_column("Δ", justify="right", no_wrap=True)
        table.add_column(f"avg/{self.rolling_window}ep", justify="right", no_wrap=True)

        if not self.history:
            return Panel(
                table,
                title=f"[bold {title_style}]{title} Losses[/bold {title_style}]",
                border_style=border,
            )

        current = self.history[-1]
        comps = current.train_components if is_train else current.val_components

        # Compute contribution percentages from loss_weights
        total_val = comps.get("total", 0.0)
        contrib_pct: dict[str, float | None] = {}
        if self.loss_weights and total_val and not math.isnan(total_val) and total_val > 0:
            # Map component name → weight key
            _COMP_TO_WEIGHT = {
                "l1": "l1", "huber": "l1", "msssim": "msssim",
                "gradient": "gradient", "chroma": "chroma", "fft": "fft",
                "texture": "texture", "zipper": "zipper", "color_bias": "color_bias",
            }
            for comp_name, comp_val in comps.items():
                wkey = _COMP_TO_WEIGHT.get(comp_name)
                if wkey and not math.isnan(comp_val):
                    w = self.loss_weights.get(wkey, 0.0)
                    if comp_name == "msssim":
                        wtd = w * (1.0 - comp_val)
                    else:
                        wtd = w * comp_val
                    contrib_pct[comp_name] = wtd / total_val * 100.0

        for name, value in comps.items():
            if name == "total":
                continue

            if math.isnan(value) or math.isinf(value):
                table.add_row(
                    name,
                    Text("NaN!", style="bold red"),
                    Text("─", style="dim"),
                    Text("─", style="dim"),
                    Text("─", style="dim"),
                )
                continue

            val_str = f"{value:.4f}"
            invert = name not in HIGHER_IS_BETTER

            # Contribution percentage
            pct = contrib_pct.get(name)
            if pct is not None:
                pct_text = Text(f"{pct:.0f}%", style="dim")
            else:
                pct_text = Text("─", style="dim")

            # Instant delta
            if len(self.history) >= 2:
                prev = self.history[-2]
                prev_comps = prev.train_components if is_train else prev.val_components
                prev_val = prev_comps.get(name, value)
                d = delta_text(value, prev_val, invert=invert, precision=4)
            else:
                d = Text("─", style="dim")

            # Rolling delta
            vals = []
            for h in self.history:
                c = h.train_components if is_train else h.val_components
                v = c.get(name)
                if v is not None and not math.isnan(v):
                    vals.append(v)

            rd = self._rolling_delta(vals)
            if rd is not None:
                if abs(rd) < 1e-5:
                    rd_text = Text(f" {abs(rd):.5f}", style="dim")
                elif (rd < 0) == invert:
                    arrow = "▼" if rd < 0 else "▲"
                    rd_text = Text(f"{arrow}{abs(rd):.5f}", style="green")
                else:
                    arrow = "▼" if rd < 0 else "▲"
                    rd_text = Text(f"{arrow}{abs(rd):.5f}", style="red")
            else:
                rd_text = Text("─", style="dim")

            table.add_row(name, val_str, pct_text, d, rd_text)

        return Panel(
            table,
            title=f"[bold {title_style}]{title} Losses[/bold {title_style}]",
            border_style=border,
        )

    # ── Log panel ────────────────────────────────────────────────────────

    def _render_log(self, max_height: int = 10) -> Panel:
        # Panel border takes 2 lines; show as many recent entries as fit
        max_lines = max(1, max_height - 2)
        visible = list(self.logs)[-max_lines:]
        text = Text()
        for entry in visible:
            ts = entry.timestamp.strftime("%H:%M:%S")
            style = LEVEL_STYLES.get(entry.level, "white")
            text.append(f"[{ts}] ", style="dim")
            text.append(f"{entry.level:7s} ", style=style)
            text.append(f"{entry.message}\n")
        if not visible:
            text.append("No log messages yet.", style="dim")
        return Panel(
            text,
            title=f"[bold {BORDER_LOG}]Log[/bold {BORDER_LOG}]",
            border_style=BORDER_LOG,
            height=max_height,
        )


# ── History loading helper ───────────────────────────────────────────────────

def _load_history_entries(history_path: str) -> list[EpochData]:
    with open(history_path) as f:
        history = json.load(f)
    return [
        EpochData(
            epoch=e["epoch"],
            train_psnr=e["train_psnr"],
            val_psnr=e["val_psnr"],
            train_components=e["train_components"],
            val_components=e["val_components"],
            lr=e["lr"],
            epoch_time=e["time"],
            train_time=e.get("train_time", 0.0),
            val_time=e.get("val_time", 0.0),
        )
        for e in history
    ]


# ── Replay ───────────────────────────────────────────────────────────────────

def replay_history(history_path: str, animate: bool = False):
    entries = _load_history_entries(history_path)
    total = len(entries)

    # Try to load loss weights from sibling config.json
    loss_weights: dict[str, float] = {}
    config_path = Path(history_path).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            for key in ("l1", "msssim", "gradient", "chroma", "fft",
                        "texture", "zipper", "color_bias"):
                w = cfg.get(f"{key}_weight", 0.0)
                if w:
                    loss_weights[key] = w
        except Exception:
            pass

    dashboard = TrainingDashboard(total_epochs=total, rolling_window=10,
                                  loss_weights=loss_weights)

    if animate:
        with dashboard:
            dashboard.log(f"Replaying {history_path} ({total} epochs)")
            for data in entries:
                dashboard.update(data)
                if dashboard.has_fatal_error:
                    dashboard.log("Replay stopped due to fatal error.", "ERROR")
                    time.sleep(2)
                    break
                time.sleep(0.02)
    else:
        dashboard.start_time = time.time()
        dashboard.bulk_load(entries)
        dashboard.print_static()


# ── Mock simulation ──────────────────────────────────────────────────────────

def mock_training(total_epochs: int = 300, fast: bool = False):
    import random
    rng = random.Random(42)
    delay = 0.05 if fast else 0.15

    def _train_psnr(ep):
        return 18 + 10 * (1 - math.exp(-ep / 80)) + rng.gauss(0, 0.3)

    def _val_psnr(ep):
        return 20 + 14 * (1 - math.exp(-ep / 100)) + rng.gauss(0, 0.5)

    def _loss(ep, base, decay):
        return max(0, base * math.exp(-ep * decay) + rng.gauss(0, 0.0005))

    comp_cfg = {
        "l1_recon": (0.05, 0.005), "l1_known": (0.02, 0.004),
        "l1": (0.05, 0.005), "gradient": (0.15, 0.003),
        "chroma": (0.01, 0.003), "color_bias": (0.008, 0.004),
    }

    dashboard = TrainingDashboard(total_epochs=total_epochs, rolling_window=10)

    with dashboard:
        dashboard.log("Training started")
        dashboard.log("Device: cuda | Mode: train | AMP: enabled")

        for ep in range(total_epochs):
            base_t = 23.5 + rng.gauss(0, 1.5)
            if rng.random() < 0.02:
                base_t += rng.uniform(5, 15)

            train_comps, val_comps = {}, {}
            for name, (base, decay) in comp_cfg.items():
                train_comps[name] = _loss(ep, base, decay)
                val_comps[name] = _loss(ep, base, decay) * 0.9
            train_comps["msssim"] = min(1.0, 0.998 + ep * 3e-6 + rng.gauss(0, 0.0001))
            val_comps["msssim"] = min(1.0, 0.998 + ep * 3e-6 + rng.gauss(0, 0.0001))

            lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * ep / total_epochs))
            if ep == 250:
                train_comps["l1_recon"] = float("nan")

            if rng.random() < 0.7:
                patches = rng.choice([8000, 9600, 11200])
                dashboard.log(f"Cache: swapped {patches} patches ({patches / 56000 * 100:.1f}%)")
            if rng.random() < 0.01:
                dashboard.log(
                    f"Gradient norm spike: {rng.uniform(8, 20):.1f} (avg: 1.3)", "WARN",
                )

            dashboard.update(EpochData(
                epoch=ep + 1, train_psnr=_train_psnr(ep), val_psnr=_val_psnr(ep),
                train_components=train_comps, val_components=val_comps,
                lr=lr, epoch_time=base_t,
                train_time=base_t * 0.85, val_time=base_t * 0.13,
            ))

            if dashboard.has_fatal_error:
                dashboard.log("Aborting training due to fatal error.", "ERROR")
                time.sleep(2)
                break
            time.sleep(delay)

    print("\nSimulation complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Dashboard mock demo")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--fast", action="store_true", help="Faster simulation")
    p.add_argument("--replay", type=str, default=None,
                    help="Replay from a history.json file")
    p.add_argument("--animate", action="store_true",
                    help="Animate replay epoch-by-epoch (default: instant render)")
    args = p.parse_args()

    if args.replay:
        replay_history(args.replay, animate=args.animate)
    else:
        mock_training(total_epochs=args.epochs, fast=args.fast)
