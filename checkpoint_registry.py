#!/usr/bin/env python3
"""Centralized checkpoint registry.

Maintains checkpoint_registry.json with structure:
  sensor_type → checkpoint_version → {base_width, major, stable?/beta?}

Each track contains best/latest slots with path, epoch, train_psnr, val_psnr,
train_loss, val_loss, history.
"""

import json
import re
import tempfile
from pathlib import Path

REGISTRY_FILENAME = "checkpoint_registry.json"


_HISTORICAL_VERSION_RE = re.compile(
    r"(?:^|_)v(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?(?P<suffix>[a-z]+)?(?:-w(?P<width>\d+))?$"
)


def _normalize_version(raw: str, *, base_width: int) -> str | None:
    """Normalize historical or canonical checkpoint tags to the new scheme.

    Examples:
      checkpoints_bayer_v6.1.4q -> v6.1.4
      checkpoints_xtrans_v6.1.4h -> v6.1.4-w32
      v6.1.5 -> v6.1.5
    """
    m = _HISTORICAL_VERSION_RE.search(raw)
    if not m:
        return None

    major = int(m.group("major"))
    if major < 6:
        return None

    minor = int(m.group("minor") or 0)
    patch = int(m.group("patch") or 0)
    width = int(m.group("width") or base_width)
    version = f"v{major}.{minor}.{patch}"
    if width != 16:
        version += f"-w{width}"
    return version


def infer_checkpoint_version(config: dict, ckpt_dir: Path) -> str | None:
    """Infer canonical checkpoint version from config or historical dirname."""
    version = config.get("checkpoint_version")
    if version:
        return version
    return _normalize_version(ckpt_dir.name, base_width=int(config.get("base_width", 16)))


def _checkpoint_major(version: str) -> int | None:
    m = re.match(r"^v(\d+)\.(\d+)\.(\d+)(?:-w\d+)?$", version)
    return int(m.group(1)) if m else None


def _version_sort_key(version: str) -> tuple[int, int, int, int]:
    m = re.match(r"^v(\d+)\.(\d+)\.(\d+)(?:-w(\d+))?$", version)
    if not m:
        return (-1, -1, -1, -1)
    return tuple(int(x or 0) for x in m.groups())


def _load_registry(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_registry(path: Path, data: dict):
    # Atomic write via temp file + rename
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    )
    try:
        json.dump(data, tmp, indent=2)
        tmp.close()
        Path(tmp.name).replace(path)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise


def update_registry(
    registry_path: Path,
    *,
    cfa_type: str,
    checkpoint_version: str,
    base_width: int,
    status: str,  # "stable" or "beta"
    slot: str,    # "best" or "latest"
    path: str,
    epoch: int,
    train_psnr: float,
    val_psnr: float,
    train_loss: float,
    val_loss: float,
    history: str,
): 
    """Update a single slot in the registry."""
    reg = _load_registry(registry_path)

    sensor = reg.setdefault(cfa_type, {})
    version_entry = sensor.setdefault(checkpoint_version, {
        "base_width": base_width,
        "major": _checkpoint_major(checkpoint_version),
    })
    version_entry["base_width"] = base_width
    version_entry["major"] = _checkpoint_major(checkpoint_version)

    st = version_entry.setdefault(status, {})

    st[slot] = {
        "path": path,
        "epoch": epoch,
        "train_psnr": round(train_psnr, 4),
        "val_psnr": round(val_psnr, 4),
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "history": history,
        "checkpoint_version": checkpoint_version,
        "base_width": base_width,
    }

    _save_registry(registry_path, reg)


def promote_to_stable(
    registry_path: Path,
    *,
    cfa_type: str,
    checkpoint_version: str,
    base_width: int,
):
    """Flip a beta entry to stable (called when training completes all epochs)."""
    reg = _load_registry(registry_path)

    try:
        version_entry = reg[cfa_type][checkpoint_version]
    except KeyError:
        return

    if version_entry.get("base_width") != base_width:
        return

    if "beta" in version_entry:
        version_entry["stable"] = version_entry.pop("beta")
        _save_registry(registry_path, reg)


def build_registry(project_root: Path) -> dict:
    """Scan checkpoint dirs and rebuild the registry using canonical versions.

    Legacy pre-v6 families are skipped unless they declare an explicit
    checkpoint_version in config.json.
    """
    reg = {}
    registry_path = project_root / REGISTRY_FILENAME

    for config_path in sorted(project_root.glob("checkpoints/**/config.json")):
        ckpt_dir = config_path.parent
        history_path = ckpt_dir / "history.json"
        if not history_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)
        with open(history_path) as f:
            history = json.load(f)

        if not history:
            continue

        cfa_type = config.get("cfa_type", "xtrans")
        base_width = config.get("base_width", 64)
        checkpoint_version = infer_checkpoint_version(config, ckpt_dir)
        if not checkpoint_version:
            continue
        total_epochs = config.get("epochs", 200)
        last_epoch = history[-1]["epoch"]

        # History stores zero-based epoch indices in newer runs. Treat a run as
        # complete when it reached the final scheduled epoch index.
        status = "stable" if last_epoch >= (total_epochs - 1) else "beta"

        # Find best epoch by val_psnr
        best_entry = max(history, key=lambda e: e.get("val_psnr", 0))
        latest_entry = history[-1]

        rel_dir = str(ckpt_dir.relative_to(project_root))
        history_rel = str(history_path.relative_to(project_root))

        def _slot(entry, pt_name):
            return {
                "path": f"{rel_dir}/{pt_name}",
                "epoch": entry["epoch"],
                "train_psnr": round(entry.get("train_psnr", 0), 4),
                "val_psnr": round(entry.get("val_psnr", 0), 4),
                "train_loss": round(entry.get("train_loss", 0), 6),
                "val_loss": round(entry.get("val_loss", 0), 6),
                "history": history_rel,
                "checkpoint_version": checkpoint_version,
                "base_width": base_width,
            }

        sensor = reg.setdefault(cfa_type, {})
        version_entry = sensor.setdefault(checkpoint_version, {
            "base_width": base_width,
            "major": _checkpoint_major(checkpoint_version),
        })
        st = version_entry.setdefault(status, {})

        if (ckpt_dir / "best.pt").exists():
            st["best"] = _slot(best_entry, "best.pt")
        if (ckpt_dir / "latest.pt").exists():
            st["latest"] = _slot(latest_entry, "latest.pt")

    # Sort versions newest-first for human readability / stable iteration order
    for sensor, versions in list(reg.items()):
        reg[sensor] = dict(sorted(
            versions.items(),
            key=lambda kv: _version_sort_key(kv[0]),
            reverse=True,
        ))

    _save_registry(registry_path, reg)
    return reg


if __name__ == "__main__":
    root = Path(__file__).parent
    reg = build_registry(root)
    print(json.dumps(reg, indent=2))
    print(f"\nWritten to {root / REGISTRY_FILENAME}")
