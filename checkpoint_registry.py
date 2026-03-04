#!/usr/bin/env python3
"""Centralized checkpoint registry.

Maintains checkpoint_registry.json with structure:
  sensor_type → base_width → variant (hl/base) → status (stable/beta) → slot (best/latest)

Each slot contains: path, epoch, train_psnr, val_psnr, val_hl_psnr, train_loss, val_loss, history
"""

import json
import tempfile
from glob import glob
from pathlib import Path

REGISTRY_FILENAME = "checkpoint_registry.json"


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
    base_width: int,
    hl_head: bool,
    status: str,  # "stable" or "beta"
    slot: str,    # "best" or "latest"
    path: str,
    epoch: int,
    train_psnr: float,
    val_psnr: float,
    val_hl_psnr: float | None,
    train_loss: float,
    val_loss: float,
    history: str,
):
    """Update a single slot in the registry."""
    reg = _load_registry(registry_path)

    variant = "hl" if hl_head else "base"
    width_key = str(base_width)

    # Navigate/create nesting
    sensor = reg.setdefault(cfa_type, {})
    width = sensor.setdefault(width_key, {})
    var = width.setdefault(variant, {})

    st = var.setdefault(status, {})

    st[slot] = {
        "path": path,
        "epoch": epoch,
        "train_psnr": round(train_psnr, 4),
        "val_psnr": round(val_psnr, 4),
        "val_hl_psnr": round(val_hl_psnr, 4) if val_hl_psnr is not None else None,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "history": history,
    }

    _save_registry(registry_path, reg)


def promote_to_stable(
    registry_path: Path,
    *,
    cfa_type: str,
    base_width: int,
    hl_head: bool,
):
    """Flip a beta entry to stable (called when training completes all epochs)."""
    reg = _load_registry(registry_path)
    variant = "hl" if hl_head else "base"
    width_key = str(base_width)

    try:
        var = reg[cfa_type][width_key][variant]
    except KeyError:
        return

    if "beta" in var and "stable" not in var:
        var["stable"] = var.pop("beta")
        _save_registry(registry_path, reg)


def build_registry(project_root: Path) -> dict:
    """Scan all checkpoint_*/ dirs and rebuild registry from config.json + history.json."""
    reg = {}
    registry_path = project_root / REGISTRY_FILENAME

    for config_path in sorted(project_root.glob("checkpoints_*/config.json")):
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
        hl_head = config.get("hl_head", False)
        total_epochs = config.get("epochs", 200)
        last_epoch = history[-1]["epoch"]

        status = "stable" if last_epoch >= total_epochs else "beta"
        variant = "hl" if hl_head else "base"
        width_key = str(base_width)

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
                "val_hl_psnr": round(entry["val_hl_psnr"], 4) if entry.get("val_hl_psnr") is not None else None,
                "train_loss": round(entry.get("train_loss", 0), 6),
                "val_loss": round(entry.get("val_loss", 0), 6),
                "history": history_rel,
            }

        sensor = reg.setdefault(cfa_type, {})
        width = sensor.setdefault(width_key, {})
        var = width.setdefault(variant, {})
        st = var.setdefault(status, {})

        if (ckpt_dir / "best.pt").exists():
            st["best"] = _slot(best_entry, "best.pt")
        if (ckpt_dir / "latest.pt").exists():
            st["latest"] = _slot(latest_entry, "latest.pt")

    _save_registry(registry_path, reg)
    return reg


if __name__ == "__main__":
    root = Path(__file__).parent
    reg = build_registry(root)
    print(json.dumps(reg, indent=2))
    print(f"\nWritten to {root / REGISTRY_FILENAME}")
