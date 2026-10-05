#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-present X-Veon contributors
"""
Export one checkpoint version to ONNX for the app.

    python export_onnx.py --version v7.0.0 [--cfa-type xtrans]
    python export_onnx.py --checkpoint path/to/best.pt

The app loads models by fixed keys, `{cfa}_w{base_width}_base` (see CHECKPOINT_POLICY.md),
so exactly one version is exported at a time and it must be named. The manifest
(models.json) is updated entry by entry: exporting one CFA type keeps the other's entry.
Every file is checked against PyTorch before it replaces the one in place, and a file
left in place because its checkpoint is unchanged is checked again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from cfa import CFA_REGISTRY, cfa_period, make_channel_masks, make_model_input
from checkpoint_registry import REGISTRY_FILENAME
from model import ARCHITECTURE_TAG, XTransUNet

DEFAULT_OUTPUT_DIR = "shared/public/checkpoints"
MANIFEST = "models.json"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def app_key(cfa_type: str, base_width: int) -> str:
    """The manifest key the app looks a model up by."""
    return f"{cfa_type}_w{base_width}_base"


def load_model(ckpt: dict[str, Any], source: str) -> XTransUNet:
    """Build the model a checkpoint belongs to and load its weights strictly."""
    tag = ckpt.get("architecture_tag")
    if tag != ARCHITECTURE_TAG:
        raise SystemExit(
            f"{source} has architecture tag {tag} (version {ckpt.get('checkpoint_version')}); "
            f"this code builds {ARCHITECTURE_TAG} models and cannot load it."
        )
    cfa_type = ckpt.get("cfa_type", "xtrans")
    model = XTransUNet(
        base_width=int(ckpt.get("base_width", 16)),
        cfa_period=cfa_period(CFA_REGISTRY[cfa_type]),
        stages=int(ckpt.get("stages", 2)),
    )
    model.load_state_dict(ckpt["model"])  # strict: a mismatched layout must fail, not load partially
    model.eval()
    return model


def _metadata(ckpt: dict[str, Any]) -> dict[str, Any]:
    state = ckpt.get("model", {})
    return {
        "epoch": ckpt.get("epoch", 0),
        "base_width": ckpt.get("base_width", 16),
        "stages": ckpt.get("stages", 2),
        "cfa_type": ckpt.get("cfa_type"),
        "checkpoint_version": ckpt.get("checkpoint_version"),
        "checkpoint_major": ckpt.get("checkpoint_major"),
        "architecture_tag": ckpt.get("architecture_tag"),
        "param_count": sum(v.numel() for v in state.values()),
    }


def export(checkpoint_path: str, output_path: str, patch_size: int = 288, opset: int = 18) -> dict[str, Any]:
    """Write a self-contained float32 ONNX file; return its manifest metadata."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_model(ckpt, checkpoint_path)

    dummy = torch.rand(1, 5, patch_size, patch_size)
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Inline the weights: ONNX Runtime Web cannot load external data files.
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)
    ext_data = Path(output_path + ".data")
    if ext_data.exists():
        ext_data.unlink()
    onnx.save(onnx_model, output_path, save_as_external_data=False)

    metadata = _metadata(ckpt)
    metadata["size_mb"] = round(Path(output_path).stat().st_size / 1024 / 1024, 1)
    metadata["dtype"] = "float32"
    return metadata


def _input_dims(onnx_path: str) -> list[list[int | str]]:
    """The shape of each input of an ONNX file; dynamic axes appear by name."""
    graph = onnx.load(onnx_path, load_external_data=False).graph
    return [[d.dim_param if d.HasField("dim_param") else d.dim_value for d in i.type.tensor_type.shape.dim]
            for i in graph.input]


def verify_tiles(patch_size: int) -> dict[str, torch.Tensor]:
    """Mosaics [1, 1, P, P] that verify() sends through both models as one batch."""
    g = torch.Generator().manual_seed(0)
    p = patch_size
    clipped = torch.rand(1, 1, p, p, generator=g)                     # half above half the clip level
    clipped[..., : p // 4, :] = 1.0                                     # saturated photosites
    clipped[..., p // 4 : p // 2, :] = 1.0 - 0.03 * torch.rand(1, 1, p // 4, p, generator=g)   # just below
    return {
        "tile": torch.rand(1, 1, p, p, generator=g) * 0.2,
        "tile with clipped photosites": clipped,
        "all-zero tile": torch.zeros(1, 1, p, p),                       # image-border padding in the app
    }


def verify(checkpoint_path: str, onnx_path: str, patch_size: int = 288) -> None:
    """Compare PyTorch and ONNX on a batch of tiles like the app's, each on its own scale. Exits on a mismatch."""
    import onnxruntime as ort

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_model(ckpt, checkpoint_path)
    masks = make_channel_masks(patch_size, patch_size, CFA_REGISTRY[ckpt.get("cfa_type", "xtrans")])
    session = ort.InferenceSession(onnx_path)
    inputs = session.get_inputs()
    if len(inputs) != 1 or list(inputs[0].shape)[1:] != [5, patch_size, patch_size]:
        raise SystemExit(f"{onnx_path}: unexpected inputs {[(i.name, i.shape) for i in inputs]}")

    tiles = verify_tiles(patch_size)
    x = make_model_input(torch.cat(list(tiles.values())), masks)
    with torch.no_grad():
        expected = model(x).numpy()
    got = session.run(None, {inputs[0].name: x.numpy()})[0]
    if got.shape != expected.shape:
        raise SystemExit(f"{onnx_path}: output shape {got.shape} for a batch of {len(tiles)}, expected {expected.shape}")
    for i, name in enumerate(tiles):
        if not np.isfinite(got[i]).all():
            raise SystemExit(f"{onnx_path}: non-finite output for the {name}")
        max_diff = float(np.max(np.abs(expected[i] - got[i])))
        scale = float(np.max(np.abs(expected[i]))) or 1.0
        print(f"  verify, {name}: max difference {max_diff:.2e} (output up to {scale:.2e})")
        if max_diff > 1e-4 * scale + 1e-7:
            raise SystemExit(f"{onnx_path}: ONNX and PyTorch disagree on the {name}")
    print("  PASS")


def select(registry: dict[str, Any], *, version: str | None, cfa_type: str | None = None,
           status: str | None = None, slot: str = "best") -> list[tuple[str, str, dict[str, Any]]]:
    """(app key, checkpoint path, registry entry) for the one named version, per CFA type.

    The registry holds one entry per sensor type and version, with one width, and the key
    names the sensor type: two selections cannot share a key.
    """
    if not version:
        raise SystemExit("--version is required: name the one checkpoint version to export, e.g. v7.0.0")
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for sensor, versions in registry.items():
        if cfa_type and sensor != cfa_type:
            continue
        meta = versions.get(version)
        if meta is None:
            continue
        for status_name in ([status] if status else ["stable", "beta"]):
            entry = meta.get(status_name, {}).get(slot)
            if entry is None:
                continue
            key = app_key(sensor, int(meta.get("base_width", 16)))
            selected.append((key, entry["path"], {**entry, "registry_status": status_name}))
            break
    if not selected:
        raise SystemExit(f"no checkpoint of version {version} in the registry"
                         + (f" for {cfa_type}" if cfa_type else ""))
    return selected


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Replace the manifest in one step, so it is never left half written."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


def export_entries(entries: list[tuple[str, str, dict[str, Any]]], out_dir: Path, *,
                   patch_size: int = 288, opset: int = 18, force: bool = False) -> dict[str, Any]:
    """Export (app key, checkpoint path, registry entry) triples and merge them into the manifest.

    The manifest is read first and only the exported keys are replaced. Each file is written
    beside its target and verified there; only a verified file replaces the target, and only
    then is its manifest entry updated. A failed verify leaves the previous file and entry.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for key, ckpt_path, entry in entries:
        onnx_path = out_dir / f"{key}.onnx"
        sha = _file_sha256(ckpt_path)
        if not force and manifest.get(key, {}).get("source_sha256") == sha and onnx_path.exists():
            # The checkpoint is unchanged, but the file in place must still be what it gives.
            print(f"--- {key}: up to date, checking {onnx_path.name}")
            dims = _input_dims(str(onnx_path))
            if len(dims) == 1 and dims[0][1:2] == [5] and dims[0][2:] != [patch_size, patch_size]:
                raise SystemExit(f"{onnx_path} was exported at a different patch size ({dims[0][2]}x{dims[0][3]}, "
                                 f"this run uses {patch_size}); re-run with --force to replace it")
            verify(ckpt_path, str(onnx_path), patch_size)
            print(f"--- {key}: up to date (skipped)")
            continue
        print(f"--- {key} from {ckpt_path}")
        tmp_path = out_dir / f".{key}.verifying.onnx"
        try:
            meta = export(ckpt_path, str(tmp_path), patch_size, opset)
            verify(ckpt_path, str(tmp_path), patch_size)
            tmp_path.replace(onnx_path)
        finally:
            tmp_path.unlink(missing_ok=True)
            Path(f"{tmp_path}.data").unlink(missing_ok=True)
        print(f"Exported: {onnx_path} ({meta['size_mb']} MB), opset {opset}, patch {patch_size}")
        manifest[key] = {
            **meta,
            "file": onnx_path.name,
            "source_sha256": sha,
            "registry_status": entry.get("registry_status"),
            "train_psnr": entry.get("train_psnr"),
            "val_psnr": entry.get("val_psnr"),
            "train_loss": entry.get("train_loss"),
            "val_loss": entry.get("val_loss"),
        }
        _write_manifest(manifest_path, manifest)

    print(f"Manifest: {manifest_path}")
    return manifest


def export_selected(registry: dict[str, Any], out_dir: Path, *, version: str | None,
                    cfa_type: str | None = None, status: str | None = None, slot: str = "best",
                    patch_size: int = 288, opset: int = 18, force: bool = False) -> dict[str, Any]:
    """Export the named version from the registry."""
    entries = select(registry, version=version, cfa_type=cfa_type, status=status, slot=slot)
    return export_entries(entries, out_dir, patch_size=patch_size, opset=opset, force=force)


def export_checkpoint(checkpoint_path: str, out_dir: Path, *, patch_size: int = 288,
                      opset: int = 18) -> dict[str, Any]:
    """Export one checkpoint file under the app key its CFA type and width give it."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = app_key(ckpt.get("cfa_type", "xtrans"), int(ckpt.get("base_width", 16)))
    return export_entries([(key, checkpoint_path, {})], out_dir, patch_size=patch_size, opset=opset,
                          force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one checkpoint version to ONNX for the app")
    parser.add_argument("--version", default=None,
                        help="Checkpoint version to export from the registry, e.g. v7.0.0 (required unless --checkpoint)")
    parser.add_argument("--checkpoint", default=None, help="Export this checkpoint file instead of a registry version")
    parser.add_argument("--cfa-type", default=None, choices=["xtrans", "bayer"], help="Export only this sensor type")
    parser.add_argument("--status", default=None, choices=["stable", "beta"], help="Registry status (default: prefer stable)")
    parser.add_argument("--slot", default="best", choices=["best", "latest"], help="Which checkpoint slot to export")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder holding the ONNX files and models.json")
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verify", action="store_true",
                        help="Accepted for compatibility; every export is checked against PyTorch")
    parser.add_argument("--force", action="store_true", help="Re-export even if the source checkpoint is unchanged")
    args = parser.parse_args()

    if args.checkpoint:
        export_checkpoint(args.checkpoint, Path(args.output_dir), patch_size=args.patch_size,
                          opset=args.opset)
        return

    registry_path = Path(__file__).parent / REGISTRY_FILENAME
    if not registry_path.exists():
        raise SystemExit(f"Registry not found: {registry_path}. Run `python checkpoint_registry.py` to build it.")
    registry = json.loads(registry_path.read_text())
    export_selected(registry, Path(args.output_dir), version=args.version, cfa_type=args.cfa_type,
                    status=args.status, slot=args.slot, patch_size=args.patch_size, opset=args.opset,
                    force=args.force)


if __name__ == "__main__":
    main()
