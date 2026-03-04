#!/usr/bin/env python3
"""Export XTransUNet to ONNX for browser inference via ONNX Runtime Web."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import onnx
from onnxconverter_common import float16

from model import XTransUNet
from checkpoint_registry import REGISTRY_FILENAME


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_metadata(ckpt: dict) -> dict:
    """Extract model metadata from checkpoint (no optimizer/scheduler)."""
    state = ckpt.get("model", {})
    return {
        "epoch": ckpt.get("epoch", 0),
        "base_width": ckpt.get("base_width", 64),
        "hl_head": ckpt.get("hl_head", False),
        "param_count": sum(v.numel() for v in state.values()),
    }


def export(checkpoint_path: str, output_path: str, patch_size: int = 288, opset: int = 18, fp16: bool = False, base_width: int | None = None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bw = base_width or ckpt.get("base_width", 64)
    hl_head = ckpt.get("hl_head", False)
    model = XTransUNet(base_width=bw, hl_head=hl_head)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 5, patch_size, patch_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Convert external data to a single self-contained file
    # (needed for ONNX Runtime Web which can't load external data files)
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)

    if fp16:
        onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)

    # Remove external data file if it exists
    ext_data = Path(output_path + ".data")
    if ext_data.exists():
        ext_data.unlink()

    # Save as single file with all weights inlined
    onnx.save(onnx_model, output_path,
              save_as_external_data=False)

    metadata = _extract_metadata(ckpt)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    dtype = "float16" if fp16 else "float32"
    metadata["size_mb"] = round(size_mb, 1)
    metadata["dtype"] = dtype

    print(f"Exported: {output_path} ({size_mb:.1f} MB, {dtype})")
    print(f"Opset: {opset}, Patch size: {patch_size}x{patch_size}")

    return metadata


def verify(checkpoint_path: str, onnx_path: str, patch_size: int = 288, base_width: int | None = None):
    """Compare PyTorch and ONNX outputs to ensure numerical equivalence."""
    import onnxruntime as ort

    # PyTorch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bw = base_width or ckpt.get("base_width", 64)
    hl_head = ckpt.get("hl_head", False)
    model = XTransUNet(base_width=bw, hl_head=hl_head)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_input = torch.randn(1, 5, patch_size, patch_size)
    with torch.no_grad():
        pt_output = model(test_input).numpy()

    # ONNX
    sess = ort.InferenceSession(onnx_path)
    ort_input = test_input.numpy()
    input_meta = sess.get_inputs()[0]
    if input_meta.type == "tensor(float16)":
        ort_input = ort_input.astype(np.float16)
    ort_output = sess.run(None, {"input": ort_input})[0].astype(np.float32)

    # Compare
    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))

    mse = np.mean((pt_output - ort_output) ** 2)
    signal_range = np.max(pt_output) - np.min(pt_output)
    psnr = 10 * np.log10(signal_range**2 / mse) if mse > 0 else float("inf")

    is_fp16 = input_meta.type == "tensor(float16)"
    psnr_threshold = 35 if is_fp16 else 60

    print(f"\nVerification ({('fp16' if is_fp16 else 'fp32')}):")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  PSNR:      {psnr:.1f} dB")

    if psnr > psnr_threshold:
        print("  PASS")
    else:
        print(f"  WARN: PSNR below {psnr_threshold} dB, outputs may not match closely")


def _iter_registry(registry: dict, *, cfa_type=None, base_width=None, variant=None, status=None, slot="best"):
    """Yield (label, checkpoint_path, meta) tuples from registry, applying filters."""
    for sensor, widths in registry.items():
        if cfa_type and sensor != cfa_type:
            continue
        for width_key, variants in widths.items():
            if base_width and width_key != str(base_width):
                continue
            for var_name, statuses in variants.items():
                if variant and var_name != variant:
                    continue
                # Pick requested status, or first available (stable preferred)
                for st_name in ([status] if status else ["stable", "beta"]):
                    if st_name not in statuses:
                        continue
                    slots = statuses[st_name]
                    if slot not in slots:
                        continue
                    entry = slots[slot]
                    label = f"{sensor}_w{width_key}_{var_name}"
                    yield label, entry["path"], entry
                    break  # only one status per variant


def main():
    parser = argparse.ArgumentParser(description="Export XTransUNet to ONNX")

    # Registry-based batch export (default)
    parser.add_argument("--cfa-type", default=None, help="Filter by sensor type (xtrans, bayer)")
    parser.add_argument("--base-width", type=int, default=None, help="Filter by base width (16, 32, 64)")
    parser.add_argument("--variant", default=None, choices=["hl", "base"], help="Filter by variant")
    parser.add_argument("--status", default=None, choices=["stable", "beta"], help="Filter by status (default: prefer stable)")
    parser.add_argument("--slot", default="best", choices=["best", "latest"], help="Which checkpoint slot to export")
    parser.add_argument("--output-dir", default="web/public/checkpoints", help="Output directory for batch export")

    # Single-checkpoint override (legacy)
    parser.add_argument("--checkpoint", default=None, help="Export a single checkpoint (skips registry)")
    parser.add_argument("--output", default=None, help="Output path (only with --checkpoint)")

    # Export options
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--fp16", action="store_true", help="Convert weights to float16")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX vs PyTorch output")
    parser.add_argument("--force", action="store_true", help="Re-export even if source checkpoint unchanged")
    args = parser.parse_args()

    if args.checkpoint:
        # Legacy single-file mode
        out = args.output or "web/public/model.onnx"
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = export(args.checkpoint, out, args.patch_size, args.opset, args.fp16)
        meta["file"] = out_path.name
        meta["source_sha256"] = _file_sha256(args.checkpoint)
        manifest = {out_path.stem: meta}
        manifest_path = out_path.parent / "models.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest: {manifest_path}")
        if args.verify:
            verify(args.checkpoint, out, args.patch_size)
        return

    # Registry-based batch export
    registry_path = Path(__file__).parent / REGISTRY_FILENAME
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        print("Run `python checkpoint_registry.py` to build it first.")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = list(_iter_registry(
        registry,
        cfa_type=args.cfa_type,
        base_width=args.base_width,
        variant=args.variant,
        status=args.status,
        slot=args.slot,
    ))

    if not entries:
        print("No matching checkpoints found in registry.")
        return

    # Load existing manifest for SHA-based skip
    manifest_path = out_dir / "models.json"
    old_manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            old_manifest = json.load(f)

    print(f"Exporting {len(entries)} checkpoint(s) to {out_dir}/\n")

    manifest = {}
    n_skipped = 0
    for label, ckpt_path, reg_entry in entries:
        onnx_file = f"{label}.onnx"
        onnx_path = str(out_dir / onnx_file)
        sha = _file_sha256(ckpt_path)

        # Skip if source unchanged and ONNX file still exists
        old_entry = old_manifest.get(label, {})
        if not args.force and old_entry.get("source_sha256") == sha and (out_dir / onnx_file).exists():
            print(f"--- {label}: up to date (skipped)")
            manifest[label] = old_entry
            n_skipped += 1
            continue

        print(f"--- {label} (epoch {reg_entry['epoch']}, val_psnr {reg_entry['val_psnr']:.2f} dB) ---")
        ckpt_meta = export(ckpt_path, onnx_path, args.patch_size, args.opset, args.fp16)
        if args.verify:
            verify(ckpt_path, onnx_path, args.patch_size)
        print()

        manifest[label] = {
            **ckpt_meta,
            "file": onnx_file,
            "source_sha256": sha,
            "train_psnr": reg_entry.get("train_psnr"),
            "val_psnr": reg_entry.get("val_psnr"),
            "val_hl_psnr": reg_entry.get("val_hl_psnr"),
            "train_loss": reg_entry.get("train_loss"),
            "val_loss": reg_entry.get("val_loss"),
        }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_exported = len(manifest) - n_skipped
    print(f"Manifest: {manifest_path} ({n_exported} exported, {n_skipped} skipped)")


if __name__ == "__main__":
    main()
