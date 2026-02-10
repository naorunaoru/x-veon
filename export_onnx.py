#!/usr/bin/env python3
"""Export XTransUNet to ONNX for browser inference via ONNX Runtime Web."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import onnx
from onnxconverter_common import float16

from model import XTransUNet


def _extract_metadata(ckpt: dict) -> dict[str, str]:
    """Extract metadata from checkpoint as string key-value pairs for ONNX."""
    meta = {}
    meta["epoch"] = str(ckpt.get("epoch", ""))
    meta["best_val_psnr"] = f"{ckpt['best_val_psnr']:.2f}" if "best_val_psnr" in ckpt else ""
    meta["base_width"] = str(ckpt.get("base_width", ""))

    # Model summary
    state = ckpt.get("model", {})
    n_params = sum(v.numel() for v in state.values())
    meta["param_count"] = str(n_params)

    # Optimizer config
    opt = ckpt.get("optimizer", {})
    if "param_groups" in opt and opt["param_groups"]:
        pg = opt["param_groups"][0]
        meta["optimizer"] = json.dumps({
            "type": "AdamW",
            "lr": pg.get("initial_lr", pg.get("lr")),
            "weight_decay": pg.get("weight_decay"),
            "betas": pg.get("betas"),
        })

    # Scheduler config
    sched = ckpt.get("scheduler", {})
    if sched:
        meta["scheduler"] = json.dumps({
            "type": "CosineAnnealingLR",
            "T_max": sched.get("T_max"),
            "eta_min": sched.get("eta_min"),
        })

    return {k: v for k, v in meta.items() if v}


def export(checkpoint_path: str, output_path: str, patch_size: int = 288, opset: int = 17, fp16: bool = False, base_width: int | None = None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bw = base_width or ckpt.get("base_width", 64)
    model = XTransUNet(base_width=bw)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 4, patch_size, patch_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
    )

    # Convert external data to a single self-contained file
    # (needed for ONNX Runtime Web which can't load external data files)
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)

    if fp16:
        onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)

    # Embed checkpoint metadata
    metadata = _extract_metadata(ckpt)
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value

    # Remove external data file if it exists
    ext_data = Path(output_path + ".data")
    if ext_data.exists():
        ext_data.unlink()

    # Save as single file with all weights inlined
    onnx.save(onnx_model, output_path,
              save_as_external_data=False)

    # Write sidecar JSON (onnxruntime-web doesn't expose metadata_props)
    meta_path = Path(output_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    dtype = "float16" if fp16 else "float32"
    print(f"Exported: {output_path} ({size_mb:.1f} MB, {dtype})")
    print(f"Opset: {opset}, Patch size: {patch_size}x{patch_size}")
    print(f"Metadata: {meta_path} {metadata}")

    return output_path


def verify(checkpoint_path: str, onnx_path: str, patch_size: int = 288, base_width: int | None = None):
    """Compare PyTorch and ONNX outputs to ensure numerical equivalence."""
    import onnxruntime as ort

    # PyTorch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bw = base_width or ckpt.get("base_width", 64)
    model = XTransUNet(base_width=bw)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_input = torch.randn(1, 4, patch_size, patch_size)
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


def main():
    parser = argparse.ArgumentParser(description="Export XTransUNet to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints_v5.1/best.pt")
    parser.add_argument("--output", default="web/public/model.onnx")
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Convert weights to float16")
    parser.add_argument("--base-width", type=int, default=None,
                        help="Base channel width (auto-detected from checkpoint if saved)")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX vs PyTorch output")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export(args.checkpoint, args.output, args.patch_size, args.opset, args.fp16, args.base_width)

    if args.verify:
        verify(args.checkpoint, args.output, args.patch_size, args.base_width)


if __name__ == "__main__":
    main()
