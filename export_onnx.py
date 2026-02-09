#!/usr/bin/env python3
"""Export XTransUNet to ONNX for browser inference via ONNX Runtime Web."""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx

from model import XTransUNet


def export(checkpoint_path: str, output_path: str, patch_size: int = 288, opset: int = 17):
    model = XTransUNet()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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

    # Remove external data file if it exists
    ext_data = Path(output_path + ".data")
    if ext_data.exists():
        ext_data.unlink()

    # Save as single file with all weights inlined
    onnx.save(onnx_model, output_path,
              save_as_external_data=False)

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")
    print(f"Opset: {opset}, Patch size: {patch_size}x{patch_size}")

    return output_path


def verify(checkpoint_path: str, onnx_path: str, patch_size: int = 288):
    """Compare PyTorch and ONNX outputs to ensure numerical equivalence."""
    import onnxruntime as ort

    # PyTorch
    model = XTransUNet()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_input = torch.randn(1, 4, patch_size, patch_size)
    with torch.no_grad():
        pt_output = model(test_input).numpy()

    # ONNX
    sess = ort.InferenceSession(onnx_path)
    ort_output = sess.run(None, {"input": test_input.numpy()})[0]

    # Compare
    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))

    mse = np.mean((pt_output - ort_output) ** 2)
    signal_range = np.max(pt_output) - np.min(pt_output)
    psnr = 10 * np.log10(signal_range**2 / mse) if mse > 0 else float("inf")

    print(f"\nVerification:")
    print(f"  Max diff:  {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  PSNR:      {psnr:.1f} dB")

    if psnr > 60:
        print("  PASS")
    else:
        print("  WARN: PSNR below 60 dB, outputs may not match closely")


def main():
    parser = argparse.ArgumentParser(description="Export XTransUNet to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints_v5.1/best.pt")
    parser.add_argument("--output", default="web/model.onnx")
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="Verify ONNX vs PyTorch output")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export(args.checkpoint, args.output, args.patch_size, args.opset)

    if args.verify:
        verify(args.checkpoint, args.output, args.patch_size)


if __name__ == "__main__":
    main()
