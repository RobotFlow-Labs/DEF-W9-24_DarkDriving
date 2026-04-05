#!/usr/bin/env python3
"""Export pipeline for DarkDriving enhancement models.

Exports: pth -> SafeTensors -> ONNX -> TensorRT FP16 + FP32

Usage:
    uv run python scripts/export.py --checkpoint best.pth --config configs/paper.toml
    uv run python scripts/export.py --checkpoint best.pth \
        --output-dir /mnt/artifacts-datai/exports/project_darkdriving
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dark_driving.model import get_model
from dark_driving.utils import load_config


def export_safetensors(model: torch.nn.Module, output_path: Path) -> Path:
    """Export model to SafeTensors format."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    path = output_path / "model.safetensors"
    save_file(state_dict, str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"[EXPORT] SafeTensors: {path} ({size_mb:.1f}MB)")
    return path


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (512, 512),
    opset: int = 17,
) -> Path:
    """Export model to ONNX format with dynamic batch size."""
    path = output_path / "model.onnx"
    dummy = torch.randn(1, 3, *input_size, device=next(model.parameters()).device)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = path.stat().st_size / 1e6
    print(f"[EXPORT] ONNX: {path} ({size_mb:.1f}MB, opset={opset})")
    return path


def validate_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    input_size: tuple[int, int] = (512, 512),
    atol: float = 1e-3,
) -> bool:
    """Validate ONNX output matches PyTorch."""
    import onnxruntime as ort

    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, *input_size, device=device)

    # PyTorch output
    model.eval()
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()

    # ONNX output
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.cpu().numpy()})[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    match = max_diff < atol
    status = "PASS" if match else "FAIL"
    print(f"[VALIDATE] ONNX vs PyTorch max_diff={max_diff:.6f} — {status}")
    return match


def export_tensorrt(
    onnx_path: Path,
    output_path: Path,
    input_size: tuple[int, int] = (512, 512),
    fp16: bool = True,
) -> Path | None:
    """Export ONNX to TensorRT engine."""
    precision = "fp16" if fp16 else "fp32"
    trt_path = output_path / f"model_{precision}.trt"

    # Try shared TRT toolkit first
    shared_trt = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if shared_trt.exists():
        import subprocess

        cmd = [
            sys.executable, str(shared_trt),
            "--onnx", str(onnx_path),
            "--output", str(trt_path),
            "--input-shape", f"1,3,{input_size[0]},{input_size[1]}",
        ]
        if fp16:
            cmd.append("--fp16")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and trt_path.exists():
            size_mb = trt_path.stat().st_size / 1e6
            print(f"[EXPORT] TensorRT {precision}: {trt_path} ({size_mb:.1f}MB)")
            return trt_path
        print(f"[WARN] Shared TRT toolkit failed: {result.stderr[:200]}")

    # Fallback: use trtexec or tensorrt Python API
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT ERROR] {parser.get_error(i)}")
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Set input profile for dynamic batch
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            (1, 3, *input_size),      # min
            (4, 3, *input_size),      # opt
            (16, 3, *input_size),     # max
        )
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("[WARN] TensorRT build failed")
            return None

        with open(trt_path, "wb") as f:
            f.write(engine_bytes)

        size_mb = trt_path.stat().st_size / 1e6
        print(f"[EXPORT] TensorRT {precision}: {trt_path} ({size_mb:.1f}MB)")
        return trt_path

    except ImportError:
        print(f"[WARN] TensorRT not available — skipping {precision} export")
        print("[HINT] Install: pip install tensorrt or use trtexec CLI")

        # Try trtexec as last resort
        import shutil
        import subprocess

        trtexec = shutil.which("trtexec")
        if trtexec:
            cmd = [
                trtexec,
                f"--onnx={onnx_path}",
                f"--saveEngine={trt_path}",
                f"--minShapes=input:1x3x{input_size[0]}x{input_size[1]}",
                f"--optShapes=input:4x3x{input_size[0]}x{input_size[1]}",
                f"--maxShapes=input:16x3x{input_size[0]}x{input_size[1]}",
            ]
            if fp16:
                cmd.append("--fp16")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and trt_path.exists():
                size_mb = trt_path.stat().st_size / 1e6
                print(f"[EXPORT] TensorRT {precision} (trtexec): {trt_path} ({size_mb:.1f}MB)")
                return trt_path
            print(f"[WARN] trtexec failed: {result.stderr[:200]}")

        return None


def benchmark_latency(
    model: torch.nn.Module,
    input_size: tuple[int, int] = (512, 512),
    batch_size: int = 1,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> float:
    """Benchmark PyTorch model latency in ms."""
    device = next(model.parameters()).device
    model.eval()
    dummy = torch.randn(batch_size, 3, *input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    print(f"[BENCH] PyTorch latency: {elapsed:.2f}ms (bs={batch_size}, {input_size})")
    return elapsed


def export_all(
    config_path: str,
    checkpoint_path: str,
    output_dir: str | None = None,
) -> dict:
    """Run the full export pipeline: pth -> safetensors -> ONNX -> TRT FP16 + FP32."""
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = tuple(data_cfg.get("input_size", [512, 512]))

    # Load model
    model = get_model(
        name=model_cfg.get("name", "retinexformer"),
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 3),
        embed_dim=model_cfg.get("embed_dim", 32),
        num_blocks=model_cfg.get("num_blocks", 4),
        num_heads=model_cfg.get("num_heads", 4),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[EXPORT] Loaded checkpoint: {checkpoint_path}")

    # Output directory
    if output_dir is None:
        output_dir = "/mnt/artifacts-datai/exports/project_darkdriving"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {"checkpoint": checkpoint_path, "formats": []}

    # 1. SafeTensors
    sf_path = export_safetensors(model, out_path)
    results["formats"].append({"format": "safetensors", "path": str(sf_path)})

    # 2. ONNX
    onnx_path = export_onnx(model, out_path, input_size=input_size)
    results["formats"].append({"format": "onnx", "path": str(onnx_path)})

    # 3. Validate ONNX
    validate_onnx(model, onnx_path, input_size=input_size)

    # 4. TensorRT FP16
    trt16_path = export_tensorrt(onnx_path, out_path, input_size=input_size, fp16=True)
    if trt16_path:
        results["formats"].append({"format": "trt_fp16", "path": str(trt16_path)})

    # 5. TensorRT FP32
    trt32_path = export_tensorrt(onnx_path, out_path, input_size=input_size, fp16=False)
    if trt32_path:
        results["formats"].append({"format": "trt_fp32", "path": str(trt32_path)})

    # 6. Benchmark
    if device.type == "cuda":
        latency = benchmark_latency(model, input_size=input_size)
        results["pytorch_latency_ms"] = latency

    print(f"\n[EXPORT] All exports saved to: {out_path}")
    print(f"[EXPORT] Formats: {[r['format'] for r in results['formats']]}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DarkDriving model export pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--output-dir", type=str, default=None, help="Export output directory")
    args = parser.parse_args()

    export_all(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
