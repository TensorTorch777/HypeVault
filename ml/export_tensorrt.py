"""
HypeVault — ONNX → TensorRT FP16 Export
=========================================
Run this on the cloud GPU after training to produce the TRT engine
that Triton Inference Server loads.

Usage:
    python export_tensorrt.py \
        --onnx checkpoints/dinov2_hypevault.onnx \
        --engine checkpoints/dinov2_hypevault_fp16.trt \
        --img_size 518 \
        --max_batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_engine(onnx_path: Path, engine_path: Path, img_size: int, max_batch: int):
    """Build a TensorRT FP16 engine from an ONNX file."""
    try:
        import tensorrt as trt
    except ImportError:
        raise SystemExit(
            "TensorRT Python bindings not found.\n"
            "Install with:  pip install tensorrt\n"
            "or use the NVIDIA container: nvcr.io/nvidia/tensorrt:24.05-py3"
        )

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # ── Builder settings ──────────────────────────────────────────
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB workspace

        # FP16 — primary precision for RTX 6000 Pro Blackwell
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled ✓")
        else:
            print("  WARNING: FP16 not supported on this GPU, using FP32")

        # ── Parse ONNX ────────────────────────────────────────────────
        print(f"Parsing ONNX: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model.")
        print("  ONNX parsed ✓")

        # ── Dynamic shapes ────────────────────────────────────────────
        profile = builder.create_optimization_profile()
        input_name = "input"
        profile.set_shape(
            input_name,
            min=(1,  3, img_size, img_size),
            opt=(8,  3, img_size, img_size),   # typical single request batch
            max=(max_batch, 3, img_size, img_size),
        )
        config.add_optimization_profile(profile)

        # ── Build engine ──────────────────────────────────────────────
        print("Building TensorRT engine (this takes 5–15 minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed.")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"\nEngine saved: {engine_path}")
        print(f"Engine size:  {engine_path.stat().st_size / 1e6:.1f} MB")


def verify_engine(engine_path: Path, img_size: int):
    """Quick sanity-check: run one forward pass through the TRT engine."""
    import numpy as np
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except ImportError:
        print("pycuda not installed — skipping verify. Install with: pip install pycuda")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("input", (1, 3, img_size, img_size))

    dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    d_input  = cuda.mem_alloc(dummy_input.nbytes)
    d_output = cuda.mem_alloc(4)  # 1 × float32 logit

    cuda.memcpy_htod(d_input, dummy_input)
    context.execute_v2([int(d_input), int(d_output)])

    result = np.empty(1, dtype=np.float32)
    cuda.memcpy_dtoh(result, d_output)

    print(f"Verify forward pass: logit = {result[0]:.4f} (sigmoid → {1/(1+np.exp(-result[0])):.4f})")
    print("TRT engine verified ✓")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",     type=Path, default=Path("checkpoints/dinov2_hypevault.onnx"))
    p.add_argument("--engine",   type=Path, default=Path("checkpoints/dinov2_hypevault_fp16.trt"))
    p.add_argument("--img_size", type=int,  default=518)
    p.add_argument("--max_batch",type=int,  default=32)
    p.add_argument("--verify",   action="store_true", help="Run a test forward pass after build")
    args = p.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX not found: {args.onnx}\nRun train.py first — it auto-exports ONNX at the end.")

    build_engine(args.onnx, args.engine, args.img_size, args.max_batch)

    if args.verify:
        verify_engine(args.engine, args.img_size)


if __name__ == "__main__":
    main()
