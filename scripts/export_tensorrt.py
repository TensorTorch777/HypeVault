#!/usr/bin/env python3
"""
Export `facebook/dinov2-giant` + binary head → ONNX → TensorRT engine + Triton model repo layout.

Requires: GPU with TensorRT, PyTorch, transformers, onnx, tensorrt (NVIDIA). This script is a
blueprint — unimplemented blocks raise clear errors until training artifacts exist.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "dinov2_finetuned.pth"
MODEL_DIR = ROOT / "models" / "dinov2_classifier" / "1"
ONNX_PATH = MODEL_DIR / "model.onnx"


def write_triton_config() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ROOT / "models" / "dinov2_classifier" / "config.pbtxt"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        textwrap.dedent(
            """
            name: "dinov2_classifier"
            backend: "tensorrt"
            input [
              {
                name: "input__0"
                data_type: TYPE_FP32
                dims: [ 3, 518, 518 ]
              }
            ]
            output [
              {
                name: "output__0"
                data_type: TYPE_FP32
                dims: [ 2 ]
              }
            ]
            dynamic_batching {
              preferred_batch_size: [ 8, 16 ]
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    write_triton_config()
    if not CKPT.exists():
        raise SystemExit(
            "Missing fine-tuned checkpoint at ./checkpoints/dinov2_finetuned.pth. "
            "Train the binary head (1536→2) before export."
        )

    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install PyTorch in an environment with CUDA before running export.") from exc

    raise SystemExit(
        "Next steps (implement locally on a GPU workstation):\n"
        "1) Load facebook/dinov2-giant backbone from Hugging Face.\n"
        "2) Attach nn.Linear(1536, 2) head; load state_dict from checkpoints/dinov2_finetuned.pth.\n"
        "3) torch.onnx.export(..., input names ['input__0'], output names ['output__0']).\n"
        f"4) Build TensorRT FP16 engine from {ONNX_PATH} and place as Triton model artifact.\n"
        "5) Point Triton --model-repository at ./models.\n"
    )


if __name__ == "__main__":
    main()
