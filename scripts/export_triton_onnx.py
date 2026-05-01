#!/usr/bin/env python3
"""
Export local DINOv2 checkpoint to Triton-ready ONNX model repository.

Usage:
  source .venv/bin/activate
  python scripts/export_triton_onnx.py \
      --checkpoint ml/checkpoints/stage2_best.pt \
      --model-name dinov2_vitg14_reg
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"

import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from inference.dinov2_model import DINOv2Classifier  # noqa: E402


class TritonOnnxWrapper(nn.Module):
    """Triton-friendly output shape: [B, 1]"""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.core(x)  # [B]
        return logits.unsqueeze(-1)  # [B, 1]


def load_model(checkpoint: Path, model_name: str) -> nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = DINOv2Classifier(model_name=model_name)
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict):
        state = ckpt.get("state_dict")
    if state is None:
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] partial load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return TritonOnnxWrapper(model).eval()


def write_config(model_root: Path) -> None:
    cfg = model_root / "config.pbtxt"
    cfg.write_text(
        textwrap.dedent(
            """
            name: "dinov2_classifier"
            backend: "onnxruntime"
            max_batch_size: 8

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
                dims: [ 1 ]
              }
            ]

            dynamic_batching {
              preferred_batch_size: [ 1, 2, 4, 8 ]
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def export_onnx(model: nn.Module, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 518, 518, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input__0"],
        output_names=["output__0"],
        dynamic_axes={"input__0": {0: "batch"}, "output__0": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ml/checkpoints/stage2_best.pt")
    p.add_argument("--model-name", default="dinov2_vitg14_reg")
    p.add_argument("--model-repo", default="models")
    args = p.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve()
    model_repo = (ROOT / args.model_repo).resolve()
    model_root = model_repo / "dinov2_classifier"
    onnx_path = model_root / "1" / "model.onnx"

    model = load_model(checkpoint, args.model_name)
    export_onnx(model, onnx_path)
    write_config(model_root)

    print(f"[ok] ONNX exported: {onnx_path}")
    print(f"[ok] Triton config: {model_root / 'config.pbtxt'}")


if __name__ == "__main__":
    main()

