#!/usr/bin/env python3
"""Export `dinov2_hypevault.onnx` from `checkpoints/best_model.pt` (no training loop)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import train as tv


def main() -> None:
    ckpt_dir = _PKG / "checkpoints"
    cfg = dict(tv.DEFAULTS)
    cfg.update(json.loads((ckpt_dir / "config.json").read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tv.DINOv2Classifier(cfg["model_name"], dropout=cfg["dropout"]).to(device)
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    tv.export_onnx(model, cfg, ckpt_dir, device)
    side = int(cfg["img_size"])
    print(f"ONNX ready for Triton: {ckpt_dir / 'dinov2_hypevault.onnx'} ({side}×{side}, input__0 / output__0)")


if __name__ == "__main__":
    main()
