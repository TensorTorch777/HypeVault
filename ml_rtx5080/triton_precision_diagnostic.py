"""Diagnose GPU Triton parity failures on synthetic inputs only.

Compares FP32 CPU PyTorch references against Triton GPU served with ONNX Runtime's
default TF32 math and with use_tf32=0. Inputs are seeded random pixel images, so the
pre-registered parity inputs stay unseen for any configuration chosen from this result.
Not a parity test and not used for routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
for path in (_PKG, _REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT = _PKG / "experiments" / "dual_model_triton_v1" / "gpu_precision_diagnostic.json"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _synthetic(n: int, side: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pixels = rng.uniform(0.0, 1.0, size=(n, 3, side, side)).astype(np.float32)
    return np.ascontiguousarray((pixels - MEAN) / STD)


def _triton(url: str, name: str, batch: np.ndarray, size: int) -> np.ndarray:
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=url)
    out = []
    for start in range(0, len(batch), size):
        chunk = batch[start : start + size]
        inputs = [grpcclient.InferInput("input__0", list(chunk.shape), "FP32")]
        inputs[0].set_data_from_numpy(chunk)
        out.append(client.infer(name, inputs, model_version="1", client_timeout=120.0).as_numpy("output__0").reshape(-1))
    return np.concatenate(out).astype(np.float64)


def _dinov2_reference(batch: np.ndarray) -> np.ndarray:
    import torch

    from inference.dinov2_model import DINOv2Classifier

    config = json.loads((_PKG / "checkpoints" / "config.json").read_text())
    model = DINOv2Classifier("vit_base_patch14_dinov2.lvd142m", dropout=config["dropout"])
    model.load_state_dict(torch.load(_PKG / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(batch[i : i + 4])).numpy() for i in range(0, len(batch), 4)]).astype(np.float64)


def _dinov3_reference(batch: np.ndarray) -> np.ndarray:
    import torch

    from reference_inference import forward_logits, load_reference_model

    model, _digest = load_reference_model(_PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt", torch.device("cpu"))
    return torch.cat([forward_logits(model, torch.from_numpy(batch[i : i + 4]), torch.device("cpu")) for i in range(0, len(batch), 4)]).numpy().astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:28001", help="Scratch Triton serving diag_*_tf32on and diag_*_tf32off")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    cases = (
        ("dinov2", 504, "diag_dinov2_tf32on", "diag_dinov2_tf32off", _dinov2_reference),
        ("dinov3", 512, "diag_dinov3_tf32on", "diag_dinov3_tf32off", _dinov3_reference),
    )
    report = {
        "purpose": "diagnose GPU parity failures; synthetic inputs only; not a parity result",
        "inputs": {"kind": "seeded uniform random pixels, ImageNet-normalized", "samples": args.samples, "seed": args.seed},
        "real_images_opened": 0,
        "models": {},
    }
    for label, side, default_name, off_name, reference_fn in cases:
        batch = _synthetic(args.samples, side, args.seed)
        reference = reference_fn(batch)
        row = {
            "reference": "FP32 CPU PyTorch",
            "triton_default_model": f"{default_name} (reviewed GPU config, renamed)",
            "triton_tf32_off_model": f"{off_name} (reviewed GPU config + use_tf32=0)",
        }
        for variant, name in (("tf32_default", default_name), ("tf32_off", off_name)):
            for size in (1, 8):
                values = _triton(args.url, name, batch, size)
                errors = np.abs(values - reference)
                row[f"{variant}_batch{size}"] = {"max_abs_logit_error": float(errors.max()), "mean_abs_logit_error": float(errors.mean())}
        row["reference_logit_range"] = [float(reference.min()), float(reference.max())]
        report["models"][label] = row
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["models"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
