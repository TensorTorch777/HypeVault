"""Parity for dinov2_vitb14_live, the live ViT-B/14 listing-check model served by Triton.

The reference is the live route's own code: backend preprocess_chw at 504, the backend
DINOv2Classifier loaded from ml_rtx5080/checkpoints/best_model.pt, logits_to_verdict, and the
0.88 minimum-authentic-confidence floor.

  --write-protocol      freeze inputs, artifact hashes, and tolerances (run once, before measuring)
  --write-gpu-protocol  same inputs and tolerances, pinned to the reviewed KIND_GPU config
  --write-gpu-protocol-v2  after attempt 1 failed on TF32: same again, pinned to the exact-FP32 GPU config
  --stage export        host onnxruntime (CPU) on the exact Triton model bytes vs the references
  --stage triton        Triton gRPC vs the references; --require-gpu uses the GPU protocol and refuses a CPU instance

Reads only non-test images. Does not train, export, or change thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
for path in (_PKG, _REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT = _PKG / "experiments" / "dual_model_triton_v1"
PROTOCOL = OUT / "dinov2_live_parity_protocol_v1.json"
PROTOCOL_SHA = OUT / "dinov2_live_parity_protocol_v1.sha256"
GPU_PROTOCOL = OUT / "dinov2_live_triton_gpu_protocol_v1.json"
GPU_PROTOCOL_SHA = OUT / "dinov2_live_triton_gpu_protocol_v1.sha256"
GPU_PROTOCOL_V2 = OUT / "dinov2_live_triton_gpu_protocol_v2.json"
GPU_PROTOCOL_V2_SHA = OUT / "dinov2_live_triton_gpu_protocol_v2.sha256"
GPU_ATTEMPT1 = OUT / "dinov2_live_triton_gpu_parity_attempt1.json"
RESULTS = {
    "export": OUT / "dinov2_live_export_parity.json",
    "triton_cpu": OUT / "dinov2_live_triton_cpu_parity.json",
    "triton_gpu": OUT / "dinov2_live_triton_gpu_parity.json",
}
INPUTS_FROM = OUT / "parity_protocol_v1.json"
SPLIT_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"

CHECKPOINT = _PKG / "checkpoints" / "best_model.pt"
TRAIN_CONFIG = _PKG / "checkpoints" / "config.json"
MODEL_DIR = _REPO / "models" / "dinov2_vitb14_live" / "1"
ONNX = MODEL_DIR / "model.onnx"
ONNX_DATA = MODEL_DIR / "dinov2_hypevault.onnx.data"
TRACKED_CONFIG = _REPO / "infra" / "triton" / "dinov2_vitb14_live" / "config.pbtxt"
TRACKED_GPU_CONFIG = _REPO / "infra" / "triton" / "dinov2_vitb14_live" / "config.gpu.pbtxt"
TRACKED_GPU_FP32_CONFIG = _REPO / "infra" / "triton" / "dinov2_vitb14_live" / "config.gpu_fp32.pbtxt"
INSTALLED_CONFIG = _REPO / "models" / "dinov2_vitb14_live" / "config.pbtxt"

MODEL_NAME = "dinov2_vitb14_live"
MODEL_VERSION = "1"
BACKBONE = "vit_base_patch14_dinov2.lvd142m"
INPUT_SIDE = 504
MIN_AUTHENTIC = 0.88
LOGIT_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
BATCH_SIZES = (1, 2, 4, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_membership() -> set[str]:
    payload = json.loads(SPLIT_MANIFEST.read_text())
    return {str(Path(p).resolve()) for p in (payload.get("membership") or {}).get("test") or []}


def _artifact_hashes(triton_config: Path = TRACKED_CONFIG) -> dict:
    return {
        "source_checkpoint": {"path": str(CHECKPOINT.relative_to(_REPO)), "sha256": _sha256(CHECKPOINT)},
        "training_config": {"path": str(TRAIN_CONFIG.relative_to(_REPO)), "sha256": _sha256(TRAIN_CONFIG)},
        "onnx_graph": {"path": str(ONNX.relative_to(_REPO)), "sha256": _sha256(ONNX)},
        "onnx_external_data": {"path": str(ONNX_DATA.relative_to(_REPO)), "sha256": _sha256(ONNX_DATA)},
        "triton_config": {"path": str(triton_config.relative_to(_REPO)), "sha256": _sha256(triton_config)},
    }


def write_protocol() -> dict:
    if PROTOCOL.exists():
        raise RuntimeError(f"{PROTOCOL} already exists. A protocol is written once, before measurement.")
    test = _test_membership()
    inputs = json.loads(INPUTS_FROM.read_text())["inputs"]
    for case in inputs:
        if case["split"] == "test" or str((_REPO / case["sample_id"]).resolve()) in test:
            raise RuntimeError(f"{case['sample_id']} is in the locked final test")
    protocol = {
        "protocol": "dinov2_live_triton_parity_v1",
        "model": {"triton_name": MODEL_NAME, "version": MODEL_VERSION, "backbone": BACKBONE, "input_side": INPUT_SIDE},
        "artifacts": _artifact_hashes(),
        "preprocessing": "backend inference.triton_client.preprocess_chw(np.array(PIL.Image.open(path).convert('RGB')), side=504)",
        "references": {
            "fp32": "backend DINOv2Classifier(vit_base_patch14_dinov2.lvd142m) with best_model.pt model_state loaded strict, CPU, FP32, no autocast",
            "live_path": "backend inference.local_torch._infer_sync on CUDA (float16 autocast), the code the live route runs",
        },
        "verdict_policy": "inference.verdict.logits_to_verdict then apply_min_authentic_confidence(min_authentic=0.88)",
        "probability": "sigmoid(logit); no temperature",
        "tolerances_vs_fp32": {"max_abs_logit_error": LOGIT_ATOL, "max_abs_probability_error": PROBABILITY_ATOL},
        "pass_rule": (
            "every batch size within both FP32 tolerances, zero final-verdict mismatches vs the FP32 reference, "
            "and zero final-verdict mismatches vs the live FP16 path. Logit deltas vs the live FP16 path are reported, not toleranced."
        ),
        "batch_sizes": list(BATCH_SIZES),
        "routing_rule": "the research selector may route to this model only after the triton stage passes with --require-gpu",
        "inputs_source": str(INPUTS_FROM.relative_to(_REPO)),
        "inputs": inputs,
        "final_test_images_opened": 0,
        "tolerances_may_change_after_results": False,
    }
    raw = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode()
    PROTOCOL.write_bytes(raw)
    PROTOCOL_SHA.write_text(f"{hashlib.sha256(raw).hexdigest()}  {PROTOCOL.name}\n")
    return protocol


def write_gpu_protocol() -> dict:
    """Pin the GPU-stage run to the reviewed KIND_GPU config. Inputs, tolerances, and rule come from v1 unchanged."""
    if GPU_PROTOCOL.exists():
        raise RuntimeError(f"{GPU_PROTOCOL} already exists. A protocol is written once, before measurement.")
    if RESULTS["triton_gpu"].exists():
        raise RuntimeError("a Triton result already exists; the GPU protocol must precede it")
    base = _load_protocol(PROTOCOL, PROTOCOL_SHA, TRACKED_CONFIG)
    protocol = dict(base)
    protocol["protocol"] = "dinov2_live_triton_gpu_parity_v1"
    protocol["derived_from"] = {"protocol": PROTOCOL.name, "sha256": PROTOCOL_SHA.read_text().split()[0]}
    protocol["artifacts"] = _artifact_hashes(TRACKED_GPU_CONFIG)
    protocol["stage"] = "triton with --require-gpu"
    protocol["why_separate"] = "v1 pins the KIND_CPU config used for the export stage; GPU serving needs the reviewed KIND_GPU config."
    raw = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode()
    GPU_PROTOCOL.write_bytes(raw)
    GPU_PROTOCOL_SHA.write_text(f"{hashlib.sha256(raw).hexdigest()}  {GPU_PROTOCOL.name}\n")
    return protocol


def write_gpu_protocol_v2() -> dict:
    """Second GPU attempt: same inputs, tolerances, and rule; config pinned to exact FP32 (use_tf32=0)."""
    if GPU_PROTOCOL_V2.exists():
        raise RuntimeError(f"{GPU_PROTOCOL_V2} already exists. A protocol is written once, before measurement.")
    attempt1 = json.loads(GPU_ATTEMPT1.read_text())
    if attempt1["status"] != "FAIL":
        raise RuntimeError("v2 exists only to follow a recorded attempt-1 failure")
    base = _load_protocol(GPU_PROTOCOL, GPU_PROTOCOL_SHA, TRACKED_GPU_CONFIG)
    protocol = dict(base)
    protocol["protocol"] = "dinov2_live_triton_gpu_parity_v2"
    protocol["derived_from"] = {"protocol": GPU_PROTOCOL.name, "sha256": GPU_PROTOCOL_SHA.read_text().split()[0]}
    protocol["artifacts"] = _artifact_hashes(TRACKED_GPU_FP32_CONFIG)
    protocol["attempt1"] = {
        "result": GPU_ATTEMPT1.name,
        "status": attempt1["status"],
        "max_abs_logit_error_vs_fp32": max(e["max_abs_logit_error_vs_fp32"] for e in attempt1["per_batch_size"].values()),
        "cause": "ONNX Runtime CUDA EP uses TF32 by default (use_tf32=1); confirmed on synthetic inputs in gpu_precision_diagnostic.json",
    }
    protocol["why_separate"] = "the only change from v1 is the serving config, which adds use_tf32=0 so GEMMs run in exact FP32"
    raw = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode()
    GPU_PROTOCOL_V2.write_bytes(raw)
    GPU_PROTOCOL_V2_SHA.write_text(f"{hashlib.sha256(raw).hexdigest()}  {GPU_PROTOCOL_V2.name}\n")
    return protocol


def _load_protocol(path: Path = PROTOCOL, sha_path: Path = PROTOCOL_SHA, config: Path = TRACKED_CONFIG) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha_path.read_text().split()[0]:
        raise RuntimeError("protocol changed after it was hashed")
    protocol = json.loads(raw)
    if protocol["tolerances_vs_fp32"] != {"max_abs_logit_error": LOGIT_ATOL, "max_abs_probability_error": PROBABILITY_ATOL}:
        raise RuntimeError("tolerances differ from the pre-registered protocol")
    current = _artifact_hashes(config)
    if current != protocol["artifacts"]:
        raise RuntimeError("an artifact changed since the protocol was written; parity not run")
    return protocol


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _verdict(logit: float) -> str:
    import numpy as np

    from inference.verdict import apply_min_authentic_confidence, logits_to_verdict

    raw, confidence = logits_to_verdict(np.asarray([logit], dtype=np.float32))
    return apply_min_authentic_confidence(raw, confidence, MIN_AUTHENTIC)[0]


def _references(protocol: dict):
    import numpy as np
    import torch
    from PIL import Image

    from inference.dinov2_model import DINOv2Classifier
    from inference.local_torch import _infer_sync
    from inference.triton_client import preprocess_chw

    test = _test_membership()
    arrays, sizes = [], []
    for case in protocol["inputs"]:
        path = (_REPO / case["sample_id"]).resolve()
        if str(path) in test:
            raise RuntimeError("refusing to open a final-test image")
        with Image.open(path) as handle:
            sizes.append([handle.width, handle.height])
            arrays.append(preprocess_chw(np.array(handle.convert("RGB")), side=INPUT_SIDE)[0])
    batch = np.ascontiguousarray(np.stack(arrays).astype(np.float32))

    model = DINOv2Classifier(BACKBONE, dropout=json.loads(TRAIN_CONFIG.read_text())["dropout"])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state"]
    load = model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        fp32 = np.concatenate([model(torch.from_numpy(batch[i : i + 4])).numpy() for i in range(0, len(batch), 4)])
    live = None
    if torch.cuda.is_available():
        model.to("cuda")
        live = np.concatenate([_infer_sync(model, torch.device("cuda"), batch[i : i + 4]) for i in range(0, len(batch), 4)])
    return batch, sizes, fp32.astype(np.float64), (None if live is None else live.astype(np.float64)), {
        "missing_keys": len(load.missing_keys),
        "unexpected_keys": len(load.unexpected_keys),
        "live_path_device": "cuda" if live is not None else None,
        "torch": torch.__version__,
    }


def _compare(protocol, candidate_by_batch: dict, fp32, live, sizes) -> dict:
    per_batch, rows = {}, []
    for size, values in candidate_by_batch.items():
        logit_err = [abs(a - b) for a, b in zip(values, fp32, strict=True)]
        prob_err = [abs(_sigmoid(a) - _sigmoid(b)) for a, b in zip(values, fp32, strict=True)]
        verdicts = [_verdict(v) for v in values]
        fp32_mismatch = sum(1 for v, r in zip(verdicts, fp32, strict=True) if v != _verdict(r))
        live_mismatch = None if live is None else sum(1 for v, r in zip(verdicts, live, strict=True) if v != _verdict(r))
        per_batch[str(size)] = {
            "samples": len(values),
            "max_abs_logit_error_vs_fp32": max(logit_err),
            "mean_abs_logit_error_vs_fp32": sum(logit_err) / len(logit_err),
            "max_abs_probability_error_vs_fp32": max(prob_err),
            "verdict_mismatches_vs_fp32": fp32_mismatch,
            "verdict_mismatches_vs_live_fp16": live_mismatch,
            "max_abs_logit_delta_vs_live_fp16": None if live is None else max(abs(a - b) for a, b in zip(values, live, strict=True)),
            "within_rule": (
                max(logit_err) <= LOGIT_ATOL
                and max(prob_err) <= PROBABILITY_ATOL
                and fp32_mismatch == 0
                and live_mismatch == 0
            ),
        }
    first = candidate_by_batch[min(candidate_by_batch)]
    for i, case in enumerate(protocol["inputs"]):
        rows.append(
            {
                "sample_id": case["sample_id"],
                "split": case["split"],
                "original_size": sizes[i],
                "fp32_logit": fp32[i],
                "live_fp16_logit": None if live is None else live[i],
                "candidate_logit": first[i],
                "fp32_verdict": _verdict(fp32[i]),
                "live_fp16_verdict": None if live is None else _verdict(live[i]),
                "candidate_verdict": _verdict(first[i]),
            }
        )
    return {"per_batch_size": per_batch, "rows": rows}


def run(stage: str, url: str, require_gpu: bool, out_path: Path | None = None, gpu_protocol: str = "v2") -> dict:
    import numpy as np

    if stage == "triton" and require_gpu:
        protocol_path, protocol_sha, config = {
            "v1": (GPU_PROTOCOL, GPU_PROTOCOL_SHA, TRACKED_GPU_CONFIG),
            "v2": (GPU_PROTOCOL_V2, GPU_PROTOCOL_V2_SHA, TRACKED_GPU_FP32_CONFIG),
        }[gpu_protocol]
        protocol = _load_protocol(protocol_path, protocol_sha, config)
        if INSTALLED_CONFIG.read_bytes() != config.read_bytes():
            raise RuntimeError(f"Triton's installed config is not {config.name}; parity not run")
    else:
        protocol_path, protocol_sha = PROTOCOL, PROTOCOL_SHA
        protocol = _load_protocol()
    batch, sizes, fp32, live, reference_meta = _references(protocol)
    if reference_meta["missing_keys"] or reference_meta["unexpected_keys"]:
        raise RuntimeError(f"checkpoint does not load strictly into the live architecture: {reference_meta}")

    candidate: dict[int, list[float]] = {}
    serving: dict = {}
    if stage == "export":
        import onnxruntime as ort

        session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
        serving = {"runtime": f"onnxruntime {ort.__version__}", "provider": session.get_providers()[0]}
        for size in protocol["batch_sizes"]:
            out = []
            for start in range(0, len(batch), size):
                out.extend(session.run(["output__0"], {"input__0": batch[start : start + size]})[0].reshape(-1).astype(np.float64).tolist())
            candidate[size] = out
    else:
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient(url=url)
        if not (client.is_server_live() and client.is_model_ready(MODEL_NAME, MODEL_VERSION)):
            raise RuntimeError(f"{MODEL_NAME} version {MODEL_VERSION} is not ready at {url}")
        config = client.get_model_config(MODEL_NAME, MODEL_VERSION, as_json=True)["config"]
        kinds = sorted({g.get("kind", "KIND_AUTO") for g in config.get("instance_group", [])})
        if require_gpu and "KIND_GPU" not in kinds:
            raise RuntimeError(f"GPU parity requested but the instance kind is {kinds}")
        serving = {"runtime": f"triton {client.get_server_metadata(as_json=True).get('version')}", "instance_kind": kinds, "url": url}
        for size in protocol["batch_sizes"]:
            out = []
            for start in range(0, len(batch), size):
                chunk = batch[start : start + size]
                inputs = [grpcclient.InferInput("input__0", list(chunk.shape), "FP32")]
                inputs[0].set_data_from_numpy(chunk)
                result = client.infer(MODEL_NAME, inputs, model_version=MODEL_VERSION, client_timeout=120.0)
                out.extend(result.as_numpy("output__0").reshape(-1).astype(np.float64).tolist())
            candidate[size] = out

    compared = _compare(protocol, candidate, fp32.tolist(), None if live is None else live.tolist(), sizes)
    passed = live is not None and all(entry["within_rule"] for entry in compared["per_batch_size"].values())
    gpu = stage == "triton" and "KIND_GPU" in serving.get("instance_kind", [])
    results = {
        "status": "PASS" if passed else "FAIL",
        "stage": stage,
        "scope": {"export": "host ONNX Runtime CPU on the Triton model bytes", "triton": "GPU Triton" if gpu else "CPU Triton"}[stage],
        "gpu_inference_claimed": gpu,
        "routing_allowed_by_this_result": bool(passed and gpu),
        "protocol": str(protocol_path.relative_to(_REPO)),
        "protocol_sha256": protocol_sha.read_text().split()[0],
        "artifacts": protocol["artifacts"],
        "serving": serving,
        "reference": reference_meta,
        "host": {"python": platform.python_version(), "machine": platform.machine()},
        "samples": len(compared["rows"]),
        "tolerances_vs_fp32": protocol["tolerances_vs_fp32"],
        "tolerances_widened_after_results": False,
        **compared,
        "final_test_images_opened": 0,
    }
    key = stage if stage == "export" else ("triton_gpu" if require_gpu and gpu else "triton_cpu")
    target = out_path if (out_path and key == "triton_gpu") else RESULTS[key]
    if key == "triton_gpu" and target.exists():
        raise RuntimeError(f"{target.name} exists; give each GPU attempt its own --out file")
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--write-gpu-protocol", action="store_true")
    parser.add_argument("--stage", choices=("export", "triton"))
    parser.add_argument("--url", default="localhost:18001")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="GPU result file for this attempt")
    parser.add_argument("--write-gpu-protocol-v2", action="store_true")
    parser.add_argument("--gpu-protocol", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    if args.write_protocol:
        protocol = write_protocol()
        print(f"wrote {PROTOCOL.name}: {len(protocol['inputs'])} inputs, sha256 {PROTOCOL_SHA.read_text().split()[0]}")
        return 0
    if args.write_gpu_protocol:
        protocol = write_gpu_protocol()
        print(f"wrote {GPU_PROTOCOL.name}: {len(protocol['inputs'])} inputs, sha256 {GPU_PROTOCOL_SHA.read_text().split()[0]}")
        return 0
    if args.write_gpu_protocol_v2:
        protocol = write_gpu_protocol_v2()
        print(f"wrote {GPU_PROTOCOL_V2.name}: {len(protocol['inputs'])} inputs, sha256 {GPU_PROTOCOL_V2_SHA.read_text().split()[0]}")
        return 0
    if not args.stage:
        parser.error("--stage is required unless --write-protocol is given")
    results = run(args.stage, args.url, args.require_gpu, args.out, args.gpu_protocol)
    summary = {k: results[k] for k in ("status", "stage", "scope", "routing_allowed_by_this_result", "reference")}
    summary["per_batch_size"] = results["per_batch_size"]
    print(json.dumps(summary, indent=2))
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
