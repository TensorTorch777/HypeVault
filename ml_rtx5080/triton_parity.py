"""Triton parity for dinov3_authenticity_candidate against the FP32 PyTorch reference.

Run once with --write-protocol to freeze inputs and tolerances, then run without it
to measure. The measurement refuses to start if the protocol file changed. It reads
only non-test images, does not train, and does not change temperature or threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

OUT = _PKG / "experiments" / "dual_model_triton_v1"
PROTOCOL = OUT / "parity_protocol_v1.json"
PROTOCOL_SHA = OUT / "parity_protocol_v1.sha256"
RESULTS = OUT / "parity_results.json"
CASES_CSV = _PKG / "experiments" / "backend_parity_v2" / "onnx_parity.csv"
SPLIT_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
CANDIDATE_ONNX = _REPO / "models" / "dinov3_authenticity_candidate" / "1" / "model.onnx"
EXPORT_IDENTITY = _REPO / "models" / "dinov3_authenticity_candidate" / "1" / "identity.json"

FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
FROZEN_TEMPERATURE = 0.24038200410185356
THRESHOLD = 0.5
LOGIT_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
BATCH_SIZES = (1, 2, 4, 8)
MODEL_NAME = "dinov3_authenticity_candidate"
MODEL_VERSION = "1"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_membership() -> set[str]:
    payload = json.loads(SPLIT_MANIFEST.read_text())
    return {str(Path(p).resolve()) for p in (payload.get("membership") or {}).get("test") or []}


def _cases() -> list[dict]:
    test = _test_membership()
    rows = []
    for row in csv.DictReader(CASES_CSV.open()):
        path = (_REPO / row["sample_id"]).resolve()
        if row["split"] == "test" or str(path) in test:
            raise RuntimeError(f"parity case {row['sample_id']} is in the locked final test")
        rows.append({"sample_id": row["sample_id"], "split": row["split"], "role": row["role"]})
    return rows


def write_protocol() -> dict:
    if PROTOCOL.exists():
        raise RuntimeError(f"{PROTOCOL} already exists. A protocol is written once, before measurement.")
    cases = _cases()
    protocol = {
        "protocol": "dinov3_triton_parity_v1",
        "model": {"triton_name": MODEL_NAME, "version": MODEL_VERSION},
        "reference": "reference_inference.load_reference_model + forward_logits, FP32, CPU, no autocast, no temperature",
        "preprocessing": "dinov3_serving.preprocess_serving_image (resize_pad_square_eval_v1, 3x512x512 FP32)",
        "frozen_checkpoint_sha256": FROZEN_SHA,
        "frozen_temperature": FROZEN_TEMPERATURE,
        "decision_threshold": THRESHOLD,
        "probability": "sigmoid(logit / frozen_temperature)",
        "tolerances": {"max_abs_logit_error": LOGIT_ATOL, "max_abs_probability_error": PROBABILITY_ATOL},
        "pass_rule": "every batch size within both tolerances and zero decision mismatches at the threshold",
        "batch_sizes": list(BATCH_SIZES),
        "inputs_source": str(CASES_CSV.relative_to(_REPO)),
        "inputs": cases,
        "final_test_images_opened": 0,
        "tolerances_may_change_after_results": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode()
    PROTOCOL.write_bytes(raw)
    PROTOCOL_SHA.write_text(f"{_sha256_bytes(raw)}  {PROTOCOL.name}\n")
    return protocol


def _load_protocol() -> dict:
    raw = PROTOCOL.read_bytes()
    recorded = PROTOCOL_SHA.read_text().split()[0]
    if _sha256_bytes(raw) != recorded:
        raise RuntimeError("parity protocol changed after it was hashed")
    protocol = json.loads(raw)
    if protocol["tolerances"] != {"max_abs_logit_error": LOGIT_ATOL, "max_abs_probability_error": PROBABILITY_ATOL}:
        raise RuntimeError("tolerances differ from the pre-registered protocol")
    return protocol


def _probability(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit / FROZEN_TEMPERATURE))


def _decision(probability: float) -> str:
    return "FAKE" if probability >= THRESHOLD else "AUTHENTIC"


def _triton_infer(client, batch):
    import tritonclient.grpc as grpcclient

    inputs = [grpcclient.InferInput("input__0", list(batch.shape), "FP32")]
    inputs[0].set_data_from_numpy(batch)
    outputs = [grpcclient.InferRequestedOutput("output__0")]
    result = client.infer(MODEL_NAME, inputs, model_version=MODEL_VERSION, outputs=outputs, client_timeout=120.0)
    return result.as_numpy("output__0").reshape(-1)


def _model_instance_kind(client) -> str:
    config = client.get_model_config(MODEL_NAME, MODEL_VERSION, as_json=True)["config"]
    kinds = {group.get("kind", "KIND_AUTO") for group in config.get("instance_group", [])}
    return ",".join(sorted(kinds)) or "KIND_AUTO"


def run(url: str, require_gpu: bool) -> dict:
    import numpy as np
    import torch
    import tritonclient.grpc as grpcclient
    from PIL import Image

    from dinov3_serving import preprocess_serving_image
    from reference_inference import forward_logits, load_reference_model

    protocol = _load_protocol()
    checkpoint_sha = _file_sha256(CHECKPOINT)
    if checkpoint_sha != FROZEN_SHA:
        raise RuntimeError("frozen checkpoint hash mismatch; parity not run")
    identity = json.loads(EXPORT_IDENTITY.read_text())
    export_sha = _file_sha256(CANDIDATE_ONNX)
    if identity["source_checkpoint_sha256"] != FROZEN_SHA or identity["export_artifact_sha256"] != export_sha:
        raise RuntimeError("candidate export identity does not match the frozen checkpoint")

    client = grpcclient.InferenceServerClient(url=url)
    if not (client.is_server_live() and client.is_model_ready(MODEL_NAME, MODEL_VERSION)):
        raise RuntimeError(f"{MODEL_NAME} version {MODEL_VERSION} is not ready at {url}")
    kind = _model_instance_kind(client)
    if require_gpu and "KIND_GPU" not in kind:
        raise RuntimeError(f"GPU parity requested but the model instance kind is {kind}")

    test = _test_membership()
    tensors, sizes = [], []
    for case in protocol["inputs"]:
        path = (_REPO / case["sample_id"]).resolve()
        if str(path) in test:
            raise RuntimeError("refusing to open a final-test image")
        with Image.open(path) as handle:
            sizes.append([handle.width, handle.height])
            tensors.append(preprocess_serving_image(handle.convert("RGB")))
    batch_all = torch.stack(tensors).float()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, ref_digest = load_reference_model(CHECKPOINT, torch.device("cpu"))
    started = time.perf_counter()
    reference = torch.cat([forward_logits(model, batch_all[i : i + 4], torch.device("cpu")) for i in range(0, len(tensors), 4)])
    reference_seconds = time.perf_counter() - started
    reference = reference.numpy().astype(np.float64)
    del model

    array_all = np.ascontiguousarray(batch_all.numpy().astype(np.float32))
    per_batch = {}
    rows = []
    for size in protocol["batch_sizes"]:
        triton_logits = []
        for start in range(0, len(array_all), size):
            triton_logits.extend(_triton_infer(client, array_all[start : start + size]).astype(np.float64).tolist())
        logit_err = [abs(a - b) for a, b in zip(triton_logits, reference.tolist(), strict=True)]
        prob_err, mismatches = [], 0
        for i, (t, r) in enumerate(zip(triton_logits, reference.tolist(), strict=True)):
            pt, pr = _probability(t), _probability(r)
            prob_err.append(abs(pt - pr))
            if _decision(pt) != _decision(pr):
                mismatches += 1
            if size == 1:
                case = protocol["inputs"][i]
                rows.append(
                    {
                        "sample_id": case["sample_id"],
                        "split": case["split"],
                        "original_size": sizes[i],
                        "reference_logit": r,
                        "triton_logit": t,
                        "abs_logit_error": abs(t - r),
                        "reference_probability": pr,
                        "triton_probability": pt,
                        "abs_probability_error": abs(pt - pr),
                        "reference_decision": _decision(pr),
                        "triton_decision": _decision(pt),
                    }
                )
        per_batch[str(size)] = {
            "samples": len(triton_logits),
            "max_abs_logit_error": max(logit_err),
            "mean_abs_logit_error": sum(logit_err) / len(logit_err),
            "max_abs_probability_error": max(prob_err),
            "mean_abs_probability_error": sum(prob_err) / len(prob_err),
            "decision_mismatches": mismatches,
            "within_tolerance": max(logit_err) <= LOGIT_ATOL and max(prob_err) <= PROBABILITY_ATOL and mismatches == 0,
        }

    passed = all(entry["within_tolerance"] for entry in per_batch.values())
    metadata = client.get_server_metadata(as_json=True)
    results = {
        "status": "PASS" if passed else "FAIL",
        "scope": "CPU Triton parity" if "KIND_GPU" not in kind else "GPU Triton parity",
        "gpu_inference_claimed": "KIND_GPU" in kind,
        "protocol": str(PROTOCOL.relative_to(_REPO)),
        "protocol_sha256": PROTOCOL_SHA.read_text().split()[0],
        "tolerances": protocol["tolerances"],
        "tolerances_widened_after_results": False,
        "model": {"triton_name": MODEL_NAME, "version": MODEL_VERSION, "instance_kind": kind},
        "triton_server": {"version": metadata.get("version"), "url": url},
        "checkpoint_sha256": checkpoint_sha,
        "reference_checkpoint_sha256": ref_digest,
        "export_artifact_sha256": export_sha,
        "reference": {"device": "cpu", "dtype": "float32", "torch": torch.__version__, "seconds": reference_seconds},
        "host": {"python": platform.python_version(), "machine": platform.machine()},
        "samples": len(rows),
        "splits": sorted({row["split"] for row in rows}),
        "original_sizes": sorted({tuple(row["original_size"]) for row in rows}),
        "per_batch_size": per_batch,
        "rows": rows,
        "final_test_images_opened": 0,
        "validated_equivalent": passed,
    }
    results["original_sizes"] = [list(size) for size in results["original_sizes"]]
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--url", default="localhost:18001")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        protocol = write_protocol()
        print(f"wrote {PROTOCOL} with {len(protocol['inputs'])} inputs; sha256 {PROTOCOL_SHA.read_text().split()[0]}")
        return 0
    results = run(args.url, args.require_gpu)
    print(json.dumps({k: results[k] for k in ("status", "scope", "samples", "per_batch_size")}, indent=2))
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
