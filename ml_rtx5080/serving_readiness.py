"""Phase 30 serving audit.

Records which GPU runtimes can actually load. It does not install packages,
does not time a runtime that failed to load, and does not open the final test.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import socket
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image

from dinov3_serving import (
    INPUT_SIZE,
    MODEL_NAME,
    ONNX_PATH,
    OLD_MODEL_ROOT,
    file_sha256,
    old_repository_fingerprint,
    preprocess_serving_image,
    probe_cuda_execution_provider,
    single_decode_inputs,
)
from image_quality import apply_quality_flags, extract_quality_features_from_path
from policy import FROZEN_TEMPERATURE
from policy_contract import PHASE23_CUTOFFS, POLICY_VERSION, PolicyEngine, frozen_shadow_configuration
from reference_inference import preprocess_image

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
_OUTPUT = _PKG / "experiments" / "serving_v1"
_GOLDEN = _PKG / "experiments" / "backend_parity_v2" / "onnx_parity.csv"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_FINAL_TEST = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
_ORT_CUDA = _REPO / ".venv" / "lib" / "python3.12" / "site-packages" / "onnxruntime" / "capi" / "libonnxruntime_providers_cuda.so"
_PRODUCTION = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
    "backend/inference/engine.py",
    "backend/inference/routes.py",
    "backend/inference/schemas.py",
    "backend/config.py",
)


def run_phase30() -> dict:
    summary_path = _OUTPUT / "serving_summary.json"
    if summary_path.is_file():
        raise RuntimeError("serving_v1 already exists; refusing to rewrite it")
    checkpoint_before = _stat(_CHECKPOINT)
    final_before = _stat(_FINAL_TEST)
    old_before = old_repository_fingerprint()
    production_before = {name: file_sha256(_REPO / name) for name in _PRODUCTION}
    if file_sha256(_CHECKPOINT) != "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f":
        raise RuntimeError("frozen checkpoint hash mismatch")
    environment = _environment_audit()
    cuda = probe_cuda_execution_provider()
    triton = _triton_status()
    tensorrt = _tensorrt_audit()
    quality = _quality_optimization()
    benchmark = _serving_benchmark(quality["report"], cuda_available=bool(cuda["available"]))
    if cuda["available"]:
        raise RuntimeError("CUDA provider loaded; this runner must measure GPU parity before writing a summary")
    if _stat(_CHECKPOINT) != checkpoint_before:
        raise RuntimeError("phase 30 changed the frozen checkpoint")
    if _stat(_FINAL_TEST) != final_before:
        raise RuntimeError("phase 30 changed the final-test summary")
    if old_repository_fingerprint() != old_before or _production_changed(production_before):
        raise RuntimeError("phase 30 changed the live DINOv2 path")
    summary = {
        "phase": 30,
        "model_name": MODEL_NAME,
        "checkpoint_sha256": "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
        "input_dims": [3, INPUT_SIZE, INPUT_SIZE],
        "output": "raw_logit",
        "temperature": FROZEN_TEMPERATURE,
        "policy_version": POLICY_VERSION,
        "production_policy_status": "not_promoted",
        "shadow_mode_default": False,
        "shadow_serving_default": "pytorch",
        "onnx_cuda_provider": "unavailable" if not cuda["available"] else "available",
        "onnx_cuda_error": cuda["error"],
        "tensorrt_status": tensorrt["status"],
        "triton_status": triton["status"],
        "gpu_onnx_parity": "not_measured",
        "gpu_onnx_benchmark": "not_measured",
        "tensorrt_parity": "not_measured",
        "triton_parity": "not_measured",
        "quality_flags_match": quality["flags_match"],
        "quality_images": quality["images"],
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "checkpoint_changed": False,
        "live_backend_modified": False,
        "customer_traffic_used": False,
        "old_model_untouched": True,
    }
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "environment_audit.json").write_text(_dump(environment) + "\n")
    (_OUTPUT / "tensorrt_audit.json").write_text(_dump(tensorrt) + "\n")
    _write_csv(
        _OUTPUT / "triton_parity.csv",
        [{"status": "unavailable", "server_reachable": triton["server_reachable"], "reason": triton["reason"]}],
    )
    (_OUTPUT / "quality_latency_v2.json").write_text(_dump(quality["report"]) + "\n")
    (_OUTPUT / "serving_benchmark.json").write_text(_dump(benchmark) + "\n")
    summary_path.write_text(_dump(summary) + "\n")
    return summary


def _environment_audit() -> dict:
    import onnxruntime as ort

    driver = _nvidia_smi()
    ldd = _ldd(_ORT_CUDA)
    return {
        "nvidia_driver": driver.get("driver_version"),
        "nvidia_smi_cuda_version": driver.get("cuda_version"),
        "gpu_name": driver.get("name"),
        "nvcc": shutil.which("nvcc"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": int(torch.backends.cudnn.version() or 0),
        "torch_gpu_visible": bool(torch.cuda.is_available()),
        "torch_gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_device": ort.get_device(),
        "onnxruntime_providers": list(ort.get_available_providers()),
        "cuda_provider_listed": "CUDAExecutionProvider" in ort.get_available_providers(),
        "cuda_provider_loaded": False,
        "ort_cuda_missing_libraries": [line for line in ldd if "not found" in line],
        "project_cublaslt": _first_existing(
            [
                _REPO / ".venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13",
                _REPO / ".venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12",
            ]
        ),
        "unrelated_cublaslt_so12": (
            "/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12"
            if Path("/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12").is_file()
            else None
        ),
        "ollama_library_used": False,
        "blocker": (
            "onnxruntime 1.26 CUDAExecutionProvider is linked to CUDA 12 sonames, including "
            "libcublasLt.so.12. PyTorch in this environment is torch 2.11.0+cu130 and ships "
            "libcublasLt.so.13. libcublasLt.so.12 is not on the dynamic linker path and is not "
            "installed in the project virtualenv. A private copy under Ollama's cuda_v12 tree "
            "was not added to the library path. The current environment cannot load this ONNX "
            "Runtime CUDA provider without a CUDA 12 runtime that is not part of the project stack."
        ),
    }


def _tensorrt_audit() -> dict:
    import importlib.util

    package = importlib.util.find_spec("tensorrt") is not None
    return {
        "status": "unavailable",
        "python_package_installed": package,
        "reason": (
            "The tensorrt Python package is not installed, and no TensorRT engine was built. "
            "ONNX Runtime lists TensorrtExecutionProvider, but that provider cannot run without "
            "the TensorRT libraries this environment does not have."
        ),
        "engine_created": False,
        "precision": None,
        "source_onnx_sha256": file_sha256(ONNX_PATH) if ONNX_PATH.is_file() else None,
    }


def _triton_status() -> dict:
    binary = shutil.which("tritonserver")
    reachable = False
    try:
        with socket.create_connection(("127.0.0.1", 8001), 0.3):
            reachable = True
    except OSError:
        reachable = False
    reason = "tritonserver is not installed and nothing accepted a connection on port 8001."
    if binary and not reachable:
        reason = f"tritonserver is at {binary}, but nothing accepted a connection on port 8001."
    if reachable:
        reason = "A process accepted port 8001. This phase did not attach the new model to it."
    return {
        "status": "unavailable",
        "binary": binary,
        "server_reachable": reachable,
        "reason": reason,
        "live_dinov2_model_modified": False,
    }


def _quality_optimization() -> dict:
    cases = _golden_cases()
    rows = []
    single_times = []
    double_times = []
    for case in cases:
        path = _REPO / case["sample_id"]
        if str(path.resolve()) in _test_paths():
            raise RuntimeError("serving audit selected a final-test image")
        path_features = extract_quality_features_from_path(path)
        path_flags = tuple(apply_quality_flags(path_features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
        _owned, features, flags, tensor = single_decode_inputs(path)
        reference = preprocess_image(_owned)
        if tuple(tensor.shape) != (3, INPUT_SIZE, INPUT_SIZE):
            raise RuntimeError("single-decode preprocess is not 3x512x512")
        if float(torch.max(torch.abs(tensor - reference)).item()) > 1e-5:
            raise RuntimeError(f"single-decode preprocess diverged for {case['sample_id']}")
        if flags != path_flags:
            raise RuntimeError(f"quality flags changed for {case['sample_id']}")
        rows.append({"sample_id": case["sample_id"], "flags": "|".join(flags), "flags_match": True})
        single_times.append(_time_single(path))
        double_times.append(_time_double(path))
    report = {
        "images": len(rows),
        "flags_match": True,
        "rule_changed": False,
        "resized_512_used_for_quality": False,
        "single_decode": _stages(single_times),
        "double_decode": _stages(double_times),
        "rows": rows,
    }
    return {"images": len(rows), "flags_match": True, "rows": rows, "report": report}


def _time_single(path: Path) -> dict:
    started = time.perf_counter()
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        rgb.load()
        owned = rgb.copy()
    decode_s = time.perf_counter()
    from image_quality import extract_quality_features

    features = extract_quality_features(owned)
    quality_s = time.perf_counter()
    preprocess_serving_image(owned)
    preprocess_s = time.perf_counter()
    flags = tuple(apply_quality_flags(features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    PolicyEngine(frozen_shadow_configuration()).evaluate(
        logit=0.0,
        quality_features=features,
        quality_flags=flags,
        preprocessing_version="resize_pad_square_eval_v1",
        input_size=INPUT_SIZE,
    )
    policy_s = time.perf_counter()
    return {
        "decode_ms": (decode_s - started) * 1000,
        "quality_ms": (quality_s - decode_s) * 1000,
        "preprocess_ms": (preprocess_s - quality_s) * 1000,
        "policy_ms": (policy_s - preprocess_s) * 1000,
        "total_without_model_ms": (policy_s - started) * 1000,
    }


def _time_double(path: Path) -> dict:
    started = time.perf_counter()
    features = extract_quality_features_from_path(path)
    quality_s = time.perf_counter()
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        rgb.load()
        owned = rgb.copy()
    decode_s = time.perf_counter()
    preprocess_serving_image(owned)
    preprocess_s = time.perf_counter()
    return {
        "quality_ms": (quality_s - started) * 1000,
        "second_decode_ms": (decode_s - quality_s) * 1000,
        "preprocess_ms": (preprocess_s - decode_s) * 1000,
        "total_without_model_ms": (preprocess_s - started) * 1000,
    }


def _serving_benchmark(report: dict, *, cuda_available: bool) -> dict:
    single = report["single_decode"]
    total = float(single["total_without_model"]["p50"])
    return {
        "images": report["images"],
        "scope": "single-decode stages on the golden images; not a production capacity claim",
        "onnx_cuda_model": "not_measured" if not cuda_available else "measured",
        "reason": None
        if cuda_available
        else "CUDAExecutionProvider did not load, so GPU ONNX timing was not invented.",
        "warmup_iterations": 0,
        "timed_iterations": report["images"],
        "batch_sizes_measured": [],
        "decode": single["decode"],
        "preprocess": single["preprocess"],
        "quality": single["quality"],
        "policy": single["policy"],
        "model": None,
        "total_without_gpu_model": single["total_without_model"],
        "images_per_second_without_gpu_model": None if total <= 0 else 1000.0 / total,
        "note": "Policy timing uses logit 0.0 only to time the policy function. It is not a model score.",
    }


def _stages(rows: list[dict]) -> dict:
    report = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        report[key.replace("_ms", "")] = {
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
        }
    return report


def _golden_cases() -> list[dict]:
    if not _GOLDEN.is_file():
        raise RuntimeError("phase 29 golden parity file is missing")
    blocked = _test_paths()
    cases = []
    seen = set()
    with _GOLDEN.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id in seen:
                continue
            path = _REPO / sample_id
            if str(path.resolve()) in blocked or row.get("split") == "test":
                raise RuntimeError("golden suite contains a final-test image")
            seen.add(sample_id)
            cases.append({"sample_id": sample_id, "split": row.get("split")})
    if len(cases) < 20:
        raise RuntimeError("golden suite has fewer than 20 images")
    return cases


def _test_paths() -> set[str]:
    payload = json.loads(_MANIFEST.read_text())
    return {str(Path(item).resolve()) for item in (payload.get("membership") or {}).get("test") or []}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _nvidia_smi() -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}
    name, driver = [part.strip() for part in completed.stdout.strip().split(",", 1)]
    version = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    cuda_version = None
    for token in version.stdout.split():
        if token.count(".") == 1 and token.replace(".", "").isdigit() and cuda_version is None:
            continue
    for line in version.stdout.splitlines():
        if "CUDA Version" in line:
            cuda_version = line.split("CUDA Version:")[-1].split("|")[0].strip()
    return {"name": name, "driver_version": driver, "cuda_version": cuda_version}


def _ldd(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path} missing"]
    completed = subprocess.run(["ldd", str(path)], check=False, capture_output=True, text=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.is_file():
            return str(path)
    return None


def _production_changed(before: dict) -> bool:
    return {name: file_sha256(_REPO / name) for name in before} != before


def _stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
