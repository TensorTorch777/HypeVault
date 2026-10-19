"""Phase 31: CUDA PyTorch shadow backend for the frozen DINOv3 model.

FP32 is authoritative. BF16 is measured and not installed as the shadow forward.
CPU ONNX is timed only as a comparison. Final-test images are not opened.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
if str(_REPO / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO / "backend"))

from inference.dinov3_model import shared_backend  # noqa: E402
from dinov3_serving import forward_onnx_logits  # noqa: E402
from image_quality import apply_quality_flags, extract_quality_features_from_path  # noqa: E402
from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit  # noqa: E402
from policy_contract import (  # noqa: E402
    PHASE23_CUTOFFS,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PolicyEngine,
    frozen_shadow_configuration,
)
from reference_inference import file_sha256, forward_logits, preprocess_image, quality_flags_for_image  # noqa: E402
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH  # noqa: E402

_OUTPUT = _PKG / "experiments" / "serving_v2"
_QUALITY = _PKG / "experiments" / "image_quality_v1" / "quality_features_nontest.csv"
_LOW_MARGIN = _PKG / "experiments" / "hard_negative_v1" / "low_margin.csv"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_FINAL_TEST = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
LOGIT_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
WARMUP = 20
TIMED = 100
_PHASE22 = (
    "Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg",
    "Label_0_Watches/Patek Philippe/9ba50fea3805.jpg",
    "Label_0_Watches/Patek Philippe/c2f778c66d15.jpg",
    "Label_0_Watches/A. Lange & Söhne/63d2c86b29d3.jpg",
    "Label_0_Watches/Audemars Piguet/99fcd3b9051c.jpg",
    "Label_0_Watches/Richard Mille/a11094843532.jpg",
    "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg",
)
_PRODUCTION = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
    "backend/config.py",
)


def run_phase31() -> dict:
    summary_path = _OUTPUT / "shadow_backend_summary.json"
    if summary_path.is_file():
        raise RuntimeError("serving_v2 already exists; refusing to rewrite it")
    checkpoint_before = _stat(_CHECKPOINT)
    final_before = _stat(_FINAL_TEST)
    production_before = {name: file_sha256(_REPO / name) for name in _PRODUCTION}
    if file_sha256(_CHECKPOINT) != "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f":
        raise RuntimeError("frozen checkpoint hash mismatch")
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("phase 31 requires the RTX 5080 CUDA device")
    cases = _select_cases()
    print(f"phase31 golden images {len(cases)}", flush=True)
    backend = shared_backend()
    backend.load()
    print("phase31 model loaded", flush=True)
    parity_rows, bf16_rows, resident = _parity(backend, cases)
    _require_parity(parity_rows)
    batch_rows = _batch_parity(backend, resident)
    benchmark_rows, memory_rows, stage_times = _benchmark(backend, resident)
    onnx_rows = _cpu_onnx(resident, parity_rows)
    if _stat(_CHECKPOINT) != checkpoint_before or file_sha256(_CHECKPOINT) != backend.checkpoint_sha256:
        raise RuntimeError("phase 31 changed the frozen checkpoint")
    if _stat(_FINAL_TEST) != final_before:
        raise RuntimeError("phase 31 changed the final-test summary")
    if {name: file_sha256(_REPO / name) for name in production_before} != production_before:
        raise RuntimeError("phase 31 changed a production inference file")
    fp32_batch1 = next(row for row in benchmark_rows if row["precision"] == "fp32" and int(row["batch_size"]) == 1)
    cpu_p50 = _percentile([row["cpu_onnx_ms"] for row in onnx_rows], 0.50)
    summary = {
        "phase": 31,
        "checkpoint_sha256": backend.checkpoint_sha256,
        "membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "manifest_file_sha256": file_sha256(_MANIFEST),
        "temperature": FROZEN_TEMPERATURE,
        "policy_version": POLICY_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "device": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "bf16_authoritative": False,
        "production_policy_status": "not_promoted",
        "shadow_mode_default": False,
        "shadow_serving_default": "pytorch",
        "images": len(parity_rows),
        "logit_atol": LOGIT_ATOL,
        "probability_atol": PROBABILITY_ATOL,
        "max_abs_logit_error": max(row["abs_logit_delta"] for row in parity_rows),
        "max_abs_probability_error": max(row["abs_probability_delta"] for row in parity_rows),
        "decision_mismatches": sum(not row["decision_match"] for row in parity_rows),
        "bf16_max_abs_logit_delta": max(row["abs_logit_delta"] for row in bf16_rows),
        "bf16_mean_abs_logit_delta": sum(row["abs_logit_delta"] for row in bf16_rows) / len(bf16_rows),
        "bf16_max_abs_probability_delta": max(row["abs_probability_delta"] for row in bf16_rows),
        "bf16_decision_mismatches": sum(not row["decision_match"] for row in bf16_rows),
        "closest_fp32_distance_to_0_50": min(abs(row["reference_probability"] - 0.50) for row in parity_rows),
        "batch_max_logit_deviation": max(row["max_logit_deviation"] for row in batch_rows),
        "batch_decision_mismatches": sum(row["decision_mismatches"] for row in batch_rows),
        "fp32_batch1_model_p50_ms": fp32_batch1["model_p50_ms"],
        "fp32_batch1_images_per_second": fp32_batch1["model_images_per_second"],
        "cpu_onnx_p50_ms": cpu_p50,
        "cpu_onnx_provider": "CPUExecutionProvider",
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "checkpoint_changed": False,
        "live_backend_modified": False,
        "customer_traffic_used": False,
        "cuda_library_remap": False,
    }
    environment = _environment(stage_times)
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "environment_check.json").write_text(_dump(environment) + "\n")
    _write_csv(_OUTPUT / "pytorch_cuda_parity.csv", parity_rows)
    _write_csv(_OUTPUT / "pytorch_bf16_comparison.csv", bf16_rows)
    _write_csv(_OUTPUT / "pytorch_cuda_benchmark.csv", benchmark_rows)
    _write_csv(_OUTPUT / "memory_benchmark.csv", memory_rows)
    _write_csv(_OUTPUT / "cpu_onnx_comparison.csv", onnx_rows)
    summary_path.write_text(_dump(summary) + "\n")
    print("phase31 artifacts written", flush=True)
    return summary


def _parity(backend, cases: list[dict]):
    engine = PolicyEngine(frozen_shadow_configuration())
    parity_rows = []
    bf16_rows = []
    resident = None
    for case in cases:
        image = _load_case(case)
        if resident is None and case["role"] != "grayscale_convertible":
            resident = image
        reference_tensor = preprocess_image(image).unsqueeze(0)
        backend_tensor = backend.preprocess(image).unsqueeze(0)
        tensor_delta = float(torch.max(torch.abs(reference_tensor - backend_tensor)).item())
        if tensor_delta > 1e-5 or tuple(backend_tensor.shape) != (1, 3, 512, 512):
            raise RuntimeError(f"preprocess parity failed for {case['sample_id']}")
        reference_logit = float(forward_logits(backend.model, reference_tensor, backend.device)[0])
        backend_logit = float(backend.predict_logits(backend_tensor)[0])
        reference_probability = calibrated_probability_from_logit(reference_logit)
        backend_probability = float(backend.predict_probability(backend_tensor)[0])
        reference_features, reference_flags = quality_flags_for_image(image)
        backend_features, backend_flags = backend.quality(image)
        if reference_flags != backend_flags:
            raise RuntimeError(f"quality flags diverged for {case['sample_id']}")
        reference = _decide(engine, reference_logit, reference_features, reference_flags)
        backend_decision = _decide(engine, backend_logit, backend_features, backend_flags)
        parity_rows.append(
            {
                "sample_id": case["sample_id"],
                "role": case["role"],
                "split": case["split"],
                "tensor_delta": tensor_delta,
                "reference_logit": reference_logit,
                "backend_logit": backend_logit,
                "abs_logit_delta": abs(reference_logit - backend_logit),
                "reference_probability": reference_probability,
                "backend_probability": backend_probability,
                "abs_probability_delta": abs(reference_probability - backend_probability),
                "quality_flags": "|".join(reference_flags),
                "reference_decision": reference.decision,
                "backend_decision": backend_decision.decision,
                "decision_match": reference.decision == backend_decision.decision,
                "ood_status": reference.ood_status,
            }
        )
        bf16_logit = float(backend.predict_logits_bf16(reference_tensor)[0])
        bf16_probability = calibrated_probability_from_logit(bf16_logit)
        bf16_decision = _decide(engine, bf16_logit, reference_features, reference_flags)
        bf16_rows.append(
            {
                "sample_id": case["sample_id"],
                "role": case["role"],
                "fp32_logit": reference_logit,
                "bf16_logit": bf16_logit,
                "abs_logit_delta": abs(reference_logit - bf16_logit),
                "fp32_probability": reference_probability,
                "bf16_probability": bf16_probability,
                "abs_probability_delta": abs(reference_probability - bf16_probability),
                "distance_to_0_50": abs(reference_probability - 0.50),
                "quality_flags": "|".join(reference_flags),
                "fp32_decision": reference.decision,
                "bf16_decision": bf16_decision.decision,
                "decision_match": reference.decision == bf16_decision.decision,
            }
        )
    if resident is None:
        raise RuntimeError("no resident image was available for the benchmark")
    return parity_rows, bf16_rows, resident


def _require_parity(rows: list[dict]) -> None:
    logit = max(row["abs_logit_delta"] for row in rows)
    probability = max(row["abs_probability_delta"] for row in rows)
    mismatches = sum(not row["decision_match"] for row in rows)
    if logit > LOGIT_ATOL or probability > PROBABILITY_ATOL or mismatches:
        raise RuntimeError(
            f"FP32 shadow parity failed: logit={logit} probability={probability} mismatches={mismatches}"
        )


def _batch_parity(backend, image) -> list[dict]:
    tensor = backend.preprocess(image).unsqueeze(0)
    base_logit = float(backend.predict_logits(tensor)[0])
    features, flags = backend.quality(image)
    engine = PolicyEngine(frozen_shadow_configuration())
    base_decision = _decide(engine, base_logit, features, flags).decision
    rows = []
    for size in (1, 2, 4, 8, 16):
        logits = backend.predict_logits(tensor.repeat(size, 1, 1, 1)).reshape(-1)
        deviations = [abs(float(logit) - base_logit) for logit in logits.tolist()]
        probability_deviations = [
            abs(calibrated_probability_from_logit(float(logit)) - calibrated_probability_from_logit(base_logit))
            for logit in logits.tolist()
        ]
        mismatches = 0
        for logit in logits.tolist():
            decision = _decide(engine, float(logit), features, flags).decision
            if decision != base_decision:
                mismatches += 1
        if max(deviations) > LOGIT_ATOL or mismatches:
            raise RuntimeError(f"batch {size} parity failed")
        rows.append(
            {
                "batch_size": size,
                "max_logit_deviation": max(deviations),
                "max_probability_deviation": max(probability_deviations),
                "decision_mismatches": mismatches,
            }
        )
    return rows


def _benchmark(backend, image) -> tuple[list[dict], list[dict], dict]:
    tensor = backend.preprocess(image).unsqueeze(0)
    features, flags = backend.quality(image)
    engine = PolicyEngine(frozen_shadow_configuration())
    stage_times = {
        "preprocess": _time_cpu(lambda: backend.preprocess(image)),
        "quality": _time_cpu(lambda: backend.quality(image)),
        "policy": _time_cpu(lambda: _decide(engine, 0.0, features, flags)),
    }
    benchmark_rows = []
    memory_rows = []
    for precision in ("fp32", "bf16"):
        predict = backend.predict_logits if precision == "fp32" else backend.predict_logits_bf16
        for size in (1, 2, 4, 8, 16):
            batch = tensor.repeat(size, 1, 1, 1)
            for _ in range(WARMUP):
                predict(batch)
            torch.cuda.synchronize()
            samples = []
            for _ in range(TIMED):
                start = time.perf_counter()
                predict(batch)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
            model = _distribution(samples)
            total_mean = (
                size * stage_times["preprocess"]["mean"]
                + model["mean"]
                + size * stage_times["quality"]["mean"]
                + size * stage_times["policy"]["mean"]
            )
            benchmark_rows.append(
                {
                    "precision": precision,
                    "batch_size": size,
                    "warmup": WARMUP,
                    "timed": TIMED,
                    "preprocess_p50_ms": stage_times["preprocess"]["p50"],
                    "preprocess_p95_ms": stage_times["preprocess"]["p95"],
                    "preprocess_p99_ms": stage_times["preprocess"]["p99"],
                    "model_p50_ms": model["p50"],
                    "model_p95_ms": model["p95"],
                    "model_p99_ms": model["p99"],
                    "model_mean_ms": model["mean"],
                    "quality_p50_ms": stage_times["quality"]["p50"],
                    "quality_p95_ms": stage_times["quality"]["p95"],
                    "quality_p99_ms": stage_times["quality"]["p99"],
                    "policy_p50_ms": stage_times["policy"]["p50"],
                    "policy_p95_ms": stage_times["policy"]["p95"],
                    "policy_p99_ms": stage_times["policy"]["p99"],
                    "total_mean_ms": total_mean,
                    "model_images_per_second": size / (model["mean"] / 1000.0),
                    "end_to_end_images_per_second": size / (total_mean / 1000.0),
                    "disk_io_included": False,
                }
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            predict(batch)
            torch.cuda.synchronize()
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
            total = int(torch.cuda.get_device_properties(0).total_memory)
            memory_rows.append(
                {
                    "precision": precision,
                    "batch_size": size,
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "total_gpu_bytes": total,
                    "allocated_bytes_per_sample": allocated / size,
                }
            )
    return benchmark_rows, memory_rows, stage_times


def _cpu_onnx(image, parity_rows: list[dict]) -> list[dict]:
    tensor = preprocess_image(image).unsqueeze(0)
    for _ in range(5):
        forward_onnx_logits(tensor)
    rows = []
    for row in parity_rows:
        if row["role"] == "grayscale_convertible":
            continue
        path = _REPO / row["sample_id"]
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            rgb.load()
            owned = rgb.copy()
        prepared = preprocess_image(owned).unsqueeze(0)
        start = time.perf_counter()
        logit = float(forward_onnx_logits(prepared)[0])
        elapsed = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "sample_id": row["sample_id"],
                "cpu_onnx_ms": elapsed,
                "cpu_onnx_logit": logit,
                "fp32_logit": row["reference_logit"],
                "abs_logit_delta": abs(logit - row["reference_logit"]),
            }
        )
    return rows


def _time_cpu(function) -> dict:
    for _ in range(WARMUP):
        function()
    samples = []
    for _ in range(TIMED):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return _distribution(samples)


def _distribution(samples: list[float]) -> dict:
    return {
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "mean": sum(samples) / len(samples),
    }


def _decide(engine, logit: float, features: dict, flags: tuple[str, ...]):
    return engine.evaluate(
        logit=logit,
        quality_features=features,
        quality_flags=flags,
        preprocessing_version=PREPROCESSING_VERSION,
        input_size=512,
        checkpoint_sha256="5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
    )


def _load_case(case: dict):
    if case["role"] == "grayscale_convertible":
        source = _open_rgb(_REPO / case["source_id"])
        return source.convert("L").convert("RGB")
    return _open_rgb(_REPO / case["sample_id"])


def _open_rgb(path: Path):
    if str(path.resolve()) in _test_paths():
        raise RuntimeError("phase 31 attempted to open a final-test image")
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        rgb.load()
        return rgb.copy()


def _select_cases() -> list[dict]:
    blocked = _test_paths()
    rows = []
    with _QUALITY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = _REPO / row["sample_id"]
            if str(path.resolve()) in blocked or row["source_split"] == "test":
                raise RuntimeError("quality catalog contains a final-test row")
            if not path.is_file():
                continue
            flags = set(filter(None, row["quality_flags"].split("|")))
            rows.append({**row, "flag_set": flags, "aspect": float(row["aspect_ratio"])})
    by_id = {row["sample_id"]: row for row in rows}
    chosen: list[dict] = []

    def add(row: dict, role: str) -> None:
        if any(item["sample_id"] == row["sample_id"] for item in chosen):
            return
        chosen.append({"sample_id": row["sample_id"], "split": row["source_split"], "role": role, "source_id": None})

    def take(predicate, role: str, count: int) -> None:
        matches = [row for row in rows if predicate(row) and all(item["sample_id"] != row["sample_id"] for item in chosen)]
        if len(matches) < count:
            raise RuntimeError(f"golden role {role} needs {count} images, found {len(matches)}")
        for row in matches[:count]:
            add(row, role)

    for sample_id in _PHASE22:
        if sample_id not in by_id:
            raise RuntimeError(f"phase 22 source {sample_id} is not in the non-test catalog")
        add(by_id[sample_id], "phase22_source")
    take(lambda row: row["label"] == "0" and not row["flag_set"], "authentic", 8)
    take(lambda row: row["label"] == "1" and not row["flag_set"], "fake", 8)
    take(lambda row: row["flag_set"] == {"LOW_RESOLUTION"}, "low_resolution", 4)
    take(lambda row: row["flag_set"] == {"LOW_CONTRAST"}, "low_contrast", 4)
    take(lambda row: {"LOW_RESOLUTION", "LOW_CONTRAST"} <= row["flag_set"], "both_quality_flags", 2)
    take(lambda row: row["aspect"] <= 0.7, "tall", 3)
    take(lambda row: row["aspect"] >= 1.45, "wide", 3)
    take(lambda row: 0.95 <= row["aspect"] <= 1.05, "square", 4)
    margin_ids = _closest_margin_ids()
    for sample_id in margin_ids:
        row = by_id.get(sample_id)
        if row is not None:
            add(row, "near_cached_margin")
    remaining = 52 - len(chosen)
    if remaining > 0:
        take(lambda row: True, "additional", remaining)
    if len(chosen) < 50:
        raise RuntimeError(f"only {len(chosen)} golden images were available")
    chosen = chosen[:52]
    source = next(item for item in chosen if item["role"] == "authentic")
    chosen.append(
        {
            "sample_id": f"{source['sample_id']}|grayscale",
            "split": source["split"],
            "role": "grayscale_convertible",
            "source_id": source["sample_id"],
        }
    )
    return chosen


def _closest_margin_ids() -> list[str]:
    ranked = []
    with _LOW_MARGIN.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ranked.append((abs(float(row["calibrated_probability"]) - 0.50), row["sample_id"]))
    ranked.sort()
    return [sample_id for _distance, sample_id in ranked[:8]]


def _environment(stage_times: dict) -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "total_gpu_bytes": int(properties.total_memory),
        "authoritative_precision": "fp32",
        "onnx_cuda_used": False,
        "cuda12_libraries_remapped": False,
        "ollama_cuda_used": False,
        "preprocess_p50_ms": stage_times["preprocess"]["p50"],
        "quality_p50_ms": stage_times["quality"]["p50"],
        "policy_p50_ms": stage_times["policy"]["p50"],
        "warmup_iterations": WARMUP,
        "timed_iterations": TIMED,
        "benchmark_scope": "100 timed iterations after 20 warmup iterations on one resident image, disk I/O excluded",
        "policy_timing_logit": 0.0,
    }


def _test_paths() -> set[str]:
    payload = json.loads(_MANIFEST.read_text())
    return {str(Path(item).resolve()) for item in (payload.get("membership") or {}).get("test") or []}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


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
