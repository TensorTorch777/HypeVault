"""Phase 32: reliability of the frozen FP32 CUDA DINOv3 shadow backend.

The checkpoint, temperature, quality rule, and live DINOv2 path stay unchanged.
BF16 is recorded as research_only. Final-test images are rejected before open.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
if str(_REPO / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO / "backend"))

from inference.dinov3_model import (  # noqa: E402
    DinoV3ShadowBackend,
    reset_backend_for_tests,
    shared_backend,
)
from inference.shadow import reset_shadow_for_tests  # noqa: E402
from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit  # noqa: E402
from policy_contract import (  # noqa: E402
    INPUT_SIZE,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PolicyConfigurationError,
    PolicyEngine,
    frozen_shadow_configuration,
)
from reference_inference import (  # noqa: E402
    file_sha256,
    forward_logits,
    preprocess_image,
    quality_flags_for_image,
)
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH  # noqa: E402

_OUTPUT = _PKG / "experiments" / "serving_stability_v1"
_QUALITY = _PKG / "experiments" / "image_quality_v1" / "quality_features_nontest.csv"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_FINAL_TEST = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
_CHECKPOINT_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
LOGIT_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
MEMORY_GROWTH_TOLERANCE_BYTES = 8 * 1024 * 1024
ALLOWED_WORKERS = (1, 2, 4, 8, 16)
PHASE31_MODEL_P50_MS = 13.870440496248193
PHASE31_E2E_MEAN_MS = 22.59588531567715
_PRODUCTION = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
    "backend/config.py",
)
_ROLES = (
    "authentic",
    "fake",
    "square",
    "wide",
    "tall",
    "tiny",
    "large",
    "low_contrast",
    "both_flags",
    "phase22_rm",
)


def validate_worker_count(workers: int) -> int:
    """Accept only the measured concurrency ladder. Other values are rejected."""
    value = int(workers)
    if value not in ALLOWED_WORKERS:
        raise ValueError(f"concurrency {workers} is outside {ALLOWED_WORKERS}")
    return value


def run_phase32() -> dict:
    summary_path = _OUTPUT / "serving_stability_summary.json"
    if summary_path.is_file():
        raise RuntimeError("serving_stability_v1 already exists; refusing to rewrite it")
    checkpoint_before = _stat(_CHECKPOINT)
    final_before = _stat(_FINAL_TEST)
    production_before = {name: file_sha256(_REPO / name) for name in _PRODUCTION}
    if file_sha256(_CHECKPOINT) != _CHECKPOINT_SHA:
        raise RuntimeError("frozen checkpoint hash mismatch")
    if not torch.cuda.is_available() or "RTX 5080" not in torch.cuda.get_device_name(0):
        raise RuntimeError("phase 32 requires the RTX 5080 CUDA device")

    blocked = _blocked_test_paths()
    cases = _select_cases(blocked)
    print(f"phase32 corpus {len(cases)}", flush=True)
    cold = _cold_start()
    backend = shared_backend()
    print("phase32 model loaded", flush=True)
    engine = PolicyEngine(frozen_shadow_configuration())
    prepared = [_prepare(backend, case) for case in cases]
    authentic = next(item for item in prepared if item["role"] == "authentic")
    fake = next(item for item in prepared if item["role"] == "fake")

    first = _timed_request(backend, engine, authentic)
    first_ten = [_timed_request(backend, engine, authentic) for _ in range(10)]
    cold["first_inference_model_ms"] = first["model_ms"]
    cold["first_inference_e2e_ms"] = first["e2e_ms"]
    cold["first_10_model_p50_ms"] = _percentile([row["model_ms"] for row in first_ten], 0.50)
    cold["first_10_e2e_mean_ms"] = sum(row["e2e_ms"] for row in first_ten) / len(first_ten)

    determinism = _determinism(backend, engine, authentic, prepared)
    isolation = _request_isolation(backend, engine, authentic, fake)
    order = _order_independence(backend, prepared)
    parity = _policy_parity(backend, engine, prepared)
    failures = _failure_isolation(backend, engine, authentic)
    print("phase32 sequential 1000", flush=True)
    sequential, sequential_memory = _sequential(backend, engine, prepared)
    print("phase32 memory runs", flush=True)
    memory_rows = _memory_runs(backend, authentic["tensor"])
    memory_rows.extend(sequential_memory)
    print("phase32 concurrency", flush=True)
    concurrency_rows, thread_safe = _concurrency(backend, engine, authentic)
    print("phase32 batch load", flush=True)
    batch_rows = _batch_load(backend, engine, authentic)
    bf16 = _bf16_research(backend, authentic["tensor"])
    print("phase32 reload cycles", flush=True)
    reload_report = _reload(authentic)
    failure_rows = failures + _cuda_failure_cases(authentic)

    memory_stable = _memory_is_stable(memory_rows)
    concurrency_ok = _concurrency_ok(concurrency_rows)
    contract_ok = (
        determinism["passed"]
        and isolation["passed"]
        and order["passed"]
        and parity["passed"]
        and all(row["isolated"] for row in failure_rows)
        and reload_report["passed"]
        and memory_stable
        and concurrency_ok
        and max(row["decision_mismatches"] for row in batch_rows) == 0
        and max(row["max_logit_deviation"] for row in batch_rows) <= LOGIT_ATOL
    )
    model_p50 = sequential["model_p50_ms"]
    summary = {
        "phase": 32,
        "checkpoint_sha256": _CHECKPOINT_SHA,
        "membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "manifest_file_sha256": file_sha256(_MANIFEST),
        "temperature": FROZEN_TEMPERATURE,
        "precision": "fp32",
        "bf16_authoritative": False,
        "bf16_label": "research_only",
        "policy_version": POLICY_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "production_policy_status": "not_promoted",
        "shadow_mode_default": False,
        "authoritative_precision": "fp32",
        "images": len(prepared),
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "customer_traffic_used": False,
        "live_backend_modified": False,
        "checkpoint_changed": False,
        "logit_atol": LOGIT_ATOL,
        "probability_atol": PROBABILITY_ATOL,
        "memory_growth_tolerance_bytes": MEMORY_GROWTH_TOLERANCE_BYTES,
        "memory_stable": memory_stable,
        "determinism_passed": determinism["passed"],
        "request_isolation_passed": isolation["passed"],
        "order_independence_passed": order["passed"],
        "policy_parity_passed": parity["passed"],
        "failure_isolation_passed": all(row["isolated"] for row in failure_rows),
        "reload_passed": reload_report["passed"],
        "thread_safe_unlocked": thread_safe,
        "concurrency_required_levels_passed": concurrency_ok,
        "concurrency_mode": "shared_model_default_stream",
        "serialization_note": (
            "Unlocked forwards on the shared model stayed within tolerance."
            if thread_safe
            else "Unlocked concurrent forwards were not safe. Throughput below uses a harness lock around the shared model. The lock is not installed in the serving module."
        ),
        "sequential_requests": sequential["requests"],
        "sequential_model_p50_ms": model_p50,
        "sequential_model_p95_ms": sequential["model_p95_ms"],
        "sequential_model_p99_ms": sequential["model_p99_ms"],
        "sequential_e2e_p50_ms": sequential["e2e_p50_ms"],
        "sequential_e2e_mean_ms": sequential["e2e_mean_ms"],
        "sequential_e2e_images_per_sec": sequential["e2e_images_per_sec"],
        "phase31_model_p50_ms": PHASE31_MODEL_P50_MS,
        "phase31_e2e_mean_ms": PHASE31_E2E_MEAN_MS,
        "model_p50_delta_ms": model_p50 - PHASE31_MODEL_P50_MS,
        "latency_comparison": _latency_note(model_p50, sequential["e2e_mean_ms"]),
        "bf16_logit_span": bf16["logit_span"],
        "contract_passed": contract_ok,
        "ood_status": "unavailable",
    }
    if _stat(_CHECKPOINT) != checkpoint_before or file_sha256(_CHECKPOINT) != _CHECKPOINT_SHA:
        raise RuntimeError("phase 32 changed the frozen checkpoint")
    if _stat(_FINAL_TEST) != final_before:
        raise RuntimeError("phase 32 changed the final-test summary")
    if {name: file_sha256(_REPO / name) for name in production_before} != production_before:
        raise RuntimeError("phase 32 changed a production inference file")

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "cold_start.json").write_text(_dump(cold) + "\n")
    _write_csv(_OUTPUT / "sequential_benchmark.csv", [sequential])
    _write_csv(_OUTPUT / "concurrency_benchmark.csv", concurrency_rows)
    _write_csv(_OUTPUT / "batch_benchmark.csv", batch_rows)
    _write_csv(_OUTPUT / "memory_stability.csv", memory_rows)
    (_OUTPUT / "determinism_check.json").write_text(_dump({"determinism": determinism, "isolation": isolation, "order": order, "parity": parity, "bf16": bf16}) + "\n")
    _write_csv(_OUTPUT / "failure_isolation.csv", failure_rows)
    (_OUTPUT / "reload_test.json").write_text(_dump(reload_report) + "\n")
    summary_path.write_text(_dump(summary) + "\n")
    print("phase32 artifacts written", flush=True)
    if not contract_ok:
        raise RuntimeError("phase 32 stability contract failed; artifacts were kept for diagnosis")
    return summary


def _cold_start() -> dict:
    already = torch.cuda.is_initialized()
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - started) * 1000.0
    reset_backend_for_tests()
    reset_shadow_for_tests()
    hash_started = time.perf_counter()
    digest = file_sha256(_CHECKPOINT)
    hash_ms = (time.perf_counter() - hash_started) * 1000.0
    if digest != _CHECKPOINT_SHA:
        raise RuntimeError("frozen checkpoint hash mismatch")
    backend = shared_backend()
    load_started = time.perf_counter()
    backend.load()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_started) * 1000.0
    warm_started = time.perf_counter()
    backend.load()
    torch.cuda.synchronize()
    warm_ms = (time.perf_counter() - warm_started) * 1000.0
    return {
        "cuda_already_initialized": already,
        "cuda_init_ms": cuda_ms,
        "checkpoint_hash_ms": hash_ms,
        "model_load_ms": load_ms,
        "warm_load_ms": warm_ms,
        "model_load_includes_second_checkpoint_hash": True,
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _prepare(backend, case: dict) -> dict:
    image = case["image"]
    tensor = backend.preprocess(image).unsqueeze(0)
    if tuple(tensor.shape) != (1, 3, INPUT_SIZE, INPUT_SIZE):
        raise RuntimeError(f"preprocessed shape {tuple(tensor.shape)} is not 1x3x512x512")
    return {**case, "tensor": tensor}


def _timed_request(backend, engine, item: dict) -> dict:
    torch.cuda.synchronize()
    started = time.perf_counter()
    tensor = backend.preprocess(item["image"]).unsqueeze(0)
    torch.cuda.synchronize()
    model_started = time.perf_counter()
    logit = float(backend.predict_logits(tensor)[0])
    torch.cuda.synchronize()
    model_ms = (time.perf_counter() - model_started) * 1000.0
    result, _flags = _decide(engine, item["image"], logit)
    finished = time.perf_counter()
    return {
        "logit": logit,
        "probability": calibrated_probability_from_logit(logit),
        "decision": result.decision,
        "model_ms": model_ms,
        "e2e_ms": (finished - started) * 1000.0,
    }


def _determinism(backend, engine, item: dict, prepared: list[dict]) -> dict:
    logits = []
    probabilities = []
    decisions = []
    flag_sets = []
    for _ in range(100):
        logit = float(backend.predict_logits(item["tensor"])[0])
        probability = calibrated_probability_from_logit(logit)
        result, flags = _decide(engine, item["image"], logit)
        logits.append(logit)
        probabilities.append(probability)
        decisions.append(result.decision)
        flag_sets.append("|".join(flags))
    single = logits[0]
    mixed = torch.cat([prepared[index % len(prepared)]["tensor"] for index in range(8)], dim=0)
    mixed_logits = [float(value) for value in backend.predict_logits(mixed).reshape(-1).tolist()]
    position_deltas = []
    for index, value in enumerate(mixed_logits):
        expected = float(backend.predict_logits(prepared[index % len(prepared)]["tensor"])[0])
        position_deltas.append(abs(value - expected))
    logit_span = max(logits) - min(logits)
    probability_span = max(probabilities) - min(probabilities)
    passed = (
        logit_span <= LOGIT_ATOL
        and probability_span <= PROBABILITY_ATOL
        and len(set(decisions)) == 1
        and len(set(flag_sets)) == 1
        and max(position_deltas) <= LOGIT_ATOL
    )
    return {
        "repeats": 100,
        "logit_span": logit_span,
        "probability_span": probability_span,
        "decisions": sorted(set(decisions)),
        "quality_flags": flag_sets[0],
        "max_batch_position_logit_delta": max(position_deltas),
        "reference_logit": single,
        "passed": passed,
    }


def _request_isolation(backend, engine, authentic: dict, fake: dict) -> dict:
    alone = {
        "authentic": _score(backend, engine, authentic),
        "fake": _score(backend, engine, fake),
    }
    forward = [_score(backend, engine, authentic), _score(backend, engine, fake)]
    reverse = [_score(backend, engine, fake), _score(backend, engine, authentic)]
    pairs = (
        (forward[0], alone["authentic"]),
        (forward[1], alone["fake"]),
        (reverse[0], alone["fake"]),
        (reverse[1], alone["authentic"]),
    )
    deltas = [abs(left["logit"] - right["logit"]) for left, right in pairs]
    passed = all(delta <= LOGIT_ATOL for delta in deltas) and all(
        left["decision"] == right["decision"] and left["flags"] == right["flags"] for left, right in pairs
    )
    return {"max_logit_delta": max(deltas), "passed": passed}


def _order_independence(backend, prepared: list[dict]) -> dict:
    def measure(item: dict) -> tuple[float, str]:
        tensor = backend.preprocess(item["image"]).unsqueeze(0)
        logit = float(backend.predict_logits(tensor)[0])
        flags = "|".join(quality_flags_for_image(item["image"])[1])
        return logit, flags

    baseline = {item["sample_id"]: measure(item) for item in prepared}
    orders = (
        prepared,
        list(reversed(prepared)),
        prepared[3:] + prepared[:3],
    )
    deltas = []
    flag_mismatch = 0
    for order in orders:
        for item in order:
            logit, flags = measure(item)
            expected_logit, expected_flags = baseline[item["sample_id"]]
            deltas.append(abs(logit - expected_logit))
            if flags != expected_flags:
                flag_mismatch += 1
    return {
        "max_logit_delta": max(deltas),
        "flag_mismatches": flag_mismatch,
        "passed": max(deltas) <= LOGIT_ATOL and flag_mismatch == 0,
    }


def _policy_parity(backend, engine, prepared: list[dict]) -> dict:
    logit_deltas = []
    probability_deltas = []
    mismatches = 0
    for item in prepared:
        reference_logit = float(forward_logits(backend.model, item["tensor"], backend.device)[0])
        backend_logit = float(backend.predict_logits(item["tensor"])[0])
        reference = _decide(engine, item["image"], reference_logit)[0]
        shadow = _decide(engine, item["image"], backend_logit)[0]
        logit_deltas.append(abs(reference_logit - backend_logit))
        probability_deltas.append(
            abs(calibrated_probability_from_logit(reference_logit) - calibrated_probability_from_logit(backend_logit))
        )
        if reference.decision != shadow.decision or reference.quality_flags != shadow.quality_flags:
            mismatches += 1
        if reference.ood_status != "unavailable":
            raise RuntimeError("shadow policy invented an OOD status")
    return {
        "images": len(prepared),
        "max_logit_delta": max(logit_deltas),
        "max_probability_delta": max(probability_deltas),
        "decision_mismatches": mismatches,
        "passed": max(logit_deltas) <= LOGIT_ATOL and max(probability_deltas) <= PROBABILITY_ATOL and mismatches == 0,
    }


def _failure_isolation(backend, engine, authentic: dict) -> list[dict]:
    golden = _score(backend, engine, authentic)
    rows = []
    probes = (
        ("valid", authentic["image"], "AUTHENTIC_OR_REVIEW_OR_FAKE"),
        ("low_resolution", _noisy_image(180, 200), "REVIEW"),
        ("low_contrast", Image.new("RGB", (800, 800), (128, 128, 128)), "REVIEW"),
        ("both_quality_flags", Image.new("RGB", (128, 96), (140, 140, 140)), "REVIEW"),
        ("malformed", b"\x00\x01not-an-image", "POLICY_ERROR"),
        ("missing_file", _REPO / "ml_rtx5080" / "experiments" / "serving_stability_v1" / "missing.jpg", "POLICY_ERROR"),
    )
    for name, payload, _expected in probes:
        outcome = _invoke(payload)
        follow = _score(backend, engine, authentic)
        isolated = (
            abs(follow["logit"] - golden["logit"]) <= LOGIT_ATOL
            and follow["decision"] == golden["decision"]
            and follow["flags"] == golden["flags"]
        )
        if name == "malformed" or name == "missing_file":
            isolated = isolated and outcome["decision"] is None and outcome["system_status"] == "POLICY_ERROR"
        elif name != "valid":
            isolated = isolated and outcome["decision"] == "REVIEW"
        rows.append(
            {
                "case": name,
                "system_status": outcome["system_status"],
                "decision": outcome["decision"],
                "reason_codes": "|".join(outcome["reason_codes"]),
                "followup_logit_delta": abs(follow["logit"] - golden["logit"]),
                "followup_decision": follow["decision"],
                "isolated": isolated,
            }
        )
    return rows


def _cuda_failure_cases(authentic: dict) -> list[dict]:
    from unittest.mock import patch

    golden_backend = shared_backend()
    golden = float(golden_backend.predict_logits(authentic["tensor"])[0])
    rows = []
    cases = (
        ("cuda_unavailable", patch("torch.cuda.is_available", return_value=False)),
        ("invalid_device", patch("torch.cuda.get_device_name", return_value="not-a-gpu")),
        (
            "model_load_failure",
            patch("inference.dinov3_model.load_reference_model", side_effect=RuntimeError("injected load failure")),
        ),
    )
    for name, context in cases:
        with context:
            if name == "model_load_failure":
                fresh = DinoV3ShadowBackend()
                try:
                    fresh.load()
                    outcome = {"system_status": "LOADED", "decision": "UNEXPECTED", "reason_codes": ()}
                except PolicyConfigurationError as exc:
                    outcome = {"system_status": "POLICY_ERROR", "decision": None, "reason_codes": exc.reason_codes}
            else:
                try:
                    shared_backend().load()
                    outcome = {"system_status": "LOADED", "decision": "UNEXPECTED", "reason_codes": ()}
                except PolicyConfigurationError as exc:
                    outcome = {"system_status": "POLICY_ERROR", "decision": None, "reason_codes": exc.reason_codes}
        follow = float(shared_backend().predict_logits(authentic["tensor"])[0])
        rows.append(
            {
                "case": name,
                "system_status": outcome["system_status"],
                "decision": outcome["decision"],
                "reason_codes": "|".join(outcome["reason_codes"]),
                "followup_logit_delta": abs(follow - golden),
                "followup_decision": "unchanged",
                "isolated": outcome["system_status"] == "POLICY_ERROR"
                and outcome["decision"] is None
                and "POLICY_INFRASTRUCTURE_ERROR" in outcome["reason_codes"]
                and abs(follow - golden) <= LOGIT_ATOL,
            }
        )
    return rows


def _sequential(backend, engine, prepared: list[dict]) -> tuple[dict, list[dict]]:
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    model_ms = []
    e2e_ms = []
    samples = []
    for index in range(1000):
        timing = _timed_request(backend, engine, prepared[index % len(prepared)])
        model_ms.append(timing["model_ms"])
        e2e_ms.append(timing["e2e_ms"])
        if index + 1 in {100, 500, 1000}:
            samples.append(_memory_row("sequential_cursor", index + 1, allocated_before, reserved_before))
    allocated_after = torch.cuda.memory_allocated()
    row = {
        "requests": 1000,
        "batch_size": 1,
        "model_p50_ms": _percentile(model_ms, 0.50),
        "model_p95_ms": _percentile(model_ms, 0.95),
        "model_p99_ms": _percentile(model_ms, 0.99),
        "model_max_ms": max(model_ms),
        "model_mean_ms": statistics.fmean(model_ms),
        "model_std_ms": statistics.stdev(model_ms),
        "e2e_p50_ms": _percentile(e2e_ms, 0.50),
        "e2e_p95_ms": _percentile(e2e_ms, 0.95),
        "e2e_p99_ms": _percentile(e2e_ms, 0.99),
        "e2e_max_ms": max(e2e_ms),
        "e2e_mean_ms": statistics.fmean(e2e_ms),
        "e2e_std_ms": statistics.stdev(e2e_ms),
        "e2e_images_per_sec": 1000.0 / (sum(e2e_ms) / 1000.0),
        "model_images_per_sec": 1000.0 / (sum(model_ms) / 1000.0),
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "growth_bytes": allocated_after - allocated_before,
        "disk_io_included": False,
    }
    return row, samples


def _memory_runs(backend, tensor: torch.Tensor) -> list[dict]:
    rows = []
    for count in (100, 500, 1000):
        for _ in range(5):
            backend.predict_logits(tensor)
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(count):
            backend.predict_logits(tensor)
        torch.cuda.synchronize()
        rows.append(_memory_row("controlled", count, allocated_before, reserved_before))
    return rows


def _memory_row(stage: str, count: int, allocated_before: int, reserved_before: int) -> dict:
    allocated_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()
    growth = allocated_after - allocated_before
    reserved_growth = reserved_after - reserved_before
    return {
        "stage": stage,
        "requests": count,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "reserved_before": reserved_before,
        "reserved_after": reserved_after,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "growth_bytes": growth,
        "reserved_growth_bytes": reserved_growth,
        "within_tolerance": abs(growth) <= MEMORY_GROWTH_TOLERANCE_BYTES
        and abs(reserved_growth) <= MEMORY_GROWTH_TOLERANCE_BYTES,
    }


def _concurrency_ok(rows: list[dict]) -> bool:
    """Workers 1, 2, and 4 must be clean. Higher levels are recorded even if they OOM."""
    required = [row for row in rows if row["requests"] >= 80 and row["workers"] <= 4]
    if len(required) != 3:
        return False
    return all(
        row["error_rate"] == 0
        and row["decision_mismatches"] == 0
        and row["logit_max_abs_delta"] is not None
        and row["logit_max_abs_delta"] <= LOGIT_ATOL
        for row in required
    )


def _memory_is_stable(rows: list[dict]) -> bool:
    controlled = [row for row in rows if row["stage"] == "controlled"]
    if len(controlled) != 3 or any(not row["within_tolerance"] for row in controlled):
        return False
    growths = [row["growth_bytes"] for row in controlled]
    if growths[0] < growths[1] < growths[2] and growths[2] - growths[0] > MEMORY_GROWTH_TOLERANCE_BYTES:
        return False
    return True


def _concurrency(backend, engine, item: dict) -> tuple[list[dict], bool]:
    reference = _score(backend, engine, item)
    probe = _concurrency_level(backend, engine, item, workers=4, requests=8, lock=None, reference=reference)
    probe_delta = probe["logit_max_abs_delta"]
    thread_safe = (
        probe["cuda_oom_count"] == 0
        and probe["policy_error_count"] == 0
        and probe["error_rate"] == 0
        and probe_delta is not None
        and probe_delta <= LOGIT_ATOL
        and probe["decision_mismatches"] == 0
    )
    rows = [probe]
    lock = None if thread_safe else threading.Lock()
    for workers in ALLOWED_WORKERS:
        rows.append(
            _concurrency_level(
                backend,
                engine,
                item,
                workers=workers,
                requests=80,
                lock=lock,
                reference=reference,
            )
        )
    return rows, thread_safe


def _concurrency_level(backend, engine, item, *, workers: int, requests: int, lock, reference: dict) -> dict:
    validate_worker_count(workers)
    images = [item["image"].copy() for _ in range(workers)]
    errors = {"oom": 0, "policy": 0, "other": 0}
    latencies = []
    logits = []
    guard = threading.Lock()

    def work(index: int) -> None:
        image = images[index % workers]
        try:
            started = time.perf_counter()
            torch.cuda.synchronize()
            tensor = backend.preprocess(image).unsqueeze(0)
            if lock is None:
                logit = float(backend.predict_logits(tensor)[0])
            else:
                with lock:
                    logit = float(backend.predict_logits(tensor)[0])
            torch.cuda.synchronize()
            _decide(engine, image, logit)
            elapsed = (time.perf_counter() - started) * 1000.0
        except torch.cuda.OutOfMemoryError:
            with guard:
                errors["oom"] += 1
            torch.cuda.empty_cache()
            return
        except PolicyConfigurationError:
            with guard:
                errors["policy"] += 1
            return
        except RuntimeError as exc:
            with guard:
                if "out of memory" in str(exc).lower():
                    errors["oom"] += 1
                    torch.cuda.empty_cache()
                else:
                    errors["other"] += 1
            return
        with guard:
            latencies.append(elapsed)
            logits.append(logit)

    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, index) for index in range(requests)]
        for future in futures:
            future.result()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_started
    completed = len(latencies)
    deltas = [abs(value - reference["logit"]) for value in logits]
    decisions = 0
    for value in logits:
        if _decide(engine, item["image"], value)[0].decision != reference["decision"]:
            decisions += 1
    total_errors = errors["oom"] + errors["policy"] + errors["other"]
    return {
        "workers": workers,
        "mode": "unlocked" if lock is None else "serialized",
        "requests": requests,
        "completed": completed,
        "throughput_per_sec": completed / wall_s if wall_s else 0.0,
        "p50_ms": _percentile(latencies, 0.50) if latencies else None,
        "p95_ms": _percentile(latencies, 0.95) if latencies else None,
        "p99_ms": _percentile(latencies, 0.99) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
        "error_rate": total_errors / requests,
        "cuda_oom_count": errors["oom"],
        "policy_error_count": errors["policy"],
        "other_error_count": errors["other"],
        "logit_max_abs_delta": max(deltas) if deltas else None,
        "decision_mismatches": decisions,
    }


def _batch_load(backend, engine, item: dict) -> list[dict]:
    image = item["image"]
    single_logit = float(backend.predict_logits(item["tensor"])[0])
    single_decision = _decide(engine, image, single_logit)[0].decision
    rows = []
    for size in (1, 2, 4, 8, 16):
        batch = item["tensor"].repeat(size, 1, 1, 1)
        for _ in range(10):
            backend.predict_logits(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        deviations = []
        mismatches = 0
        for _ in range(40):
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = [float(value) for value in backend.predict_logits(batch).reshape(-1).tolist()]
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
            for logit in logits:
                deviations.append(abs(logit - single_logit))
                if _decide(engine, image, logit)[0].decision != single_decision:
                    mismatches += 1
        mean_ms = statistics.fmean(latencies)
        rows.append(
            {
                "batch_size": size,
                "repeats": 40,
                "warmup": 10,
                "latency_p50_ms": _percentile(latencies, 0.50),
                "latency_p95_ms": _percentile(latencies, 0.95),
                "latency_p99_ms": _percentile(latencies, 0.99),
                "latency_mean_ms": mean_ms,
                "latency_per_image_p50_ms": _percentile(latencies, 0.50) / size,
                "throughput_images_per_sec": size / (mean_ms / 1000.0),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "max_logit_deviation": max(deviations),
                "decision_mismatches": mismatches,
            }
        )
    return rows


def _bf16_research(backend, tensor: torch.Tensor) -> dict:
    logits = [float(backend.predict_logits_bf16(tensor)[0]) for _ in range(20)]
    return {
        "label": "research_only",
        "authoritative": False,
        "repeats": 20,
        "logit_min": min(logits),
        "logit_max": max(logits),
        "logit_span": max(logits) - min(logits),
        "used_for_shadow_contract": False,
    }


def _reload(item: dict) -> dict:
    baseline = float(shared_backend().predict_logits(item["tensor"])[0])
    cycles = []
    for cycle in range(1, 5):
        reset_shadow_for_tests()
        reset_backend_for_tests()
        gc.collect()
        torch.cuda.synchronize()
        after_cleanup = torch.cuda.memory_allocated()
        backend = shared_backend()
        backend.load()
        if backend.checkpoint_sha256 != _CHECKPOINT_SHA:
            raise RuntimeError("reload produced a different checkpoint hash")
        logit = float(backend.predict_logits(item["tensor"])[0])
        cycles.append(
            {
                "cycle": cycle,
                "allocated_after_cleanup": after_cleanup,
                "allocated_after_load": torch.cuda.memory_allocated(),
                "logit_delta": abs(logit - baseline),
                "checkpoint_sha256": backend.checkpoint_sha256,
            }
        )
    cleanup = [row["allocated_after_cleanup"] for row in cycles]
    growth = max(cleanup) - min(cleanup)
    passed = growth <= MEMORY_GROWTH_TOLERANCE_BYTES and all(row["logit_delta"] <= LOGIT_ATOL for row in cycles)
    return {
        "cycles": cycles,
        "cleanup_growth_bytes": growth,
        "duplicate_instances_intended": False,
        "passed": passed,
    }


def _invoke(payload) -> dict:
    from inference.shadow import run_shadow_image

    if isinstance(payload, Path):
        result, _timings = run_shadow_image(payload)
    else:
        result, _timings = run_shadow_image(payload)
    return {
        "system_status": result.system_status,
        "decision": result.decision,
        "reason_codes": result.reason_codes,
    }


def _score(backend, engine, item: dict) -> dict:
    logit = float(backend.predict_logits(item["tensor"])[0])
    result, flags = _decide(engine, item["image"], logit)
    return {
        "logit": logit,
        "probability": calibrated_probability_from_logit(logit),
        "decision": result.decision,
        "flags": "|".join(flags),
    }


def _decide(engine, image, logit: float):
    features, flags = quality_flags_for_image(image)
    result = engine.evaluate(
        logit=float(logit),
        quality_features=features,
        quality_flags=flags,
        preprocessing_version=PREPROCESSING_VERSION,
        input_size=INPUT_SIZE,
        checkpoint_sha256=_CHECKPOINT_SHA,
    )
    return result, flags


def _noisy_image(height: int, width: int) -> Image.Image:
    import numpy as np

    values = np.random.default_rng(32).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(values, mode="RGB")


def _select_cases(blocked: set[str]) -> list[dict]:
    picks: dict[str, dict] = {}
    with _QUALITY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_split"] == "test":
                raise RuntimeError("quality table contains a final-test row")
            path = (_REPO / row["sample_id"]).resolve()
            if str(path) in blocked:
                raise RuntimeError("final-test path reached the selector")
            flags = set(filter(None, row["quality_flags"].split("|")))
            aspect = float(row["aspect_ratio"])
            min_dimension = float(row["min_dimension"])
            candidates = {
                "authentic": row["sample_id"] == "Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg",
                "fake": row["label"] == "1" and not flags,
                "square": 0.98 <= aspect <= 1.02 and min_dimension >= 512 and not flags,
                "wide": aspect >= 1.45 and not flags,
                "tall": aspect <= 0.7,
                "tiny": min_dimension <= 256,
                "large": min_dimension >= 2400 and not flags,
                "low_contrast": "LOW_CONTRAST" in flags and "LOW_RESOLUTION" not in flags,
                "both_flags": {"LOW_RESOLUTION", "LOW_CONTRAST"} <= flags,
                "phase22_rm": row["sample_id"] == "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg",
            }
            for role, matched in candidates.items():
                if matched and role not in picks:
                    picks[role] = row
    missing = [role for role in _ROLES if role not in picks]
    if missing:
        raise RuntimeError(f"corpus is missing {missing}")
    cases = []
    for role in _ROLES:
        row = picks[role]
        image = _open_rgb(row["sample_id"], blocked)
        cases.append({"sample_id": row["sample_id"], "split": row["source_split"], "role": role, "image": image})
    return cases


def _open_rgb(sample_id: str, blocked: set[str]) -> Image.Image:
    path = (_REPO / sample_id).resolve()
    if str(path) in blocked:
        raise RuntimeError("final-test sample rejected before open")
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.load()
    return image


def _blocked_test_paths() -> set[str]:
    payload = json.loads(_MANIFEST.read_text())
    return {str(Path(item).resolve()) for item in (payload.get("membership") or {}).get("test") or []}


def _latency_note(model_p50: float, e2e_mean: float) -> str:
    model_clause = (
        f"Batch-1 model p50 is {model_p50:.2f} ms against the Phase 31 baseline of {PHASE31_MODEL_P50_MS:.2f} ms."
    )
    e2e_clause = (
        f"End-to-end mean is {e2e_mean:.2f} ms against the Phase 31 single-image mean of {PHASE31_E2E_MEAN_MS:.2f} ms. "
        "Phase 32 end-to-end time includes preprocess, the FP32 forward, quality, and policy on a mixed-resolution corpus."
    )
    return model_clause + " " + e2e_clause


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
