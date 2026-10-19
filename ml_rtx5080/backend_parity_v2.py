"""Phase 29: parallel DINOv3 artifact parity.

Creates the new ONNX/Triton repository entry, compares it with the fp32
reference, and profiles quality latency. It does not promote the policy,
does not open the final test, and does not rewrite the DINOv2 artifact.
"""

from __future__ import annotations

import csv
import json
import math
import socket
import time
from pathlib import Path

import torch
from PIL import Image

from dinov3_serving import (
    CHECKPOINT,
    DETERMINISM_ATOL,
    INPUT_SIZE,
    LOGIT_ATOL,
    MODEL_NAME,
    OLD_MODEL_ROOT,
    ONNX_PATH,
    PROBABILITY_ATOL,
    TENSOR_ATOL,
    export_frozen_onnx,
    file_sha256,
    forward_onnx_logits,
    load_identity,
    old_repository_fingerprint,
    preprocess_serving_image,
    reset_onnx_session,
    tensor_sha256,
)
from image_quality import (
    apply_quality_flags,
    extract_quality_features,
    extract_quality_features_from_path,
    foreground_occupancy,
    luminance_image,
    sharpness_laplacian_variance,
)
from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit
from policy_contract import (
    PHASE23_CUTOFFS,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PolicyEngine,
    frozen_shadow_configuration,
)
from reference_inference import forward_logits, load_reference_model, preprocess_image, quality_flags_for_image

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_QUALITY_CSV = _PKG / "experiments" / "image_quality_v1" / "quality_features_nontest.csv"
_LOW_MARGIN = _PKG / "experiments" / "hard_negative_v1" / "low_margin.csv"
_FINAL_TEST = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_OUTPUT = _PKG / "experiments" / "backend_parity_v2"
_AUDIT_IMAGE = "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg"
_STORED_LOGIT = -0.83984375
_MANIFEST_IMAGE_SHA = "26c7795df6c093cb67ab491496fad618cd68e1f17fa72661b6435880981c932a"
_PHASE22_SOURCES = (
    "Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg",
    "Label_0_Watches/Patek Philippe/9ba50fea3805.jpg",
    "Label_0_Watches/Patek Philippe/c2f778c66d15.jpg",
    "Label_0_Watches/A. Lange & Söhne/63d2c86b29d3.jpg",
    "Label_0_Watches/Audemars Piguet/99fcd3b9051c.jpg",
    "Label_0_Watches/Richard Mille/a11094843532.jpg",
    _AUDIT_IMAGE,
)
_PRODUCTION_FILES = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
    "backend/inference/engine.py",
    "backend/inference/routes.py",
    "backend/inference/schemas.py",
    "backend/config.py",
)


def run_phase29() -> dict:
    summary_path = _OUTPUT / "shadow_model_summary.json"
    if summary_path.is_file():
        raise RuntimeError("backend_parity_v2 already exists; refusing to rewrite it")
    if file_sha256(CHECKPOINT) != "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f":
        raise RuntimeError("frozen checkpoint hash mismatch; export was not started")
    checkpoint_before = _stat(CHECKPOINT)
    final_before = _stat(_FINAL_TEST)
    old_before = old_repository_fingerprint()
    production_before = _production_hashes()
    test_paths = _test_paths()
    cases = _select_cases(test_paths)
    print(f"phase29 exporting {len(cases)} golden images after the artifact", flush=True)
    export_info = export_frozen_onnx()
    print(f"phase29 exported {export_info['artifact_bytes']} bytes", flush=True)
    if export_info["source_checkpoint_sha256"] != file_sha256(CHECKPOINT):
        raise RuntimeError("export identity does not match the frozen checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, digest = load_reference_model(CHECKPOINT, device)
    model.eval()
    audit = _numerical_audit(model, device)
    preprocess_rows = _preprocess_parity()
    if any(row["max_abs_delta"] > TENSOR_ATOL for row in preprocess_rows):
        raise RuntimeError("serving preprocess diverged from the reference")
    parity_rows, timings = _onnx_parity(model, device, cases)
    max_logit = max(row["abs_logit_delta"] for row in parity_rows)
    mean_logit = sum(row["abs_logit_delta"] for row in parity_rows) / len(parity_rows)
    max_probability = max(row["abs_probability_delta"] for row in parity_rows)
    disagreements = sum(not row["decision_match"] for row in parity_rows)
    if max_logit > LOGIT_ATOL or disagreements:
        raise RuntimeError(
            f"ONNX parity failed: max_logit={max_logit} disagreements={disagreements}"
        )
    batch = _batch_parity(model, device, cases[0])
    determinism = _determinism(model, device, cases[0])
    quality = _quality_profile(cases)
    triton = _triton_status()
    if _stat(CHECKPOINT) != checkpoint_before or file_sha256(CHECKPOINT) != digest:
        raise RuntimeError("phase 29 changed the frozen checkpoint")
    if _stat(_FINAL_TEST) != final_before:
        raise RuntimeError("phase 29 changed the final-test summary")
    if old_repository_fingerprint() != old_before:
        raise RuntimeError("phase 29 changed the DINOv2 Triton artifact")
    if _production_hashes() != production_before:
        raise RuntimeError("phase 29 changed a production inference file")
    if OLD_MODEL_ROOT.joinpath("config.pbtxt").read_text().find('name: "dinov2_classifier"') < 0:
        raise RuntimeError("DINOv2 Triton config no longer names dinov2_classifier")
    if "518" not in (OLD_MODEL_ROOT / "config.pbtxt").read_text():
        raise RuntimeError("DINOv2 Triton config lost its 518 input")
    latency = _latency_summary(timings)
    summary = {
        "phase": 29,
        "model_name": MODEL_NAME,
        "checkpoint_sha256": digest,
        "source_checkpoint_sha256": export_info["source_checkpoint_sha256"],
        "export_artifact_sha256": export_info["export_artifact_sha256"],
        "input_dims": [3, INPUT_SIZE, INPUT_SIZE],
        "preprocessing_version": PREPROCESSING_VERSION,
        "policy_version": POLICY_VERSION,
        "temperature": FROZEN_TEMPERATURE,
        "temperature_in_graph": False,
        "production_policy_status": "not_promoted",
        "shadow_mode_default": False,
        "shadow_serving_default": "pytorch",
        "old_model_untouched": True,
        "live_backend_modified": False,
        "customer_traffic_used": False,
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "checkpoint_changed": False,
        "tolerances": {
            "preprocess_tensor_atol": TENSOR_ATOL,
            "logit_atol": LOGIT_ATOL,
            "probability_atol": PROBABILITY_ATOL,
            "determinism_atol": DETERMINISM_ATOL,
            "bitwise_equality_across_engines": False,
        },
        "onnx_parity": {
            "images": len(parity_rows),
            "max_abs_logit_error": max_logit,
            "mean_abs_logit_error": mean_logit,
            "max_abs_probability_error": max_probability,
            "decision_disagreements": disagreements,
            "status": "passed",
        },
        "batch_parity": batch,
        "determinism": determinism,
        "triton_parity_status": triton["status"],
        "triton_reason": triton["reason"],
        "tensorrt_status": "unavailable",
        "onnx_providers": _onnx_providers(),
        "latency": latency,
        "quality_profile_note": quality["conclusion"],
        "authoritative_forward": "fp32 reference, no autocast",
        "numerical_reproducibility": {
            "reproduced_stored_phase22_logit": audit["reproduced_stored_logit"],
            "authoritative": audit["authoritative_path"],
        },
    }
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "export_audit.json").write_text(_dump({"export": export_info, "old_dinov2": old_before}) + "\n")
    _write_csv(_OUTPUT / "onnx_parity.csv", parity_rows)
    _write_csv(
        _OUTPUT / "triton_parity.csv",
        [
            {
                "status": triton["status"],
                "server_reachable": triton["server_reachable"],
                "model_name": MODEL_NAME,
                "reason": triton["reason"],
            }
        ],
    )
    _write_csv(_OUTPUT / "preprocessing_parity.csv", preprocess_rows)
    (_OUTPUT / "numerical_reproducibility_audit.json").write_text(_dump(audit) + "\n")
    (_OUTPUT / "latency_v2.json").write_text(_dump(latency) + "\n")
    (_OUTPUT / "quality_latency_profile.json").write_text(_dump(quality) + "\n")
    summary_path.write_text(_dump(summary) + "\n")
    reset_onnx_session()
    print("phase29 artifacts written", flush=True)
    return summary


def _onnx_parity(model, device, cases: list[dict]) -> tuple[list[dict], list[dict]]:
    engine = PolicyEngine(frozen_shadow_configuration())
    checkpoint_sha = file_sha256(CHECKPOINT)
    rows = []
    timings = []
    for case in cases:
        path = _REPO / case["sample_id"]
        if str(path.resolve()) in _test_paths():
            raise RuntimeError("phase 29 selected a final-test image")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image.load()
            owned = image.copy()
        started = time.perf_counter()
        reference_tensor = preprocess_image(owned).unsqueeze(0)
        preprocess_s = time.perf_counter()
        serving_tensor = preprocess_serving_image(owned).unsqueeze(0)
        if torch.max(torch.abs(reference_tensor - serving_tensor)).item() > TENSOR_ATOL:
            raise RuntimeError(f"preprocess mismatch for {case['sample_id']}")
        reference_logit = float(forward_logits(model, reference_tensor, device)[0])
        model_s = time.perf_counter()
        features, flags = quality_flags_for_image(owned)
        quality_s = time.perf_counter()
        reference_probability = calibrated_probability_from_logit(reference_logit)
        reference = engine.evaluate(
            logit=reference_logit,
            quality_features=features,
            quality_flags=flags,
            preprocessing_version=PREPROCESSING_VERSION,
            input_size=INPUT_SIZE,
            checkpoint_sha256=checkpoint_sha,
            split=case["split"],
            sample_id=case["sample_id"],
        )
        policy_s = time.perf_counter()
        onnx_started = time.perf_counter()
        onnx_logit = float(forward_onnx_logits(serving_tensor)[0])
        onnx_s = time.perf_counter()
        onnx_probability = calibrated_probability_from_logit(onnx_logit)
        onnx_result = engine.evaluate(
            logit=onnx_logit,
            quality_features=features,
            quality_flags=flags,
            preprocessing_version=PREPROCESSING_VERSION,
            input_size=INPUT_SIZE,
            checkpoint_sha256=load_identity()["source_checkpoint_sha256"],
            split=case["split"],
            sample_id=case["sample_id"],
        )
        rows.append(
            {
                "sample_id": case["sample_id"],
                "role": case["role"],
                "split": case["split"],
                "reference_logit": reference_logit,
                "onnx_logit": onnx_logit,
                "abs_logit_delta": abs(reference_logit - onnx_logit),
                "reference_probability": reference_probability,
                "onnx_probability": onnx_probability,
                "abs_probability_delta": abs(reference_probability - onnx_probability),
                "quality_flags": "|".join(flags),
                "reference_decision": reference.decision,
                "onnx_decision": onnx_result.decision,
                "decision_match": reference.decision == onnx_result.decision,
                "ood_status": reference.ood_status,
            }
        )
        timings.append(
            {
                "sample_id": case["sample_id"],
                "preprocess_ms": (preprocess_s - started) * 1000,
                "model_ms": (model_s - preprocess_s) * 1000,
                "onnx_model_ms": (onnx_s - onnx_started) * 1000,
                "quality_ms": (quality_s - model_s) * 1000,
                "policy_ms": (policy_s - quality_s) * 1000,
                "total_ms": (
                    (preprocess_s - started)
                    + (onnx_s - onnx_started)
                    + (quality_s - model_s)
                    + (policy_s - quality_s)
                )
                * 1000,
            }
        )
    return rows, timings


def _batch_parity(model, device, case: dict) -> dict:
    path = _REPO / case["sample_id"]
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.load()
        owned = image.copy()
    tensor = preprocess_image(owned).unsqueeze(0)
    reference = forward_logits(model, tensor, device)
    base = float(reference[0])
    rows = []
    worst_logit = 0.0
    worst_probability = 0.0
    mismatches = 0
    engine = PolicyEngine(frozen_shadow_configuration())
    features, flags = quality_flags_for_image(owned)
    base_decision = engine.evaluate(
        logit=base,
        quality_features=features,
        quality_flags=flags,
        preprocessing_version=PREPROCESSING_VERSION,
        input_size=INPUT_SIZE,
    ).decision
    for size in (1, 2, 4, 8, 16):
        batch = tensor.repeat(size, 1, 1, 1)
        pytorch = forward_logits(model, batch, device)
        onnx = forward_onnx_logits(batch)
        pytorch_delta = float(torch.max(torch.abs(pytorch - base)).item())
        onnx_delta = float(torch.max(torch.abs(onnx - base)).item())
        worst_logit = max(worst_logit, pytorch_delta, onnx_delta)
        for logit in pytorch.tolist() + onnx.tolist():
            probability_delta = abs(calibrated_probability_from_logit(float(logit)) - calibrated_probability_from_logit(base))
            worst_probability = max(worst_probability, probability_delta)
            decision = engine.evaluate(
                logit=float(logit),
                quality_features=features,
                quality_flags=flags,
                preprocessing_version=PREPROCESSING_VERSION,
                input_size=INPUT_SIZE,
            ).decision
            if decision != base_decision:
                mismatches += 1
        rows.append(
            {
                "batch_size": size,
                "pytorch_max_abs_logit_delta": pytorch_delta,
                "onnx_max_abs_logit_delta": onnx_delta,
            }
        )
    if worst_logit > LOGIT_ATOL or mismatches:
        raise RuntimeError(f"batch parity failed: logit={worst_logit} mismatches={mismatches}")
    return {
        "rows": rows,
        "max_logit_deviation": worst_logit,
        "max_probability_deviation": worst_probability,
        "decision_mismatches": mismatches,
        "atol": LOGIT_ATOL,
    }


def _determinism(model, device, case: dict) -> dict:
    path = _REPO / case["sample_id"]
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.load()
        owned = image.copy()
    tensor = preprocess_image(owned).unsqueeze(0)
    pytorch = [float(forward_logits(model, tensor, device)[0]) for _ in range(5)]
    onnx = [float(forward_onnx_logits(tensor)[0]) for _ in range(5)]
    pytorch_span = max(pytorch) - min(pytorch)
    onnx_span = max(onnx) - min(onnx)
    if pytorch_span > DETERMINISM_ATOL or onnx_span > DETERMINISM_ATOL:
        raise RuntimeError("repeated inference exceeded the determinism tolerance")
    return {
        "repeats": 5,
        "tolerance": DETERMINISM_ATOL,
        "pytorch_span": pytorch_span,
        "onnx_span": onnx_span,
        "triton_span": None,
        "triton_status": "unavailable",
    }


def _numerical_audit(model, device) -> dict:
    import timm
    from hard_negative import _logits_from_features

    path = _REPO / _AUDIT_IMAGE
    image_sha = file_sha256(path)
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.load()
        owned = image.copy()
    tensor = preprocess_image(owned).unsqueeze(0)
    direct = preprocess_serving_image(owned).unsqueeze(0)
    dropout_training = [module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    fp32 = float(forward_logits(model, tensor, device)[0])
    manual = _manual_logit(model, tensor, device, _logits_from_features, autocast=False)
    bf16 = _manual_logit(model, tensor, device, _logits_from_features, autocast=True)
    bf16_forward = _autocast_forward(model, tensor, device)
    tf32_disabled = _fp32_without_tf32(model, tensor, device)
    stored = _stored_logit()
    image_matches_manifest = image_sha == _MANIFEST_IMAGE_SHA
    reproduced = abs(fp32 - stored) <= LOGIT_ATOL
    if image_matches_manifest:
        image_clause = "The image bytes still match the Phase 16 manifest hash."
    else:
        image_clause = "The image bytes do not match the Phase 16 manifest hash."
    finding = (
        f"{image_clause} The checkpoint hash matches, eval mode is on, and the current "
        f"preprocess tensor is stable. The stored Phase 22 logit {stored} "
        f"{'matches' if reproduced else 'is not reproduced by'} "
        f"the current fp32 forward {fp32}. A fresh bf16 forward returned {bf16_forward}. "
        "The stored value is also not the bf16 rounding of the current fp32 logit. It stays "
        "a historical result. The authoritative forward is the current fp32 reference."
    )
    return {
        "sample_id": _AUDIT_IMAGE,
        "authoritative_path": "frozen checkpoint + resize_pad_square_eval_v1 + fp32 reference forward + frozen temperature + shadow_v1",
        "image_sha256": image_sha,
        "manifest_image_sha256": _MANIFEST_IMAGE_SHA,
        "image_bytes_match_manifest": image_matches_manifest,
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "model_training": bool(model.training),
        "dropout_modules_training": dropout_training,
        "preprocess_tensors_match": torch.equal(tensor, direct),
        "preprocess_tensor_sha256": tensor_sha256(tensor),
        "parameter_sha256": _parameter_sha(model),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "cuda": torch.cuda.is_available(),
        "stored_phase22_bf16_logit": stored,
        "fresh_fp32_forward": fp32,
        "fresh_fp32_manual_head": manual,
        "fresh_bf16_manual_head": bf16,
        "fresh_bf16_model_forward": bf16_forward,
        "fresh_fp32_tf32_disabled": tf32_disabled,
        "fp32_matches_manual_head": abs(fp32 - manual) <= LOGIT_ATOL,
        "reproduced_stored_logit": reproduced,
        "stored_is_not_the_bf16_cast_of_current_fp32": abs(_bf16_float(fp32) - stored) > 1e-6,
        "checks": {
            "checkpoint_hash_matches_frozen": True,
            "image_bytes_match_phase16_manifest": image_matches_manifest,
            "preprocess_function_matches_reference": bool(torch.equal(tensor, direct)),
            "checkpoint_loading_path": "evaluation.build_eval_model strict load of epoch_018.pt",
            "autocast_changes_the_fresh_logit": abs(fp32 - bf16) > LOGIT_ATOL,
            "stored_logit_source": "ml_rtx5080/experiments/hard_negative_v1/low_margin.csv",
            "train_mode": bool(model.training),
            "dropout_in_train_mode": any(dropout_training),
        },
        "finding": finding,
    }


def _manual_logit(model, tensor, device, head, *, autocast: bool) -> float:
    moved = tensor.to(device)
    with torch.inference_mode():
        if autocast and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model.backbone.forward_features(moved)
                cls_token = model.backbone.get_cls_token(features)
                output = head(model, features, cls_token)
        else:
            features = model.backbone.forward_features(moved)
            cls_token = model.backbone.get_cls_token(features)
            output = head(model, features, cls_token)
    return float(output.detach().float().reshape(-1)[0].cpu())


def _autocast_forward(model, tensor, device) -> float:
    moved = tensor.to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(moved)
        else:
            output = model(moved)
    return float(output.detach().float().reshape(-1)[0].cpu())


def _fp32_without_tf32(model, tensor, device) -> float:
    if device.type != "cuda":
        return float(forward_logits(model, tensor, device)[0])
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        return float(forward_logits(model, tensor, device)[0])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def _bf16_float(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float())


def _stored_logit() -> float:
    with _LOW_MARGIN.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] == _AUDIT_IMAGE:
                return float(row["logit"])
    raise RuntimeError("stored Phase 22 logit row is missing")


def _parameter_sha(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _preprocess_parity() -> list[dict]:
    specs = {
        "square": Image.new("RGB", (512, 512), (20, 40, 60)),
        "wide": Image.new("RGB", (900, 200), (180, 30, 30)),
        "tall": Image.new("RGB", (180, 960), (30, 30, 180)),
        "tiny": Image.new("RGB", (8, 8), (255, 255, 255)),
        "large": Image.new("RGB", (1600, 900), (10, 120, 10)),
        "rgb": Image.new("RGB", (640, 480), (200, 100, 50)),
        "grayscale_convertible": Image.new("L", (700, 300), 40).convert("RGB"),
    }
    rows = []
    for name, image in specs.items():
        reference = preprocess_image(image)
        serving = preprocess_serving_image(image)
        delta = float(torch.max(torch.abs(reference - serving)).item())
        rows.append(
            {
                "case": name,
                "source_size": f"{image.size[0]}x{image.size[1]}",
                "output_shape": "3x512x512",
                "max_abs_delta": delta,
                "within_tolerance": delta <= TENSOR_ATOL,
            }
        )
    return rows


def _quality_profile(cases: list[dict]) -> dict:
    samples = []
    for case in cases[:3]:
        path = _REPO / case["sample_id"]
        samples.append(_profile_one(path))
    shared = _shared_decode_benchmark(cases[0])
    resized = _resized_buffer_check(cases[0])
    return {
        "samples": samples,
        "shared_decode": shared,
        "resized_model_buffer": resized,
        "rule_changed": False,
        "authoritative_function": "image_quality.extract_quality_features",
        "conclusion": shared["conclusion"],
    }


def _profile_one(path: Path) -> dict:
    import numpy as np

    started = time.perf_counter()
    with Image.open(path) as handle:
        decode_s = time.perf_counter()
        rgb = handle.convert("RGB")
        rgb.load()
        owned = rgb.copy()
    convert_s = time.perf_counter()
    luminance = luminance_image(owned)
    luminance_s = time.perf_counter()
    contrast = float(np.std(luminance, ddof=0))
    contrast_s = time.perf_counter()
    _low = float(np.percentile(luminance, 5, method="linear"))
    _high = float(np.percentile(luminance, 95, method="linear"))
    percentile_s = time.perf_counter()
    sharpness = sharpness_laplacian_variance(luminance)
    laplacian_s = time.perf_counter()
    occupancy = foreground_occupancy(luminance)
    occupancy_s = time.perf_counter()
    function_started = time.perf_counter()
    features = extract_quality_features(owned)
    function_s = time.perf_counter()
    flags = tuple(apply_quality_flags(features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    flags_s = time.perf_counter()
    return {
        "sample_id": str(path.relative_to(_REPO)),
        "width": owned.size[0],
        "height": owned.size[1],
        "decode_ms": (decode_s - started) * 1000,
        "rgb_conversion_ms": (convert_s - decode_s) * 1000,
        "luminance_ms": (luminance_s - convert_s) * 1000,
        "contrast_ms": (contrast_s - luminance_s) * 1000,
        "percentile_ms": (percentile_s - contrast_s) * 1000,
        "laplacian_ms": (laplacian_s - percentile_s) * 1000,
        "occupancy_ms": (occupancy_s - laplacian_s) * 1000,
        "full_feature_function_ms": (function_s - function_started) * 1000,
        "flags_ms": (flags_s - function_s) * 1000,
        "total_profiled_ms": (flags_s - started) * 1000,
        "percentile_spread": _high - _low,
        "sharpness": sharpness,
        "occupancy": occupancy,
        "contrast": contrast,
        "flags": list(flags),
    }


def _shared_decode_benchmark(case: dict) -> dict:
    path = _REPO / case["sample_id"]
    path_times = []
    shared_times = []
    for _ in range(5):
        started = time.perf_counter()
        from_path = extract_quality_features_from_path(path)
        path_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            rgb.load()
            owned = rgb.copy()
        shared = extract_quality_features(
            owned,
            file_bytes=path.stat().st_size,
            image_format=from_path["image_format"],
            jpeg_quantization_tables_present=from_path["jpeg_quantization_tables_present"],
        )
        shared_times.append((time.perf_counter() - started) * 1000)
    path_flags = tuple(apply_quality_flags(from_path, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    shared_flags = tuple(apply_quality_flags(shared, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    numeric_match = all(
        from_path[name] == shared[name]
        for name in from_path
        if name not in {"image_format", "jpeg_quantization_tables_present"}
    )
    return {
        "flags_match": path_flags == shared_flags,
        "numeric_features_match": numeric_match,
        "path_decode_p50_ms": _percentile(path_times, 0.50),
        "shared_decode_p50_ms": _percentile(shared_times, 0.50),
        "adopted": False,
        "conclusion": (
            "Quality flags are computed from the original image. A single decode can feed "
            "both preprocess and quality, and the shadow path already does that. The resized "
            "512 model tensor is not a valid quality input. The authoritative feature function "
            "was not replaced."
        ),
    }


def _resized_buffer_check(case: dict) -> dict:
    path = _REPO / case["sample_id"]
    with Image.open(path) as handle:
        original = handle.convert("RGB")
        original.load()
        owned = original.copy()
    original_features = extract_quality_features(owned)
    original_flags = tuple(apply_quality_flags(original_features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    from augmentations import CONSERVATIVE, ResizePadToSquare, image_size_for_family

    resized = ResizePadToSquare(image_size_for_family("dinov3"))(owned)
    resized_features = extract_quality_features(resized)
    resized_flags = tuple(apply_quality_flags(resized_features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    small = Image.new("RGB", (200, 180), (80, 80, 80))
    small_flags = tuple(apply_quality_flags(extract_quality_features(small), {"cutoffs": dict(PHASE23_CUTOFFS)}))
    small_resized_flags = tuple(
        apply_quality_flags(
            extract_quality_features(ResizePadToSquare(image_size_for_family("dinov3"))(small)),
            {"cutoffs": dict(PHASE23_CUTOFFS)},
        )
    )
    return {
        "original_flags": list(original_flags),
        "resized_flags": list(resized_flags),
        "flags_match": original_flags == resized_flags,
        "min_dimension_original": original_features["min_dimension"],
        "min_dimension_resized": resized_features["min_dimension"],
        "small_original_flags": list(small_flags),
        "small_resized_flags": list(small_resized_flags),
        "small_flags_match": small_flags == small_resized_flags,
        "used_for_decisions": False,
    }


def _select_cases(test_paths: set[str]) -> list[dict]:
    rows = []
    with _QUALITY_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = _REPO / row["sample_id"]
            if str(path.resolve()) in test_paths or row["source_split"] == "test":
                raise RuntimeError("quality catalog contains a final-test row")
            if not path.is_file():
                continue
            flags = set(filter(None, row["quality_flags"].split("|")))
            rows.append({**row, "flag_set": flags})
    by_id = {row["sample_id"]: row for row in rows}

    def take(predicate, role: str, count: int, chosen: list[dict]) -> None:
        have = {item["sample_id"] for item in chosen}
        matches = [row for row in rows if row["sample_id"] not in have and predicate(row)]
        for row in matches[:count]:
            chosen.append(_case(row, role))

    chosen: list[dict] = []
    for sample_id in _PHASE22_SOURCES:
        row = by_id.get(sample_id)
        if row is None:
            raise RuntimeError(f"phase 22 source {sample_id} is not in the non-test quality catalog")
        chosen.append(_case(row, "phase22_source"))
    take(lambda row: row["label"] == "0" and not row["flag_set"], "authentic", 4, chosen)
    take(lambda row: row["label"] == "1" and not row["flag_set"], "fake", 4, chosen)
    take(lambda row: row["flag_set"] == {"LOW_RESOLUTION"}, "low_resolution", 2, chosen)
    take(lambda row: row["flag_set"] == {"LOW_CONTRAST"}, "low_contrast", 2, chosen)
    take(lambda row: {"LOW_RESOLUTION", "LOW_CONTRAST"} <= row["flag_set"], "both_quality_flags", 1, chosen)
    take(lambda row: True, "additional", 24, chosen)
    if len(chosen) < 20:
        raise RuntimeError(f"only {len(chosen)} non-test golden images were available")
    return chosen[:24]


def _case(row: dict, role: str) -> dict:
    return {"sample_id": row["sample_id"], "split": row["source_split"], "role": role, "label": int(row["label"])}


def _latency_summary(timings: list[dict]) -> dict:
    keys = ("preprocess_ms", "model_ms", "onnx_model_ms", "quality_ms", "policy_ms", "total_ms")
    report = {"images": len(timings), "scope": "one forward per golden image, not a capacity claim"}
    for key in keys:
        values = [row[key] for row in timings]
        report[key.replace("_ms", "")] = {
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
        }
    return report


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _onnx_providers() -> list[str]:
    from dinov3_serving import onnx_session

    return list(onnx_session().get_providers())


def _triton_status() -> dict:
    reachable = False
    try:
        with socket.create_connection(("127.0.0.1", 8001), 0.3):
            reachable = True
    except OSError:
        reachable = False
    if not reachable:
        return {
            "status": "blocked",
            "server_reachable": False,
            "reason": "Triton is not accepting connections on port 8001. Parity was not fabricated.",
        }
    return {
        "status": "blocked",
        "server_reachable": True,
        "reason": "A server answered on port 8001, but this phase does not attach the new model to a live Triton instance.",
    }


def _test_paths() -> set[str]:
    payload = json.loads(_MANIFEST.read_text())
    return {str(Path(path).resolve()) for path in (payload.get("membership") or {}).get("test") or []}


def _production_hashes() -> dict:
    return {relative: file_sha256(_REPO / relative) for relative in _PRODUCTION_FILES}


def _stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
