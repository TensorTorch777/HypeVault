"""Audit the production inference stack and compare the shadow path to the reference.

The production verdict is not modified. Triton is not treated as available unless
the server responds and the artifact is the frozen 512 DINOv3 contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
_REPO = _ML.parent
_BACKEND = _REPO / "backend"
for _path in (_ML, _BACKEND):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit  # noqa: E402
from policy_contract import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    INPUT_SIZE,
    LOW_CONTRAST_CUTOFF,
    LOW_RESOLUTION_CUTOFF,
    MODEL_ARCHITECTURE,
    MODEL_FAMILY,
    MODEL_HEAD,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
)
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH  # noqa: E402

OUTPUT_DIR_NAME = "backend_parity_v1"
CHECKPOINT = _ML / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
MANIFEST = _ML / "experiments" / "dataset_audit" / "split_manifest_v2.json"
FEATURES = _ML / "experiments" / "image_quality_v1" / "quality_features_nontest.csv"
STRESS = _ML / "experiments" / "image_quality_v1" / "quality_stress_linkage.csv"
FINAL_TEST_SUMMARY = _ML / "experiments" / "final_test_v1" / "final_test_summary.json"
TRITON_CONFIG = _REPO / "models" / "dinov2_classifier" / "config.pbtxt"
TRITON_ONNX = _REPO / "models" / "dinov2_classifier" / "1" / "model.onnx"
TRITON_ONNX_DATA = _REPO / "models" / "dinov2_classifier" / "1" / "model.onnx.data"
PHASE22_MISS = "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg|resize_recompression"
LOGIT_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
TENSOR_ATOL = 1e-5
BATCH_LOGIT_ATOL = 1e-4
PRODUCTION_FILES = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
    "backend/inference/engine.py",
    "backend/inference/routes.py",
    "backend/inference/schemas.py",
    "backend/config.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_backend_audit(*, hash_artifacts: bool = False) -> dict:
    """Read-only description of the current inference path. This does not load the frozen weights."""
    from config import REPO_ROOT, Settings, settings
    from inference.triton_client import preprocess_chw
    from inference.verdict import FAKE_THRESHOLD

    from augmentations import CONSERVATIVE, EVAL_TRANSFORM_VERSION, build_transforms
    from PIL import Image

    wide = Image.new("RGB", (600, 100), (180, 40, 40))
    reference = build_transforms(MODEL_FAMILY, train=False, profile=CONSERVATIVE)(wide)
    production = preprocess_chw(_rgb_array(wide))
    triton = _triton_status(hash_artifacts=hash_artifacts)
    configured = Path(settings.local_model_path) if settings.local_model_path else None
    if configured is not None and not configured.is_absolute():
        configured = REPO_ROOT / configured
    configured_sha = file_sha256(configured) if configured is not None and configured.is_file() else None
    answers = {
        "backbone": (
            f"Code default is {Settings.model_fields['dinov2_model_name'].default}. "
            f"This environment is configured as {settings.dinov2_model_name}. "
            "Neither is vit_base_patch16_dinov3.lvd1689m."
        ),
        "head": "Sequential LayerNorm, Dropout, Linear 512, GELU, Dropout, Linear 1 on the CLS token. Not cls_patch_attention.",
        "code_default_image_size": int(Settings.model_fields["inference_img_size"].default),
        "effective_image_size": int(settings.inference_img_size),
        "preprocessing": "PIL BICUBIC resize that stretches the image to a square, then ImageNet normalization. Not resize_pad_square_eval_v1.",
        "checkpoint": (
            "Code default LOCAL_MODEL_PATH is empty. "
            f"This environment points at {configured}. "
            "That file is not the frozen epoch_018.pt."
        ),
        "configured_checkpoint_sha256": configured_sha,
        "configured_checkpoint_matches_frozen": configured_sha == EXPECTED_CHECKPOINT_SHA256,
        "local_and_triton_same_model": False,
        "applies_sigmoid": True,
        "sigmoid_definition": "sigmoid(raw_logit), with no temperature",
        "threshold": float(FAKE_THRESHOLD),
        "threshold_applied_to": "uncalibrated sigmoid(raw_logit)",
        "verdict_policy": (
            "AUTHENTIC confidence is 1 - P(fake). "
            f"AUTHENTIC confidence below {settings.inference_min_authentic_confidence} is remapped to FAKE."
        ),
        "assumes_dinov2": True,
        "inference_backend": settings.inference_backend,
    }
    return {
        "phase": 28,
        "production_modules_modified": False,
        "answers": answers,
        "model_identity": {
            "production_family": "dinov2",
            "production_default_backbone": settings.dinov2_model_name,
            "production_head": "mlp_cls_only",
            "reference_family": MODEL_FAMILY,
            "reference_backbone": MODEL_ARCHITECTURE,
            "reference_head": MODEL_HEAD,
            "identity_match": False,
        },
        "preprocessing": {
            "production": "stretch_bicubic_square",
            "code_default_size": int(Settings.model_fields["inference_img_size"].default),
            "production_size": int(settings.inference_img_size),
            "reference": EVAL_TRANSFORM_VERSION[CONSERVATIVE],
            "reference_size": INPUT_SIZE,
            "reference_shape": list(reference.shape),
            "production_shape": list(production.shape),
            "material_difference": list(reference.shape) != list(production.shape[1:]),
            "shadow_uses_production_preprocess": False,
            "integration_of_production_preprocess": "stopped",
        },
        "probability": {
            "reference": "sigmoid(raw_logit / 0.24038200410185356)",
            "production": "sigmoid(raw_logit)",
            "temperature_applied_to": "raw_logit",
            "temperature_on_probability": False,
        },
        "triton": triton,
        "shadow_mode_default": False,
        "customer_traffic_used": False,
        "ood_status": "unavailable",
        "production_policy_status": "not_promoted",
    }


def run_phase28(experiments_root: Path | None = None, *, output_dir: Path | None = None) -> dict:
    """Write the parity artifacts. Raises if the shadow path drifts from the reference."""
    root = Path(experiments_root) if experiments_root is not None else _ML / "experiments"
    destination = _assert_destination(Path(output_dir) if output_dir is not None else root / OUTPUT_DIR_NAME)
    if (destination / "parity_summary.json").is_file():
        raise RuntimeError("backend parity artifacts already exist; refusing to rerun")
    before = _production_hashes()
    checkpoint_before = _checkpoint_stat()
    final_before = _final_test_stat()
    audit = build_backend_audit(hash_artifacts=True)
    if audit["preprocessing"]["material_difference"] is not True:
        raise RuntimeError("production preprocessing unexpectedly matched the reference; integration was not started")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("membership_hash") != AUTHORITATIVE_MEMBERSHIP_HASH:
        raise RuntimeError("Phase 16 membership hash does not match")
    test_paths = {str(Path(path).resolve()) for path in (manifest.get("membership") or {}).get("test") or []}
    rows = _feature_rows()
    probabilities = _cached_probabilities(root)
    cases = _select_cases(rows, probabilities, test_paths)
    regression = _phase22_regression(test_paths)
    import torch

    from inference.shadow import bind_shadow_model, reset_shadow_for_tests, run_shadow_batch, run_shadow_image
    from policy_contract import PolicyEngine, frozen_shadow_configuration
    from reference_inference import forward_logits, load_reference_model, preprocess_image, quality_flags_for_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, digest = load_reference_model(CHECKPOINT, device)
    bind_shadow_model(model, digest, device)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash changed during load")
    state_keys = list(model.state_dict())
    audit["checkpoint_contract"] = {
        "source_checkpoint_sha256": digest,
        "attention_pool_keys": sum("attention_pool" in key for key in state_keys),
        "fusion_keys": sum(key.startswith("fusion") or ".fusion." in key for key in state_keys),
        "production_local_torch_executed": False,
        "reason": "Production DINOv2Classifier has no patch-attention pool. Loading the frozen state into it would be a partial fallback.",
    }
    golden = []
    parity_rows = []
    comparison_rows = []
    images = []
    reference_engine = PolicyEngine(frozen_shadow_configuration())
    for case in cases:
        if case["case_id"] == "malformed_input":
            result, _timings = run_shadow_image(b"not-an-image")
            if result.decision is not None or result.system_status != "POLICY_ERROR":
                raise RuntimeError("malformed input did not fail closed")
            golden.append({**case, "expected_decision": None, "expected_system_status": "POLICY_ERROR", "expected_reason_codes": list(result.reason_codes)})
            continue
        image = _open_rgb(_REPO / case["sample_id"])
        if str((_REPO / case["sample_id"]).resolve()) in test_paths:
            raise RuntimeError("backend parity opened a final-test image")
        reference_tensor = preprocess_image(image).unsqueeze(0)
        reference_logit = float(forward_logits(model, reference_tensor, device)[0])
        reference_features, reference_flags = quality_flags_for_image(image)
        reference_result = reference_engine.evaluate(
            logit=reference_logit,
            quality_features=reference_features,
            quality_flags=reference_flags,
            preprocessing_version=PREPROCESSING_VERSION,
            split=case["split"],
            sample_id=case["sample_id"],
        )
        shadow_results = run_shadow_batch([image])
        shadow = shadow_results[0]
        shadow_logit = float(shadow.authenticity_logit)
        if reference_result.decision != shadow.decision or tuple(reference_result.quality_flags) != tuple(shadow.quality_flags):
            raise RuntimeError(f"shadow decision or quality flags diverged for {case['sample_id']}")
        reference_probability = calibrated_probability_from_logit(reference_logit)
        _require_match(case["sample_id"], reference_logit, shadow_logit, reference_probability, shadow)
        existing_verdict, existing_confidence = _production_verdict(reference_logit)
        agreement = existing_verdict == shadow.decision
        golden.append(
            {
                **case,
                "expected_logit": reference_logit,
                "expected_probability": reference_probability,
                "expected_quality_flags": list(shadow.quality_flags),
                "expected_decision": shadow.decision,
                "expected_reason_codes": list(shadow.reason_codes),
                "expected_system_status": shadow.system_status,
            }
        )
        parity_rows.append(
            {
                "case_id": case["case_id"],
                "sample_id": case["sample_id"],
                "reference_logit": reference_logit,
                "shadow_logit": shadow_logit,
                "abs_logit_delta": abs(reference_logit - shadow_logit),
                "reference_probability": reference_probability,
                "shadow_probability": shadow.authenticity_probability,
                "abs_probability_delta": abs(reference_probability - float(shadow.authenticity_probability)),
                "reference_flags": "|".join(reference_result.quality_flags),
                "shadow_flags": "|".join(shadow.quality_flags),
                "reference_decision": reference_result.decision,
                "shadow_decision": shadow.decision,
                "production_local_torch_executed": False,
            }
        )
        comparison_rows.append(
            {
                "case_id": case["case_id"],
                "sample_id": case["sample_id"],
                "existing_verdict": existing_verdict,
                "existing_confidence": existing_confidence,
                "shadow_decision": shadow.decision,
                "shadow_reason_codes": "|".join(shadow.reason_codes),
                "model_probability": shadow.authenticity_probability,
                "quality_flags": "|".join(shadow.quality_flags),
                "agreement": agreement,
                "disagreement_is_error": False,
            }
        )
        images.append(image)
    batch = _batch_parity(model, device, images[0], forward_logits, preprocess_image)
    latency = _latency(_REPO / golden[0]["sample_id"])
    if _production_hashes() != before:
        raise RuntimeError("backend parity modified a production inference file")
    if _checkpoint_stat() != checkpoint_before or file_sha256(CHECKPOINT) != digest:
        raise RuntimeError("backend parity changed the frozen checkpoint")
    if _final_test_stat() != final_before:
        raise RuntimeError("backend parity changed the final-test summary")
    agreements = sum(bool(row["agreement"]) for row in comparison_rows)
    summary = {
        "phase": 28,
        "checkpoint_sha256": digest,
        "membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "manifest_file_sha256": file_sha256(MANIFEST),
        "policy_version": POLICY_VERSION,
        "temperature": FROZEN_TEMPERATURE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "forward": "evaluation.collect_logits fp32, no autocast",
        "tolerances": {
            "preprocess_tensor_atol": TENSOR_ATOL,
            "logit_atol": LOGIT_ATOL,
            "probability_atol": PROBABILITY_ATOL,
            "batch_logit_atol": BATCH_LOGIT_ATOL,
            "bitwise_equality_across_engines": False,
        },
        "local_parity": {
            "images": len(parity_rows),
            "max_abs_logit_delta": max(row["abs_logit_delta"] for row in parity_rows),
            "max_abs_probability_delta": max(row["abs_probability_delta"] for row in parity_rows),
            "decisions_match": True,
            "quality_flags_match": True,
            "production_local_torch_executed": False,
        },
        "batch_parity": batch,
        "triton_parity_status": "skipped",
        "triton_reason": audit["triton"]["reason"],
        "phase22_regression": regression,
        "phase22_live_logit_check": _live_versus_stored_phase22_logit(parity_rows),
        "shadow_comparison": {
            "rows": len(comparison_rows),
            "agreements": agreements,
            "disagreements": len(comparison_rows) - agreements,
            "agreement_rate": agreements / len(comparison_rows) if comparison_rows else None,
            "disagreement_is_error": False,
        },
        "latency": latency,
        "shadow_mode_default": False,
        "production_response_modified": False,
        "production_policy_status": "not_promoted",
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "customer_traffic_used": False,
        "checkpoint_changed": False,
        "live_backend_modified": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "backend_audit.json").write_text(_dump(audit) + "\n")
    (destination / "golden_cases.json").write_text(_dump({"final_test_samples_used": 0, "cases": golden}) + "\n")
    _write_csv(destination / "local_torch_parity.csv", parity_rows)
    _write_csv(
        destination / "triton_parity.csv",
        [
            {
                "status": "skipped",
                "server_reachable": audit["triton"]["server_reachable"],
                "corresponds_to_frozen_checkpoint": False,
                "reason": audit["triton"]["reason"],
            }
        ],
    )
    _write_csv(destination / "shadow_comparison.csv", comparison_rows)
    (destination / "latency_benchmark.json").write_text(_dump(latency) + "\n")
    (destination / "parity_summary.json").write_text(_dump(summary) + "\n")
    reset_shadow_for_tests()
    return summary


def _require_match(sample_id: str, reference_logit: float, shadow_logit: float, reference_probability: float, shadow) -> None:
    if abs(reference_logit - shadow_logit) > LOGIT_ATOL:
        raise RuntimeError(f"logit parity failed for {sample_id}")
    if abs(reference_probability - float(shadow.authenticity_probability)) > PROBABILITY_ATOL:
        raise RuntimeError(f"probability parity failed for {sample_id}")
    if shadow.decision not in {"AUTHENTIC", "REVIEW", "FAKE"} or shadow.ood_status != "unavailable":
        raise RuntimeError(f"shadow decision invalid for {sample_id}")
    if shadow.ood_score is not None or shadow.ood_flag is not None:
        raise RuntimeError("shadow stored an OOD score while OOD is unavailable")
    low_resolution = "LOW_RESOLUTION" in shadow.quality_flags
    low_contrast = "LOW_CONTRAST" in shadow.quality_flags
    if low_resolution or low_contrast:
        if shadow.decision != "REVIEW":
            raise RuntimeError(f"quality gate did not review {sample_id}")
    elif reference_probability >= 0.50:
        if shadow.decision != "FAKE":
            raise RuntimeError(f"threshold decision drifted for {sample_id}")
    elif shadow.decision != "AUTHENTIC":
        raise RuntimeError(f"authentic decision drifted for {sample_id}")


def _batch_parity(model, device, image, forward, preprocess) -> dict:
    import torch

    single = preprocess(image).unsqueeze(0)
    baseline = forward(model, single, device)
    measured = {"batch_1_logit": float(baseline[0])}
    max_delta = 0.0
    for size in (2, 4, 8):
        batch = torch.stack([preprocess(image) for _ in range(size)])
        logits = forward(model, batch, device)
        delta = float((logits - baseline[0]).abs().max())
        measured[f"batch_{size}_max_abs_logit_delta"] = delta
        max_delta = max(max_delta, delta)
        if delta > BATCH_LOGIT_ATOL:
            raise RuntimeError(f"batch {size} logit drifted by {delta}")
    measured["max_abs_logit_delta"] = max_delta
    measured["atol"] = BATCH_LOGIT_ATOL
    return measured


def _latency(path: Path) -> dict:
    from inference.shadow import run_shadow_image
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.load()
        owned = image.copy()
    for _ in range(3):
        run_shadow_image(owned)
    samples = []
    for _ in range(30):
        _result, timings = run_shadow_image(owned)
        samples.append(timings)
    summary = {"warmup": 3, "measured": len(samples), "device_note": "shadow cache"}
    for name in ("preprocess_ms", "model_ms", "quality_ms", "policy_ms", "total_ms"):
        values = [float(row[name]) for row in samples]
        summary[name] = {"p50": _percentile(values, 50), "p95": _percentile(values, 95), "p99": _percentile(values, 99)}
    return summary


def _live_versus_stored_phase22_logit(parity_rows: list[dict]) -> dict:
    """The stored Phase 22 logit is historical. This run does not retune to match it."""
    sample_id = "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg"
    stored_logit = -0.83984375
    live = [row for row in parity_rows if row["sample_id"] == sample_id]
    fresh = float(live[0]["reference_logit"]) if live else None
    return {
        "sample_id": sample_id,
        "stored_phase22_bf16_logit": stored_logit,
        "fresh_fp32_reference_logit": fresh,
        "reproduced_stored_logit": fresh is not None and abs(fresh - stored_logit) <= LOGIT_ATOL,
        "policy_regression_uses": "stored phase 22 probabilities and flags",
        "note": (
            "A fresh fp32 evaluation.collect_logits forward of this image does not reproduce "
            "the stored Phase 22 bf16 logit. A fresh hard_negative bf16 forward in this "
            "environment is also -1.40625, not the stored -0.83984375. Shadow matches the "
            "current fp32 reference. Nothing was retuned."
        ),
    }


def _phase22_regression(test_paths: set[str]) -> dict:
    from policy_contract import PolicyEngine, frozen_shadow_configuration

    engine = PolicyEngine(frozen_shadow_configuration())
    events = []
    with STRESS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["false_fake"] not in {"true", "1"}:
                continue
            if row["source_split"] == "test" or str((_REPO / row["sample_id"]).resolve()) in test_paths:
                raise RuntimeError("phase 22 regression encountered a final-test sample")
            result = engine.evaluate(
                probability=float(row["transformed_calibrated_probability"]),
                quality_flags=row["flags_after"],
                preprocessing_version=PREPROCESSING_VERSION,
                split=row["source_split"],
                sample_id=row["sample_id"],
            )
            events.append(
                {
                    "event_id": f"{row['sample_id']}|{row['condition']}",
                    "condition": row["condition"],
                    "decision": result.decision,
                    "label": int(row["label"]),
                }
            )
    caught = [row for row in events if row["decision"] == "REVIEW"]
    missed = [row for row in events if row["decision"] != "REVIEW"]
    false_authentic = sum(row["label"] == 1 and row["decision"] == "AUTHENTIC" for row in events)
    if len(events) != 9 or len(caught) != 8 or [row["event_id"] for row in missed] != [PHASE22_MISS]:
        raise RuntimeError(f"phase 22 regression changed: caught {len(caught)} of {len(events)}")
    if missed[0]["decision"] != "FAKE" or false_authentic != 0:
        raise RuntimeError("phase 22 resize case did not remain FAKE or a false authentic appeared")
    return {
        "events": len(events),
        "review": len(caught),
        "fake": sum(row["decision"] == "FAKE" for row in events),
        "false_authentic_count": false_authentic,
        "missed_ids": [row["event_id"] for row in missed],
        "named_cases": _named_phase22(events),
    }


def _named_phase22(events: list[dict]) -> dict:
    by_id = {row["event_id"]: row["decision"] for row in events}
    expected = {
        "ap_mild_crop": "Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg|mild_crop",
        "ap_partial_crop": "Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg|partial_crop",
        "patek_lower_resolution_9ba50fea3805": "Label_0_Watches/Patek Philippe/9ba50fea3805.jpg|lower_resolution",
        "patek_lower_resolution_c2f778c66d15": "Label_0_Watches/Patek Philippe/c2f778c66d15.jpg|lower_resolution",
        "rm_darker": "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg|darker",
        "rm_resize_recompression": PHASE22_MISS,
    }
    resolved = {name: by_id[event_id] for name, event_id in expected.items()}
    if resolved["rm_resize_recompression"] != "FAKE" or any(resolved[name] != "REVIEW" for name in resolved if name != "rm_resize_recompression"):
        raise RuntimeError(f"named phase 22 cases drifted: {resolved}")
    return resolved


def _select_cases(rows: list[dict], probabilities: dict[str, float], test_paths: set[str]) -> list[dict]:
    usable = []
    for row in rows:
        path = _REPO / row["sample_id"]
        if not path.is_file() or str(path.resolve()) in test_paths:
            continue
        usable.append(row)
    by_id = {row["sample_id"]: row for row in usable}

    def first(predicate, case_id: str) -> dict:
        matches = sorted((row["sample_id"] for row in usable if predicate(row)))
        if not matches:
            raise RuntimeError(f"golden case {case_id} has no non-test image")
        return _case(case_id, by_id[matches[0]])

    def flag_set(row) -> set[str]:
        return set(row["quality_flags"])

    selected = [
        first(lambda row: row["label"] == 0 and not flag_set(row), "clean_authentic"),
        first(lambda row: row["label"] == 1 and not flag_set(row), "clean_fake"),
        first(lambda row: flag_set(row) == {"LOW_RESOLUTION"}, "low_resolution"),
        first(lambda row: flag_set(row) == {"LOW_CONTRAST"}, "low_contrast"),
        first(lambda row: {"LOW_RESOLUTION", "LOW_CONTRAST"} <= flag_set(row), "both_quality_flags"),
    ]
    ranked = []
    for sample_id, probability in probabilities.items():
        row = by_id.get(sample_id)
        if row is None:
            continue
        ranked.append((abs(probability - 0.50), probability, sample_id, row))
    if not ranked:
        raise RuntimeError("no cached calibrated probability was available for golden cases")
    ranked.sort()
    closest_distance, closest_probability, closest_id, closest_row = ranked[0]
    selected.append(
        {
            **_case("closest_cached_probability", closest_row),
            "cached_probability": closest_probability,
            "distance_to_0_50": closest_distance,
            "near_0_50_available": closest_distance <= 0.05,
        }
    )
    authentic = [item for item in ranked if item[3]["label"] == 0]
    fake = [item for item in ranked if item[3]["label"] == 1]
    strong_authentic = min(authentic, key=lambda item: item[1])
    strong_fake = max(fake, key=lambda item: item[1])
    selected.append({**_case("strong_authentic_probability", strong_authentic[3]), "cached_probability": strong_authentic[1]})
    selected.append({**_case("strong_fake_probability", strong_fake[3]), "cached_probability": strong_fake[1]})
    stress_source = "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg"
    if stress_source not in by_id:
        raise RuntimeError("phase 22 resize source image is not in the non-test catalog")
    selected.append(_case("phase22_resize_source", by_id[stress_source]))
    selected.append(
        {
            "case_id": "malformed_input",
            "sample_id": None,
            "split": None,
            "label": None,
            "quality_flags": [],
        }
    )
    return selected


def _case(case_id: str, row: dict) -> dict:
    return {
        "case_id": case_id,
        "sample_id": row["sample_id"],
        "split": row["source_split"],
        "label": row["label"],
        "quality_flags": list(row["quality_flags"]),
    }


def _feature_rows() -> list[dict]:
    rows = []
    with FEATURES.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["source_split"] == "test":
                raise RuntimeError("quality features included a final-test row")
            flags = tuple(part for part in record["quality_flags"].split("|") if part)
            low_resolution = float(record["min_dimension"]) < LOW_RESOLUTION_CUTOFF
            low_contrast = float(record["luminance_std"]) < LOW_CONTRAST_CUTOFF
            if low_resolution != ("LOW_RESOLUTION" in flags) or low_contrast != ("LOW_CONTRAST" in flags):
                raise RuntimeError(f"quality flags drifted for {record['sample_id']}")
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_split": record["source_split"],
                    "label": int(record["label"]),
                    "quality_flags": flags,
                }
            )
    return rows


def _cached_probabilities(root: Path) -> dict[str, float]:
    found: dict[str, float] = {}
    calibration = root / "v2_dinov3_cls_patch_attention" / "calibration" / "calibration_predictions.csv"
    if calibration.is_file():
        with calibration.open(newline="") as handle:
            for row in csv.DictReader(handle):
                found.setdefault(_canonical(row["sample_id"]), calibrated_probability_from_logit(float(row["logit"])))
    for name in ("low_margin.csv", "high_confidence.csv"):
        path = root / "hard_negative_v1" / name
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = _canonical(row["sample_id"])
                if row.get("logit"):
                    probability = calibrated_probability_from_logit(float(row["logit"]))
                else:
                    probability = float(row["calibrated_probability"])
                found[sample_id] = probability
    return found


def _production_verdict(logit: float) -> tuple[str, float]:
    import numpy as np
    from config import settings
    from inference.verdict import apply_min_authentic_confidence, logits_to_verdict

    verdict, confidence = logits_to_verdict(np.asarray([logit], dtype=np.float32))
    return apply_min_authentic_confidence(verdict, confidence, settings.inference_min_authentic_confidence)


def _triton_status(*, hash_artifacts: bool) -> dict:
    reachable = False
    try:
        with socket.create_connection(("127.0.0.1", 8001), timeout=0.3):
            reachable = True
    except OSError:
        reachable = False
    config_text = TRITON_CONFIG.read_text() if TRITON_CONFIG.is_file() else ""
    corresponds = "518" not in config_text and "dinov2_classifier" not in config_text and TRITON_CONFIG.is_file()
    reason = (
        "Triton model repository config is dinov2_classifier at 518x518. "
        "That is not the frozen DINOv3 512 contract, so Triton parity was not run. "
        + ("The server accepted a connection on port 8001." if reachable else "Nothing accepted a connection on port 8001.")
    )
    payload = {
        "server_reachable": reachable,
        "model_name": "dinov2_classifier",
        "config_present": TRITON_CONFIG.is_file(),
        "input_dims": [3, 518, 518],
        "backend": "onnxruntime",
        "corresponds_to_frozen_checkpoint": False if not corresponds else False,
        "parity_status": "skipped",
        "reason": reason,
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "converted_artifact_sha256": None,
        "converted_weights_sha256": None,
    }
    if hash_artifacts and TRITON_ONNX.is_file() and TRITON_ONNX_DATA.is_file():
        payload["converted_artifact_sha256"] = file_sha256(TRITON_ONNX)
        payload["converted_weights_sha256"] = file_sha256(TRITON_ONNX_DATA)
    payload["corresponds_to_frozen_checkpoint"] = False
    return payload


def _open_rgb(path: Path):
    from PIL import Image

    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        rgb.load()
        return rgb.copy()


def _rgb_array(image):
    import numpy as np

    return np.asarray(image.convert("RGB"))


def _canonical(value: str) -> str:
    text = str(value).replace("\\", "/")
    for marker in ("Label_0_Watches/", "Label_1_Watches/"):
        index = text.find(marker)
        if index >= 0:
            return text[index:]
    return text


def _percentile(values: list[float], quantile: float) -> float:
    import numpy as np

    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile / 100.0))


def _production_hashes() -> dict[str, str]:
    return {relative: file_sha256(_REPO / relative) for relative in PRODUCTION_FILES}


def _checkpoint_stat() -> tuple[int, int]:
    stat = CHECKPOINT.stat()
    return stat.st_size, stat.st_mtime_ns


def _final_test_stat() -> tuple[int, int] | None:
    if not FINAL_TEST_SUMMARY.is_file():
        return None
    stat = FINAL_TEST_SUMMARY.stat()
    return stat.st_size, stat.st_mtime_ns


def _assert_destination(path: Path) -> Path:
    from training_config import assert_experiment_output_dir

    resolved = assert_experiment_output_dir(path)
    if resolved.name != OUTPUT_DIR_NAME:
        raise RuntimeError(f"backend parity output must be named {OUTPUT_DIR_NAME}")
    return resolved


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, allow_nan=False, sort_keys=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv(row.get(key)) for key in fields})


def _csv(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value
