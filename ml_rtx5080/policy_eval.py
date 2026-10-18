"""Simulate the shadow policy on stored non-test scores. No live deployment.

Calibration and validation logits are converted with the frozen temperature.
Where the full logit was not retained, the side of 0.50 is taken from the
Phase 22 result that every non-test image was classified exactly. Numeric
probabilities are not invented. The final test is not opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from policy import (
    AUTHENTICITY_THRESHOLD,
    EXPECTED_CHECKPOINT_SHA256,
    FROZEN_TEMPERATURE,
    LOW_CONTRAST_CUTOFF,
    LOW_RESOLUTION_CUTOFF,
    OOD_UNAVAILABLE,
    PRODUCTION_POLICY_STATUS,
    QUALITY_RULE,
    QUALITY_RULE_STATUS,
    THRESHOLD_STATUS,
    authenticity_only_decision,
    build_policy_trace,
    calibrated_probability_from_logit,
    evaluate_quality_gate,
    parse_flags,
    screen_splits,
    summarize_policy,
)
from policy_contract import (
    PHASE23_CUTOFFS,
    POLICY_STATUS,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PRECEDENCE,
    PolicyEngine,
    frozen_shadow_configuration,
    policy_result_schema,
)
from quality_gate import assign_quality_level
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH, PHASE15_EXPERIMENT_DIRS, SPLIT_NAMES
from training_config import assert_experiment_output_dir

MEMBERSHIP_HASH = AUTHORITATIVE_MEMBERSHIP_HASH
OUTPUT_DIR_NAME = "policy_shadow_v1"
CONTRACT_DIR_NAME = "policy_contract_v1"
PHASE26_AUTHENTIC_COUNT = 11917
PHASE26_REVIEW_COUNT = 1855
PHASE26_FAKE_COUNT = 13222
PHASE26_SAMPLE_COUNT = 26994
PHASE26_STRESS_CAUGHT = 8
PHASE26_STRESS_EVENTS = 9
PHASE26_STRESS_MISS = "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg|resize_recompression"
BACKEND_FILES = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_SPLITS = ("train", "calibration", "validation")
TRACE_FIELDS = (
    "sample_id",
    "split",
    "brand",
    "label",
    "calibrated_probability",
    "probability_source",
    "authenticity_threshold",
    "quality_flags",
    "quality_review",
    "ood_status",
    "ood_score",
    "ood_flag",
    "decision",
    "reason_codes",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase26(
    experiments_root: Path,
    manifest_path: Path,
    *,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    quality_features: Path | None = None,
    stress_linkage: Path | None = None,
    phase22_summary: Path | None = None,
    calibration_predictions: Path | None = None,
    validation_predictions: Path | None = None,
    validation_checkpoint: Path | None = None,
    low_margin: Path | None = None,
    high_confidence: Path | None = None,
    quality_thresholds: Path | None = None,
    input_splits: tuple[str, ...] = SOURCE_SPLITS,
    threshold: float = AUTHENTICITY_THRESHOLD,
    quality_candidate: str = QUALITY_RULE,
    mode: str = "shadow",
    enforce_expected_sha: bool = True,
) -> dict:
    """Write the shadow simulation. Refuses a promoted production mode."""
    if mode != "shadow":
        raise ValueError("only --mode shadow is supported; production mode is not available")
    if quality_candidate != QUALITY_RULE:
        raise ValueError(f"shadow quality candidate must stay {QUALITY_RULE}")
    if float(threshold) != AUTHENTICITY_THRESHOLD:
        raise ValueError("shadow threshold candidate is 0.50 and is not tuned in this phase")
    for split in input_splits:
        if split == "test" or split not in SOURCE_SPLITS:
            raise RuntimeError("shadow policy encountered a final-test sample")
    root = Path(experiments_root)
    destination = _assert_destination(Path(output_dir) if output_dir is not None else root / OUTPUT_DIR_NAME)
    if (destination / "policy_summary.json").is_file():
        raise RuntimeError("shadow policy simulation already completed; refusing to rerun")
    manifest_file = Path(manifest_path)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else root / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
    final_test_summary = root / "final_test_v1" / "final_test_summary.json"
    protected = {
        "checkpoint": file_sha256(checkpoint_path),
        "manifest_file": file_sha256(manifest_file),
        "final_test_summary": file_sha256(final_test_summary) if final_test_summary.is_file() else None,
    }
    if enforce_expected_sha and protected["checkpoint"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash does not match the Phase 21 checkpoint")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("membership_hash") != MEMBERSHIP_HASH:
        raise RuntimeError("Phase 16 membership hash does not match the authoritative hash")
    test_ids = _test_sample_ids(manifest)
    features_path = Path(quality_features) if quality_features is not None else root / "image_quality_v1" / "quality_features_nontest.csv"
    linkage_path = Path(stress_linkage) if stress_linkage is not None else root / "image_quality_v1" / "quality_stress_linkage.csv"
    summary_path = Path(phase22_summary) if phase22_summary is not None else root / "hard_negative_v1" / "challenge_summary.json"
    threshold_path = Path(quality_thresholds) if quality_thresholds is not None else root / "image_quality_v1" / "quality_thresholds.json"
    _assert_quality_cutoffs(threshold_path)
    rows = _load_feature_rows(features_path, set(input_splits))
    screen_splits(rows, test_ids)
    probabilities = _cached_probabilities(
        root,
        calibration_predictions=calibration_predictions,
        validation_predictions=validation_predictions,
        validation_checkpoint=validation_checkpoint,
        low_margin=low_margin,
        high_confidence=high_confidence,
        checkpoint_sha256=protected["checkpoint"],
    )
    exact_pool = _phase22_exact_pool(summary_path)
    traces = [_reference_trace(row, probabilities, exact_pool) for row in rows]
    screen_splits(traces, test_ids)
    stress = _stress_traces(linkage_path, test_ids) if linkage_path.is_file() else []
    reference_summary = summarize_policy(traces)
    only_rows = [_with_decision(row, authenticity_only_decision(row["calibrated_probability"], row["probability_side"])) for row in _reference_sides(rows, probabilities, exact_pool)]
    quality_rows = traces
    stress_only = [_with_decision(row, authenticity_only_decision(row["calibrated_probability"], None)) for row in stress]
    stress_quality = stress
    summary = {
        "phase": 26,
        "mode": "shadow",
        "production_policy_status": PRODUCTION_POLICY_STATUS,
        "production_authenticity_threshold": None,
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "threshold_status": THRESHOLD_STATUS,
        "temperature": FROZEN_TEMPERATURE,
        "temperature_changed": False,
        "calibrated_probability_definition": "sigmoid(logit / T)",
        "quality_rule": QUALITY_RULE,
        "quality_rule_status": QUALITY_RULE_STATUS,
        "quality_review_definition": "LOW_RESOLUTION or LOW_CONTRAST",
        "quality_cutoffs": {
            "LOW_RESOLUTION": f"min_dimension < {LOW_RESOLUTION_CUTOFF}",
            "LOW_CONTRAST": f"luminance_std < {LOW_CONTRAST_CUTOFF}",
            "comparison": "strict inequality",
        },
        "shadow_precedence": [
            "quality_review_flag -> REVIEW",
            "else calibrated P(fake) >= 0.50 -> FAKE",
            "else AUTHENTIC",
        ],
        "ood_status": OOD_UNAVAILABLE,
        "ood_score": None,
        "ood_flag": None,
        "production_ood_threshold": None,
        "research_ood_threshold": None,
        "ood_protection": "unavailable",
        "in_distribution_assumed": False,
        "checkpoint_sha256": protected["checkpoint"],
        "membership_hash": MEMBERSHIP_HASH,
        "manifest_file_sha256": protected["manifest_file"],
        "checkpoint_changed": False,
        "training_performed": False,
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "live_backend_modified": False,
        "source_splits": list(input_splits),
        "reference": reference_summary,
        "quality_levels": _level_summary(rows, traces),
        "brands": _brand_summary(traces),
        "reason_codes": _reason_distribution(traces),
        "review_budget": [
            _budget_row("reference", "authenticity_only", summarize_policy(only_rows)),
            _budget_row("reference", "quality_gated", summarize_policy(quality_rows)),
            _budget_row("phase22_stress", "authenticity_only", summarize_policy(stress_only)),
            _budget_row("phase22_stress", "quality_gated", summarize_policy(stress_quality)),
        ],
        "stress": _stress_summary(stress),
        "clean_review": _clean_review(rows, traces),
        "probability_sources": dict(Counter(row["probability_source"] for row in traces)),
    }
    _write_outputs(destination, summary, traces, stress)
    if file_sha256(checkpoint_path) != protected["checkpoint"]:
        raise RuntimeError("shadow policy changed checkpoint bytes")
    if file_sha256(manifest_file) != protected["manifest_file"]:
        raise RuntimeError("shadow policy changed the split manifest file")
    if protected["final_test_summary"] is not None and file_sha256(final_test_summary) != protected["final_test_summary"]:
        raise RuntimeError("shadow policy changed the final-test summary")
    return json.loads((destination / "policy_summary.json").read_text())


def _reference_trace(row: dict, probabilities: dict[str, dict], exact_pool: bool) -> dict:
    cached = probabilities.get(row["sample_id"])
    probability = None if cached is None else cached["probability"]
    source = None if cached is None else cached["source"]
    side = None
    if probability is None:
        if not exact_pool:
            raise ValueError(f"no calibrated probability for {row['sample_id']}")
        side = "below" if int(row["label"]) == 0 else "above"
        source = "phase22_exact_calibrated_side"
    trace = build_policy_trace(
        {**row, "probability_source": source},
        probability=probability,
        probability_side=side,
        ood_status=OOD_UNAVAILABLE,
    )
    trace["probability_side"] = side if probability is None else ("above" if probability >= AUTHENTICITY_THRESHOLD else "below")
    trace["quality_level"] = assign_quality_level(parse_flags(row["quality_flags"]), row)
    if probability is not None and ((int(row["label"]) == 0) != (probability < AUTHENTICITY_THRESHOLD)):
        raise RuntimeError("cached calibrated probability contradicts the Phase 22 exact pool")
    return trace


def _reference_sides(rows, probabilities, exact_pool) -> list[dict]:
    prepared = []
    for row in rows:
        cached = probabilities.get(row["sample_id"])
        probability = None if cached is None else cached["probability"]
        side = None if probability is not None else ("below" if int(row["label"]) == 0 else "above")
        if probability is None and not exact_pool:
            raise ValueError(f"no calibrated probability for {row['sample_id']}")
        prepared.append({**row, "calibrated_probability": probability, "probability_side": side, "decision": "AUTHENTIC"})
    return prepared


def _with_decision(row: dict, decision: str) -> dict:
    return {**row, "decision": decision}


def _stress_traces(path: Path, test_ids: set[str]) -> list[dict]:
    rows = []
    with Path(path).open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["source_split"] not in SOURCE_SPLITS or record["sample_id"] in test_ids:
                raise RuntimeError("shadow policy encountered a final-test sample")
            before_flags = parse_flags(record["flags_before"])
            after_flags = parse_flags(record["flags_after"])
            before = float(record["original_calibrated_probability"])
            after = float(record["transformed_calibrated_probability"])
            before_decision = evaluate_quality_gate(before_flags)
            after_policy = build_policy_trace(
                {
                    "sample_id": record["sample_id"],
                    "source_split": record["source_split"],
                    "brand": record["brand"],
                    "label": int(record["label"]),
                    "quality_flags": after_flags,
                    "probability_source": "phase22_stress_calibrated_probability",
                },
                probability=after,
                ood_status=OOD_UNAVAILABLE,
            )
            false_fake = record["false_fake"] in {"true", "1"}
            caught = false_fake and after_policy["decision"] == "REVIEW"
            rows.append(
                {
                    **after_policy,
                    "condition": record["condition"],
                    "probability_before": before,
                    "probability_after": after,
                    "flags_before": list(before_flags),
                    "flags_after": list(after_flags),
                    "decision_before": "REVIEW" if before_decision else authenticity_only_decision(before, None),
                    "decision_after": after_policy["decision"],
                    "false_fake_event": false_fake,
                    "caught_by_quality_policy": caught,
                    "quality_review_before": before_decision,
                }
            )
    return rows


def _stress_summary(rows: list[dict]) -> dict:
    failures = [row for row in rows if row["false_fake_event"]]
    caught = [row for row in failures if row["caught_by_quality_policy"]]
    missed = [row for row in failures if not row["caught_by_quality_policy"]]
    return {
        "stress_rows": len(rows),
        "demonstrated_false_fake_events": len(failures),
        "caught_by_quality_policy": len(caught),
        "missed_by_quality_policy": len(missed),
        "missed_ids": [f"{row['sample_id']}|{row['condition']}" for row in missed],
        "false_authentic_count": sum(int(row["label"]) == 1 and row["decision"] == "AUTHENTIC" for row in rows),
    }


def _level_summary(features: list[dict], traces: list[dict]) -> list[dict]:
    by_id = {row["sample_id"]: row for row in traces}
    grouped = {level: [] for level in (0, 1, 2)}
    for feature in features:
        level = assign_quality_level(parse_flags(feature["quality_flags"]), feature)
        grouped[level].append(by_id[feature["sample_id"]])
    rows = []
    for level in (0, 1, 2):
        summary = summarize_policy(grouped[level])
        rows.append({"quality_level": level, **summary})
    return rows


def _brand_summary(traces: list[dict]) -> list[dict]:
    brands = sorted({row["brand"] for row in traces})
    rows = []
    for brand in brands:
        summary = summarize_policy([row for row in traces if row["brand"] == brand])
        rows.append({"brand": brand, **summary})
    return rows


def _reason_distribution(traces: list[dict]) -> dict:
    counts = Counter("|".join(row["reason_codes"]) for row in traces)
    return dict(sorted(counts.items()))


def _budget_row(dataset: str, policy_name: str, summary: dict) -> dict:
    return {
        "dataset": dataset,
        "policy": policy_name,
        "authentic_rate": summary["authentic_rate"],
        "fake_rate": summary["fake_rate"],
        "review_rate": summary["review_rate"],
        "false_authentic_count": summary["false_authentic_count"],
        "false_fake_count": summary["false_fake_count"],
        "review_count": summary["review_count"],
        "sample_count": summary["sample_count"],
    }


def _clean_review(features: list[dict], traces: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in traces}
    clean = [feature for feature in features if not parse_flags(feature["quality_flags"])]
    reviewed = [by_id[feature["sample_id"]] for feature in clean if by_id[feature["sample_id"]]["decision"] == "REVIEW"]
    return {
        "clean_count": len(clean),
        "clean_reviewed_count": len(reviewed),
        "clean_review_rate": 0.0 if not clean else len(reviewed) / len(clean),
    }


def _cached_probabilities(
    root: Path,
    *,
    calibration_predictions,
    validation_predictions,
    validation_checkpoint,
    low_margin,
    high_confidence,
    checkpoint_sha256: str,
) -> dict[str, dict]:
    found: dict[str, dict] = {}
    calibration = Path(calibration_predictions) if calibration_predictions is not None else root / "v2_dinov3_cls_patch_attention" / "calibration" / "calibration_predictions.csv"
    if calibration.is_file():
        _load_logit_file(calibration, found, "phase20_calibration_logit")
    validation_file = Path(validation_predictions) if validation_predictions is not None else root / "v2_dinov3_cls_patch_attention" / "evaluation" / "predictions.csv"
    validation_weights = Path(validation_checkpoint) if validation_checkpoint is not None else root / "v2_dinov3_cls_patch_attention" / "best_model.pt"
    if validation_file.is_file() and validation_weights.is_file() and file_sha256(validation_weights) == checkpoint_sha256:
        _load_logit_file(validation_file, found, "frozen_validation_logit")
    for path, source in (
        (Path(low_margin) if low_margin is not None else root / "hard_negative_v1" / "low_margin.csv", "phase22_low_margin"),
        (Path(high_confidence) if high_confidence is not None else root / "hard_negative_v1" / "high_confidence.csv", "phase22_high_confidence"),
    ):
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = _canonical_sample_id(row["sample_id"])
                if "logit" in row and row["logit"] != "":
                    probability = calibrated_probability_from_logit(float(row["logit"]))
                else:
                    probability = float(row["calibrated_probability"])
                found[sample_id] = {"probability": probability, "source": source}
    return found


def _load_logit_file(path: Path, found: dict[str, dict], source: str) -> None:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = _canonical_sample_id(row["sample_id"])
            found.setdefault(sample_id, {"probability": calibrated_probability_from_logit(float(row["logit"])), "source": source})


def _phase22_exact_pool(path: Path) -> bool:
    if not path.is_file():
        return False
    summary = json.loads(path.read_text())
    pool = summary.get("non_test_pool") or {}
    distribution = summary.get("non_test_distribution") or {}
    return (
        int(summary.get("test_samples_used", 1)) == 0
        and float(pool.get("accuracy", 0.0)) == 1.0
        and int(pool.get("false_authentic_count", 1)) == 0
        and int(pool.get("false_fake_count", 1)) == 0
        and int(distribution.get("misclassified_count", 1)) == 0
        and int(pool.get("sample_count", 0)) == 26994
    )


def _load_feature_rows(path: Path, splits: set[str]) -> list[dict]:
    rows = []
    with Path(path).open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["source_split"] not in splits:
                continue
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_split": record["source_split"],
                    "brand": record["brand"],
                    "label": int(record["label"]),
                    "quality_flags": parse_flags(record.get("quality_flags")),
                    "min_dimension": float(record["min_dimension"]),
                    "luminance_std": float(record["luminance_std"]),
                }
            )
    return rows


def _assert_quality_cutoffs(path: Path) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    cutoffs = payload["cutoffs"]
    if float(cutoffs["min_dimension_p5"]) != LOW_RESOLUTION_CUTOFF or float(cutoffs["luminance_std_p5"]) != LOW_CONTRAST_CUTOFF:
        raise RuntimeError("Phase 23 quality cutoffs do not match the shadow quality candidate")


def _test_sample_ids(manifest: dict) -> set[str]:
    membership = manifest.get("membership") or {}
    test_paths = {str(Path(path).resolve()) for path in membership.get("test") or []}
    identifiers = set()
    for row in manifest.get("samples") or []:
        path = row.get("path")
        if isinstance(path, str) and str(Path(path).resolve()) in test_paths and isinstance(row.get("sample_id"), str):
            identifiers.add(row["sample_id"])
    return identifiers


def _canonical_sample_id(value: str) -> str:
    text = str(value).replace("\\", "/")
    for marker in ("Label_0_Watches/", "Label_1_Watches/"):
        index = text.find(marker)
        if index >= 0:
            return text[index:]
    return text


def _write_outputs(destination: Path, summary: dict, traces: list[dict], stress: list[dict]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "shadow",
        "production_policy_status": PRODUCTION_POLICY_STATUS,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "membership_hash": summary["membership_hash"],
        "manifest_file_sha256": summary["manifest_file_sha256"],
        "temperature": FROZEN_TEMPERATURE,
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "threshold_status": THRESHOLD_STATUS,
        "production_authenticity_threshold": None,
        "quality_rule": QUALITY_RULE,
        "quality_rule_status": QUALITY_RULE_STATUS,
        "quality_cutoffs": summary["quality_cutoffs"],
        "ood_status": OOD_UNAVAILABLE,
        "production_ood_threshold": None,
        "research_ood_threshold": None,
        "shadow_precedence": summary["shadow_precedence"],
    }
    (destination / "policy_config.json").write_text(_dump(config) + "\n")
    (destination / "policy_summary.json").write_text(_dump(summary) + "\n")
    (destination / "policy_confusion.json").write_text(_dump(summary["reference"]["confusion"]) + "\n")
    _write_csv(destination / "policy_predictions.csv", [_flat_trace(row) for row in traces], TRACE_FIELDS)
    _write_csv(
        destination / "policy_brand_summary.csv",
        summary["brands"],
        (
            "brand",
            "sample_count",
            "authentic_rate",
            "fake_rate",
            "review_rate",
            "false_authentic_count",
            "false_fake_count",
        ),
    )
    _write_csv(
        destination / "policy_stress_summary.csv",
        [
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "condition": row["condition"],
                "probability_before": row["probability_before"],
                "probability_after": row["probability_after"],
                "flags_before": "|".join(row["flags_before"]),
                "flags_after": "|".join(row["flags_after"]),
                "decision_before": row["decision_before"],
                "decision_after": row["decision_after"],
                "false_fake_event": row["false_fake_event"],
                "caught_by_quality_policy": row["caught_by_quality_policy"],
            }
            for row in stress
        ],
        (
            "sample_id",
            "split",
            "condition",
            "probability_before",
            "probability_after",
            "flags_before",
            "flags_after",
            "decision_before",
            "decision_after",
            "false_fake_event",
            "caught_by_quality_policy",
        ),
    )
    _write_csv(
        destination / "policy_review_budget.csv",
        summary["review_budget"],
        (
            "dataset",
            "policy",
            "authentic_rate",
            "fake_rate",
            "review_rate",
            "false_authentic_count",
            "false_fake_count",
            "review_count",
            "sample_count",
        ),
    )


def _flat_trace(row: dict) -> dict:
    return {
        **row,
        "quality_flags": "|".join(row["quality_flags"]),
        "reason_codes": "|".join(row["reason_codes"]),
    }


def _assert_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != OUTPUT_DIR_NAME:
        raise RuntimeError(f"shadow policy output must be named {OUTPUT_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | set(SPLIT_NAMES) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        "final_test_v1",
        "final_model_freeze",
        "hard_negative_v1",
        "image_quality_v1",
        "image_quality_v2",
        "ood_v1",
        "v2_dinov2_cls_only",
        "v2_dinov2_cls_patch_attention",
        "v2_dinov3_cls_only",
        "v2_dinov3_cls_patch_attention",
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the shadow policy into {resolved}")
    return resolved


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, allow_nan=False, sort_keys=True)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def simulate_contract(
    experiments_root: Path,
    manifest_path: Path,
    *,
    checkpoint: Path | None = None,
    quality_features: Path | None = None,
    stress_linkage: Path | None = None,
    phase22_summary: Path | None = None,
    calibration_predictions: Path | None = None,
    validation_predictions: Path | None = None,
    validation_checkpoint: Path | None = None,
    low_margin: Path | None = None,
    high_confidence: Path | None = None,
    quality_thresholds: Path | None = None,
    input_splits: tuple[str, ...] = SOURCE_SPLITS,
    threshold: float = AUTHENTICITY_THRESHOLD,
    quality_candidate: str = QUALITY_RULE,
    mode: str = "shadow",
    enforce_expected_sha: bool = True,
) -> dict:
    """Replay the shadow contract on cached non-test rows. A count mismatch raises."""
    if mode != "shadow":
        raise ValueError("only --mode shadow is supported; production mode is not available")
    if quality_candidate != QUALITY_RULE or float(threshold) != AUTHENTICITY_THRESHOLD:
        raise ValueError("the shadow contract does not retune the threshold or the quality candidate")
    for split in input_splits:
        if split == "test" or split not in SOURCE_SPLITS:
            raise RuntimeError("policy contract encountered a final-test sample")
    root = Path(experiments_root)
    manifest_file = Path(manifest_path)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else root / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
    checkpoint_stat = checkpoint_path.stat()
    checkpoint_sha = file_sha256(checkpoint_path)
    if enforce_expected_sha and checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash does not match the Phase 21 checkpoint")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("membership_hash") != MEMBERSHIP_HASH:
        raise RuntimeError("Phase 16 membership hash does not match the authoritative hash")
    threshold_path = Path(quality_thresholds) if quality_thresholds is not None else root / "image_quality_v1" / "quality_thresholds.json"
    _assert_quality_cutoffs(threshold_path)
    if threshold_path.is_file():
        cutoffs = json.loads(threshold_path.read_text())["cutoffs"]
        for name, expected in PHASE23_CUTOFFS.items():
            if float(cutoffs[name]) != float(expected):
                raise RuntimeError(f"Phase 23 cutoff {name} does not match the policy contract")
    test_ids = _test_sample_ids(manifest)
    features_path = Path(quality_features) if quality_features is not None else root / "image_quality_v1" / "quality_features_nontest.csv"
    linkage_path = Path(stress_linkage) if stress_linkage is not None else root / "image_quality_v1" / "quality_stress_linkage.csv"
    summary_path = Path(phase22_summary) if phase22_summary is not None else root / "hard_negative_v1" / "challenge_summary.json"
    rows = _load_feature_rows(features_path, set(input_splits))
    screen_splits(rows, test_ids)
    for row in rows:
        _assert_decision_flags(row)
    probabilities = _cached_probabilities(
        root,
        calibration_predictions=calibration_predictions,
        validation_predictions=validation_predictions,
        validation_checkpoint=validation_checkpoint,
        low_margin=low_margin,
        high_confidence=high_confidence,
        checkpoint_sha256=checkpoint_sha,
    )
    exact_pool = _phase22_exact_pool(summary_path)
    engine = PolicyEngine(frozen_shadow_configuration())
    traces = []
    for row in rows:
        trace = _reference_trace(row, probabilities, exact_pool)
        result = engine.evaluate(
            probability=trace["calibrated_probability"],
            recorded_probability_side=trace["probability_side"] if trace["calibrated_probability"] is None else None,
            quality_flags=trace["quality_flags"],
            preprocessing_version=PREPROCESSING_VERSION,
            split=trace["split"],
            sample_id=trace["sample_id"],
            test_sample_ids=test_ids,
        )
        if result.system_status != "ok" or result.decision != trace["decision"]:
            raise RuntimeError(
                f"policy contract decision {result.decision!r} diverges from Phase 26 for {trace['sample_id']}"
            )
        if trace["calibrated_probability"] is None and result.authenticity_probability is not None:
            raise RuntimeError("policy contract invented a calibrated probability")
        if result.ood_status != OOD_UNAVAILABLE or result.ood_score is not None or result.ood_flag is not None:
            raise RuntimeError("policy contract stored an OOD score while OOD is unavailable")
        if result.decision not in {"AUTHENTIC", "REVIEW", "FAKE"} or not result.reason_codes:
            raise RuntimeError("policy contract emitted a verdict without a reason code")
        traces.append(trace)
    reference = summarize_policy(traces)
    _require_phase26_counts(reference)
    stress = _stress_traces(linkage_path, test_ids) if linkage_path.is_file() else []
    stress_engine = PolicyEngine(frozen_shadow_configuration())
    for row in stress:
        result = stress_engine.evaluate(
            probability=row["probability_after"],
            quality_flags=row["flags_after"],
            preprocessing_version=PREPROCESSING_VERSION,
            split=row["split"],
            sample_id=row["sample_id"],
            test_sample_ids=test_ids,
        )
        if result.decision != row["decision_after"] or result.system_status != "ok":
            raise RuntimeError(f"policy contract stress decision diverges for {row['sample_id']}|{row['condition']}")
        if f"{row['sample_id']}|{row['condition']}" == PHASE26_STRESS_MISS and result.decision != "FAKE":
            raise RuntimeError("the resize/recompression stress miss did not remain FAKE")
    stress_summary = _stress_summary(stress)
    _require_phase26_stress(stress_summary)
    if file_sha256(checkpoint_path) != checkpoint_sha or checkpoint_path.stat().st_mtime != checkpoint_stat.st_mtime:
        raise RuntimeError("policy contract changed checkpoint bytes")
    return {
        "phase": 27,
        "mode": "shadow",
        "policy_version": POLICY_VERSION,
        "policy_status": POLICY_STATUS,
        "production_policy_status": PRODUCTION_POLICY_STATUS,
        "production_authenticity_threshold": None,
        "production_quality_gate": None,
        "production_ood_threshold": None,
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "threshold_status": THRESHOLD_STATUS,
        "temperature": FROZEN_TEMPERATURE,
        "quality_policy": QUALITY_RULE,
        "quality_policy_version": frozen_shadow_configuration().quality_policy_version,
        "quality_rule_status": QUALITY_RULE_STATUS,
        "ood_status": OOD_UNAVAILABLE,
        "ood_score": None,
        "ood_flag": None,
        "in_distribution_assumed": False,
        "precedence": list(PRECEDENCE),
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_size": 512,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_stat.st_size,
        "checkpoint_mtime": checkpoint_stat.st_mtime,
        "membership_hash": MEMBERSHIP_HASH,
        "manifest_file_sha256": file_sha256(manifest_file),
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "live_backend_modified": False,
        "temperature_changed": False,
        "checkpoint_changed": False,
        "counts_match": True,
        "stress_match": True,
        "observed": {
            "authentic_count": reference["authentic_count"],
            "review_count": reference["review_count"],
            "fake_count": reference["fake_count"],
            "false_authentic_count": reference["false_authentic_count"],
            "false_fake_count": reference["false_fake_count"],
            "false_authentic_rate": reference["false_authentic_rate"],
            "false_fake_rate": reference["false_fake_rate"],
            "review_rate": reference["review_rate"],
            "sample_count": reference["sample_count"],
        },
        "phase26_expected": {
            "authentic_count": PHASE26_AUTHENTIC_COUNT,
            "review_count": PHASE26_REVIEW_COUNT,
            "fake_count": PHASE26_FAKE_COUNT,
            "false_authentic_count": 0,
            "false_fake_count": 0,
            "sample_count": PHASE26_SAMPLE_COUNT,
        },
        "counters": engine.counters.to_dict(),
        "stress": stress_summary,
        "stress_counters": stress_engine.counters.to_dict(),
        "probability_sources": dict(Counter(row["probability_source"] for row in traces)),
        "policy_error_count": engine.counters.policy_error,
    }


def run_phase27(
    experiments_root: Path,
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    **kwargs,
) -> dict:
    """Write the contract audit after the shadow counts reproduce Phase 26."""
    root = Path(experiments_root)
    destination = _assert_contract_destination(Path(output_dir) if output_dir is not None else root / CONTRACT_DIR_NAME)
    if (destination / "policy_simulation_summary.json").is_file():
        raise RuntimeError("policy contract simulation already completed; refusing to rerun")
    repo = Path(__file__).resolve().parents[1]
    backend_before = _backend_hashes(repo)
    final_test_summary = root / "final_test_v1" / "final_test_summary.json"
    final_before = file_sha256(final_test_summary) if final_test_summary.is_file() else None
    observed = simulate_contract(root, manifest_path, **kwargs)
    if not observed["counts_match"] or not observed["stress_match"]:
        raise RuntimeError("refusing to write a policy contract that does not reproduce Phase 26")
    backend_after = _backend_hashes(repo)
    if backend_before != backend_after:
        raise RuntimeError("policy contract modified a live backend file")
    if final_before is not None and file_sha256(final_test_summary) != final_before:
        raise RuntimeError("policy contract changed the final-test summary")
    config = frozen_shadow_configuration().to_dict()
    config.update(
        {
            "checkpoint_sha256": observed["checkpoint_sha256"],
            "membership_hash": observed["membership_hash"],
            "manifest_file_sha256": observed["manifest_file_sha256"],
            "precedence": observed["precedence"],
            "final_test_samples_used": 0,
        }
    )
    audit = {
        "phase": 27,
        "policy_version": POLICY_VERSION,
        "policy_status": POLICY_STATUS,
        "production_policy_status": PRODUCTION_POLICY_STATUS,
        "production_authenticity_threshold": None,
        "production_quality_gate": None,
        "production_ood_threshold": None,
        "checkpoint_sha256": observed["checkpoint_sha256"],
        "checkpoint_bytes": observed["checkpoint_bytes"],
        "checkpoint_mtime": observed["checkpoint_mtime"],
        "membership_hash": observed["membership_hash"],
        "manifest_file_sha256": observed["manifest_file_sha256"],
        "temperature": FROZEN_TEMPERATURE,
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "threshold_status": THRESHOLD_STATUS,
        "quality_policy": QUALITY_RULE,
        "ood_status": OOD_UNAVAILABLE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "batch_absolute_tolerance": 1e-5,
        "precedence": observed["precedence"],
        "fail_closed": [
            "non-finite logit or probability",
            "invalid temperature or threshold at initialization",
            "malformed or missing quality feature",
            "corrupted image or missing file",
            "model family, head, input size, or checkpoint mismatch",
            "preprocessing other than resize_pad_square_eval_v1 at 512",
            "OOD score or flag while status is unavailable",
        ],
        "final_test_samples_used": 0,
        "training_performed": False,
        "live_backend_modified": False,
        "backend_sha256": backend_after,
        "counts_match_phase26": True,
        "stress_match_phase26": True,
        "phase26_expected": observed["phase26_expected"],
        "observed": observed["observed"],
        "stress": observed["stress"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "policy_schema.json").write_text(_dump(policy_result_schema()) + "\n")
    (destination / "policy_config.json").write_text(_dump(config) + "\n")
    (destination / "policy_contract_audit.json").write_text(_dump(audit) + "\n")
    (destination / "policy_simulation_summary.json").write_text(_dump(observed) + "\n")
    return json.loads((destination / "policy_simulation_summary.json").read_text())


def dry_run_images(
    paths: list[Path],
    *,
    logits: dict[str, float] | None = None,
    batch_size: int = 1,
    test_paths: set[str] | None = None,
    checkpoint: Path | None = None,
    enforce_expected_sha: bool = True,
) -> list[dict]:
    """Read-only image dry run. A missing logit fails closed. Source bytes are not modified."""
    if checkpoint is not None and enforce_expected_sha and file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash does not match the Phase 21 checkpoint")
    engine = PolicyEngine(frozen_shadow_configuration())
    before = {str(path): Path(path).read_bytes() if Path(path).is_file() else None for path in paths}
    rows = engine.evaluate_paths(paths, logits=logits, batch_size=batch_size, test_paths=test_paths)
    for path in paths:
        if Path(path).is_file() and Path(path).read_bytes() != before[str(path)]:
            raise RuntimeError("policy dry run modified a source image")
    return rows


def _require_phase26_counts(summary: dict) -> None:
    observed = (
        summary["authentic_count"],
        summary["review_count"],
        summary["fake_count"],
        summary["false_authentic_count"],
        summary["false_fake_count"],
        summary["sample_count"],
    )
    expected = (
        PHASE26_AUTHENTIC_COUNT,
        PHASE26_REVIEW_COUNT,
        PHASE26_FAKE_COUNT,
        0,
        0,
        PHASE26_SAMPLE_COUNT,
    )
    if observed != expected or summary["review_rate"] != PHASE26_REVIEW_COUNT / PHASE26_SAMPLE_COUNT:
        raise RuntimeError(
            "Phase 27 shadow counts "
            f"{observed} do not reproduce Phase 26 {expected}; the threshold was not changed"
        )


def _require_phase26_stress(summary: dict) -> None:
    if summary["caught_by_quality_policy"] != PHASE26_STRESS_CAUGHT or summary["demonstrated_false_fake_events"] != PHASE26_STRESS_EVENTS:
        raise RuntimeError(
            "Phase 27 stress coverage "
            f"{summary['caught_by_quality_policy']}/{summary['demonstrated_false_fake_events']} "
            "does not reproduce Phase 26; the policy was not tuned"
        )
    if summary["missed_ids"] != [PHASE26_STRESS_MISS] or summary["false_authentic_count"] != 0:
        raise RuntimeError(f"Phase 27 stress miss changed: {summary['missed_ids']}")


def _assert_decision_flags(row: dict) -> None:
    flags = set(parse_flags(row["quality_flags"]))
    low_resolution = float(row["min_dimension"]) < LOW_RESOLUTION_CUTOFF
    low_contrast = float(row["luminance_std"]) < LOW_CONTRAST_CUTOFF
    if low_resolution != ("LOW_RESOLUTION" in flags) or low_contrast != ("LOW_CONTRAST" in flags):
        raise RuntimeError(f"stored quality flags disagree with the Phase 23 cutoffs for {row['sample_id']}")


def _backend_hashes(repo: Path) -> dict[str, str]:
    hashes = {}
    for relative in BACKEND_FILES:
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"protected backend file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def _assert_contract_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != CONTRACT_DIR_NAME:
        raise RuntimeError(f"policy contract output must be named {CONTRACT_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | set(SPLIT_NAMES) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        "final_test_v1",
        "final_model_freeze",
        "hard_negative_v1",
        "image_quality_v1",
        "image_quality_v2",
        "ood_v1",
        "policy_shadow_v1",
        "v2_dinov2_cls_only",
        "v2_dinov2_cls_patch_attention",
        "v2_dinov3_cls_only",
        "v2_dinov3_cls_patch_attention",
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the policy contract into {resolved}")
    return resolved


def _collect_images(images: list[str] | None, directory: Path | None) -> list[Path]:
    selected = [Path(path) for path in images] if images else []
    if directory is not None:
        selected.extend(path for path in sorted(Path(directory).iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES)
    if not selected:
        raise ValueError("dry run needs --image or --image-dir")
    return selected


def _logit_map(paths: list[Path], logits: list[str] | None) -> dict[str, float]:
    if not logits:
        return {}
    if len(logits) != len(paths):
        raise ValueError("each dry-run image needs its own finite logit; missing logits are not guessed")
    mapped = {}
    for path, logit in zip(paths, logits, strict=True):
        mapped[str(path.resolve())] = float(logit)
    return mapped


def _test_paths(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    manifest = json.loads(manifest_path.read_text())
    return {str(Path(path).resolve()) for path in (manifest.get("membership") or {}).get("test") or []}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shadow-mode authenticity policy contract")
    parser.add_argument("--mode", choices=("shadow",), required=True)
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--input-split", action="append", dest="input_splits", default=None)
    parser.add_argument("--threshold", type=float, default=AUTHENTICITY_THRESHOLD)
    parser.add_argument("--quality-candidate", default=QUALITY_RULE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image", action="append", default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--logit", action="append", default=None)
    args = parser.parse_args(argv)
    root = Path(args.experiment) if args.experiment is not None else Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    manifest = Path(args.split_manifest) if args.split_manifest is not None else root / "dataset_audit" / "split_manifest_v2.json"
    if args.image or args.image_dir is not None:
        paths = _collect_images(args.image, args.image_dir)
        checkpoint = Path(args.checkpoint) if args.checkpoint is not None else root / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        rows = dry_run_images(
            paths,
            logits=_logit_map(paths, args.logit),
            batch_size=args.batch_size,
            test_paths=_test_paths(manifest),
            checkpoint=checkpoint,
        )
        print(json.dumps(rows, allow_nan=False))
        return
    splits = tuple(args.input_splits) if args.input_splits else SOURCE_SPLITS
    summary = run_phase27(
        root,
        manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        input_splits=splits,
        threshold=args.threshold,
        quality_candidate=args.quality_candidate,
        mode=args.mode,
    )
    observed = summary["observed"]
    print(
        json.dumps(
            {
                "mode": "shadow",
                "policy_version": summary["policy_version"],
                "policy_status": summary["policy_status"],
                "authentic_count": observed["authentic_count"],
                "review_count": observed["review_count"],
                "fake_count": observed["fake_count"],
                "false_authentic_count": observed["false_authentic_count"],
                "false_fake_count": observed["false_fake_count"],
                "review_rate": observed["review_rate"],
                "stress_caught": summary["stress"]["caught_by_quality_policy"],
                "stress_events": summary["stress"]["demonstrated_false_fake_events"],
                "ood_status": summary["ood_status"],
                "final_test_samples_used": summary["final_test_samples_used"],
                "production_policy_status": summary["production_policy_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
