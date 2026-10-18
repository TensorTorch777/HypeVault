"""Freeze one checkpoint from validation and calibration, then score it once.

The final test split is not an input to model selection. Selection uses the
Phase 19 validation metrics and the cached Phase 20 calibration logits.
Temperature is not fit again. A production threshold is not chosen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from calibration import (
    DEFAULT_BINS,
    binary_nll,
    brier_score,
    expected_calibration_error,
    probabilities_from_temperature,
)
from calibration_phase20 import (
    MODELS,
    PREPROCESSING_VERSION,
    collect_calibration_logits,
    file_sha256,
)
from checkpoint_selection import validate_checkpoint_artifact
from evaluation import evaluate_logits, label_from_dataset_path
from split_v2 import (
    AUTHORITATIVE_COUNTS,
    AUTHORITATIVE_MEMBERSHIP_HASH,
    CATALOG_BRANDS,
    PHASE15_EXPERIMENT_DIRS,
    SPLIT_NAMES,
    membership_hash,
)
from training_config import assert_experiment_output_dir

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 42
CI_LOW_QUANTILE = 0.025
CI_HIGH_QUANTILE = 0.975
FAR_LIMIT = 0.02
EVALUATION_THRESHOLD = 0.50
FREEZE_DIR_NAME = "final_model_freeze"
FINAL_TEST_DIR_NAME = "final_test_v1"
SELECTION_RULE = (
    "Validation safety first: false-authentic rate <= 0.02. "
    "The corrected validation classification metrics are tied, so the "
    "predeclared secondary criterion is paired bootstrap of calibrated "
    "calibration-set NLL. A model is materially better only when its mean "
    "calibrated NLL is lower and the 95% percentile interval for "
    "NLL_A - NLL_B lies entirely below zero. Otherwise the models are "
    "calibration-tied. Ties then use lower calibrated ECE, lower calibrated "
    "Brier, cls_only before cls_patch_attention, lower measured peak "
    "allocated VRAM, higher measured training throughput, then the "
    "architecture name. Final-test metrics are not a selection input."
)
PREDICTION_FIELDS = (
    "sample_id",
    "brand",
    "label",
    "logit",
    "raw_probability",
    "calibrated_probability",
    "baseline_prediction",
    "calibrated_prediction",
)
ERROR_FIELDS = (
    "sample_id",
    "brand",
    "true_label",
    "prediction",
    "raw_probability",
    "calibrated_probability",
    "logit",
    "error_type",
)


def per_sample_nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> np.ndarray:
    """Float64 per-sample binary NLL. Does not modify logits or fit T."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    scaled = logits.detach().double().reshape(-1) / float(temperature)
    target = labels.detach().double().reshape(-1)
    if scaled.numel() != target.numel():
        raise ValueError("logits and labels differ in length")
    loss = F.binary_cross_entropy_with_logits(scaled, target, reduction="none")
    if not torch.isfinite(loss).all():
        raise RuntimeError("per-sample calibration NLL is not finite")
    return loss.cpu().numpy()


def paired_bootstrap_nll_difference(
    nll_a: np.ndarray,
    nll_b: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    indices: np.ndarray | None = None,
) -> dict:
    """Paired percentile interval for mean(NLL_A - NLL_B) on shared resamples."""
    left = np.asarray(nll_a, dtype=np.float64).reshape(-1)
    right = np.asarray(nll_b, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired NLL vectors must be non-empty and the same length")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired NLL vectors must be finite")
    count = int(resamples)
    if count < 1:
        raise ValueError("bootstrap resamples must be >= 1")
    if indices is None:
        indices = np.random.default_rng(int(seed)).integers(0, left.size, size=(count, left.size))
    else:
        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != left.size:
            raise ValueError("bootstrap indices must have shape (resamples, n)")
    delta = left[indices].mean(axis=1) - right[indices].mean(axis=1)
    observed = float(left.mean() - right.mean())
    ci_low = float(np.quantile(delta, CI_LOW_QUANTILE))
    ci_high = float(np.quantile(delta, CI_HIGH_QUANTILE))
    return {
        "observed_delta_nll": observed,
        "mean_nll_a": float(left.mean()),
        "mean_nll_b": float(right.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "resamples": int(indices.shape[0]),
        "seed": int(seed),
        "ci_method": "percentile",
        "quantiles": [CI_LOW_QUANTILE, CI_HIGH_QUANTILE],
    }


def materially_better(comparison: dict) -> bool:
    """True when A has lower mean NLL and the paired interval lies below zero."""
    return bool(comparison["observed_delta_nll"] < 0.0 and comparison["ci_high"] < 0.0)


def select_frozen_candidate(candidates: list[dict], comparisons: list[dict]) -> dict:
    """Apply the predeclared rule. Ignores any final-test fields."""
    safe = [row for row in candidates if float(row["validation_false_authentic_rate"]) <= FAR_LIMIT]
    if not safe:
        raise RuntimeError("no validation-safe candidate")
    by_name = {row["name"]: row for row in safe}
    beaten: set[str] = set()
    for comparison in comparisons:
        if comparison["model_a"] not in by_name or comparison["model_b"] not in by_name:
            continue
        if materially_better(comparison):
            beaten.add(comparison["model_b"])
        reverse = {
            "observed_delta_nll": -float(comparison["observed_delta_nll"]),
            "ci_low": -float(comparison["ci_high"]),
            "ci_high": -float(comparison["ci_low"]),
        }
        if materially_better(reverse):
            beaten.add(comparison["model_a"])
    survivors = [row for row in safe if row["name"] not in beaten]
    if not survivors:
        survivors = list(safe)
        cycle = True
    else:
        cycle = False
    ordered = sorted(survivors, key=_tie_break_key)
    winner = ordered[0]
    validation_values = {
        (row["validation_pr_auc"], row["validation_f1"], row["validation_false_authentic_rate"], row["validation_roc_auc"])
        for row in safe
    }
    return {
        "selected_name": winner["name"],
        "selected": winner,
        "validation_safe": [row["name"] for row in safe],
        "calibration_survivors": [row["name"] for row in survivors],
        "calibration_cycle": cycle,
        "validation_metrics_tied": len(validation_values) == 1,
        "tie_break_applied": len(survivors) > 1,
        "production_threshold": None,
        "test_used_for_selection": False,
    }


def _tie_break_key(row: dict) -> tuple:
    throughput = row.get("throughput_images_per_sec")
    peak = row.get("peak_allocated_bytes")
    return (
        float(row["calibrated_ece"]),
        float(row["calibrated_brier"]),
        0 if row["head"] == "cls_only" else 1,
        math.inf if peak is None else int(peak),
        math.inf if throughput is None else -float(throughput),
        str(row["architecture_name"]),
    )


def build_freeze_decision(experiments_root: Path) -> dict:
    """Read cached validation and calibration files. Does not open images."""
    root = Path(experiments_root)
    candidates, comparisons, shared_indices = _load_calibration_candidates(root)
    decision = select_frozen_candidate(candidates, comparisons)
    winner = decision["selected"]
    return {
        "phase": 21,
        "selection_rule": SELECTION_RULE,
        "validation_metrics_tied": decision["validation_metrics_tied"],
        "calibration_used_as_secondary_criterion": True,
        "paired_calibration_bootstrap": True,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_indices_shared_across_pairs": True,
        "final_test_opened_at_selection": False,
        "test_images_loaded": False,
        "test_used_for_selection": False,
        "threshold_tuned_on_test": False,
        "temperature_fitted_on_test": False,
        "temperature_refit": False,
        "ood_used": False,
        "training_performed": False,
        "production_threshold": None,
        "architecture_ranking_source": "validation_safety_then_paired_calibration_nll",
        "phase15_used_for_ranking": False,
        "selected_model": winner["name"],
        "selected_family": winner["family"],
        "selected_head": winner["head"],
        "selected_checkpoint": winner["checkpoint"],
        "selected_temperature": winner["temperature"],
        "selected_calibrated_nll": winner["calibrated_nll"],
        "selected_calibrated_ece": winner["calibrated_ece"],
        "selected_calibrated_brier": winner["calibrated_brier"],
        "validation_safe": decision["validation_safe"],
        "calibration_survivors": decision["calibration_survivors"],
        "calibration_cycle": decision["calibration_cycle"],
        "tie_break_applied": decision["tie_break_applied"],
        "candidates": [_public_candidate(row) for row in candidates],
        "comparisons": comparisons,
        "shared_resample_count": int(shared_indices),
        "checkpoint_sha256": file_sha256(winner["checkpoint"]),
        "manifest_membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
    }


def write_model_freeze(experiments_root: Path, destination_dir: Path | None = None) -> Path:
    """Write the freeze record before any final-test image is opened."""
    root = Path(experiments_root)
    destination = _assert_freeze_destination(
        Path(destination_dir) if destination_dir is not None else root / FREEZE_DIR_NAME
    )
    if (destination / "model_selection_freeze.json").is_file():
        return destination
    decision = build_freeze_decision(root)
    return emit_freeze_files(destination, decision)


def emit_freeze_files(destination: Path, decision: dict) -> Path:
    """Write an already computed freeze decision. Does not read the test split."""
    destination = _assert_freeze_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_selection_freeze.json").write_text(_dump(decision) + "\n")
    _write_freeze_csv(destination / "model_selection_freeze.csv", decision["candidates"], decision["selected_model"])
    _write_bootstrap_csv(destination / "calibration_bootstrap.csv", decision["comparisons"])
    (destination / "model_selection_freeze.md").write_text(_freeze_markdown(decision))
    return destination


def validate_final_test_membership(manifest: dict) -> list[dict]:
    """Check the locked test membership. Does not open images."""
    failures: list[str] = []
    samples = manifest.get("samples")
    membership = manifest.get("membership") if isinstance(manifest.get("membership"), dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("split manifest has no samples")
    if membership is None or any(name not in membership for name in SPLIT_NAMES):
        raise ValueError("split manifest is missing a required membership list")
    recorded = {name: [str(path) for path in list(membership.get(name) or [])] for name in SPLIT_NAMES}
    digest = membership_hash(recorded)
    if digest != manifest.get("membership_hash") or digest != AUTHORITATIVE_MEMBERSHIP_HASH:
        failures.append("membership hash does not match the Phase 16 manifest")
    by_path = {}
    for row in samples:
        path = row.get("path")
        if isinstance(path, str) and path:
            by_path[str(Path(path).resolve())] = row
    resolved = {name: [str(Path(path).resolve()) for path in recorded[name]] for name in SPLIT_NAMES}
    test_paths = resolved["test"]
    test_set = set(test_paths)
    if len(test_set) != len(test_paths):
        failures.append("test membership contains duplicate paths")
    for other in ("train", "calibration", "validation"):
        overlap = test_set & set(resolved[other])
        if overlap:
            failures.append(f"test overlaps {other} ({len(overlap)} paths)")
    records = []
    authentic = 0
    fake = 0
    brand_class: dict[tuple[str, int], int] = {}
    groups: dict[str, set[str]] = {}
    for path in test_paths:
        row = by_path.get(path)
        if row is None or row.get("split") != "test":
            failures.append("test path is missing from samples")
            continue
        label = row.get("label")
        try:
            path_label = label_from_dataset_path(Path(path))
        except ValueError:
            path_label = None
        if label not in (0, 1) or path_label != label:
            failures.append("test label does not match the image path")
            continue
        brand = row.get("brand")
        group_id = row.get("duplicate_group_id")
        sample_id = row.get("sample_id")
        if not isinstance(brand, str) or not isinstance(group_id, str) or not isinstance(sample_id, str):
            failures.append("test sample is missing brand, sample_id, or duplicate_group_id")
            continue
        records.append(
            {
                "sample_id": sample_id,
                "path": path,
                "label": int(label),
                "brand": brand,
                "duplicate_group_id": group_id,
            }
        )
        authentic += int(label == 0)
        fake += int(label == 1)
        brand_class[(brand, int(label))] = brand_class.get((brand, int(label)), 0) + 1
        groups.setdefault(group_id, set()).add("test")
    expected = AUTHORITATIVE_COUNTS["test"]
    if (len(records), authentic, fake) != (expected["samples"], expected["authentic"], expected["fake"]):
        failures.append(
            f"test counts are {len(records)}/{authentic}/{fake}, "
            f"expected {expected['samples']}/{expected['authentic']}/{expected['fake']}"
        )
    for brand in CATALOG_BRANDS:
        if brand_class.get((brand, 0), 0) == 0 or brand_class.get((brand, 1), 0) == 0:
            failures.append(f"test is missing a class for {brand}")
    for row in samples:
        group_id = row.get("duplicate_group_id")
        split = row.get("split")
        if isinstance(group_id, str) and split in SPLIT_NAMES:
            groups.setdefault(group_id, set()).add(split)
    crossing = [group_id for group_id, owners in groups.items() if len(owners) > 1]
    if crossing:
        failures.append(f"{len(crossing)} duplicate groups cross splits")
    if failures:
        raise ValueError("; ".join(failures))
    return records


def score_holdout(
    records: list[dict],
    logits: torch.Tensor,
    temperature: float,
) -> dict:
    """Score stored logits at the frozen temperature. Does not fit T or a threshold."""
    labels = torch.tensor([int(row["label"]) for row in records], dtype=torch.float32)
    vector = logits.detach().float().reshape(-1).cpu()
    if vector.numel() != len(records):
        raise ValueError("logits and test records differ in length")
    frozen = float(temperature)
    if not math.isfinite(frozen) or frozen <= 0.0:
        raise ValueError(f"frozen temperature must be finite and > 0, got {temperature}")
    raw = probabilities_from_temperature(vector, 1.0)
    calibrated = probabilities_from_temperature(vector, frozen)
    baseline_eval = evaluate_logits(vector, labels, threshold=EVALUATION_THRESHOLD)
    calibrated_eval = evaluate_logits(vector / frozen, labels, threshold=EVALUATION_THRESHOLD)
    baseline_probability = raw
    calibrated_probability = calibrated
    baseline_ece, baseline_bins = expected_calibration_error(baseline_probability, labels, DEFAULT_BINS)
    calibrated_ece, calibrated_bins = expected_calibration_error(calibrated_probability, labels, DEFAULT_BINS)
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "sample_id": record["sample_id"],
                "brand": record["brand"],
                "label": int(record["label"]),
                "logit": float(vector[index]),
                "raw_probability": float(raw[index]),
                "calibrated_probability": float(calibrated[index]),
                "baseline_prediction": int(float(raw[index]) >= EVALUATION_THRESHOLD),
                "calibrated_prediction": int(float(calibrated[index]) >= EVALUATION_THRESHOLD),
            }
        )
    return {
        "sample_count": len(records),
        "authentic_count": sum(row["label"] == 0 for row in records),
        "fake_count": sum(row["label"] == 1 for row in records),
        "temperature": frozen,
        "temperature_refit": False,
        "evaluation_threshold": EVALUATION_THRESHOLD,
        "evaluation_threshold_role": "fixed_evaluation_threshold",
        "production_threshold": None,
        "baseline": _metric_block(vector, labels, 1.0, baseline_eval, baseline_ece, baseline_bins),
        "calibrated": _metric_block(vector, labels, frozen, calibrated_eval, calibrated_ece, calibrated_bins),
        "predictions": rows,
        "brands": _brand_report(rows, frozen),
        "error_audit": _error_audit(rows),
        "severe_false_authentic": severe_false_authentic_counts(
            np.asarray([row["label"] for row in rows]),
            np.asarray([row["calibrated_probability"] for row in rows], dtype=np.float64),
        ),
        "test_used_for_selection": False,
        "threshold_tuned_on_test": False,
        "temperature_fitted_on_test": False,
        "ood_used": False,
        "training_performed": False,
    }


def severe_false_authentic_counts(labels: np.ndarray, calibrated_probability: np.ndarray) -> dict:
    """Diagnostic counts. These cutoffs are not production thresholds."""
    fake = np.asarray(labels).reshape(-1) == 1
    probability = np.asarray(calibrated_probability, dtype=np.float64).reshape(-1)
    if fake.shape != probability.shape:
        raise ValueError("labels and calibrated probabilities differ in length")
    return {
        "actual_fake_and_calibrated_p_fake_below_0_10": int(np.sum(fake & (probability < 0.10))),
        "actual_fake_and_calibrated_p_fake_below_0_25": int(np.sum(fake & (probability < 0.25))),
        "actual_fake_and_calibrated_p_fake_below_0_50": int(np.sum(fake & (probability < 0.50))),
        "role": "diagnostic_severity_only",
        "production_threshold": None,
    }


def write_final_test_artifacts(
    destination_dir: Path,
    *,
    score: dict,
    freeze: dict,
    freeze_sha256: str,
    test_membership_sha256: str,
    checkpoint_sha256: str,
) -> Path:
    """Write the one-time test report. Refuses to overwrite a completed test."""
    destination = _assert_final_destination(destination_dir)
    summary_path = destination / "final_test_summary.json"
    if summary_path.is_file():
        raise RuntimeError(f"final test already completed at {destination}; refusing to rerun")
    destination.mkdir(parents=True, exist_ok=True)
    _write_prediction_csv(destination / "final_test_predictions.csv", score["predictions"])
    prediction_sha256 = file_sha256(destination / "final_test_predictions.csv")
    (destination / "final_test_metrics.json").write_text(
        _dump(
            {
                "baseline": score["baseline"],
                "calibrated": score["calibrated"],
                "evaluation_threshold": EVALUATION_THRESHOLD,
                "production_threshold": None,
                "temperature": score["temperature"],
                "temperature_refit": False,
            }
        )
        + "\n"
    )
    _write_brand_csv(destination / "final_test_brand_metrics.csv", score["brands"])
    (destination / "final_test_brand_metrics.json").write_text(_dump(score["brands"]) + "\n")
    (destination / "final_test_confusion_matrix.json").write_text(
        _dump(
            {
                "evaluation_threshold": EVALUATION_THRESHOLD,
                "production_threshold": None,
                "baseline": score["baseline"]["confusion_matrix"],
                "calibrated": score["calibrated"]["confusion_matrix"],
            }
        )
        + "\n"
    )
    _write_error_csv(destination / "final_test_error_audit.csv", score["error_audit"])
    reference = {
        "freeze_directory": FREEZE_DIR_NAME,
        "selected_model": freeze["selected_model"],
        "selected_checkpoint": freeze["selected_checkpoint"],
        "selected_temperature": freeze["selected_temperature"],
        "freeze_sha256": freeze_sha256,
        "production_threshold": None,
        "model_selection_completed_before_test": True,
        "test_used_for_selection": False,
    }
    (destination / "model_freeze_reference.json").write_text(_dump(reference) + "\n")
    summary = {
        "model_selection_completed_before_test": True,
        "test_used_for_selection": False,
        "threshold_tuned_on_test": False,
        "temperature_fitted_on_test": False,
        "checkpoint_changed_after_test": False,
        "production_threshold": None,
        "ood_used": False,
        "training_performed": False,
        "selected_model": freeze["selected_model"],
        "selected_checkpoint": freeze["selected_checkpoint"],
        "selected_temperature": freeze["selected_temperature"],
        "freeze_sha256": freeze_sha256,
        "test_membership_sha256": test_membership_sha256,
        "prediction_sha256": prediction_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_count": score["sample_count"],
        "authentic_count": score["authentic_count"],
        "fake_count": score["fake_count"],
        "baseline": _summary_metrics(score["baseline"]),
        "calibrated": _summary_metrics(score["calibrated"]),
        "brands": score["brands"],
        "severe_false_authentic": score["severe_false_authentic"],
        "error_count": len(score["error_audit"]),
    }
    summary_path.write_text(_dump(summary) + "\n")
    return destination


def run_locked_final_test(
    experiments_root: Path,
    manifest_path: Path,
    freeze_dir: Path,
    destination_dir: Path | None = None,
    *,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str | None = None,
) -> dict:
    """Score only the frozen checkpoint. Raises if the freeze or a prior test is missing or present."""
    root = Path(experiments_root)
    freeze_path = Path(freeze_dir) / "model_selection_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("model selection freeze is missing; refusing to open the final test")
    freeze_digest = file_sha256(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("final_test_opened_at_selection") is not False:
        raise RuntimeError("freeze record does not show that selection preceded the test")
    if freeze.get("production_threshold") is not None:
        raise RuntimeError("freeze record contains a production threshold")
    destination = _assert_final_destination(
        Path(destination_dir) if destination_dir is not None else root / FINAL_TEST_DIR_NAME
    )
    if (destination / "final_test_summary.json").is_file():
        raise RuntimeError("final test already completed; refusing to rerun")
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text())
    records = validate_final_test_membership(manifest)
    test_membership_sha256 = _path_list_sha256([row["path"] for row in records])
    spec = next(row for row in MODELS if row["directory"] == freeze["selected_model"])
    checkpoint = Path(freeze["selected_checkpoint"])
    if checkpoint.name != spec["checkpoint"] or spec["family"] != freeze["selected_family"]:
        raise RuntimeError("frozen checkpoint does not match the Phase 19 candidate table")
    before_checkpoint = file_sha256(checkpoint)
    if before_checkpoint != freeze["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint bytes changed after selection")
    preflight = validate_checkpoint_artifact(
        checkpoint,
        expected_family=spec["family"],
        expected_classifier_arch=spec["head"],
        expected_epoch=int(spec["epoch"]),
        expected_input_size=int(spec["input_size"]),
        expected_membership_hash=AUTHORITATIVE_MEMBERSHIP_HASH,
        expected_preprocessing_version=PREPROCESSING_VERSION,
        expected_experiment_dir=root / spec["directory"],
    )
    if not preflight["checkpoint_load_valid"]:
        raise RuntimeError("selected checkpoint failed preflight: " + "; ".join(preflight["reasons"]))
    temperature = float(freeze["selected_temperature"])
    logits = collect_calibration_logits(
        records,
        checkpoint,
        family=spec["family"],
        head=spec["head"],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    score = score_holdout(records, logits, temperature)
    after_checkpoint = file_sha256(checkpoint)
    if after_checkpoint != before_checkpoint or file_sha256(freeze_path) != freeze_digest:
        raise RuntimeError("final test changed the checkpoint or the selection freeze")
    write_final_test_artifacts(
        destination,
        score=score,
        freeze=freeze,
        freeze_sha256=freeze_digest,
        test_membership_sha256=test_membership_sha256,
        checkpoint_sha256=after_checkpoint,
    )
    return score


def _load_calibration_candidates(root: Path) -> tuple[list[dict], list[dict], int]:
    policy = json.loads((root / "checkpoint_policy_v3" / "checkpoint_policy_v3.json").read_text())
    policy_rows = {row["directory"]: row for row in policy["experiments"]}
    loaded = []
    aligned_ids: list[str] | None = None
    aligned_labels: np.ndarray | None = None
    for spec in MODELS:
        directory = root / spec["directory"]
        policy_row = policy_rows[spec["directory"]]
        temperature_payload = json.loads((directory / "calibration" / "temperature.json").read_text())
        metrics = json.loads((directory / "calibration" / "calibration_metrics.json").read_text())["calibrated"]
        resource = json.loads((directory / "resource.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        selection = json.loads((directory / "selection" / "checkpoint_selection.json").read_text())
        epoch = int(policy_row["artifact_selected_epoch"])
        validation = next(row for row in selection["candidates"] if int(row["epoch"]) == epoch)
        throughput = next(
            (float(row["throughput_images_per_sec"]) for row in history["epochs"] if int(row["epoch"]) == epoch),
            None,
        )
        sample_ids, labels, logits = _read_calibration_csv(directory / "calibration" / "calibration_predictions.csv")
        if aligned_ids is None:
            aligned_ids = sample_ids
            aligned_labels = labels
        elif sample_ids != aligned_ids or not np.array_equal(labels, aligned_labels):
            raise RuntimeError(f"{spec['directory']} calibration rows are not paired with the other models")
        nll = per_sample_nll(torch.tensor(logits), torch.tensor(labels), float(temperature_payload["temperature"]))
        loaded.append(
            {
                "name": spec["directory"],
                "family": spec["family"],
                "head": spec["head"],
                "architecture_name": f"{spec['family']}:{spec['head']}",
                "checkpoint": str(directory / spec["checkpoint"]),
                "epoch": epoch,
                "temperature": float(temperature_payload["temperature"]),
                "validation_false_authentic_rate": float(policy_row["false_authentic_rate"]),
                "validation_pr_auc": float(policy_row["pr_auc"]),
                "validation_f1": float(policy_row["f1"]),
                "validation_roc_auc": float(validation["roc_auc"]),
                "calibrated_nll": float(nll.mean()),
                "stored_calibrated_nll": float(metrics["nll"]),
                "calibrated_ece": float(metrics["ece"]),
                "calibrated_brier": float(metrics["brier"]),
                "peak_allocated_bytes": resource.get("peak_allocated_bytes"),
                "throughput_images_per_sec": throughput,
                "per_sample_nll": nll,
            }
        )
    assert aligned_ids is not None
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(aligned_ids), size=(BOOTSTRAP_RESAMPLES, len(aligned_ids))
    )
    comparisons = []
    names = [row["name"] for row in loaded]
    nll_by_name = {row["name"]: row["per_sample_nll"] for row in loaded}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            comparison = paired_bootstrap_nll_difference(
                nll_by_name[left_name],
                nll_by_name[right_name],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
                indices=indices,
            )
            comparison["model_a"] = left_name
            comparison["model_b"] = right_name
            comparison["materially_better_a"] = materially_better(comparison)
            reverse = {
                "observed_delta_nll": -comparison["observed_delta_nll"],
                "ci_low": -comparison["ci_high"],
                "ci_high": -comparison["ci_low"],
            }
            comparison["materially_better_b"] = materially_better(reverse)
            comparisons.append(comparison)
    for row in loaded:
        row.pop("per_sample_nll")
    return loaded, comparisons, int(indices.shape[0])


def _read_calibration_csv(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids = []
    labels = []
    logits = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample_ids.append(row["sample_id"])
            labels.append(int(row["label"]))
            logits.append(float(row["logit"]))
    order = sorted(range(len(sample_ids)), key=lambda index: sample_ids[index])
    return (
        [sample_ids[index] for index in order],
        np.asarray([labels[index] for index in order], dtype=np.float64),
        np.asarray([logits[index] for index in order], dtype=np.float64),
    )


def _public_candidate(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "per_sample_nll"}


def _metric_block(logits, labels, temperature, evaluation: dict, ece: float, bins: list[dict]) -> dict:
    return {
        "nll": binary_nll(logits, labels, temperature),
        "brier": brier_score(probabilities_from_temperature(logits, temperature), labels),
        "ece": ece,
        "ece_bins": bins,
        "roc_auc": evaluation["roc_auc"],
        "pr_auc": evaluation["pr_auc"],
        "sample_count": evaluation["sample_count"],
        "authentic_count": evaluation["authentic_count"],
        "fake_count": evaluation["fake_count"],
        "threshold_0_50": {
            "threshold": evaluation["threshold"],
            "threshold_role": "fixed_evaluation_threshold",
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "false_authentic_count": evaluation["false_authentic"],
            "false_authentic_rate": evaluation["false_authentic_rate"],
            "false_fake_count": evaluation["false_fake"],
            "false_fake_rate": evaluation["false_fake_rate"],
        },
        "confusion_matrix": evaluation["confusion_matrix"],
    }


def _summary_metrics(block: dict) -> dict:
    operating = block["threshold_0_50"]
    return {
        "roc_auc": block["roc_auc"],
        "pr_auc": block["pr_auc"],
        "nll": block["nll"],
        "brier": block["brier"],
        "ece": block["ece"],
        "accuracy": operating["accuracy"],
        "precision": operating["precision"],
        "recall": operating["recall"],
        "f1": operating["f1"],
        "false_authentic_count": operating["false_authentic_count"],
        "false_authentic_rate": operating["false_authentic_rate"],
        "false_fake_count": operating["false_fake_count"],
        "false_fake_rate": operating["false_fake_rate"],
    }


def _brand_report(rows: list[dict], temperature: float) -> dict:
    brands = []
    for brand in CATALOG_BRANDS:
        subset = [row for row in rows if row["brand"] == brand]
        if not subset:
            continue
        brands.append({"brand": brand, **_brand_metrics(subset, temperature)})
    rates = [float(row["calibrated_false_authentic_rate"]) for row in brands if row["calibrated_false_authentic_rate"] is not None]
    worst = None if not rates else max(rates)
    worst_brands = [
        row["brand"]
        for row in brands
        if row["calibrated_false_authentic_rate"] is not None and float(row["calibrated_false_authentic_rate"]) == worst
    ]
    return {
        "brands": brands,
        "macro_brand_far": None if not rates else sum(rates) / len(rates),
        "macro_brand_f1": _mean(brands, "calibrated_f1"),
        "macro_brand_recall": _mean(brands, "calibrated_recall"),
        "worst_brand_far": worst,
        "worst_brands_by_far": worst_brands,
        "far_spread": None if not rates else max(rates) - min(rates),
        "highest_false_authentic_count_brands": _extreme_brands(brands, "calibrated_false_authentic_count", high=True),
        "lowest_recall_brands": _extreme_brands(brands, "calibrated_recall", high=False),
        "production_threshold": None,
        "tuned_from_brand_results": False,
    }


def _brand_metrics(rows: list[dict], temperature: float) -> dict:
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    logits = torch.tensor([row["logit"] for row in rows], dtype=torch.float32)
    raw = evaluate_logits(logits, labels, threshold=EVALUATION_THRESHOLD)
    calibrated = evaluate_logits(logits / float(temperature), labels, threshold=EVALUATION_THRESHOLD)
    return {
        "sample_count": len(rows),
        "authentic_count": int((labels == 0).sum()),
        "fake_count": int((labels == 1).sum()),
        "baseline_pr_auc": raw["pr_auc"],
        "baseline_f1": raw["f1"],
        "baseline_recall": raw["recall"],
        "baseline_false_authentic_count": raw["false_authentic"],
        "baseline_false_authentic_rate": raw["false_authentic_rate"],
        "baseline_false_fake_count": raw["false_fake"],
        "baseline_false_fake_rate": raw["false_fake_rate"],
        "calibrated_pr_auc": calibrated["pr_auc"],
        "calibrated_f1": calibrated["f1"],
        "calibrated_recall": calibrated["recall"],
        "calibrated_false_authentic_count": calibrated["false_authentic"],
        "calibrated_false_authentic_rate": calibrated["false_authentic_rate"],
        "calibrated_false_fake_count": calibrated["false_fake"],
        "calibrated_false_fake_rate": calibrated["false_fake_rate"],
    }


def _error_audit(rows: list[dict]) -> list[dict]:
    audit = []
    for row in rows:
        label = int(row["label"])
        calibrated_prediction = int(row["calibrated_prediction"])
        baseline_prediction = int(row["baseline_prediction"])
        calibrated_wrong = calibrated_prediction != label
        baseline_wrong = baseline_prediction != label
        if not calibrated_wrong and not baseline_wrong:
            continue
        if calibrated_wrong:
            error_type = "false_authentic" if label == 1 else "false_fake"
            prediction = calibrated_prediction
        else:
            error_type = "baseline_only_false_authentic" if label == 1 else "baseline_only_false_fake"
            prediction = baseline_prediction
        audit.append(
            {
                "sample_id": row["sample_id"],
                "brand": row["brand"],
                "true_label": label,
                "prediction": prediction,
                "raw_probability": row["raw_probability"],
                "calibrated_probability": row["calibrated_probability"],
                "logit": row["logit"],
                "error_type": error_type,
            }
        )
    order = {
        "false_authentic": 0,
        "false_fake": 1,
        "baseline_only_false_authentic": 2,
        "baseline_only_false_fake": 3,
    }
    audit.sort(
        key=lambda row: (
            order[row["error_type"]],
            row["calibrated_probability"] if "authentic" in row["error_type"] else -row["calibrated_probability"],
            row["sample_id"],
        )
    )
    return audit


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _extreme_brands(rows: list[dict], key: str, *, high: bool) -> list[str]:
    usable = [row for row in rows if row[key] is not None]
    if not usable:
        return []
    target = max(float(row[key]) for row in usable) if high else min(float(row[key]) for row in usable)
    return [row["brand"] for row in usable if float(row[key]) == target]


def _freeze_markdown(decision: dict) -> str:
    winner = decision["selected_model"]
    lines = [
        "# Final model freeze",
        "",
        "Validation metrics were tied.",
        "Calibration was used only as the predeclared secondary criterion.",
        "Paired calibration bootstrap was used.",
        "The final test had not yet been opened at selection time.",
        "",
        f"Selected model: {winner}",
        f"Selected checkpoint: {decision['selected_checkpoint']}",
        f"Selected temperature: {decision['selected_temperature']}",
        "Production threshold: null",
        "",
        SELECTION_RULE,
        "",
        f"Calibration survivors: {', '.join(decision['calibration_survivors'])}",
        f"Tie-break applied: {decision['tie_break_applied']}",
        "",
    ]
    return "\n".join(lines)


def _write_freeze_csv(path: Path, candidates: list[dict], selected: str) -> None:
    fields = [
        "name",
        "family",
        "head",
        "checkpoint",
        "temperature",
        "validation_false_authentic_rate",
        "validation_pr_auc",
        "validation_f1",
        "validation_roc_auc",
        "calibrated_nll",
        "calibrated_ece",
        "calibrated_brier",
        "peak_allocated_bytes",
        "throughput_images_per_sec",
        "selected",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in candidates:
            writer.writerow({**row, "selected": row["name"] == selected})


def _write_bootstrap_csv(path: Path, comparisons: list[dict]) -> None:
    fields = [
        "model_a",
        "model_b",
        "mean_nll_a",
        "mean_nll_b",
        "observed_delta_nll",
        "ci_low",
        "ci_high",
        "excludes_zero",
        "materially_better_a",
        "materially_better_b",
        "resamples",
        "seed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in comparisons:
            writer.writerow(row)


def _write_prediction_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in PREDICTION_FIELDS if key not in {"logit", "raw_probability", "calibrated_probability"}},
                    "logit": repr(float(row["logit"])),
                    "raw_probability": repr(float(row["raw_probability"])),
                    "calibrated_probability": repr(float(row["calibrated_probability"])),
                }
            )


def _write_brand_csv(path: Path, report: dict) -> None:
    rows = report["brands"]
    fields = list(rows[0].keys()) if rows else ["brand"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_error_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ERROR_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "raw_probability": repr(float(row["raw_probability"])),
                    "calibrated_probability": repr(float(row["calibrated_probability"])),
                    "logit": repr(float(row["logit"])),
                }
            )


def _path_list_sha256(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_freeze_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != FREEZE_DIR_NAME:
        raise RuntimeError(f"model freeze output must be named {FREEZE_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {spec["directory"] for spec in MODELS}
    blocked.update({"checkpoints", "checkpoints_watches", "dataset_audit", FINAL_TEST_DIR_NAME})
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the model freeze into {resolved}")
    return resolved


def _assert_final_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != FINAL_TEST_DIR_NAME:
        raise RuntimeError(f"final test output must be named {FINAL_TEST_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {spec["directory"] for spec in MODELS}
    blocked.update({"checkpoints", "checkpoints_watches", "dataset_audit", FREEZE_DIR_NAME})
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the final test into {resolved}")
    return resolved


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    manifest = root / "dataset_audit" / "split_manifest_v2.json"
    freeze_dir = write_model_freeze(root)
    print(f"freeze written: {freeze_dir}", flush=True)
    score = run_locked_final_test(root, manifest, freeze_dir)
    calibrated = score["calibrated"]["threshold_0_50"]
    print(
        json.dumps(
            {
                "selected_temperature": score["temperature"],
                "sample_count": score["sample_count"],
                "pr_auc": score["calibrated"]["pr_auc"],
                "roc_auc": score["calibrated"]["roc_auc"],
                "f1": calibrated["f1"],
                "false_authentic_rate": calibrated["false_authentic_rate"],
                "production_threshold": None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
