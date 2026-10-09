"""Fit one global temperature on the frozen Phase 16 calibration split.

Checkpoint selection is already finished. This module does not train, does not
rerun selection, does not open the final test split, and does not choose a
production threshold. A lower NLL, Brier score, or ECE is a change in
probability calibration, not a claim that authenticity classification improved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from augmentations import CONSERVATIVE, build_transforms
from calibration import (
    DEFAULT_BINS,
    DEFAULT_FAR_LIMIT,
    THRESHOLD_GRID,
    binary_nll,
    brand_calibration_summaries,
    calibration_threshold_candidates,
    compare_calibration,
    fit_temperature,
    probabilities_from_temperature,
    scaled_logits,
    score_temperature,
)
from checkpoint_selection import validate_checkpoint_artifact
from evaluation import build_eval_model, evaluate_logits, label_from_dataset_path
from model import DINOV2_IMAGE_SIZE, DINOV3_IMAGE_SIZE
from split_v2 import (
    AUTHORITATIVE_COUNTS,
    AUTHORITATIVE_MEMBERSHIP_HASH,
    CATALOG_BRANDS,
    PHASE15_EXPERIMENT_DIRS,
    SPLIT_NAMES,
    membership_hash,
)
from training_config import assert_experiment_output_dir

FITTING_METHOD = "lbfgs_log_temperature_nll"
PREPROCESSING_VERSION = "resize_pad_square_eval_v1"
INITIAL_TEMPERATURE = 1.0
TEMPERATURE_REPEAT_TOLERANCE = 1e-6
COMPARISON_DIR_NAME = "calibration_comparison_v1"
INFERENCE_BATCH_SIZE = 16
INFERENCE_WORKERS = 4
MODELS = (
    {
        "directory": "v2_dinov2_cls_only",
        "checkpoint": "epoch_008.pt",
        "family": "dinov2",
        "head": "cls_only",
        "epoch": 8,
        "input_size": DINOV2_IMAGE_SIZE,
    },
    {
        "directory": "v2_dinov2_cls_patch_attention",
        "checkpoint": "epoch_016.pt",
        "family": "dinov2",
        "head": "cls_patch_attention",
        "epoch": 16,
        "input_size": DINOV2_IMAGE_SIZE,
    },
    {
        "directory": "v2_dinov3_cls_only",
        "checkpoint": "epoch_030.pt",
        "family": "dinov3",
        "head": "cls_only",
        "epoch": 30,
        "input_size": DINOV3_IMAGE_SIZE,
    },
    {
        "directory": "v2_dinov3_cls_patch_attention",
        "checkpoint": "epoch_018.pt",
        "family": "dinov3",
        "head": "cls_patch_attention",
        "epoch": 18,
        "input_size": DINOV3_IMAGE_SIZE,
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_phase16_calibration_manifest(manifest: dict) -> list[dict]:
    """Check the frozen calibration membership. Does not open images or carve a split."""
    failures: list[str] = []
    samples = manifest.get("samples")
    membership = manifest.get("membership") if isinstance(manifest.get("membership"), dict) else None
    if not isinstance(samples, list) or not samples:
        failures.append("manifest has no samples")
        samples = []
    if membership is None or any(name not in membership for name in SPLIT_NAMES):
        raise ValueError("split manifest is missing train, calibration, validation, or test membership")
    recorded = {
        name: [str(path) for path in list(membership.get(name) or [])] for name in SPLIT_NAMES
    }
    digest = membership_hash(recorded)
    if digest != manifest.get("membership_hash"):
        failures.append("stored membership hash does not match the membership lists")
    if digest != AUTHORITATIVE_MEMBERSHIP_HASH:
        failures.append("membership hash does not match the Phase 16 manifest")

    by_path: dict[str, dict] = {}
    for row in samples:
        path = row.get("path")
        if not isinstance(path, str) or not path:
            failures.append("sample is missing a path")
            continue
        by_path[str(Path(path).resolve())] = row
    resolved = {name: [str(Path(path).resolve()) for path in recorded[name]] for name in SPLIT_NAMES}
    calibration_paths = resolved["calibration"]
    calibration_set = set(calibration_paths)
    if len(calibration_set) != len(calibration_paths):
        failures.append("calibration membership contains duplicate paths")
    for other in ("train", "validation", "test"):
        overlap = calibration_set & set(resolved[other])
        if overlap:
            failures.append(f"calibration overlaps {other} ({len(overlap)} paths)")

    records = []
    authentic = 0
    fake = 0
    brand_class: dict[tuple[str, int], int] = {}
    groups: dict[str, set[str]] = {}
    for path in calibration_paths:
        row = by_path.get(path)
        if row is None:
            failures.append("calibration path is missing from samples")
            continue
        if row.get("split") != "calibration":
            failures.append("calibration membership path is not labeled calibration")
        label = row.get("label")
        try:
            path_label = label_from_dataset_path(Path(path))
        except ValueError:
            path_label = None
        if label not in (0, 1) or path_label != label:
            failures.append("calibration label does not match the image path")
            continue
        brand = row.get("brand")
        group_id = row.get("duplicate_group_id")
        sample_id = row.get("sample_id")
        if not isinstance(brand, str) or not brand:
            failures.append("calibration sample is missing a brand")
            continue
        if not isinstance(group_id, str) or not group_id:
            failures.append("calibration sample is missing duplicate_group_id")
            continue
        if not isinstance(sample_id, str) or not sample_id:
            failures.append("calibration sample is missing sample_id")
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
        groups.setdefault(group_id, set()).add("calibration")

    expected = AUTHORITATIVE_COUNTS["calibration"]
    if (len(records), authentic, fake) != (expected["samples"], expected["authentic"], expected["fake"]):
        failures.append(
            f"calibration counts are {len(records)}/{authentic}/{fake}, "
            f"expected {expected['samples']}/{expected['authentic']}/{expected['fake']}"
        )
    for brand in CATALOG_BRANDS:
        if brand_class.get((brand, 0), 0) == 0 or brand_class.get((brand, 1), 0) == 0:
            failures.append(f"calibration is missing a class for {brand}")
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


def preflight_checkpoint(spec: dict, experiment_dir: Path) -> dict:
    """Load one checkpoint and check identity. Does not run the dataset."""
    checkpoint = Path(experiment_dir) / spec["checkpoint"]
    return validate_checkpoint_artifact(
        checkpoint,
        expected_family=spec["family"],
        expected_classifier_arch=spec["head"],
        expected_epoch=int(spec["epoch"]),
        expected_input_size=int(spec["input_size"]),
        expected_membership_hash=AUTHORITATIVE_MEMBERSHIP_HASH,
        expected_preprocessing_version=PREPROCESSING_VERSION,
        expected_experiment_dir=Path(experiment_dir),
    )


def choose_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Fit one global T. Keep T = 1 when the fit does not strictly improve NLL."""
    unchanged = logits.detach().float().reshape(-1).cpu().clone()
    fit = fit_temperature(unchanged, labels)
    if not torch.equal(unchanged, logits.detach().float().reshape(-1).cpu()):
        raise RuntimeError("temperature fitting modified the stored logits")
    optimized = float(fit["optimizer_temperature"])
    if not math.isfinite(optimized) or optimized <= 0.0:
        raise RuntimeError(f"optimized temperature is not finite and positive: {optimized}")
    baseline_nll = binary_nll(unchanged, labels, INITIAL_TEMPERATURE)
    optimized_nll = binary_nll(unchanged, labels, optimized)
    if not math.isfinite(baseline_nll) or not math.isfinite(optimized_nll):
        raise RuntimeError("calibration NLL is not finite")
    improved = optimized_nll < baseline_nll
    temperature = optimized if improved else INITIAL_TEMPERATURE
    calibrated_nll = optimized_nll if improved else baseline_nll
    return {
        "temperature": temperature,
        "initial_temperature": INITIAL_TEMPERATURE,
        "optimizer_temperature": optimized,
        "baseline_nll": baseline_nll,
        "calibrated_nll": calibrated_nll,
        "temperature_improved_nll": improved,
        "fitting_method": FITTING_METHOD,
        "fitting_status": "fitted" if improved else "baseline_retained",
    }


def calibration_result(
    records: list[dict],
    logits: torch.Tensor,
    *,
    spec: dict,
    checkpoint_path: Path,
    bins: int = DEFAULT_BINS,
    max_false_authentic_rate: float = DEFAULT_FAR_LIMIT,
) -> dict:
    """Score stored logits. Does not open images or update weights."""
    labels = torch.tensor([int(row["label"]) for row in records], dtype=torch.float32)
    vector = logits.detach().float().reshape(-1).cpu()
    if vector.numel() != len(records):
        raise ValueError("logits and calibration records differ in length")
    if not torch.isfinite(vector).all():
        raise RuntimeError("calibration logits are not finite")
    chosen = choose_temperature(vector, labels)
    temperature = float(chosen["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(f"temperature must be finite and > 0, got {temperature}")
    baseline = score_temperature(vector, labels, INITIAL_TEMPERATURE, bins)
    calibrated = score_temperature(vector, labels, temperature, bins)
    baseline_at_half = _operating_point(vector, labels, 0.5, INITIAL_TEMPERATURE)
    calibrated_at_half = _operating_point(vector, labels, 0.5, temperature)
    comparison = compare_calibration(baseline, calibrated)
    comparison["roc_auc_unchanged"] = _close(baseline["roc_auc"], calibrated["roc_auc"])
    comparison["pr_auc_unchanged"] = _close(baseline["pr_auc"], calibrated["pr_auc"])
    comparison["classification_change_claimed"] = False
    baseline_sweep = _sweep(vector, labels, INITIAL_TEMPERATURE)
    calibrated_sweep = _sweep(vector, labels, temperature)
    baseline_candidates = calibration_threshold_candidates(
        baseline_sweep, max_false_authentic_rate=max_false_authentic_rate
    )
    calibrated_candidates = calibration_threshold_candidates(
        calibrated_sweep, max_false_authentic_rate=max_false_authentic_rate
    )
    brands = [row["brand"] for row in records]
    prediction_rows = _prediction_rows(records, vector)
    if [row["logit"] for row in prediction_rows] != [float(value) for value in vector]:
        raise RuntimeError("prediction rows changed the raw logits")
    brand_input = [
        {"brand": row["brand"], "label": row["label"], "logit": row["logit"]} for row in prediction_rows
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "model_family": spec["family"],
        "model_architecture": spec["head"],
        "head": spec["head"],
        "input_size": int(spec["input_size"]),
        "manifest_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "preprocessing_version": PREPROCESSING_VERSION,
        "calibration_sample_count": len(records),
        "authentic_count": sum(1 for row in records if row["label"] == 0),
        "fake_count": sum(1 for row in records if row["label"] == 1),
        "brands": brands,
        "temperature_fit": chosen,
        "baseline": baseline,
        "calibrated": calibrated,
        "baseline_at_0_50": baseline_at_half,
        "calibrated_at_0_50": calibrated_at_half,
        "comparison": comparison,
        "baseline_sweep": baseline_sweep,
        "calibrated_sweep": calibrated_sweep,
        "baseline_candidates": baseline_candidates,
        "calibrated_candidates": calibrated_candidates,
        "baseline_sensitivity": _sensitivity(vector, labels, INITIAL_TEMPERATURE, baseline_candidates),
        "calibrated_sensitivity": _sensitivity(vector, labels, temperature, calibrated_candidates),
        "brand_baseline": brand_calibration_summaries(brand_input, INITIAL_TEMPERATURE, n_bins=bins),
        "brand_calibrated": brand_calibration_summaries(brand_input, temperature, n_bins=bins),
        "prediction_rows": prediction_rows,
        "checkpoint_selection_completed": True,
        "calibration_fitted_after_selection": True,
        "checkpoint_selection_rerun": False,
        "production_threshold": None,
        "architecture_ranking_affected": False,
        "test_images_loaded": False,
        "ood_used": False,
        "training_performed": False,
    }


def write_model_artifacts(experiment_dir: Path, result: dict) -> Path:
    """Write one model's calibration files. Does not write checkpoints."""
    root = _assert_model_destination(experiment_dir)
    destination = root / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    fit = result["temperature_fit"]
    temperature_payload = {
        "checkpoint": result["checkpoint"],
        "model_family": result["model_family"],
        "model_architecture": result["model_architecture"],
        "head": result["head"],
        "input_size": result["input_size"],
        "manifest_hash": result["manifest_hash"],
        "preprocessing_version": result["preprocessing_version"],
        "calibration_sample_count": result["calibration_sample_count"],
        "authentic_count": result["authentic_count"],
        "fake_count": result["fake_count"],
        "temperature": fit["temperature"],
        "initial_temperature": fit["initial_temperature"],
        "baseline_nll": fit["baseline_nll"],
        "calibrated_nll": fit["calibrated_nll"],
        "temperature_improved_nll": fit["temperature_improved_nll"],
        "fitting_method": fit["fitting_method"],
        "fitting_status": fit["fitting_status"],
        "checkpoint_selection_completed": True,
        "calibration_fitted_after_selection": True,
        "production_threshold": None,
        "architecture_ranking_affected": False,
    }
    (destination / "temperature.json").write_text(_dump(temperature_payload) + "\n")
    (destination / "calibration_metrics.json").write_text(
        _dump(
            {
                "baseline": _public_metrics(result["baseline"], result["baseline_at_0_50"]),
                "calibrated": _public_metrics(result["calibrated"], result["calibrated_at_0_50"]),
                "comparison": result["comparison"],
                "ece_bins": DEFAULT_BINS,
                "brand_baseline": result["brand_baseline"],
                "brand_calibrated": result["brand_calibrated"],
                "global_temperature": fit["temperature"],
                "per_brand_temperature_fitted": False,
                "classification_change_claimed": False,
                "production_threshold": None,
            }
        )
        + "\n"
    )
    (destination / "calibration_summary.json").write_text(
        _dump(_summary_payload(result)) + "\n"
    )
    _write_prediction_csv(destination / "calibration_predictions.csv", result["prediction_rows"])
    _write_threshold_csv(
        destination / "threshold_candidates.csv",
        result["baseline_sweep"],
        result["calibrated_sweep"],
        result["baseline_candidates"],
        result["calibrated_candidates"],
    )
    _write_reliability_csv(
        destination / "reliability_bins_baseline.csv", result["baseline"]["reliability_bins"]
    )
    _write_reliability_csv(
        destination / "reliability_bins_calibrated.csv", result["calibrated"]["reliability_bins"]
    )
    return destination


def write_comparison(destination_dir: Path, rows: list[dict]) -> Path:
    """Write the cross-model calibration table. Does not rank architectures."""
    destination = _assert_comparison_destination(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 20,
        "architecture_ranking_affected": False,
        "architecture_winner": None,
        "production_threshold": None,
        "checkpoint_selection_completed": True,
        "calibration_fitted_after_selection": True,
        "checkpoint_selection_rerun": False,
        "test_images_loaded": False,
        "ood_used": False,
        "training_performed": False,
        "temperature_meaning": (
            "Each temperature is one global positive scalar. It changes probability "
            "calibration. It is not an architecture-selection score."
        ),
        "models": [_comparison_row(row) for row in rows],
    }
    (destination / "calibration_comparison.json").write_text(_dump(payload) + "\n")
    fieldnames = [
        "model",
        "checkpoint",
        "temperature",
        "baseline_nll",
        "calibrated_nll",
        "delta_nll",
        "baseline_brier",
        "calibrated_brier",
        "baseline_ece",
        "calibrated_ece",
        "baseline_roc_auc",
        "calibrated_roc_auc",
        "baseline_pr_auc",
        "calibrated_pr_auc",
        "baseline_f1_at_0_50",
        "calibrated_f1_at_0_50",
        "baseline_far_at_0_50",
        "calibrated_far_at_0_50",
        "production_threshold",
        "architecture_ranking_affected",
    ]
    with (destination / "calibration_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in payload["models"]:
            writer.writerow(row)
    return destination


def collect_calibration_logits(
    records: list[dict],
    checkpoint: Path,
    *,
    family: str,
    head: str,
    batch_size: int = INFERENCE_BATCH_SIZE,
    num_workers: int = INFERENCE_WORKERS,
    device: str | None = None,
) -> torch.Tensor:
    """Run the calibration images once. Returns raw logits, with no temperature applied."""
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    chosen = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _image_size = build_eval_model(family, head, Path(checkpoint), chosen)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    transform = build_transforms(family, train=False, profile=CONSERVATIVE)
    loader = DataLoader(
        _CalibrationImages(records, transform),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=chosen.type == "cuda",
    )
    logits = _forward_logits(model, loader, chosen)
    after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key, value in after.items():
        if not torch.equal(value, before[key]):
            raise RuntimeError(f"inference changed model parameter {key}")
    del model
    if chosen.type == "cuda":
        torch.cuda.empty_cache()
    return logits


def run_phase20(
    experiments_root: Path,
    manifest_path: Path,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
    num_workers: int = INFERENCE_WORKERS,
    device: str | None = None,
) -> dict:
    """Calibrate the four Phase 19 checkpoints. Refuses a failed preflight before inference."""
    root = Path(experiments_root)
    manifest_file = Path(manifest_path)
    manifest_digest = file_sha256(manifest_file)
    checkpoint_digests = {}
    for spec in MODELS:
        checkpoint = root / spec["directory"] / spec["checkpoint"]
        checkpoint_digests[str(checkpoint)] = file_sha256(checkpoint)
    manifest = json.loads(manifest_file.read_text())
    records = validate_phase16_calibration_manifest(manifest)
    blocked = {str(Path(path).resolve()) for path in manifest["membership"]["test"]}
    if any(row["path"] in blocked for row in records):
        raise RuntimeError("calibration records include a final-test path")
    preflight = []
    for spec in MODELS:
        report = preflight_checkpoint(spec, root / spec["directory"])
        report["directory"] = spec["directory"]
        preflight.append(report)
        if not report["checkpoint_load_valid"]:
            reasons = "; ".join(report["reasons"]) or "checkpoint preflight failed"
            raise RuntimeError(f"{spec['directory']} failed checkpoint preflight: {reasons}")
    results = []
    for spec in MODELS:
        experiment_dir = root / spec["directory"]
        checkpoint = experiment_dir / spec["checkpoint"]
        print(f"collecting logits for {spec['directory']}", flush=True)
        logits = collect_calibration_logits(
            records,
            checkpoint,
            family=spec["family"],
            head=spec["head"],
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        result = calibration_result(records, logits, spec=spec, checkpoint_path=checkpoint)
        write_model_artifacts(experiment_dir, result)
        results.append(result)
        print(
            f"{spec['directory']} T={result['temperature_fit']['temperature']:.6f} "
            f"status={result['temperature_fit']['fitting_status']}",
            flush=True,
        )
    comparison_dir = write_comparison(root / COMPARISON_DIR_NAME, results)
    if file_sha256(manifest_file) != manifest_digest:
        raise RuntimeError("calibration rewrote the Phase 16 manifest")
    for path, digest in checkpoint_digests.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"calibration changed checkpoint bytes: {path}")
    return {
        "comparison_dir": str(comparison_dir),
        "preflight": preflight,
        "checkpoint_sha256": checkpoint_digests,
        "manifest_sha256": manifest_digest,
        "models": [_comparison_row(row) for row in results],
        "architecture_ranking_affected": False,
        "production_threshold": None,
        "test_images_loaded": False,
        "ood_used": False,
        "training_performed": False,
    }


def _forward_logits(model, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    collected: list[torch.Tensor] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    total = len(loader)
    with torch.inference_mode():
        for batch_index, (images, _labels, _sample_ids, _brands) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(images)
            else:
                output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            collected.append(output.detach().float().cpu().reshape(-1))
            if batch_index == 1 or batch_index == total or batch_index % 25 == 0:
                print(f"  calibration batch {batch_index}/{total}", flush=True)
    if not collected:
        raise RuntimeError("calibration loader produced no batches")
    return torch.cat(collected)


def _sweep(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> list[dict]:
    return [_operating_point(logits, labels, threshold, temperature) for threshold in THRESHOLD_GRID]


def _operating_point(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    temperature: float,
) -> dict:
    metrics = evaluate_logits(scaled_logits(logits, temperature), labels, threshold=threshold)
    return {
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_authentic": metrics["false_authentic"],
        "false_authentic_count": metrics["false_authentic"],
        "false_authentic_rate": metrics["false_authentic_rate"],
        "false_fake": metrics["false_fake"],
        "false_fake_count": metrics["false_fake"],
        "false_fake_rate": metrics["false_fake_rate"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "fake_count": metrics["fake_count"],
    }


def _sensitivity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    candidates: dict,
) -> dict:
    reported = {}
    for name in ("highest_f1", "highest_recall", "lowest_far", "closest_to_0_50"):
        candidate = candidates.get(name)
        if candidate is None:
            reported[name] = []
            continue
        center = float(candidate["threshold"])
        rows = []
        for threshold in (center - 0.02, center, center + 0.02):
            if not 0.0 <= threshold <= 1.0:
                continue
            point = _operating_point(logits, labels, threshold, temperature)
            point["relation"] = "candidate" if threshold == center else "neighbor"
            rows.append(
                {
                    "threshold": point["threshold"],
                    "relation": point["relation"],
                    "false_authentic_rate": point["false_authentic_rate"],
                    "recall": point["recall"],
                    "f1": point["f1"],
                    "false_fake_rate": point["false_fake_rate"],
                }
            )
        reported[name] = rows
    return reported


def _prediction_rows(records: list[dict], logits: torch.Tensor) -> list[dict]:
    raw = probabilities_from_temperature(logits, INITIAL_TEMPERATURE)
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "sample_id": record["sample_id"],
                "brand": record["brand"],
                "label": int(record["label"]),
                "logit": float(logits[index]),
                "raw_probability": float(raw[index]),
            }
        )
    return rows


def _public_metrics(scored: dict, operating: dict) -> dict:
    return {
        "nll": scored["nll"],
        "brier": scored["brier"],
        "ece": scored["ece"],
        "roc_auc": scored["roc_auc"],
        "pr_auc": scored["pr_auc"],
        "threshold_0_50": {
            "threshold": 0.5,
            "accuracy": operating["accuracy"],
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "false_authentic_count": operating["false_authentic_count"],
            "false_authentic_rate": operating["false_authentic_rate"],
            "false_fake_count": operating["false_fake_count"],
            "false_fake_rate": operating["false_fake_rate"],
            "fake_count": operating["fake_count"],
        },
    }


def _summary_payload(result: dict) -> dict:
    fit = result["temperature_fit"]
    return {
        "checkpoint": result["checkpoint"],
        "model_family": result["model_family"],
        "head": result["head"],
        "input_size": result["input_size"],
        "manifest_hash": result["manifest_hash"],
        "calibration_sample_count": result["calibration_sample_count"],
        "authentic_count": result["authentic_count"],
        "fake_count": result["fake_count"],
        "temperature": fit["temperature"],
        "temperature_improved_nll": fit["temperature_improved_nll"],
        "fitting_status": fit["fitting_status"],
        "baseline_nll": fit["baseline_nll"],
        "calibrated_nll": fit["calibrated_nll"],
        "baseline_brier": result["baseline"]["brier"],
        "calibrated_brier": result["calibrated"]["brier"],
        "baseline_ece": result["baseline"]["ece"],
        "calibrated_ece": result["calibrated"]["ece"],
        "baseline_roc_auc": result["baseline"]["roc_auc"],
        "calibrated_roc_auc": result["calibrated"]["roc_auc"],
        "baseline_pr_auc": result["baseline"]["pr_auc"],
        "calibrated_pr_auc": result["calibrated"]["pr_auc"],
        "threshold_0_50_baseline": result["baseline_at_0_50"],
        "threshold_0_50_calibrated": result["calibrated_at_0_50"],
        "baseline_calibration_candidates": result["baseline_candidates"],
        "calibrated_calibration_candidates": result["calibrated_candidates"],
        "baseline_threshold_sensitivity": result["baseline_sensitivity"],
        "calibrated_threshold_sensitivity": result["calibrated_sensitivity"],
        "brand_baseline": result["brand_baseline"],
        "brand_calibrated": result["brand_calibrated"],
        "checkpoint_selection_completed": True,
        "calibration_fitted_after_selection": True,
        "checkpoint_selection_rerun": False,
        "production_threshold": None,
        "architecture_ranking_affected": False,
        "test_images_loaded": False,
        "ood_used": False,
        "training_performed": False,
        "probability_quality_is_not_classification_accuracy": True,
    }


def _comparison_row(result: dict) -> dict:
    fit = result["temperature_fit"]
    baseline = result["baseline"]
    calibrated = result["calibrated"]
    return {
        "model": Path(result["checkpoint"]).parent.name,
        "checkpoint": result["checkpoint"],
        "temperature": fit["temperature"],
        "temperature_improved_nll": fit["temperature_improved_nll"],
        "fitting_status": fit["fitting_status"],
        "baseline_nll": fit["baseline_nll"],
        "calibrated_nll": fit["calibrated_nll"],
        "delta_nll": fit["calibrated_nll"] - fit["baseline_nll"],
        "baseline_brier": baseline["brier"],
        "calibrated_brier": calibrated["brier"],
        "baseline_ece": baseline["ece"],
        "calibrated_ece": calibrated["ece"],
        "baseline_roc_auc": baseline["roc_auc"],
        "calibrated_roc_auc": calibrated["roc_auc"],
        "baseline_pr_auc": baseline["pr_auc"],
        "calibrated_pr_auc": calibrated["pr_auc"],
        "roc_auc_unchanged": result["comparison"]["roc_auc_unchanged"],
        "pr_auc_unchanged": result["comparison"]["pr_auc_unchanged"],
        "baseline_f1_at_0_50": baseline["threshold_0_50"]["f1"],
        "calibrated_f1_at_0_50": calibrated["threshold_0_50"]["f1"],
        "baseline_far_at_0_50": baseline["threshold_0_50"]["false_authentic_rate"],
        "calibrated_far_at_0_50": calibrated["threshold_0_50"]["false_authentic_rate"],
        "baseline_safe_thresholds": result["baseline_candidates"]["safe_thresholds"],
        "calibrated_safe_thresholds": result["calibrated_candidates"]["safe_thresholds"],
        "baseline_candidates": {
            name: _candidate_brief(result["baseline_candidates"].get(name))
            for name in ("highest_f1", "highest_recall", "lowest_far", "closest_to_0_50")
        },
        "calibrated_candidates": {
            name: _candidate_brief(result["calibrated_candidates"].get(name))
            for name in ("highest_f1", "highest_recall", "lowest_far", "closest_to_0_50")
        },
        "brand_calibrated": result["brand_calibrated"],
        "production_threshold": None,
        "architecture_ranking_affected": False,
    }


def _candidate_brief(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "threshold": candidate["threshold"],
        "f1": candidate.get("f1"),
        "recall": candidate.get("recall"),
        "false_authentic_rate": candidate.get("false_authentic_rate"),
        "role": "calibration_candidate",
    }


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= 1e-8


def _assert_model_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {"checkpoints", "checkpoints_watches", "dataset_audit"}
    if resolved.name in blocked or any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write calibration into {resolved}")
    return resolved


def _assert_comparison_destination(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.name != COMPARISON_DIR_NAME:
        raise RuntimeError(f"Phase 20 comparison output must be named {COMPARISON_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        *(spec["directory"] for spec in MODELS),
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the calibration comparison into {resolved}")
    return resolved


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("refusing to write a non-finite calibration value")
    return repr(float(value))


def _write_prediction_csv(path: Path, rows: list[dict]) -> None:
    fields = ["sample_id", "brand", "label", "logit", "raw_probability"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "brand": row["brand"],
                    "label": row["label"],
                    "logit": _format_float(float(row["logit"])),
                    "raw_probability": _format_float(float(row["raw_probability"])),
                }
            )


def _write_threshold_csv(
    path: Path,
    baseline_sweep: list[dict],
    calibrated_sweep: list[dict],
    baseline_candidates: dict,
    calibrated_candidates: dict,
) -> None:
    fields = [
        "probabilities",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_authentic_count",
        "false_authentic_rate",
        "false_fake_count",
        "false_fake_rate",
        "calibration_candidate",
        "highest_f1",
        "highest_recall",
        "lowest_far",
        "closest_to_0_50",
        "production_threshold",
    ]
    groups = (
        ("baseline", baseline_sweep, baseline_candidates),
        ("calibrated", calibrated_sweep, calibrated_candidates),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, sweep, candidates in groups:
            safe = set(candidates["safe_thresholds"])
            roles = {
                role: None if candidates.get(role) is None else float(candidates[role]["threshold"])
                for role in ("highest_f1", "highest_recall", "lowest_far", "closest_to_0_50")
            }
            for row in sweep:
                writer.writerow(
                    {
                        "probabilities": name,
                        "threshold": _format_float(float(row["threshold"])),
                        "accuracy": _format_float(float(row["accuracy"])),
                        "precision": "" if row["precision"] is None else _format_float(float(row["precision"])),
                        "recall": "" if row["recall"] is None else _format_float(float(row["recall"])),
                        "f1": "" if row["f1"] is None else _format_float(float(row["f1"])),
                        "false_authentic_count": row["false_authentic_count"],
                        "false_authentic_rate": _format_float(float(row["false_authentic_rate"])),
                        "false_fake_count": row["false_fake_count"],
                        "false_fake_rate": _format_float(float(row["false_fake_rate"])),
                        "calibration_candidate": row["threshold"] in safe,
                        "highest_f1": roles["highest_f1"] == row["threshold"],
                        "highest_recall": roles["highest_recall"] == row["threshold"],
                        "lowest_far": roles["lowest_far"] == row["threshold"],
                        "closest_to_0_50": roles["closest_to_0_50"] == row["threshold"],
                        "production_threshold": "",
                    }
                )


def _write_reliability_csv(path: Path, bins: list[dict]) -> None:
    fields = [
        "bin_lower",
        "bin_upper",
        "sample_count",
        "mean_confidence",
        "observed_accuracy",
        "absolute_gap",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in bins:
            writer.writerow(
                {
                    "bin_lower": _format_float(float(row["lower"])),
                    "bin_upper": _format_float(float(row["upper"])),
                    "sample_count": row["sample_count"],
                    "mean_confidence": "" if row["confidence_mean"] is None else _format_float(float(row["confidence_mean"])),
                    "observed_accuracy": "" if row["observed_accuracy"] is None else _format_float(float(row["observed_accuracy"])),
                    "absolute_gap": "" if row["absolute_gap"] is None else _format_float(float(row["absolute_gap"])),
                }
            )


class _CalibrationImages(Dataset):
    def __init__(self, records: list[dict], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        return self.transform(image), int(record["label"]), record["sample_id"], record["brand"]


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    manifest = root / "dataset_audit" / "split_manifest_v2.json"
    summary = run_phase20(root, manifest)
    print(json.dumps({"comparison_dir": summary["comparison_dir"], "models": summary["models"]}, indent=2))


if __name__ == "__main__":
    main()
