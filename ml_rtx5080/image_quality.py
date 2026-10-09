"""Deterministic input-quality checks for the frozen authenticity model.

Quality features and flags are not authenticity scores. They are not multiplied
by, or subtracted from, the calibrated probability. This module does not train,
does not read the final test split, and does not choose a production threshold.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from dataset import IMAGE_SUFFIXES
from hard_negative import (
    FROZEN_TEMPERATURE,
    SOURCE_SPLITS,
    apply_challenge_transform,
    false_authentic_rate,
    file_sha256,
    prepare_source_records,
)
from split_v2 import PHASE15_EXPERIMENT_DIRS, SPLIT_NAMES
from training_config import assert_experiment_output_dir

EXPECTED_CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
QUALITY_DIR_NAME = "image_quality_v1"
REFERENCE_SPLITS = SOURCE_SPLITS
MIN_REFERENCE_SAMPLES = 20
PERCENTILE_METHOD = "linear"
STD_DDOF = 0
LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)
DARK_LUMINANCE = 0.05
BRIGHT_LUMINANCE = 0.95
NEAR_BLACK_LUMINANCE = 5.0 / 255.0
NEAR_WHITE_LUMINANCE = 250.0 / 255.0
FOREGROUND_BORDER_FRACTION = 0.04
FOREGROUND_DELTA = 0.08
FLAG_NAMES = (
    "LOW_RESOLUTION",
    "BLURRY",
    "TOO_DARK",
    "TOO_BRIGHT",
    "LOW_CONTRAST",
    "EXTREME_ASPECT_RATIO",
    "EXCESSIVE_CLIPPING",
    "SUSPICIOUS_FRAMING",
)
NUMERIC_FEATURES = (
    "width",
    "height",
    "pixel_count",
    "min_dimension",
    "aspect_ratio",
    "mean_luminance",
    "luminance_std",
    "luminance_percentile_spread",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
    "near_black_fraction",
    "near_white_fraction",
    "sharpness_laplacian_variance",
    "foreground_occupancy",
    "file_bytes",
    "bytes_per_pixel",
)
FEATURE_SCHEMA = NUMERIC_FEATURES + ("image_format", "jpeg_quantization_tables_present")
LINK_FEATURES = (
    "min_dimension",
    "pixel_count",
    "sharpness_laplacian_variance",
    "mean_luminance",
    "luminance_std",
    "foreground_occupancy",
    "near_black_fraction",
    "near_white_fraction",
)
CONDITION_ORDER = (
    "jpeg_recompression",
    "gaussian_blur",
    "brightness_shift",
    "contrast_shift",
    "mild_crop",
    "resize_recompression",
    "small_rotation",
    "perspective",
    "lower_resolution",
    "darker",
    "brighter",
    "moderate_jpeg",
    "mild_blur",
    "partial_crop",
)
THRESHOLD_RULES = {
    "LOW_RESOLUTION": "min_dimension strictly below the reference 5th percentile",
    "BLURRY": "Laplacian variance strictly below the reference 5th percentile",
    "TOO_DARK": "mean luminance strictly below the reference 5th percentile",
    "TOO_BRIGHT": "mean luminance strictly above the reference 95th percentile",
    "LOW_CONTRAST": "luminance standard deviation strictly below the reference 5th percentile",
    "EXTREME_ASPECT_RATIO": "width/height strictly outside the reference 5th to 95th percentile",
    "EXCESSIVE_CLIPPING": "near-black or near-white fraction strictly above its reference 95th percentile",
    "SUSPICIOUS_FRAMING": "foreground-occupancy proxy strictly below the reference 5th percentile",
    "percentile_method": PERCENTILE_METHOD,
    "std_ddof": STD_DDOF,
    "comparison": "strict inequality, so a value equal to the percentile is not flagged",
    "not_fit_to_f1": True,
    "final_test_used": False,
}


class QualityReadError(ValueError):
    """An image could not be read into RGB quality features."""


def luminance_image(rgb: Image.Image) -> np.ndarray:
    """BT.601 luminance in 0..1 from an RGB image. Normalization is not applied."""
    array = np.asarray(rgb.convert("RGB"), dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise QualityReadError(f"expected an RGB image, got shape {array.shape}")
    return (LUMINANCE_WEIGHTS[0] * array[:, :, 0] + LUMINANCE_WEIGHTS[1] * array[:, :, 1] + LUMINANCE_WEIGHTS[2] * array[:, :, 2]) / 255.0


def sharpness_laplacian_variance(luminance: np.ndarray) -> float:
    """Variance of a 4-neighbor Laplacian. A constant image scores 0."""
    if luminance.shape[0] < 3 or luminance.shape[1] < 3:
        return 0.0
    center = luminance[1:-1, 1:-1]
    response = (
        luminance[:-2, 1:-1]
        + luminance[2:, 1:-1]
        + luminance[1:-1, :-2]
        + luminance[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(response))


def foreground_occupancy(luminance: np.ndarray) -> float:
    """Fraction of pixels unlike the border median.

    This is a lightweight occupancy proxy. It is not watch-boundary detection.
    """
    height, width = luminance.shape
    border = max(1, int(round(FOREGROUND_BORDER_FRACTION * min(height, width))))
    if border * 2 >= height or border * 2 >= width:
        border = 1
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    background = float(np.median(luminance[mask]))
    return float(np.mean(np.abs(luminance - background) > FOREGROUND_DELTA))


def extract_quality_features(
    image: Image.Image,
    *,
    file_bytes: int | None = None,
    image_format: str | None = None,
    jpeg_quantization_tables_present: bool | None = None,
) -> dict:
    """Features from the original RGB image, before model normalization."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise QualityReadError("image has no pixels")
    luminance = luminance_image(rgb)
    pixel_count = int(width * height)
    tables_present = bool(jpeg_quantization_tables_present)
    bytes_per_pixel = None if file_bytes is None or pixel_count == 0 else float(file_bytes) / float(pixel_count)
    dark = float(np.mean(luminance < DARK_LUMINANCE))
    bright = float(np.mean(luminance > BRIGHT_LUMINANCE))
    near_black = float(np.mean(luminance <= NEAR_BLACK_LUMINANCE))
    near_white = float(np.mean(luminance >= NEAR_WHITE_LUMINANCE))
    low = float(np.percentile(luminance, 5, method=PERCENTILE_METHOD))
    high = float(np.percentile(luminance, 95, method=PERCENTILE_METHOD))
    luminance_std = float(np.std(luminance, ddof=STD_DDOF))
    if luminance_std < 1e-12:
        luminance_std = 0.0
    return {
        "width": int(width),
        "height": int(height),
        "pixel_count": pixel_count,
        "min_dimension": int(min(width, height)),
        "aspect_ratio": float(width) / float(height),
        "mean_luminance": float(np.mean(luminance)),
        "luminance_std": luminance_std,
        "luminance_percentile_spread": high - low,
        "dark_pixel_fraction": dark,
        "bright_pixel_fraction": bright,
        "near_black_fraction": near_black,
        "near_white_fraction": near_white,
        "sharpness_laplacian_variance": sharpness_laplacian_variance(luminance),
        "foreground_occupancy": foreground_occupancy(luminance),
        "file_bytes": None if file_bytes is None else int(file_bytes),
        "bytes_per_pixel": bytes_per_pixel,
        "image_format": None if image_format is None else str(image_format),
        "jpeg_quantization_tables_present": tables_present,
    }


def extract_quality_features_from_path(path: Path) -> dict:
    """Read one supported image. Missing and corrupt files raise QualityReadError."""
    file_path = Path(path)
    if file_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise QualityReadError(f"unsupported image suffix {file_path.suffix}")
    if not file_path.is_file():
        raise QualityReadError("missing image file")
    try:
        file_bytes = file_path.stat().st_size
        with Image.open(file_path) as handle:
            image_format = handle.format
            tables = getattr(handle, "quantization", None)
            rgb = handle.convert("RGB")
            rgb.load()
            owned = rgb.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise QualityReadError(str(exc)) from exc
    return extract_quality_features(
        owned,
        file_bytes=file_bytes,
        image_format=image_format,
        jpeg_quantization_tables_present=bool(tables),
    )


def feature_delta(before: dict, after: dict) -> dict:
    """After minus before for numeric quality features."""
    delta = {}
    for name in NUMERIC_FEATURES:
        left = before.get(name)
        right = after.get(name)
        if left is None or right is None:
            delta[name] = None
        else:
            delta[name] = float(right) - float(left)
    return delta


def reference_distribution(rows: list[dict]) -> dict:
    """Descriptive percentiles. This does not fit a classifier."""
    report = {}
    for name in NUMERIC_FEATURES:
        values = np.asarray(
            [float(row[name]) for row in rows if row.get(name) is not None and np.isfinite(row[name])],
            dtype=np.float64,
        )
        report[name] = _distribution(values)
    return report


def derive_thresholds(distribution: dict, sample_count: int) -> dict:
    """Fixed percentile rules. Thresholds are not optimized for F1."""
    if int(sample_count) < MIN_REFERENCE_SAMPLES:
        raise ValueError(f"quality reference needs at least {MIN_REFERENCE_SAMPLES} images")
    required = (
        "min_dimension",
        "sharpness_laplacian_variance",
        "mean_luminance",
        "luminance_std",
        "aspect_ratio",
        "near_black_fraction",
        "near_white_fraction",
        "foreground_occupancy",
    )
    for name in required:
        if distribution[name]["count"] < MIN_REFERENCE_SAMPLES:
            raise ValueError(f"quality reference is missing {name}")
    return {
        "rules": dict(THRESHOLD_RULES),
        "reference_sample_count": int(sample_count),
        "final_test_used": False,
        "production_threshold": None,
        "cutoffs": {
            "min_dimension_p5": distribution["min_dimension"]["p5"],
            "sharpness_p5": distribution["sharpness_laplacian_variance"]["p5"],
            "mean_luminance_p5": distribution["mean_luminance"]["p5"],
            "mean_luminance_p95": distribution["mean_luminance"]["p95"],
            "luminance_std_p5": distribution["luminance_std"]["p5"],
            "aspect_ratio_p5": distribution["aspect_ratio"]["p5"],
            "aspect_ratio_p95": distribution["aspect_ratio"]["p95"],
            "near_black_fraction_p95": distribution["near_black_fraction"]["p95"],
            "near_white_fraction_p95": distribution["near_white_fraction"]["p95"],
            "foreground_occupancy_p5": distribution["foreground_occupancy"]["p5"],
        },
    }


def apply_quality_flags(features: dict, thresholds: dict) -> list[str]:
    """Independent flags. A value exactly at a percentile is not flagged."""
    cutoffs = thresholds["cutoffs"]
    flags = []
    if float(features["min_dimension"]) < float(cutoffs["min_dimension_p5"]):
        flags.append("LOW_RESOLUTION")
    if float(features["sharpness_laplacian_variance"]) < float(cutoffs["sharpness_p5"]):
        flags.append("BLURRY")
    if float(features["mean_luminance"]) < float(cutoffs["mean_luminance_p5"]):
        flags.append("TOO_DARK")
    if float(features["mean_luminance"]) > float(cutoffs["mean_luminance_p95"]):
        flags.append("TOO_BRIGHT")
    if float(features["luminance_std"]) < float(cutoffs["luminance_std_p5"]):
        flags.append("LOW_CONTRAST")
    aspect = float(features["aspect_ratio"])
    if aspect < float(cutoffs["aspect_ratio_p5"]) or aspect > float(cutoffs["aspect_ratio_p95"]):
        flags.append("EXTREME_ASPECT_RATIO")
    if float(features["near_black_fraction"]) > float(cutoffs["near_black_fraction_p95"]) or float(
        features["near_white_fraction"]
    ) > float(cutoffs["near_white_fraction_p95"]):
        flags.append("EXCESSIVE_CLIPPING")
    if float(features["foreground_occupancy"]) < float(cutoffs["foreground_occupancy_p5"]):
        flags.append("SUSPICIOUS_FRAMING")
    return flags


def quality_state(flags: list[str]) -> str:
    """Diagnostic label only. This is not a production verdict."""
    count = len(flags)
    if count <= 0:
        return "clean"
    if count == 1:
        return "low_quality"
    return "degraded"


def classification_by_quality(rows: list[dict]) -> dict:
    """Authenticity metrics inside quality groups. Empty-class rates stay undefined."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["quality_state"], []).append(row)
        for flag in row["quality_flags"]:
            grouped.setdefault(f"flag:{flag}", []).append(row)
    return {name: _classification_metrics(group) for name, group in sorted(grouped.items())}


def robustness_table(stress_rows: list[dict]) -> list[dict]:
    """Phase 22 stress rows already contain probabilities. No model rerun."""
    table = []
    for condition in CONDITION_ORDER:
        subset = [row for row in stress_rows if row["condition"] == condition]
        flips = [row for row in subset if int(row["prediction_flip"]) == 1]
        false_authentic = [
            row for row in subset if int(row["label"]) == 1 and float(row["transformed_calibrated_probability"]) < 0.5
        ]
        false_fake = [
            row for row in subset if int(row["label"]) == 0 and float(row["transformed_calibrated_probability"]) >= 0.5
        ]
        changes = [float(row["probability_change"]) for row in subset]
        table.append(
            {
                "condition": condition,
                "sample_count": len(subset),
                "prediction_flips": len(flips),
                "false_authentic_count": len(false_authentic),
                "false_fake_count": len(false_fake),
                "mean_probability_change": None if not changes else float(np.mean(changes)),
                "maximum_absolute_probability_change": None if not changes else float(np.max(np.abs(changes))),
                "unique_images": len({row["sample_id"] for row in subset}),
                "unique_flipped_images": len({row["sample_id"] for row in flips}),
            }
        )
    return table


def rank_failure_modes(table: list[dict]) -> list[dict]:
    """Rank stress conditions. The rank does not change the frozen model."""
    ranked = sorted(
        table,
        key=lambda row: (
            -int(row["false_fake_count"]),
            -1.0 if row["maximum_absolute_probability_change"] is None else -float(row["maximum_absolute_probability_change"]),
            -1.0 if row["mean_probability_change"] is None else -abs(float(row["mean_probability_change"])),
            -int(row["unique_flipped_images"]),
            row["condition"],
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["model_updated"] = False
    return ranked


def associate_feature_changes(linked_rows: list[dict]) -> list[dict]:
    """Compare feature movement on flips and non-flips. Association is not causation."""
    report = []
    for condition in CONDITION_ORDER:
        subset = [row for row in linked_rows if row["condition"] == condition]
        flips = [row for row in subset if int(row["prediction_flip"]) == 1]
        stable = [row for row in subset if int(row["prediction_flip"]) == 0]
        entry = {"condition": condition, "flip_count": len(flips), "stable_count": len(stable), "association_only": True}
        for name in LINK_FEATURES:
            flip_values = _finite_deltas(flips, name)
            stable_values = _finite_deltas(stable, name)
            entry[f"{name}_flip_mean_delta"] = None if not flip_values else float(np.mean(flip_values))
            entry[f"{name}_stable_mean_delta"] = None if not stable_values else float(np.mean(stable_values))
        report.append(entry)
    return report


def reject_final_test_paths(paths: list[str], test_paths: set[str]) -> None:
    blocked = {str(Path(path).resolve()) for path in paths}
    if blocked & {str(Path(path).resolve()) for path in test_paths}:
        raise RuntimeError("quality analysis encountered a final-test sample")


def load_stress_rows(challenge_dir: Path) -> list[dict]:
    rows = []
    for filename in ("transform_stability.csv", "quality_stress.csv"):
        path = Path(challenge_dir) / filename
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                rows.append(
                    {
                        "sample_id": record["source_sample_id"],
                        "source_split": record["source_split"],
                        "brand": record["brand"],
                        "label": int(record["label"]),
                        "condition": record["condition"],
                        "original_calibrated_probability": float(record["original_calibrated_probability"]),
                        "transformed_calibrated_probability": float(record["transformed_calibrated_probability"]),
                        "probability_change": float(record["probability_change"]),
                        "prediction_flip": int(record["prediction_flip"]),
                    }
                )
    return rows


def brand_quality_summary(rows: list[dict], stress_rows: list[dict]) -> list[dict]:
    summary = []
    for brand in sorted({row["brand"] for row in rows}):
        subset = [row for row in rows if row["brand"] == brand]
        stress = [row for row in stress_rows if row["brand"] == brand]
        counts = {flag: sum(flag in row["quality_flags"] for row in subset) for flag in FLAG_NAMES}
        most_common = None
        if any(counts.values()):
            most_common = sorted(counts, key=lambda flag: (-counts[flag], flag))[0]
        authentic = sum(int(row["label"]) == 0 for row in subset)
        false_fake = sum(int(row["label"]) == 0 and row["authenticity_prediction"] == "fake" for row in subset)
        summary.append(
            {
                "brand": brand,
                "sample_count": len(subset),
                "quality_flag_rate": None if not subset else sum(bool(row["quality_flags"]) for row in subset) / len(subset),
                "false_fake_rate": None if authentic == 0 else false_fake / authentic,
                "prediction_flip_rate": None if not stress else sum(int(row["prediction_flip"]) for row in stress) / len(stress),
                "most_common_quality_issue": most_common,
                "flag_counts": counts,
                "brand_specific_threshold": False,
            }
        )
    return summary


def run_phase23(
    experiments_root: Path,
    manifest_path: Path,
    *,
    workers: int | None = None,
) -> dict:
    """Build the non-test quality reference and link stored Phase 22 stress rows."""
    root = Path(experiments_root)
    destination = _assert_quality_destination(root / QUALITY_DIR_NAME)
    if (destination / "quality_summary.json").is_file():
        raise RuntimeError("image-quality analysis already completed; refusing to rerun")
    manifest_file = Path(manifest_path)
    checkpoint = root / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
    challenge_summary_path = root / "hard_negative_v1" / "challenge_summary.json"
    final_test_summary = root / "final_test_v1" / "final_test_summary.json"
    freeze_path = root / "final_model_freeze" / "model_selection_freeze.json"
    protected = {
        "checkpoint": file_sha256(checkpoint),
        "manifest": file_sha256(manifest_file),
        "final_test_summary": file_sha256(final_test_summary),
        "freeze": file_sha256(freeze_path),
        "challenge_summary": file_sha256(challenge_summary_path),
    }
    if protected["checkpoint"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash does not match the Phase 21 checkpoint")
    manifest = json.loads(manifest_file.read_text())
    test_paths = {str(Path(path).resolve()) for path in manifest["membership"]["test"]}
    records, rejected_records = prepare_source_records(
        manifest,
        enforce_authoritative_hash=True,
        require_files=True,
    )
    reject_final_test_paths([row["path"] for row in records], test_paths)
    print(f"quality reference images {len(records)}", flush=True)
    measured, read_errors = _measure_records(records, workers=workers)
    distribution = reference_distribution(measured)
    thresholds = derive_thresholds(distribution, len(measured))
    phase22 = json.loads(challenge_summary_path.read_text())
    _require_exact_nontest_pool(phase22, len(records))
    probabilities = _cached_probabilities(root / "hard_negative_v1")
    for row in measured:
        row["quality_flags"] = apply_quality_flags(row, thresholds)
        row["quality_state"] = quality_state(row["quality_flags"])
        row["authenticity_prediction"] = "fake" if int(row["label"]) == 1 else "authentic"
        row["authenticity_probability"] = probabilities.get(row["sample_id"])
        row["prediction_source"] = "phase22_nontest_pool_exact_classification"
        row["probability_source"] = "phase22_cached_prediction" if row["authenticity_probability"] is not None else "not_rerun"
    stress_rows = load_stress_rows(root / "hard_negative_v1")
    reject_final_test_paths([], test_paths)
    if any(row["source_split"] not in REFERENCE_SPLITS for row in stress_rows):
        raise RuntimeError("quality analysis encountered a final-test sample")
    by_id = {row["sample_id"]: row for row in measured}
    linked = _link_stress_rows(stress_rows, by_id, thresholds)
    table = robustness_table(stress_rows)
    ranked = rank_failure_modes(table)
    associations = associate_feature_changes(linked)
    brands = brand_quality_summary(measured, stress_rows)
    stratified = classification_by_quality(measured)
    flag_counts = {flag: sum(flag in row["quality_flags"] for row in measured) for flag in FLAG_NAMES}
    review = [row for row in measured if row["quality_flags"]]
    summary = {
        "phase": 23,
        "production_authenticity_threshold": None,
        "production_verdict_implemented": False,
        "temperature": FROZEN_TEMPERATURE,
        "temperature_changed": False,
        "checkpoint_changed": False,
        "training_performed": False,
        "ood_used": False,
        "final_test_opened": False,
        "final_test_samples_used": 0,
        "checkpoint_sha256": protected["checkpoint"],
        "split_manifest_sha256": protected["manifest"],
        "challenge_manifest_sha256": phase22["challenge_manifest_sha256"],
        "reference_sample_count": len(measured),
        "read_error_count": len(read_errors),
        "rejected_manifest_count": len(rejected_records),
        "source_splits": list(REFERENCE_SPLITS),
        "flag_counts": flag_counts,
        "quality_state_counts": {
            state: sum(row["quality_state"] == state for row in measured) for state in ("clean", "low_quality", "degraded")
        },
        "quality_review_candidate_count": len(review),
        "quality_stratified_metrics": stratified,
        "false_authentic_safety": _false_authentic_safety(measured),
        "cross_table": _cross_table(measured),
        "robustness": table,
        "failure_mode_ranking": ranked,
        "feature_associations": associations,
        "linked_flips": [row for row in linked if int(row["prediction_flip"]) == 1],
        "brands": brands,
        "quality_score_created": False,
        "composite_score": None,
    }
    if file_sha256(checkpoint) != protected["checkpoint"]:
        raise RuntimeError("image-quality analysis changed checkpoint bytes")
    if file_sha256(manifest_file) != protected["manifest"]:
        raise RuntimeError("image-quality analysis changed the split manifest")
    if file_sha256(final_test_summary) != protected["final_test_summary"] or file_sha256(freeze_path) != protected["freeze"]:
        raise RuntimeError("image-quality analysis changed the final test or the freeze")
    if file_sha256(challenge_summary_path) != protected["challenge_summary"]:
        raise RuntimeError("image-quality analysis changed the Phase 22 summary")
    _write_artifacts(
        destination,
        measured=measured,
        distribution=distribution,
        thresholds=thresholds,
        review=review,
        summary=summary,
        brands=brands,
        ranked=ranked,
        linked=linked,
    )
    return json.loads((destination / "quality_summary.json").read_text())


def _measure_records(records: list[dict], *, workers: int | None) -> tuple[list[dict], list[dict]]:
    worker_count = os.cpu_count() or 1 if workers is None else int(workers)
    worker_count = max(1, min(worker_count, 8))
    if worker_count == 1:
        return _collect_measurements(map(_measure_one, records), len(records))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        return _collect_measurements(pool.map(_measure_one, records, chunksize=16), len(records))


def _collect_measurements(results, total: int) -> tuple[list[dict], list[dict]]:
    measured = []
    errors = []
    for index, result in enumerate(results, start=1):
        if result.get("read_error"):
            errors.append(result)
        else:
            measured.append(result)
        if index == 1 or index % 2000 == 0 or index == total:
            print(f"  quality features {index}/{total}", flush=True)
    return measured, errors


def _measure_one(record: dict) -> dict:
    identity = {
        "sample_id": record["sample_id"],
        "source_split": record["source_split"],
        "brand": record["brand"],
        "label": int(record["label"]),
        "path": record["path"],
    }
    if record.get("source_split") not in REFERENCE_SPLITS:
        raise RuntimeError("quality analysis encountered a final-test sample")
    try:
        features = extract_quality_features_from_path(record["path"])
    except QualityReadError as exc:
        return {**identity, "read_error": str(exc)}
    return {**identity, **features, "read_error": ""}


def _link_stress_rows(stress_rows: list[dict], by_id: dict[str, dict], thresholds: dict) -> list[dict]:
    linked = []
    cache: dict[tuple[str, str], dict] = {}
    total = len(stress_rows)
    for index, row in enumerate(stress_rows, start=1):
        source = by_id.get(row["sample_id"])
        if source is None:
            raise RuntimeError(f"Phase 22 stress sample is missing from the non-test reference: {row['sample_id']}")
        if row["source_split"] not in REFERENCE_SPLITS:
            raise RuntimeError("quality analysis encountered a final-test sample")
        key = (row["sample_id"], row["condition"])
        if key not in cache:
            with Image.open(source["path"]) as handle:
                owned = handle.convert("RGB")
                owned.load()
                original = owned.copy()
            after_image = apply_challenge_transform(original, row["condition"])
            cache[key] = extract_quality_features(after_image)
        after = cache[key]
        delta = feature_delta(source, after)
        false_authentic = int(row["label"]) == 1 and float(row["transformed_calibrated_probability"]) < 0.5
        false_fake = int(row["label"]) == 0 and float(row["transformed_calibrated_probability"]) >= 0.5
        linked.append(
            {
                **row,
                "flags_before": "|".join(source["quality_flags"]),
                "flags_after": "|".join(apply_quality_flags(after, thresholds)),
                "delta": delta,
                "false_authentic": false_authentic,
                "false_fake": false_fake,
                "before_min_dimension": source["min_dimension"],
                "after_min_dimension": after["min_dimension"],
                "before_sharpness": source["sharpness_laplacian_variance"],
                "after_sharpness": after["sharpness_laplacian_variance"],
                "before_mean_luminance": source["mean_luminance"],
                "after_mean_luminance": after["mean_luminance"],
                "delta_min_dimension": delta["min_dimension"],
                "delta_sharpness": delta["sharpness_laplacian_variance"],
                "delta_mean_luminance": delta["mean_luminance"],
                "delta_luminance_std": delta["luminance_std"],
                "delta_pixel_count": delta["pixel_count"],
                "delta_foreground_occupancy": delta["foreground_occupancy"],
            }
        )
        if index == 1 or index % 400 == 0 or index == total:
            print(f"  stress quality linkage {index}/{total}", flush=True)
    return linked


def _cached_probabilities(challenge_dir: Path) -> dict[str, float]:
    found: dict[str, float] = {}
    low_margin = challenge_dir / "low_margin.csv"
    high_confidence = challenge_dir / "high_confidence.csv"
    for path, field in ((low_margin, "calibrated_probability"), (high_confidence, "calibrated_probability")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                found[row["sample_id"]] = float(row[field])
    for filename in ("transform_stability.csv", "quality_stress.csv"):
        with (challenge_dir / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                found.setdefault(row["source_sample_id"], float(row["original_calibrated_probability"]))
    return found


def _require_exact_nontest_pool(summary: dict, record_count: int) -> None:
    pool = summary["non_test_pool"]
    distribution = summary["non_test_distribution"]
    if int(pool["sample_count"]) != int(record_count):
        raise RuntimeError("Phase 22 non-test pool does not match the quality reference")
    if float(pool["accuracy"]) != 1.0 or int(distribution["misclassified_count"]) != 0:
        raise RuntimeError("Phase 22 non-test pool is not exact; refusing to invent predictions")
    if int(summary["test_samples_used"]) != 0:
        raise RuntimeError("Phase 22 summary reports final-test samples")


def _false_authentic_safety(rows: list[dict]) -> dict:
    groups = {"all": rows}
    for state in ("clean", "low_quality", "degraded"):
        groups[state] = [row for row in rows if row["quality_state"] == state]
    for flag in FLAG_NAMES:
        groups[flag] = [row for row in rows if flag in row["quality_flags"]]
    report = {}
    for name, subset in groups.items():
        fake_count = sum(int(row["label"]) == 1 for row in subset)
        cases = [
            row
            for row in subset
            if int(row["label"]) == 1 and row["authenticity_prediction"] == "authentic"
        ]
        report[name] = {
            "sample_count": len(subset),
            "fake_count": fake_count,
            "false_authentic_count": len(cases),
            "false_authentic_rate": false_authentic_rate(len(cases), fake_count),
            "p_fake_below_0.10": 0,
            "p_fake_from_0.10_to_0.25": 0,
            "p_fake_from_0.25_to_0.50": 0,
            "hidden": False,
        }
    return report


def _cross_table(rows: list[dict]) -> list[dict]:
    table = []
    for state in ("clean", "low_quality", "degraded"):
        subset = [row for row in rows if row["quality_state"] == state]
        table.append(
            {
                "quality_state": state,
                "predicted_authentic": sum(row["authenticity_prediction"] == "authentic" for row in subset),
                "predicted_fake": sum(row["authenticity_prediction"] == "fake" for row in subset),
                "role": "diagnostic_only",
                "production_verdict": None,
            }
        )
    return table


def _classification_metrics(rows: list[dict]) -> dict:
    authentic = sum(int(row["label"]) == 0 for row in rows)
    fake = sum(int(row["label"]) == 1 for row in rows)
    false_authentic = sum(int(row["label"]) == 1 and row["authenticity_prediction"] == "authentic" for row in rows)
    false_fake = sum(int(row["label"]) == 0 and row["authenticity_prediction"] == "fake" for row in rows)
    true_positive = fake - false_authentic
    true_negative = authentic - false_fake
    total = len(rows)
    precision = None if true_positive + false_fake == 0 else true_positive / (true_positive + false_fake)
    recall = None if fake == 0 else true_positive / fake
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0 and authentic > 0 and fake > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "sample_count": total,
        "authentic_count": authentic,
        "fake_count": fake,
        "accuracy": None if total == 0 else (true_positive + true_negative) / total,
        "f1": f1,
        "false_authentic_count": false_authentic,
        "false_authentic_rate": false_authentic_rate(false_authentic, fake),
        "false_fake_count": false_fake,
        "false_fake_rate": None if authentic == 0 else false_fake / authentic,
    }


def _distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    percentiles = np.percentile(values, [5, 25, 50, 75, 95], method=PERCENTILE_METHOD)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(percentiles[2]),
        "std": float(np.std(values, ddof=STD_DDOF)),
        "p5": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def _finite_deltas(rows: list[dict], name: str) -> list[float]:
    values = []
    for row in rows:
        value = row["delta"].get(name)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _write_artifacts(
    destination: Path,
    *,
    measured: list[dict],
    distribution: dict,
    thresholds: dict,
    review: list[dict],
    summary: dict,
    brands: list[dict],
    ranked: list[dict],
    linked: list[dict],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    definitions = {
        "phase": 23,
        "separates_quality_from_authenticity": True,
        "composite_score": None,
        "luminance": "BT.601 Y = (0.299 R + 0.587 G + 0.114 B) / 255 on the original RGB image",
        "sharpness": "variance of the 4-neighbor Laplacian of luminance",
        "foreground_occupancy": (
            f"fraction of pixels whose luminance differs from the {FOREGROUND_BORDER_FRACTION:.0%} "
            f"border median by more than {FOREGROUND_DELTA}. Not a watch detector."
        ),
        "compression": "file bytes and bytes per pixel only. JPEG quality is not estimated.",
        "clipping": {
            "near_black": NEAR_BLACK_LUMINANCE,
            "near_white": NEAR_WHITE_LUMINANCE,
            "dark_pixel": DARK_LUMINANCE,
            "bright_pixel": BRIGHT_LUMINANCE,
        },
        "features": list(FEATURE_SCHEMA),
        "distribution": distribution,
        "reference_sample_count": len(measured),
    }
    (destination / "quality_features_reference.json").write_text(_dump(definitions) + "\n")
    (destination / "quality_thresholds.json").write_text(_dump(thresholds) + "\n")
    _write_distribution_csv(destination / "quality_reference_stats.csv", distribution)
    _write_feature_csv(destination / "quality_features_nontest.csv", measured)
    _write_review_csv(destination / "quality_review_candidates.csv", review)
    _write_failure_csv(destination / "quality_failure_modes.csv", ranked)
    _write_brand_csv(destination / "quality_brand_summary.csv", brands)
    _write_robustness_csv(destination / "quality_robustness_summary.csv", summary["robustness"])
    _write_linkage_csv(destination / "quality_stress_linkage.csv", linked)
    (destination / "quality_summary.json").write_text(_dump(summary) + "\n")


def _write_distribution_csv(path: Path, distribution: dict) -> None:
    fields = ["feature", "count", "mean", "median", "std", "p5", "p25", "p75", "p95"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature, stats in distribution.items():
            writer.writerow({"feature": feature, **{key: _csv_value(stats[key]) for key in fields if key != "feature"}})


def _write_feature_csv(path: Path, rows: list[dict]) -> None:
    fields = ["sample_id", "source_split", "brand", "label", "quality_state", "quality_flags", *FEATURE_SCHEMA]
    _write_dicts(path, rows, fields)


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "brand",
        "source_split",
        "quality_flags",
        "quality_state",
        "authenticity_probability",
        "authenticity_prediction",
        "probability_source",
        *NUMERIC_FEATURES,
    ]
    _write_dicts(path, rows, fields)


def _write_failure_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "rank",
        "condition",
        "sample_count",
        "prediction_flips",
        "false_authentic_count",
        "false_fake_count",
        "mean_probability_change",
        "maximum_absolute_probability_change",
        "unique_images",
        "unique_flipped_images",
        "model_updated",
    ]
    _write_dicts(path, rows, fields)


def _write_brand_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "brand",
        "sample_count",
        "quality_flag_rate",
        "false_fake_rate",
        "prediction_flip_rate",
        "most_common_quality_issue",
    ]
    _write_dicts(path, rows, fields)


def _write_robustness_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "condition",
        "sample_count",
        "prediction_flips",
        "false_authentic_count",
        "false_fake_count",
        "mean_probability_change",
        "maximum_absolute_probability_change",
        "unique_images",
    ]
    _write_dicts(path, rows, fields)


def _write_linkage_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "source_split",
        "brand",
        "label",
        "condition",
        "prediction_flip",
        "false_authentic",
        "false_fake",
        "original_calibrated_probability",
        "transformed_calibrated_probability",
        "probability_change",
        "flags_before",
        "flags_after",
        "before_min_dimension",
        "after_min_dimension",
        "before_sharpness",
        "after_sharpness",
        "before_mean_luminance",
        "after_mean_luminance",
        "delta_min_dimension",
        "delta_sharpness",
        "delta_mean_luminance",
        "delta_luminance_std",
        "delta_pixel_count",
        "delta_foreground_occupancy",
    ]
    _write_dicts(path, rows, fields)


def _write_dicts(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {}
            for field in fields:
                value = row.get(field)
                if field == "quality_flags" and isinstance(value, list):
                    value = "|".join(value)
                payload[field] = _csv_value(value)
            writer.writerow(payload)


def _csv_value(value):
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return value


def _dump(payload: dict) -> str:
    return json.dumps(_json_ready(payload), indent=2, allow_nan=False)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _assert_quality_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != QUALITY_DIR_NAME:
        raise RuntimeError(f"image-quality output must be named {QUALITY_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | set(SPLIT_NAMES) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        "final_test_v1",
        "final_model_freeze",
        "hard_negative_v1",
        "v2_dinov2_cls_only",
        "v2_dinov2_cls_patch_attention",
        "v2_dinov3_cls_only",
        "v2_dinov3_cls_patch_attention",
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write image quality into {resolved}")
    return resolved


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    manifest = root / "dataset_audit" / "split_manifest_v2.json"
    summary = run_phase23(root, manifest)
    print(
        json.dumps(
            {
                "final_test_samples_used": summary["final_test_samples_used"],
                "reference_sample_count": summary["reference_sample_count"],
                "quality_review_candidate_count": summary["quality_review_candidate_count"],
                "flag_counts": summary["flag_counts"],
                "production_authenticity_threshold": None,
                "temperature": summary["temperature"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
