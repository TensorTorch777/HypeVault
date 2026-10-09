"""Hard-negative mining for the frozen Phase 21 candidate.

Inference and feature extraction only. The final test split is rejected.
Temperature, checkpoint bytes, and the production threshold stay unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import perspective as perspective_transform

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from augmentations import CONSERVATIVE, build_transforms, imagenet_mean_fill
from calibration import probabilities_from_temperature
from calibration_phase20 import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_WORKERS,
    PREPROCESSING_VERSION,
    file_sha256,
)
from checkpoint_selection import validate_checkpoint_artifact
from evaluation import build_eval_model, evaluate_logits, label_from_dataset_path
from split_v2 import (
    AUTHORITATIVE_MEMBERSHIP_HASH,
    PHASE15_EXPERIMENT_DIRS,
    SPLIT_NAMES,
    membership_hash,
)
from training_config import assert_experiment_output_dir

FROZEN_DIRECTORY = "v2_dinov3_cls_patch_attention"
FROZEN_CHECKPOINT_NAME = "epoch_018.pt"
FROZEN_FAMILY = "dinov3"
FROZEN_HEAD = "cls_patch_attention"
FROZEN_EPOCH = 18
FROZEN_INPUT_SIZE = 512
FROZEN_TEMPERATURE = 0.24038200410185356
EVALUATION_THRESHOLD = 0.50
SOURCE_SPLITS = ("train", "calibration", "validation")
CHALLENGE_DIR_NAME = "hard_negative_v1"
LOW_MARGIN_FRACTIONS = (0.01, 0.05, 0.10)
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)
HIGH_CONFIDENCE_FAKE = 0.99
HIGH_CONFIDENCE_AUTHENTIC = 0.01
DEFAULT_LIMITS = {
    "low_margin": 500,
    "high_confidence": 500,
    "same_brand_pairs": 500,
    "cross_brand_pairs": 500,
    "transform_originals": 200,
    "quality_originals": 200,
}
TRANSFORM_SPECS = (
    {"name": "jpeg_recompression", "quality": 85},
    {"name": "gaussian_blur", "radius": 0.4},
    {"name": "brightness_shift", "factor": 1.10},
    {"name": "contrast_shift", "factor": 1.10},
    {"name": "mild_crop", "keep": 0.90},
    {"name": "small_rotation", "degrees": 3.0},
    {"name": "resize_recompression", "scale": 0.85, "quality": 80},
    {"name": "perspective", "shift": 0.02},
)
QUALITY_SPECS = (
    {"name": "lower_resolution", "longest_side": 256},
    {"name": "moderate_jpeg", "quality": 60},
    {"name": "mild_blur", "radius": 0.8},
    {"name": "partial_crop", "keep": 0.75},
    {"name": "darker", "factor": 0.85},
    {"name": "brighter", "factor": 1.15},
)


def false_authentic_rate(false_authentic_count: int, fake_count: int) -> float | None:
    """FAR is false authentic divided by the number of actual fakes."""
    if int(fake_count) <= 0:
        return None
    return int(false_authentic_count) / int(fake_count)


def build_signal_rows(records: list[dict], logits, temperature: float = FROZEN_TEMPERATURE) -> list[dict]:
    """Attach frozen-temperature probabilities. Does not fit a temperature."""
    if float(temperature) != FROZEN_TEMPERATURE:
        raise ValueError("hard-negative mining must use the frozen temperature")
    vector = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    if int(vector.numel()) != len(records):
        raise ValueError("logits and source records differ in length")
    if not torch.isfinite(vector).all():
        raise ValueError("logits must be finite")
    baseline = torch.sigmoid(vector)
    calibrated = probabilities_from_temperature(vector, FROZEN_TEMPERATURE)
    rows = []
    for index, record in enumerate(records):
        if record.get("source_split") not in SOURCE_SPLITS:
            raise RuntimeError("challenge candidate belongs to membership.test")
        probability = float(calibrated[index])
        prediction = 1 if probability >= EVALUATION_THRESHOLD else 0
        label = int(record["label"])
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_split": record["source_split"],
                "path": record.get("path"),
                "brand": record["brand"],
                "label": label,
                "logit": float(vector[index]),
                "baseline_probability": float(baseline[index]),
                "calibrated_probability": probability,
                "confidence": float(max(probability, 1.0 - probability)),
                "predicted_label": prediction,
                "predicted_class": "fake" if prediction == 1 else "authentic",
                "margin": abs(probability - EVALUATION_THRESHOLD),
                "misclassified": prediction != label,
            }
        )
    return rows


def rank_low_margin(rows: list[dict], limit: int | None = None) -> list[dict]:
    """Smallest distance to 0.50 first. A correct row is not marked as an error."""
    ordered = sorted(rows, key=lambda row: (float(row["margin"]), row["sample_id"], row["source_split"]))
    if limit is None:
        return ordered
    return ordered[: max(0, int(limit))]


def low_margin_slices(rows: list[dict], fractions: tuple[float, ...] = LOW_MARGIN_FRACTIONS) -> dict:
    ranked = rank_low_margin(rows)
    report = {}
    total = len(ranked)
    for fraction in fractions:
        count = min(total, max(1, int(math.ceil(total * float(fraction))))) if total else 0
        report[f"top_{int(round(fraction * 100))}_percent"] = distribution(ranked[:count])
    return report


def select_high_confidence(rows: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Correct and extreme probabilities only. The stored slice is class-balanced."""
    eligible = [
        row
        for row in rows
        if not row["misclassified"] and _is_extreme_confidence(row)
    ]
    eligible.sort(key=lambda row: (-float(row["confidence"]), row["sample_id"], row["source_split"]))
    per_class = max(0, int(limit)) // 2
    selected: list[dict] = []
    for label in (0, 1):
        selected.extend([row for row in eligible if int(row["label"]) == label][:per_class])
    chosen = {row["sample_id"] for row in selected}
    if len(selected) < int(limit):
        for row in eligible:
            if row["sample_id"] in chosen:
                continue
            selected.append(row)
            chosen.add(row["sample_id"])
            if len(selected) >= int(limit):
                break
    selected.sort(key=lambda row: (-float(row["confidence"]), int(row["label"]), row["sample_id"]))
    return eligible, selected[: max(0, int(limit))]


def confidence_level_report(rows: list[dict]) -> list[dict]:
    """Analysis bands. These are not production thresholds."""
    report = []
    for level in CONFIDENCE_LEVELS:
        chosen = [row for row in rows if float(row["confidence"]) + 1e-12 >= level]
        false_authentic = sum(row["misclassified"] and int(row["label"]) == 1 for row in chosen)
        false_fake = sum(row["misclassified"] and int(row["label"]) == 0 for row in chosen)
        report.append(
            {
                "minimum_confidence": level,
                "count": len(chosen),
                "authentic_count": sum(int(row["label"]) == 0 for row in chosen),
                "fake_count": sum(int(row["label"]) == 1 for row in chosen),
                "predicted_authentic_count": sum(row["predicted_class"] == "authentic" for row in chosen),
                "predicted_fake_count": sum(row["predicted_class"] == "fake" for row in chosen),
                "error_count": false_authentic + false_fake,
                "false_authentic_count": false_authentic,
                "false_fake_count": false_fake,
                "role": "analysis_only",
                "production_threshold": None,
            }
        )
    return report


def rank_opposite_pairs(
    rows: list[dict],
    embeddings,
    *,
    same_brand: bool,
    limit: int,
) -> list[dict]:
    """Nearest opposite-class pairs by cosine similarity. Authentic is side A."""
    matrix = _l2_normalize(np.asarray(embeddings, dtype=np.float64))
    if matrix.shape[0] != len(rows):
        raise ValueError("embeddings and rows differ in length")
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault((row["brand"], int(row["label"])), []).append(index)
    brands = sorted({row["brand"] for row in rows})
    collected: list[dict] = []
    keep = max(0, int(limit))
    if keep == 0:
        return []
    for brand_a in brands:
        for brand_b in brands:
            matched = brand_a == brand_b
            if same_brand != matched:
                continue
            left = grouped.get((brand_a, 0), [])
            right = grouped.get((brand_b, 1), [])
            collected.extend(_top_block(rows, matrix, left, right, keep))
    collected.sort(key=lambda row: (-float(row["cosine_similarity"]), row["sample_id_a"], row["sample_id_b"]))
    return collected[:keep]


def apply_challenge_transform(image: Image.Image, name: str) -> Image.Image:
    """Deterministic mild transform. The original image is not modified."""
    rgb = image.convert("RGB")
    if name == "jpeg_recompression":
        return _jpeg(rgb, 85)
    if name == "gaussian_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=0.4))
    if name == "brightness_shift":
        return ImageEnhance.Brightness(rgb).enhance(1.10)
    if name == "contrast_shift":
        return ImageEnhance.Contrast(rgb).enhance(1.10)
    if name == "mild_crop":
        return _center_keep(rgb, 0.90)
    if name == "small_rotation":
        return rgb.rotate(3.0, resample=Image.Resampling.BICUBIC, fillcolor=imagenet_mean_fill(), expand=False)
    if name == "resize_recompression":
        return _jpeg(_scale(rgb, 0.85), 80)
    if name == "perspective":
        return _perspective(rgb, 0.02)
    if name == "lower_resolution":
        return _longest_side(rgb, 256)
    if name == "moderate_jpeg":
        return _jpeg(rgb, 60)
    if name == "mild_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=0.8))
    if name == "partial_crop":
        return _center_keep(rgb, 0.75)
    if name == "darker":
        return ImageEnhance.Brightness(rgb).enhance(0.85)
    if name == "brighter":
        return ImageEnhance.Brightness(rgb).enhance(1.15)
    raise ValueError(f"unknown challenge transform {name}")


def transform_measurement(
    original: dict,
    transformed_logit: float,
    condition: str,
    temperature: float = FROZEN_TEMPERATURE,
) -> dict:
    """Compare one transformed logit with its source. Lineage stays on the source sample."""
    if float(temperature) != FROZEN_TEMPERATURE:
        raise ValueError("hard-negative mining must use the frozen temperature")
    if original.get("source_split") not in SOURCE_SPLITS:
        raise RuntimeError("challenge candidate belongs to membership.test")
    original_probability = float(original["calibrated_probability"])
    transformed_probability = float(probabilities_from_temperature(torch.tensor([float(transformed_logit)]), FROZEN_TEMPERATURE)[0])
    original_prediction = 1 if original_probability >= EVALUATION_THRESHOLD else 0
    transformed_prediction = 1 if transformed_probability >= EVALUATION_THRESHOLD else 0
    flipped = int(original_prediction != transformed_prediction)
    original_confidence = float(max(original_probability, 1.0 - original_probability))
    transformed_confidence = float(max(transformed_probability, 1.0 - transformed_probability))
    return {
        "source_sample_id": original["sample_id"],
        "source_split": original["source_split"],
        "brand": original["brand"],
        "label": int(original["label"]),
        "condition": condition,
        "original_logit": float(original["logit"]),
        "transformed_logit": float(transformed_logit),
        "logit_change": float(transformed_logit) - float(original["logit"]),
        "original_calibrated_probability": original_probability,
        "transformed_calibrated_probability": transformed_probability,
        "calibrated_probability": transformed_probability,
        "probability_change": transformed_probability - original_probability,
        "confidence_change": transformed_confidence - original_confidence,
        "prediction_flip": flipped,
        "prediction_stable": int(flipped == 0),
        "logit": float(transformed_logit),
        "predicted_label": transformed_prediction,
        "predicted_class": "fake" if transformed_prediction == 1 else "authentic",
        "misclassified": transformed_prediction != int(original["label"]),
        "confidence": transformed_confidence,
        "sample_id": original["sample_id"],
    }


def condition_report(rows: list[dict]) -> list[dict]:
    names = []
    for row in rows:
        if row["condition"] not in names:
            names.append(row["condition"])
    report = []
    for name in names:
        subset = [row for row in rows if row["condition"] == name]
        fake_count = sum(int(row["label"]) == 1 for row in subset)
        false_authentic = sum(row["misclassified"] and int(row["label"]) == 1 for row in subset)
        changes = [abs(float(row["probability_change"])) for row in subset]
        logit_changes = [abs(float(row["logit_change"])) for row in subset]
        signed = [float(row["probability_change"]) for row in subset]
        report.append(
            {
                "quality_condition": name,
                "condition": name,
                "sample_count": len(subset),
                "fake_count": fake_count,
                "authentic_count": sum(int(row["label"]) == 0 for row in subset),
                "prediction_flip_rate": sum(int(row["prediction_flip"]) for row in subset) / len(subset),
                "mean_probability_shift": float(np.mean(signed)),
                "mean_absolute_probability_change": float(np.mean(changes)),
                "maximum_probability_change": float(np.max(np.abs(signed))),
                "mean_absolute_logit_change": float(np.mean(logit_changes)),
                "false_authentic_rate": false_authentic_rate(false_authentic, fake_count),
                "false_authentic_count": false_authentic,
                "production_threshold": None,
            }
        )
    return report


def distribution(rows: list[dict]) -> dict:
    brands: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = brands.setdefault(row["brand"], {"authentic": 0, "fake": 0})
        bucket["authentic" if int(row["label"]) == 0 else "fake"] += 1
    return {
        "count": len(rows),
        "authentic_count": sum(int(row["label"]) == 0 for row in rows),
        "fake_count": sum(int(row["label"]) == 1 for row in rows),
        "misclassified_count": sum(bool(row.get("misclassified")) for row in rows),
        "brands": brands,
    }


def score_rows(rows: list[dict], temperature: float = FROZEN_TEMPERATURE) -> dict:
    if float(temperature) != FROZEN_TEMPERATURE:
        raise ValueError("hard-negative mining must use the frozen temperature")
    if not rows:
        return {
            "sample_count": 0,
            "authentic_count": 0,
            "fake_count": 0,
            "accuracy": None,
            "f1": None,
            "recall": None,
            "false_authentic_count": 0,
            "false_authentic_rate": None,
            "false_fake_count": 0,
            "false_fake_rate": None,
            "mean_calibrated_probability": None,
            "median_calibrated_probability": None,
        }
    logits = torch.tensor([float(row["logit"]) for row in rows], dtype=torch.float32)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    metrics = evaluate_logits(logits / FROZEN_TEMPERATURE, labels, threshold=EVALUATION_THRESHOLD)
    probabilities = [float(row["calibrated_probability"]) for row in rows]
    return {
        "sample_count": metrics["sample_count"],
        "authentic_count": metrics["authentic_count"],
        "fake_count": metrics["fake_count"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "recall": metrics["recall"],
        "false_authentic_count": metrics["false_authentic"],
        "false_authentic_rate": metrics["false_authentic_rate"],
        "false_fake_count": metrics["false_fake"],
        "false_fake_rate": metrics["false_fake_rate"],
        "mean_calibrated_probability": float(np.mean(probabilities)),
        "median_calibrated_probability": float(np.median(probabilities)),
    }


def brand_reports(rows: list[dict]) -> list[dict]:
    table = []
    for brand in sorted({row["brand"] for row in rows}):
        subset = [row for row in rows if row["brand"] == brand]
        scored = score_rows(subset)
        flips = [row["prediction_flip"] for row in subset if "prediction_flip" in row]
        table.append(
            {
                "brand": brand,
                "challenge_sample_count": len(subset),
                "false_authentic_rate": scored["false_authentic_rate"],
                "f1": scored["f1"],
                "recall": scored["recall"],
                "prediction_flip_rate": None if not flips else sum(int(value) for value in flips) / len(flips),
                "production_threshold": None,
            }
        )
    return table


def worst_brand_far(table: list[dict]) -> dict:
    scored = [row for row in table if row["false_authentic_rate"] is not None]
    if not scored:
        return {"worst_brand_far": None, "brands": []}
    worst = max(float(row["false_authentic_rate"]) for row in scored)
    return {
        "worst_brand_far": worst,
        "brands": [row["brand"] for row in scored if float(row["false_authentic_rate"]) == worst],
    }


def false_authentic_cases(rows: list[dict], category: str) -> list[dict]:
    cases = []
    for row in rows:
        probability = float(row["calibrated_probability"])
        if int(row["label"]) != 1 or probability >= EVALUATION_THRESHOLD:
            continue
        if probability < 0.10:
            severity = "p_fake_below_0.10"
        elif probability < 0.25:
            severity = "p_fake_from_0.10_to_0.25"
        else:
            severity = "p_fake_from_0.25_to_0.50"
        cases.append(
            {
                "category": category,
                "sample_id": row.get("source_sample_id", row["sample_id"]),
                "source_split": row["source_split"],
                "brand": row["brand"],
                "condition": row.get("condition"),
                "calibrated_probability": probability,
                "severity": severity,
                "error_type": "false_authentic",
            }
        )
    cases.sort(key=lambda row: (row["calibrated_probability"], row["sample_id"], row["category"]))
    return cases


def pair_metrics(pairs: list[dict]) -> dict:
    if not pairs:
        return {
            "pair_count": 0,
            "authentic_count": 0,
            "fake_count": 0,
            "accuracy": None,
            "f1": None,
            "false_authentic_rate": None,
            "false_fake_rate": None,
            "mean_cosine_similarity": None,
        }
    false_authentic = sum(float(row["probability_b"]) < EVALUATION_THRESHOLD for row in pairs)
    false_fake = sum(float(row["probability_a"]) >= EVALUATION_THRESHOLD for row in pairs)
    count = len(pairs)
    true_positive = count - false_authentic
    precision = true_positive / (true_positive + false_fake) if (true_positive + false_fake) else None
    recall = true_positive / count
    f1 = None if precision is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    both_correct = sum(
        float(row["probability_a"]) < EVALUATION_THRESHOLD and float(row["probability_b"]) >= EVALUATION_THRESHOLD
        for row in pairs
    )
    return {
        "pair_count": count,
        "authentic_count": count,
        "fake_count": count,
        "accuracy": both_correct / count,
        "f1": f1,
        "recall": recall,
        "false_authentic_count": false_authentic,
        "false_authentic_rate": false_authentic_rate(false_authentic, count),
        "false_fake_count": false_fake,
        "false_fake_rate": false_fake / count,
        "mean_cosine_similarity": float(np.mean([row["cosine_similarity"] for row in pairs])),
        "maximum_cosine_similarity": float(max(row["cosine_similarity"] for row in pairs)),
        "mean_calibrated_probability": float(
            np.mean([row["probability_a"] for row in pairs] + [row["probability_b"] for row in pairs])
        ),
        "median_calibrated_probability": float(
            np.median([row["probability_a"] for row in pairs] + [row["probability_b"] for row in pairs])
        ),
    }


def change_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "sample_count": 0,
            "prediction_flip_rate": None,
            "mean_absolute_logit_change": None,
            "mean_absolute_probability_change": None,
            "maximum_probability_change": None,
        }
    probability_change = np.asarray([abs(float(row["probability_change"])) for row in rows], dtype=np.float64)
    logit_change = np.asarray([abs(float(row["logit_change"])) for row in rows], dtype=np.float64)
    return {
        "sample_count": len(rows),
        "authentic_count": sum(int(row["label"]) == 0 for row in rows),
        "fake_count": sum(int(row["label"]) == 1 for row in rows),
        "prediction_flip_rate": sum(int(row["prediction_flip"]) for row in rows) / len(rows),
        "prediction_stability": sum(int(row["prediction_stable"]) for row in rows) / len(rows),
        "mean_absolute_logit_change": float(logit_change.mean()),
        "mean_absolute_probability_change": float(probability_change.mean()),
        "maximum_probability_change": float(probability_change.max()),
        **score_rows(rows),
    }


def prepare_source_records(
    manifest: dict,
    *,
    enforce_authoritative_hash: bool = False,
    require_files: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Read train, calibration, and validation. A final-test path raises."""
    membership = manifest.get("membership")
    samples = manifest.get("samples")
    if not isinstance(membership, dict) or any(name not in membership for name in SPLIT_NAMES):
        raise ValueError("split manifest is missing train, calibration, validation, or test membership")
    if not isinstance(samples, list):
        raise ValueError("split manifest is missing samples")
    recorded = {name: [str(path) for path in list(membership.get(name) or [])] for name in SPLIT_NAMES}
    if enforce_authoritative_hash:
        digest = membership_hash(recorded)
        if digest != manifest.get("membership_hash") or digest != AUTHORITATIVE_MEMBERSHIP_HASH:
            raise ValueError("membership hash does not match the Phase 16 manifest")
    test_paths = {str(Path(path).resolve()) for path in recorded["test"]}
    by_path = {}
    for row in samples:
        path = row.get("path")
        if isinstance(path, str) and path:
            by_path[str(Path(path).resolve())] = row
    records = []
    rejected = []
    for split in SOURCE_SPLITS:
        for raw_path in recorded[split]:
            path = str(Path(raw_path).resolve())
            if path in test_paths:
                raise RuntimeError("challenge candidate belongs to membership.test")
            row = by_path.get(path)
            reason = _reject_reason(row, split, path, require_files=require_files)
            if reason is not None:
                rejected.append({"path": path, "source_split": split, "reason": reason})
                continue
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "source_split": split,
                    "path": path,
                    "brand": row["brand"],
                    "label": int(row["label"]),
                }
            )
    seen = set()
    unique = []
    for record in records:
        if record["sample_id"] in seen:
            rejected.append(
                {
                    "path": record["path"],
                    "source_split": record["source_split"],
                    "reason": "duplicate sample_id",
                }
            )
            continue
        seen.add(record["sample_id"])
        unique.append(record)
    return unique, rejected


def build_challenge_manifest(
    *,
    low_margin: list[dict],
    high_confidence: list[dict],
    same_brand_pairs: list[dict],
    cross_brand_pairs: list[dict],
    transform_rows: list[dict],
    quality_rows: list[dict],
    checkpoint: str,
) -> dict:
    items = []
    items.extend(_sample_items(low_margin, "low_margin"))
    items.extend(_sample_items(high_confidence, "high_confidence"))
    items.extend(_pair_items(same_brand_pairs, "same_brand_pair"))
    items.extend(_pair_items(cross_brand_pairs, "cross_brand_pair"))
    items.extend(_stress_items(transform_rows, "transform"))
    items.extend(_stress_items(quality_rows, "quality_stress"))
    for item in items:
        if item.get("source_split") == "test" or "test" in str(item.get("source_split_a", "")):
            raise RuntimeError("challenge candidate belongs to membership.test")
        if item.get("source_split_b") == "test":
            raise RuntimeError("challenge candidate belongs to membership.test")
    items.sort(key=_manifest_sort_key)
    return {
        "frozen_checkpoint": checkpoint,
        "frozen_temperature": FROZEN_TEMPERATURE,
        "production_threshold": None,
        "test_samples_used": 0,
        "source_splits": list(SOURCE_SPLITS),
        "final_test_opened": False,
        "items": items,
    }


def manifest_sha256(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def assemble_challenge(
    records: list[dict],
    logits,
    embeddings,
    *,
    checkpoint: str,
    limits: dict | None = None,
    transformed_logits: dict[str, np.ndarray] | None = None,
    quality_logits: dict[str, np.ndarray] | None = None,
) -> dict:
    """Build challenge tables from cached signals. Does not open images or the test split."""
    chosen_limits = dict(DEFAULT_LIMITS)
    if limits:
        chosen_limits.update(limits)
    signals = build_signal_rows(records, logits, FROZEN_TEMPERATURE)
    low_margin = rank_low_margin(signals, int(chosen_limits["low_margin"]))
    high_pool, high_confidence = select_high_confidence(signals, int(chosen_limits["high_confidence"]))
    same_brand = rank_opposite_pairs(
        signals,
        embeddings,
        same_brand=True,
        limit=int(chosen_limits["same_brand_pairs"]),
    )
    cross_brand = rank_opposite_pairs(
        signals,
        embeddings,
        same_brand=False,
        limit=int(chosen_limits["cross_brand_pairs"]),
    )
    difficult = rank_low_margin(signals, int(chosen_limits["transform_originals"]))
    by_id = {row["sample_id"]: row for row in signals}
    originals = [by_id[row["sample_id"]] for row in difficult]
    transform_rows = _measurement_rows(originals, transformed_logits or {}, [spec["name"] for spec in TRANSFORM_SPECS])
    quality_limit = int(chosen_limits["quality_originals"])
    quality_originals = originals[:quality_limit]
    quality_rows = _measurement_rows(quality_originals, quality_logits or {}, [spec["name"] for spec in QUALITY_SPECS])
    manifest = build_challenge_manifest(
        low_margin=low_margin,
        high_confidence=high_confidence,
        same_brand_pairs=same_brand,
        cross_brand_pairs=cross_brand,
        transform_rows=transform_rows,
        quality_rows=quality_rows,
        checkpoint=checkpoint,
    )
    low_brands = brand_reports(low_margin)
    transform_brands = brand_reports(transform_rows)
    quality_brands = brand_reports(quality_rows)
    clean_failures = false_authentic_cases(signals, "non_test_pool")
    stress_failures = false_authentic_cases(transform_rows, "transform") + false_authentic_cases(quality_rows, "quality_stress")
    return {
        "signals": signals,
        "low_margin": low_margin,
        "low_margin_slices": low_margin_slices(signals),
        "high_confidence_pool": high_pool,
        "high_confidence": high_confidence,
        "same_brand_pairs": same_brand,
        "cross_brand_pairs": cross_brand,
        "transform_rows": transform_rows,
        "quality_rows": quality_rows,
        "manifest": manifest,
        "summary_groups": {
            "low_margin": {**score_rows(low_margin), "distribution": distribution(low_margin)},
            "high_confidence": {**score_rows(high_confidence), "pool_count": len(high_pool)},
            "same_brand_pairs": pair_metrics(same_brand),
            "cross_brand_pairs": pair_metrics(cross_brand),
            "transform": change_summary(transform_rows),
            "quality_stress": change_summary(quality_rows),
        },
        "conditions": {
            "transform": condition_report(transform_rows),
            "quality_stress": condition_report(quality_rows),
        },
        "confidence_levels": confidence_level_report(signals),
        "brands": {
            "low_margin": low_brands,
            "transform": transform_brands,
            "quality_stress": quality_brands,
        },
        "worst_brands": {
            "low_margin": worst_brand_far(low_brands),
            "transform": worst_brand_far(transform_brands),
            "quality_stress": worst_brand_far(quality_brands),
        },
        "false_authentic": {
            "non_test_pool": _failure_summary(clean_failures),
            "stress": _failure_summary(stress_failures),
        },
        "production_threshold": None,
        "test_samples_used": 0,
    }


def write_challenge_artifacts(destination_dir: Path, assembled: dict, *, metadata: dict) -> Path:
    destination = _assert_challenge_destination(destination_dir)
    if (destination / "challenge_summary.json").is_file():
        raise RuntimeError("hard-negative challenge already completed; refusing to rerun")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = assembled["manifest"]
    (destination / "challenge_manifest.json").write_text(_dump(manifest) + "\n")
    _write_rows(
        destination / "low_margin.csv",
        assembled["low_margin"],
        (
            "sample_id",
            "source_split",
            "brand",
            "label",
            "logit",
            "baseline_probability",
            "calibrated_probability",
            "confidence",
            "predicted_class",
            "margin",
            "misclassified",
        ),
    )
    _write_rows(
        destination / "high_confidence.csv",
        assembled["high_confidence"],
        (
            "sample_id",
            "source_split",
            "brand",
            "label",
            "calibrated_probability",
            "confidence",
            "predicted_class",
        ),
    )
    pair_fields = (
        "sample_id_a",
        "sample_id_b",
        "source_split_a",
        "source_split_b",
        "brand_a",
        "brand_b",
        "label_a",
        "label_b",
        "cosine_similarity",
        "probability_a",
        "probability_b",
    )
    _write_rows(destination / "same_brand_hard_negative_pairs.csv", assembled["same_brand_pairs"], pair_fields)
    _write_rows(destination / "cross_brand_hard_negative_pairs.csv", assembled["cross_brand_pairs"], pair_fields)
    stress_fields = (
        "source_sample_id",
        "source_split",
        "brand",
        "label",
        "condition",
        "original_logit",
        "transformed_logit",
        "logit_change",
        "original_calibrated_probability",
        "transformed_calibrated_probability",
        "probability_change",
        "confidence_change",
        "prediction_flip",
        "prediction_stable",
    )
    _write_rows(destination / "transform_stability.csv", assembled["transform_rows"], stress_fields)
    _write_rows(destination / "quality_stress.csv", assembled["quality_rows"], stress_fields)
    summary = {
        "phase": 22,
        "production_threshold": None,
        "temperature": FROZEN_TEMPERATURE,
        "temperature_changed": False,
        "threshold_selected": False,
        "training_performed": False,
        "ood_used": False,
        "final_test_opened": False,
        "test_samples_used": 0,
        "checkpoint_changed": False,
        "low_margin_note": "Low-margin rows are nearest the 0.50 boundary. They are errors only when misclassified is true.",
        "transform_parameters": list(TRANSFORM_SPECS),
        "quality_parameters": list(QUALITY_SPECS),
        "groups": assembled["summary_groups"],
        "low_margin_slices": assembled["low_margin_slices"],
        "conditions": assembled["conditions"],
        "confidence_levels": assembled["confidence_levels"],
        "brands": assembled["brands"],
        "worst_brands": assembled["worst_brands"],
        "false_authentic": assembled["false_authentic"],
        "high_confidence_pool_count": len(assembled["high_confidence_pool"]),
        "high_confidence_stored_count": len(assembled["high_confidence"]),
        "challenge_manifest_sha256": manifest_sha256(manifest),
        **metadata,
    }
    summary.pop("signals", None)
    _write_plots(destination, assembled)
    (destination / "challenge_summary.json").write_text(_dump(summary) + "\n")
    return destination


def run_phase22(
    experiments_root: Path,
    manifest_path: Path,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
    num_workers: int = INFERENCE_WORKERS,
    device: str | None = None,
) -> dict:
    """Mine train, calibration, and validation once. Refuses a second completed run."""
    root = Path(experiments_root)
    destination = _assert_challenge_destination(root / CHALLENGE_DIR_NAME)
    if (destination / "challenge_summary.json").is_file():
        raise RuntimeError("hard-negative challenge already completed; refusing to rerun")
    manifest_file = Path(manifest_path)
    checkpoint = root / FROZEN_DIRECTORY / FROZEN_CHECKPOINT_NAME
    freeze_path = root / "final_model_freeze" / "model_selection_freeze.json"
    final_test_summary = root / "final_test_v1" / "final_test_summary.json"
    _assert_freeze(freeze_path, checkpoint)
    protected = {
        "checkpoint": file_sha256(checkpoint),
        "manifest": file_sha256(manifest_file),
        "freeze": file_sha256(freeze_path),
        "final_test_summary": file_sha256(final_test_summary),
    }
    manifest = json.loads(manifest_file.read_text())
    records, rejected = prepare_source_records(
        manifest,
        enforce_authoritative_hash=True,
        require_files=True,
    )
    test_paths = {str(Path(path).resolve()) for path in manifest["membership"]["test"]}
    preflight = validate_checkpoint_artifact(
        checkpoint,
        expected_family=FROZEN_FAMILY,
        expected_classifier_arch=FROZEN_HEAD,
        expected_epoch=FROZEN_EPOCH,
        expected_input_size=FROZEN_INPUT_SIZE,
        expected_membership_hash=AUTHORITATIVE_MEMBERSHIP_HASH,
        expected_preprocessing_version=PREPROCESSING_VERSION,
        expected_experiment_dir=root / FROZEN_DIRECTORY,
    )
    if not preflight["checkpoint_load_valid"]:
        raise RuntimeError("frozen checkpoint failed preflight: " + "; ".join(preflight["reasons"]))
    print(f"hard-negative sources {len(records)} rejected {len(rejected)}", flush=True)
    inferred = collect_challenge_inference(
        records,
        checkpoint,
        test_paths=test_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        transform_limit=int(DEFAULT_LIMITS["transform_originals"]),
        quality_limit=int(DEFAULT_LIMITS["quality_originals"]),
    )
    assembled = assemble_challenge(
        records,
        inferred["logits"],
        inferred["embeddings"],
        checkpoint=str(checkpoint),
        transformed_logits=inferred["transformed_logits"],
        quality_logits=inferred["quality_logits"],
    )
    if file_sha256(checkpoint) != protected["checkpoint"]:
        raise RuntimeError("hard-negative mining changed checkpoint bytes")
    if file_sha256(manifest_file) != protected["manifest"]:
        raise RuntimeError("hard-negative mining changed the split manifest")
    if file_sha256(freeze_path) != protected["freeze"] or file_sha256(final_test_summary) != protected["final_test_summary"]:
        raise RuntimeError("hard-negative mining changed the freeze or the final test")
    metadata = {
        "checkpoint_sha256": protected["checkpoint"],
        "split_manifest_sha256": protected["manifest"],
        "freeze_sha256": protected["freeze"],
        "source_counts": _source_counts(records),
        "rejected_count": len(rejected),
        "rejected_reasons": _reason_counts(rejected),
        "non_test_pool": score_rows(assembled["signals"]),
        "non_test_distribution": distribution(assembled["signals"]),
    }
    write_challenge_artifacts(destination, assembled, metadata=metadata)
    return json.loads((destination / "challenge_summary.json").read_text())


def collect_challenge_inference(
    records: list[dict],
    checkpoint: Path,
    *,
    test_paths: set[str],
    batch_size: int,
    num_workers: int,
    device: str | None,
    transform_limit: int,
    quality_limit: int,
) -> dict:
    """One eval forward for source images, then mild stresses on the low-margin subset."""
    if any(record["source_split"] not in SOURCE_SPLITS for record in records):
        raise RuntimeError("challenge candidate belongs to membership.test")
    if any(str(Path(record["path"]).resolve()) in test_paths for record in records):
        raise RuntimeError("challenge candidate belongs to membership.test")
    chosen = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _image_size = build_eval_model(FROZEN_FAMILY, FROZEN_HEAD, Path(checkpoint), chosen)
    before = _parameter_digest(model)
    transform = build_transforms(FROZEN_FAMILY, train=False, profile=CONSERVATIVE)
    logits, embeddings = _forward_records(
        model,
        records,
        transform,
        test_paths,
        chosen,
        batch_size=batch_size,
        num_workers=num_workers,
        label="source",
    )
    difficult_ids = [row["sample_id"] for row in rank_low_margin(build_signal_rows(records, logits), transform_limit)]
    by_id = {record["sample_id"]: record for record in records}
    originals = [by_id[sample_id] for sample_id in difficult_ids]
    quality_originals = originals[:quality_limit]
    transformed = {
        spec["name"]: _forward_records(
            model,
            originals,
            transform,
            test_paths,
            chosen,
            batch_size=batch_size,
            num_workers=num_workers,
            label=spec["name"],
            pil_name=spec["name"],
        )[0]
        for spec in TRANSFORM_SPECS
    }
    quality = {
        spec["name"]: _forward_records(
            model,
            quality_originals,
            transform,
            test_paths,
            chosen,
            batch_size=batch_size,
            num_workers=num_workers,
            label=spec["name"],
            pil_name=spec["name"],
        )[0]
        for spec in QUALITY_SPECS
    }
    if _parameter_digest(model) != before:
        raise RuntimeError("inference changed frozen model parameters")
    del model
    if chosen.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "logits": logits,
        "embeddings": embeddings,
        "transformed_logits": transformed,
        "quality_logits": quality,
        "original_ids": difficult_ids,
    }


def _forward_records(
    model,
    records: list[dict],
    eval_transform,
    test_paths: set[str],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    label: str,
    pil_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    loader = DataLoader(
        _ChallengeImages(records, eval_transform, test_paths, pil_name),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    model.eval()
    logit_batches = []
    embedding_batches = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    total = len(loader)
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    features = model.backbone.forward_features(images)
                    cls_token = model.backbone.get_cls_token(features)
                    output = _logits_from_features(model, features, cls_token)
            else:
                features = model.backbone.forward_features(images)
                cls_token = model.backbone.get_cls_token(features)
                output = _logits_from_features(model, features, cls_token)
            logit_batches.append(output.detach().float().cpu().reshape(-1))
            embedding_batches.append(cls_token.detach().float().cpu())
            if batch_index == 1 or batch_index == total or batch_index % 50 == 0:
                print(f"  {label} batch {batch_index}/{total}", flush=True)
    return torch.cat(logit_batches).numpy(), torch.cat(embedding_batches).numpy()


def _logits_from_features(model, features: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    if model.classifier_arch == "cls_only":
        hidden = cls_token
    else:
        pooled, _weights = model.attention_pool(model.backbone.get_patch_tokens(features))
        hidden = model.fusion(torch.cat((cls_token, pooled), dim=-1))
    return model.trunk(hidden)


def _measurement_rows(originals: list[dict], logits_by_condition: dict[str, np.ndarray], names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        if name not in logits_by_condition:
            continue
        logits = np.asarray(logits_by_condition[name], dtype=np.float64).reshape(-1)
        if logits.shape[0] != len(originals):
            raise ValueError(f"{name} logits do not match the original subset")
        for index, original in enumerate(originals):
            rows.append(transform_measurement(original, float(logits[index]), name))
    return rows


def _top_block(rows: list[dict], matrix: np.ndarray, left: list[int], right: list[int], limit: int) -> list[dict]:
    if not left or not right:
        return []
    similarity = matrix[left] @ matrix[right].T
    flat = similarity.reshape(-1)
    keep = min(int(limit), int(flat.size))
    chosen = np.argpartition(flat, -keep)[-keep:]
    chosen = chosen[np.argsort(flat[chosen])[::-1]]
    width = len(right)
    pairs = []
    for flat_index in chosen.tolist():
        left_index = left[flat_index // width]
        right_index = right[flat_index % width]
        left_row = rows[left_index]
        right_row = rows[right_index]
        pairs.append(
            {
                "sample_id_a": left_row["sample_id"],
                "sample_id_b": right_row["sample_id"],
                "source_split_a": left_row["source_split"],
                "source_split_b": right_row["source_split"],
                "brand_a": left_row["brand"],
                "brand_b": right_row["brand"],
                "label_a": int(left_row["label"]),
                "label_b": int(right_row["label"]),
                "cosine_similarity": float(flat[flat_index]),
                "probability_a": float(left_row["calibrated_probability"]),
                "probability_b": float(right_row["calibrated_probability"]),
            }
        )
    return pairs


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _is_extreme_confidence(row: dict) -> bool:
    probability = float(row["calibrated_probability"])
    if int(row["label"]) == 1:
        return probability >= HIGH_CONFIDENCE_FAKE
    return probability <= HIGH_CONFIDENCE_AUTHENTIC


def _reject_reason(row: dict | None, split: str, path: str, *, require_files: bool) -> str | None:
    if row is None:
        return "missing sample"
    if row.get("split") != split:
        return "split mismatch"
    label = row.get("label")
    try:
        path_label = label_from_dataset_path(Path(path))
    except ValueError:
        path_label = None
    if label not in (0, 1) or path_label != label:
        return "invalid label"
    if not isinstance(row.get("brand"), str) or not row.get("brand"):
        return "missing brand"
    if not isinstance(row.get("sample_id"), str) or not row.get("sample_id"):
        return "missing sample_id"
    if require_files and not Path(path).is_file():
        return "missing image file"
    return None


def _sample_items(rows: list[dict], category: str) -> list[dict]:
    return [
        {
            "challenge_category": category,
            "source_split": row["source_split"],
            "sample_id": row["sample_id"],
            "brand": row["brand"],
            "label": int(row["label"]),
            "model_probability": float(row["calibrated_probability"]),
            "confidence": float(row["confidence"]),
            "embedding_similarity": None,
        }
        for row in rows
    ]


def _pair_items(pairs: list[dict], category: str) -> list[dict]:
    return [
        {
            "challenge_category": category,
            "source_split": row["source_split_a"],
            "source_split_a": row["source_split_a"],
            "source_split_b": row["source_split_b"],
            "sample_id": row["sample_id_a"],
            "paired_sample_id": row["sample_id_b"],
            "brand": row["brand_a"],
            "paired_brand": row["brand_b"],
            "label": int(row["label_a"]),
            "paired_label": int(row["label_b"]),
            "model_probability": float(row["probability_a"]),
            "paired_model_probability": float(row["probability_b"]),
            "confidence": float(max(row["probability_a"], 1.0 - row["probability_a"])),
            "embedding_similarity": float(row["cosine_similarity"]),
        }
        for row in pairs
    ]


def _stress_items(rows: list[dict], category: str) -> list[dict]:
    return [
        {
            "challenge_category": category,
            "source_split": row["source_split"],
            "sample_id": row["source_sample_id"],
            "brand": row["brand"],
            "label": int(row["label"]),
            "condition": row["condition"],
            "model_probability": float(row["transformed_calibrated_probability"]),
            "confidence": float(row["confidence"]),
            "embedding_similarity": None,
            "prediction_flip": int(row["prediction_flip"]),
        }
        for row in rows
    ]


def _manifest_sort_key(item: dict) -> tuple:
    return (
        item["challenge_category"],
        item.get("condition") or "",
        item["sample_id"],
        item.get("paired_sample_id") or "",
        item.get("source_split") or "",
    )


def _failure_summary(cases: list[dict]) -> dict:
    return {
        "count": len(cases),
        "p_fake_below_0.10": sum(row["severity"] == "p_fake_below_0.10" for row in cases),
        "p_fake_from_0.10_to_0.25": sum(row["severity"] == "p_fake_from_0.10_to_0.25" for row in cases),
        "p_fake_from_0.25_to_0.50": sum(row["severity"] == "p_fake_from_0.25_to_0.50" for row in cases),
        "cases": cases[:200],
        "production_threshold": None,
    }


def _source_counts(records: list[dict]) -> dict:
    return {
        split: {
            "count": sum(record["source_split"] == split for record in records),
            "authentic": sum(record["source_split"] == split and record["label"] == 0 for record in records),
            "fake": sum(record["source_split"] == split and record["label"] == 1 for record in records),
        }
        for split in SOURCE_SPLITS
    }


def _reason_counts(rejected: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rejected:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return counts


def _assert_freeze(freeze_path: Path, checkpoint: Path) -> None:
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze.get("selected_temperature") != FROZEN_TEMPERATURE:
        raise RuntimeError("freeze temperature does not match the Phase 21 temperature")
    if Path(freeze.get("selected_checkpoint", "")).resolve() != Path(checkpoint).resolve():
        raise RuntimeError("freeze checkpoint does not match the Phase 21 checkpoint")
    if freeze.get("production_threshold") is not None:
        raise RuntimeError("freeze record contains a production threshold")
    if freeze.get("final_test_opened_at_selection") is not False:
        raise RuntimeError("freeze record does not show that selection preceded the test")


def _parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        array = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def _assert_challenge_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != CHALLENGE_DIR_NAME:
        raise RuntimeError(f"hard-negative output must be named {CHALLENGE_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        "final_test_v1",
        "final_model_freeze",
        "v2_dinov2_cls_only",
        "v2_dinov2_cls_patch_attention",
        "v2_dinov3_cls_only",
        "v2_dinov3_cls_patch_attention",
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the challenge set into {resolved}")
    return resolved


def _write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


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


def _write_plots(destination: Path, assembled: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = assembled["signals"]
    authentic = [row["calibrated_probability"] for row in signals if int(row["label"]) == 0]
    fake = [row["calibrated_probability"] for row in signals if int(row["label"]) == 1]
    figure, axis = plt.subplots(figsize=(8, 4))
    if authentic:
        axis.hist(authentic, bins=40, alpha=0.7, label="authentic")
    if fake:
        axis.hist(fake, bins=40, alpha=0.7, label="fake")
    axis.set_xlabel("calibrated P(fake)")
    axis.set_ylabel("samples")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "probability_distribution.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist([row["calibrated_probability"] for row in assembled["low_margin"]], bins=40)
    axis.set_xlabel("calibrated P(fake)")
    axis.set_ylabel("low-margin samples")
    figure.tight_layout()
    figure.savefig(destination / "low_margin_histogram.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8, 6))
    for axis, key, title in (
        (axes[0], "transform", "Transformation probability drift"),
        (axes[1], "quality_stress", "Quality-stress probability drift"),
    ):
        conditions = assembled["conditions"][key]
        axis.bar(
            [row["condition"] for row in conditions],
            [row["mean_absolute_probability_change"] for row in conditions],
        )
        axis.tick_params(axis="x", labelrotation=30)
        axis.set_ylabel("mean |ΔP|")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination / "transform_probability_drift.png", dpi=120)
    plt.close(figure)

    brands = assembled["brands"]["low_margin"]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        [row["brand"] for row in brands],
        [0.0 if row["false_authentic_rate"] is None else row["false_authentic_rate"] for row in brands],
    )
    axis.tick_params(axis="x", labelrotation=20)
    axis.set_ylabel("FAR")
    figure.tight_layout()
    figure.savefig(destination / "brand_far.png", dpi=120)
    plt.close(figure)

    groups = assembled["summary_groups"]
    names = ["low_margin", "high_confidence", "transform", "quality_stress"]
    rates = []
    for name in names:
        block = groups[name]
        count = block.get("sample_count") or 0
        false_authentic = block.get("false_authentic_count") or 0
        false_fake = block.get("false_fake_count") or 0
        rates.append(0.0 if not count else (false_authentic + false_fake) / count)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(names, rates)
    axis.set_ylabel("error rate")
    figure.tight_layout()
    figure.savefig(destination / "category_error_rate.png", dpi=120)
    plt.close(figure)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def _center_keep(image: Image.Image, keep: float) -> Image.Image:
    width, height = image.size
    kept_w = max(1, int(round(width * keep)))
    kept_h = max(1, int(round(height * keep)))
    left = (width - kept_w) // 2
    top = (height - kept_h) // 2
    return image.crop((left, top, left + kept_w, top + kept_h))


def _scale(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.Resampling.BICUBIC)


def _longest_side(image: Image.Image, longest: int) -> Image.Image:
    width, height = image.size
    scale = float(longest) / float(max(width, height))
    if scale >= 1.0:
        scale = 0.5
    return _scale(image, scale)


def _perspective(image: Image.Image, shift: float) -> Image.Image:
    width, height = image.size
    dx = max(1, int(round(shift * width)))
    dy = max(1, int(round(shift * height)))
    start = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    end = [[dx, dy], [width - 1 - dx, dy], [width - 1, height - 1], [0, height - 1 - dy]]
    return perspective_transform(
        image,
        startpoints=start,
        endpoints=end,
        interpolation=InterpolationMode.BICUBIC,
        fill=imagenet_mean_fill(),
    )


class _ChallengeImages(Dataset):
    def __init__(self, records: list[dict], eval_transform, test_paths: set[str], pil_name: str | None) -> None:
        self.records = records
        self.eval_transform = eval_transform
        self.test_paths = test_paths
        self.pil_name = pil_name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = str(Path(record["path"]).resolve())
        if path in self.test_paths or record["source_split"] not in SOURCE_SPLITS:
            raise RuntimeError("challenge candidate belongs to membership.test")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if self.pil_name is not None:
            image = apply_challenge_transform(image, self.pil_name)
        return self.eval_transform(image)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    manifest = root / "dataset_audit" / "split_manifest_v2.json"
    summary = run_phase22(root, manifest)
    print(
        json.dumps(
            {
                "test_samples_used": summary["test_samples_used"],
                "production_threshold": summary["production_threshold"],
                "temperature": summary["temperature"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "challenge_manifest_sha256": summary["challenge_manifest_sha256"],
                "low_margin": summary["groups"]["low_margin"]["sample_count"],
                "same_brand_pairs": summary["groups"]["same_brand_pairs"]["pair_count"],
                "cross_brand_pairs": summary["groups"]["cross_brand_pairs"]["pair_count"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
