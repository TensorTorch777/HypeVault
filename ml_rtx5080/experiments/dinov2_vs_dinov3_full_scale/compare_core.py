"""Frozen paired-comparison rules for the live DINOv2 and experimental DINOv3 checkpoints.

Membership, decisions, and metrics live here so they can be tested without opening
the watch corpus or the locked final test. This module does not train, write
weights, or change a threshold.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

PROTOCOL_NAME = "dinov2_vs_dinov3_full_scale_v1"
DINOV2_ID = "dinov2_legacy"
DINOV3_ID = "dinov3_experimental"
DINOV2_SHA256 = "fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66"
DINOV3_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
DINOV2_PREPROCESS = "legacy_square_resize_504_imagenet"
DINOV3_PREPROCESS = "resize_pad_square_eval_v1"
DINOV2_FAKE_THRESHOLD = 0.5
DINOV2_MIN_AUTHENTIC = 0.88
DINOV3_TEMPERATURE = 0.24038200410185356
DINOV3_THRESHOLD = 0.5
LOGIT_ATOL = 1e-4
PRIMARY_COUNT = 315
PRIMARY_AUTHENTIC = 315
PRIMARY_FAKE = 0
PRIMARY_BRAND = "A. Lange & Söhne"
CORPUS_COUNT = 30000
LOCKED_TEST_COUNT = 3006
SWEEP_COUNT = 26994
POSITIVE_LABEL = 1


class ComparisonGuardError(RuntimeError):
    """A stop condition from the frozen protocol."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ComparisonGuardError(f"{label} hash {actual} != {expected}")
    return actual


def sigmoid(logit: float) -> float:
    value = float(logit)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def dinov2_decision(logit: float) -> tuple[str, float]:
    """Native live policy. REVIEW is not produced. The sigmoid is uncalibrated."""
    if not math.isfinite(float(logit)):
        raise ComparisonGuardError("DINOv2 logit is non-finite")
    fake_score = sigmoid(logit)
    if fake_score >= DINOV2_FAKE_THRESHOLD:
        return "FAKE", fake_score
    authentic_confidence = 1.0 - fake_score
    if authentic_confidence < DINOV2_MIN_AUTHENTIC:
        return "FAKE", fake_score
    return "AUTHENTIC", fake_score


def dinov3_fake_score(logit: float) -> float:
    """Temperature-scaled sigmoid. Not on the DINOv2 score scale."""
    if not math.isfinite(float(logit)):
        raise ComparisonGuardError("DINOv3 logit is non-finite")
    return sigmoid(float(logit) / DINOV3_TEMPERATURE)


def dinov3_decision(logit: float, quality_flags: tuple[str, ...] | list[str]) -> tuple[str, float]:
    """shadow_v1: quality review, then the frozen threshold. OOD stays unavailable."""
    from policy import AUTHENTICITY_THRESHOLD, FROZEN_TEMPERATURE, evaluate_shadow_policy

    if float(FROZEN_TEMPERATURE) != DINOV3_TEMPERATURE or float(AUTHENTICITY_THRESHOLD) != DINOV3_THRESHOLD:
        raise ComparisonGuardError("DINOv3 policy constants drifted from the frozen protocol")
    score = dinov3_fake_score(logit)
    decision = evaluate_shadow_policy(probability=score, quality_flags=quality_flags)
    label = decision["decision"]
    if label not in {"AUTHENTIC", "FAKE", "REVIEW"}:
        raise ComparisonGuardError(f"DINOv3 decision {label!r} is outside shadow_v1")
    return label, score


def assign_role(
    *,
    parent_folder: str,
    resolved_path: str,
    dinov2_train_folders: set[str],
    dinov2_val_folders: set[str],
    dinov3_split: str,
    blocked: set[str],
) -> str:
    """Return excluded, primary_validation, or diagnostic. Does not open a file."""
    if resolved_path in blocked or dinov3_split == "test":
        return "excluded"
    if parent_folder in dinov2_val_folders:
        dinov2_split = "validation"
    elif parent_folder in dinov2_train_folders:
        dinov2_split = "train"
    else:
        dinov2_split = "outside_manifest"
    dinov2_held_out = dinov2_split == "validation"
    dinov3_held_out = dinov3_split == "validation"
    dinov2_fit = dinov2_split == "train"
    dinov3_fit = dinov3_split in {"train", "calibration"}
    if dinov2_held_out and dinov3_held_out and not dinov2_fit and not dinov3_fit:
        return "primary_validation"
    return "diagnostic"


def select_catalog(records: list[dict], blocked: set[str]) -> dict[str, list[dict]]:
    """Assign every catalog row. Duplicate sample ids are a stop condition.

    Each record needs sample_id, resolved_path, parent_folder, label, brand,
    dinov2_train_folders/val are passed inside the record as dinov2_split_hint
    already resolved to a split name, plus dinov3_split. The caller resolves
    folders so this function stays deterministic.
    """
    seen: set[str] = set()
    grouped = {"primary_validation": [], "diagnostic": [], "excluded": []}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ComparisonGuardError(f"duplicate sample id {sample_id}")
        seen.add(sample_id)
        if int(record["label"]) not in {0, 1}:
            raise ComparisonGuardError(f"{sample_id} has no explicit directory label")
        role = record["role"]
        if role not in grouped:
            raise ComparisonGuardError(f"unknown role {role}")
        if record["resolved_path"] in blocked and role != "excluded":
            raise ComparisonGuardError("a blocked path was assigned to the sweep")
        grouped[role].append(record)
    for name in grouped:
        grouped[name] = sorted(grouped[name], key=lambda row: row["sample_id"])
    return grouped


def assert_primary_cohort(rows: list[dict]) -> None:
    brands = {row["brand"] for row in rows}
    authentic = sum(1 for row in rows if int(row["label"]) == 0)
    fake = sum(1 for row in rows if int(row["label"]) == 1)
    if len(rows) != PRIMARY_COUNT or authentic != PRIMARY_AUTHENTIC or fake != PRIMARY_FAKE:
        raise ComparisonGuardError(
            f"primary cohort {len(rows)} authentic={authentic} fake={fake} does not match the protocol"
        )
    if brands != {PRIMARY_BRAND}:
        raise ComparisonGuardError(f"primary brands {sorted(brands)} are not {[PRIMARY_BRAND]}")


def open_if_allowed(path: str, blocked: set[str], opener):
    """Reject a locked path before the opener sees it."""
    resolved = str(Path(path).resolve())
    if resolved in blocked:
        raise ComparisonGuardError("refusing to open a locked final-test or excluded OOD image")
    return opener(path)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == POSITIVE_LABEL]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label != POSITIVE_LABEL]
    if not positives or not negatives:
        return None
    correct = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                correct += 1.0
            elif positive == negative:
                correct += 0.5
    return correct / (len(positives) * len(negatives))


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: (-item[0], item[1]))
    total_positive = sum(1 for label in labels if label == POSITIVE_LABEL)
    if total_positive == 0 or total_positive == len(labels):
        return None
    hit = 0
    area = 0.0
    for rank, (_score, label) in enumerate(pairs, start=1):
        if label == POSITIVE_LABEL:
            hit += 1
            area += hit / rank
    return area / total_positive


def score_rows(rows: list[dict]) -> dict:
    """Aggregate one model. Failed rows stay out of denominators and are counted."""
    if len({row["model_id"] for row in rows}) > 1:
        raise ComparisonGuardError("score_rows received more than one model")
    ok = [row for row in rows if row.get("error") in {None, ""}]
    failed = [row for row in rows if row.get("error") not in {None, ""}]
    review = [row for row in ok if row["decision"] == "REVIEW"]
    binary = [row for row in ok if row["decision"] in {"AUTHENTIC", "FAKE"}]
    for row in ok:
        if row["decision"] not in {"AUTHENTIC", "FAKE", "REVIEW"}:
            raise ComparisonGuardError(f"decision {row['decision']!r} is not a native outcome")
        if row["model_id"] == DINOV2_ID and row["decision"] == "REVIEW":
            raise ComparisonGuardError("DINOv2 produced REVIEW")
    matrix = {
        "authentic_labeled": {"AUTHENTIC": 0, "FAKE": 0, "REVIEW": 0},
        "fake_labeled": {"AUTHENTIC": 0, "FAKE": 0, "REVIEW": 0},
    }
    for row in ok:
        key = "fake_labeled" if int(row["label"]) == POSITIVE_LABEL else "authentic_labeled"
        matrix[key][row["decision"]] += 1
    authentic_n = sum(matrix["authentic_labeled"].values())
    fake_n = sum(matrix["fake_labeled"].values())
    true_positive = matrix["fake_labeled"]["FAKE"]
    false_positive = matrix["authentic_labeled"]["FAKE"]
    true_negative = matrix["authentic_labeled"]["AUTHENTIC"]
    false_negative = matrix["fake_labeled"]["AUTHENTIC"]
    binary_n = len(binary)
    correct = true_positive + true_negative
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    true_positive_rate = _safe_div(true_positive, true_positive + false_negative)
    true_negative_rate = _safe_div(true_negative, true_negative + false_positive)
    balanced = None
    if true_positive_rate is not None and true_negative_rate is not None:
        balanced = (true_positive_rate + true_negative_rate) / 2
    labels = [int(row["label"]) for row in ok]
    scores = [float(row["raw_logit"]) for row in ok]
    return {
        "rows": len(rows),
        "successful": len(ok),
        "failed": len(failed),
        "authentic_labeled": authentic_n,
        "fake_labeled": fake_n,
        "confusion": matrix,
        "review_count": len(review),
        "review_rate": _safe_div(len(review), len(ok)),
        "decision_coverage": _safe_div(binary_n, len(ok)),
        "selective_accuracy": _safe_div(correct, binary_n),
        "accuracy": _safe_div(correct, binary_n),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "balanced_accuracy": balanced,
        "false_authentic_count": matrix["fake_labeled"]["AUTHENTIC"],
        "false_authentic_rate": _safe_div(matrix["fake_labeled"]["AUTHENTIC"], fake_n),
        "false_fake_count": matrix["authentic_labeled"]["FAKE"],
        "false_fake_rate": _safe_div(matrix["authentic_labeled"]["FAKE"], authentic_n),
        "roc_auc_raw_logit": roc_auc(labels, scores),
        "pr_auc_raw_logit": average_precision(labels, scores),
        "binary_denominator": binary_n,
        "review_excluded_from_binary_metrics": True,
        "directory_labels_are_verified_authenticity": False,
    }


def paired_agreement(left: list[dict], right: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in right}
    if len(by_id) != len(right):
        raise ComparisonGuardError("duplicate sample ids in a paired side")
    both_ok = 0
    agree = 0
    disagree = 0
    both_wrong = 0
    for row in left:
        other = by_id.get(row["sample_id"])
        if other is None:
            raise ComparisonGuardError(f"missing paired row for {row['sample_id']}")
        if row.get("error") or other.get("error"):
            continue
        both_ok += 1
        if row["decision"] == other["decision"]:
            agree += 1
        else:
            disagree += 1
        if (
            row["decision"] in {"AUTHENTIC", "FAKE"}
            and other["decision"] in {"AUTHENTIC", "FAKE"}
            and (row["decision"] == "FAKE") != (int(row["label"]) == POSITIVE_LABEL)
            and (other["decision"] == "FAKE") != (int(other["label"]) == POSITIVE_LABEL)
        ):
            both_wrong += 1
    if {row["sample_id"] for row in left} != set(by_id):
        raise ComparisonGuardError("paired outputs do not cover the same samples")
    return {
        "samples": len(left),
        "both_successful": both_ok,
        "agreement": agree,
        "disagreement": disagree,
        "both_binary_and_both_wrong": both_wrong,
    }


def assert_complete_pairs(rows: list[dict], sample_ids: list[str]) -> None:
    expected = set(sample_ids)
    found: dict[str, set[str]] = {sample_id: set() for sample_id in sample_ids}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in expected:
            raise ComparisonGuardError("prediction outside the selected sample list")
        model_id = row["model_id"]
        if model_id not in {DINOV2_ID, DINOV3_ID}:
            raise ComparisonGuardError(f"unknown model {model_id}")
        if model_id in found[sample_id]:
            raise ComparisonGuardError(f"duplicate {model_id} row for {sample_id}")
        found[sample_id].add(model_id)
    missing = [sample_id for sample_id, models in found.items() if models != {DINOV2_ID, DINOV3_ID}]
    if missing:
        raise ComparisonGuardError(f"{len(missing)} samples are missing a model result")


def latency_summary(times_ms: list[float], infer_seconds: float, successful: int) -> dict:
    return {
        "images": len(times_ms),
        "p50_ms": percentile(times_ms, 50),
        "p95_ms": percentile(times_ms, 95),
        "throughput_images_per_s": None if infer_seconds <= 0 else successful / infer_seconds,
        "triton_seconds": infer_seconds,
    }


def brand_breakdown(rows: list[dict]) -> list[dict]:
    brands = sorted({row["brand"] for row in rows})
    report = []
    for brand in brands:
        subset = [row for row in rows if row["brand"] == brand]
        metrics = score_rows(subset)
        report.append({"brand": brand, "sample_count": len(subset), "metrics": metrics})
    return report
