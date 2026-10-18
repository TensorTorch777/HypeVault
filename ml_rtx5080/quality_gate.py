"""Research operating points for an image-quality gate.

This reads Phase 23 features and Phase 22 stress rows. It does not score
authenticity again, does not open the final test, and does not select a
production gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH, PHASE15_EXPERIMENT_DIRS, SPLIT_NAMES
from training_config import assert_experiment_output_dir

FROZEN_TEMPERATURE = 0.24038200410185356
EXPECTED_CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
MEMBERSHIP_HASH = AUTHORITATIVE_MEMBERSHIP_HASH
SOURCE_SPLITS = ("train", "calibration", "validation")
OUTPUT_DIR_NAME = "image_quality_v2"
REVIEW_BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25)
RESEARCH_CANDIDATE_MAX_REVIEW_RATE = 0.10
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
# Flags added on a Phase 22 false-fake flip. Aspect was already present on the
# resize case while the original prediction was still correct, so it is not in
# this pair.
FAILURE_LINKED_FLAGS = ("LOW_RESOLUTION", "LOW_CONTRAST")
RESOLUTION_BINS = (
    (None, 256.0, "<256"),
    (256.0, 384.0, "256-383"),
    (384.0, 512.0, "384-511"),
    (512.0, 768.0, "512-767"),
    (768.0, 1024.0, "768-1023"),
    (1024.0, None, ">=1024"),
)
LEVEL_RULES = {
    "0": "No quality flag.",
    "1": "One or two flags, excluding low resolution, and not both too-dark and low-contrast.",
    "2": "LOW_RESOLUTION, or LOW_CONTRAST together with TOO_DARK, or three or more flags, or minimum side below 512.",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_flags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in str(value).split("|") if part)


def assign_quality_level(flags, features: dict) -> int:
    """Map flags to a research severity. This is not a production verdict."""
    present = set(flags)
    if "LOW_RESOLUTION" in present or float(features["min_dimension"]) < 512.0:
        return 2
    if "LOW_CONTRAST" in present and "TOO_DARK" in present:
        return 2
    if len(present) >= 3:
        return 2
    if present:
        return 1
    return 0


def rule_matches(rule_id: str, flags, features: dict) -> bool:
    present = set(flags)
    minimum = float(features["min_dimension"])
    if rule_id == "any_flag":
        return bool(present)
    if rule_id == "low_resolution":
        return "LOW_RESOLUTION" in present
    if rule_id == "low_resolution_or_suspicious_framing":
        return bool(present & {"LOW_RESOLUTION", "SUSPICIOUS_FRAMING"})
    if rule_id == "low_resolution_or_low_contrast":
        return bool(present & {"LOW_RESOLUTION", "LOW_CONTRAST"})
    if rule_id == "low_resolution_or_too_dark":
        return bool(present & {"LOW_RESOLUTION", "TOO_DARK"})
    if rule_id == "low_resolution_or_extreme_aspect":
        return bool(present & {"LOW_RESOLUTION", "EXTREME_ASPECT_RATIO"})
    if rule_id == "two_or_more_flags":
        return len(present) >= 2
    if rule_id == "three_or_more_flags":
        return len(present) >= 3
    if rule_id == "critical_resolution_or_dark_contrast_or_three_flags":
        return "LOW_RESOLUTION" in present or {"LOW_CONTRAST", "TOO_DARK"} <= present or len(present) >= 3
    if rule_id == "level_2":
        return assign_quality_level(present, features) == 2
    if rule_id == "min_dimension_below_256":
        return minimum < 256.0
    if rule_id == "min_dimension_below_384":
        return minimum < 384.0
    if rule_id == "min_dimension_below_512":
        return minimum < 512.0
    if rule_id == "min_dimension_below_768":
        return minimum < 768.0
    raise KeyError(rule_id)


RULE_DEFINITIONS = {
    "any_flag": "Any Phase 23 quality flag.",
    "low_resolution": "LOW_RESOLUTION, minimum side strictly below the Phase 23 5th percentile (512).",
    "low_resolution_or_suspicious_framing": "LOW_RESOLUTION or SUSPICIOUS_FRAMING. Framing remains a border-median proxy.",
    "low_resolution_or_low_contrast": "LOW_RESOLUTION or LOW_CONTRAST.",
    "low_resolution_or_too_dark": "LOW_RESOLUTION or TOO_DARK.",
    "low_resolution_or_extreme_aspect": "LOW_RESOLUTION or EXTREME_ASPECT_RATIO. Aspect is not watch-boundary detection.",
    "two_or_more_flags": "Phase 23 degraded state: two or more flags.",
    "three_or_more_flags": "Three or more flags.",
    "critical_resolution_or_dark_contrast_or_three_flags": "LOW_RESOLUTION, or TOO_DARK together with LOW_CONTRAST, or three or more flags.",
    "level_2": "Research severity level 2.",
    "min_dimension_below_256": "Submitted minimum side below 256.",
    "min_dimension_below_384": "Submitted minimum side below 384.",
    "min_dimension_below_512": "Submitted minimum side below 512. Matches LOW_RESOLUTION when flags use that cutoff.",
    "min_dimension_below_768": "Submitted minimum side below 768.",
}
RULE_IDS = tuple(RULE_DEFINITIONS)


def review_rate(flagged_count: int, sample_count: int) -> float | None:
    if int(sample_count) <= 0:
        return None
    return int(flagged_count) / int(sample_count)


def false_authentic_rate(false_authentic_count: int, fake_count: int) -> float | None:
    """False authentic divided by actual fakes. Zero fakes leaves the rate undefined."""
    if int(fake_count) <= 0:
        return None
    return int(false_authentic_count) / int(fake_count)


def false_fake_rate(false_fake_count: int, authentic_count: int) -> float | None:
    if int(authentic_count) <= 0:
        return None
    return int(false_fake_count) / int(authentic_count)


def subset_classification(rows: list[dict]) -> dict:
    authentic = sum(int(row["label"]) == 0 for row in rows)
    fake = sum(int(row["label"]) == 1 for row in rows)
    false_authentic = sum(int(row["label"]) == 1 and row["authenticity_prediction"] == "authentic" for row in rows)
    false_fake = sum(int(row["label"]) == 0 and row["authenticity_prediction"] == "fake" for row in rows)
    true_positive = fake - false_authentic
    total = len(rows)
    precision = None if true_positive + false_fake == 0 else true_positive / (true_positive + false_fake)
    recall = None if fake == 0 else true_positive / fake
    f1 = None
    if precision is not None and recall is not None and authentic > 0 and fake > 0 and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "sample_count": total,
        "authentic_count": authentic,
        "fake_count": fake,
        "accuracy": None if total == 0 else (total - false_authentic - false_fake) / total,
        "f1": f1,
        "false_authentic_count": false_authentic,
        "false_authentic_rate": false_authentic_rate(false_authentic, fake),
        "false_fake_count": false_fake,
        "false_fake_rate": false_fake_rate(false_fake, authentic),
    }


def resolution_bin(min_dimension: float) -> str:
    value = float(min_dimension)
    for lower, upper, name in RESOLUTION_BINS:
        if lower is not None and value < lower:
            continue
        if upper is not None and value >= upper:
            continue
        return name
    raise RuntimeError(f"minimum side {value} did not fall in a resolution bin")


def evaluate_rule(rule_id: str, records: list[dict], failures: list[dict]) -> dict:
    flagged = [row for row in records if rule_matches(rule_id, row["quality_flags"], row)]
    flagged_ids = {id(row) for row in flagged}
    passed = [row for row in records if id(row) not in flagged_ids]
    flagged_metrics = subset_classification(flagged)
    passed_metrics = subset_classification(passed)
    caught = [row for row in failures if rule_matches(rule_id, row["flags_after"], row["features_after"])]
    caught_ids = {row["event_id"] for row in caught}
    missed = [row for row in failures if row["event_id"] not in caught_ids]
    linked = [row for row in failures if _failure_linked_catch(rule_id, row)]
    rate = review_rate(len(flagged), len(records))
    return {
        "rule_id": rule_id,
        "definition": RULE_DEFINITIONS[rule_id],
        "label": "research_quality_gate_candidate",
        "production_quality_gate": None,
        "review_rate": rate,
        "clean_pass_rate": None if rate is None else 1.0 - rate,
        "affected_images": len(flagged),
        "coverage": rate,
        "flagged_classification": flagged_metrics,
        "passed_classification": passed_metrics,
        "false_authentic_count": flagged_metrics["false_authentic_count"],
        "false_fake_count": flagged_metrics["false_fake_count"],
        "hidden_false_authentic_count": passed_metrics["false_authentic_count"],
        "hidden_false_fake_count": passed_metrics["false_fake_count"],
        "stress_failures_caught": len(caught),
        "failure_linked_catches": len(linked),
        "stress_failures_missed": len(missed),
        "stress_failure_count": len(failures),
        "caught_ids": [row["event_id"] for row in caught],
        "missed_ids": [row["event_id"] for row in missed],
        "association_only": True,
    }


def rank_rules(rules: list[dict]) -> list[dict]:
    """More stress coverage first, then a smaller review rate, then the rule id."""
    ordered = sorted(
        rules,
        key=lambda row: (
            -int(row["stress_failures_caught"]),
            float("inf") if row["review_rate"] is None else float(row["review_rate"]),
            row["rule_id"],
        ),
    )
    for index, row in enumerate(ordered, start=1):
        row["research_rank"] = index
    return ordered


def match_review_budget(rules: list[dict], budget: float) -> dict:
    """Closest research rule to a review load. Not a production optimum."""
    def sort_key(row: dict) -> tuple:
        rate = float(row["review_rate"])
        within = 0 if rate <= float(budget) + 1e-12 else 1
        return (
            abs(rate - float(budget)),
            within,
            -int(row["stress_failures_caught"]),
            row["rule_id"],
        )

    chosen = min(rules, key=sort_key)
    return {
        "review_budget": float(budget),
        "rule_id": chosen["rule_id"],
        "definition": chosen["definition"],
        "review_rate": chosen["review_rate"],
        "absolute_gap": abs(float(chosen["review_rate"]) - float(budget)),
        "within_budget": float(chosen["review_rate"]) <= float(budget) + 1e-12,
        "stress_failures_caught": chosen["stress_failures_caught"],
        "stress_failures_missed": chosen["stress_failures_missed"],
        "false_authentic_count": chosen["false_authentic_count"],
        "false_fake_count": chosen["false_fake_count"],
        "hidden_false_authentic_count": chosen["hidden_false_authentic_count"],
        "label": "research_quality_gate_candidate",
        "production_quality_gate": None,
        "production_optimal": False,
    }


def _failure_linked_catch(rule_id: str, failure: dict) -> bool:
    """Count a catch only when a flag added by the stress is one the rule uses.

    A pre-existing flag that did not appear at the flip does not count. The raw
    stress-failure count is kept separately.
    """
    added = tuple(
        flag
        for flag in failure["flags_after"]
        if flag not in failure["flags_before"] and flag in FAILURE_LINKED_FLAGS
    )
    if not added:
        return False
    return rule_matches(rule_id, added, failure["features_after"])


def _linked_catch_count(row: dict) -> int:
    if "failure_linked_catches" in row:
        return int(row["failure_linked_catches"])
    return int(row["stress_failures_caught"])


def recommend_research_candidate(rules: list[dict]) -> dict | None:
    """Best failure-linked coverage at or below 10% review. Partial coverage stays visible."""
    eligible = [row for row in rules if row["review_rate"] is not None and float(row["review_rate"]) <= RESEARCH_CANDIDATE_MAX_REVIEW_RATE]
    if not eligible or max(_linked_catch_count(row) for row in eligible) <= 0:
        return None
    chosen = min(
        eligible,
        key=lambda row: (
            -_linked_catch_count(row),
            float(row["review_rate"]),
            row["rule_id"],
        ),
    )
    return {
        "rule_id": chosen["rule_id"],
        "definition": chosen["definition"],
        "label": "research_quality_gate_candidate",
        "production_quality_gate": None,
        "review_rate": chosen["review_rate"],
        "stress_failures_caught": chosen["stress_failures_caught"],
        "failure_linked_catches": _linked_catch_count(chosen),
        "stress_failures_missed": chosen["stress_failures_missed"],
        "missed_ids": list(chosen["missed_ids"]),
        "false_authentic_count": chosen["false_authentic_count"],
        "hidden_false_authentic_count": chosen["hidden_false_authentic_count"],
        "false_fake_count": chosen["false_fake_count"],
        "selection_limit": RESEARCH_CANDIDATE_MAX_REVIEW_RATE,
        "selection_key": "maximum catches explained by a failure-linked flag added at the flip, then lower review rate, then rule id, among rules at or below 10% review",
        "not_production_optimal": True,
    }


def analyze(records: list[dict], failures: list[dict], darker_observations: list[dict] | None = None) -> dict:
    """Quantify research gates. Authenticity probabilities are not modified."""
    screen_records(records, set())
    screen_failures(failures)
    for row in records:
        row["quality_level"] = assign_quality_level(row["quality_flags"], row)
    total = len(records)
    flag_rows = []
    for flag in FLAG_NAMES:
        subset = [row for row in records if flag in row["quality_flags"]]
        metrics = subset_classification(subset)
        flag_rows.append(
            {
                "flag": flag,
                "flagged_count": len(subset),
                "flagged_rate": review_rate(len(subset), total),
                "failure_linked": flag in FAILURE_LINKED_FLAGS,
                **metrics,
                "association_only": True,
            }
        )
    rules = [evaluate_rule(rule_id, records, failures) for rule_id in RULE_IDS]
    ranked = rank_rules(rules)
    budgets = [match_review_budget(ranked, budget) for budget in REVIEW_BUDGETS]
    levels = []
    for level in (0, 1, 2):
        subset = [row for row in records if int(row["quality_level"]) == level]
        levels.append({"level": level, "definition": LEVEL_RULES[str(level)], **subset_classification(subset)})
    return {
        "sample_count": total,
        "any_flag_count": sum(bool(row["quality_flags"]) for row in records),
        "any_flag_rate": review_rate(sum(bool(row["quality_flags"]) for row in records), total),
        "flag_analysis": flag_rows,
        "rules": ranked,
        "review_budgets": budgets,
        "recommended_research_candidate": recommend_research_candidate(ranked),
        "production_quality_gate": None,
        "quality_levels": levels,
        "resolution_bins": _resolution_table(records),
        "framing": _framing_study(records, failures),
        "brightness_contrast": _brightness_study(records, failures, darker_observations or []),
        "compression": _compression_study(records, failures),
        "failure_coverage": _failure_coverage(failures, ranked),
        "reference_false_authentic_count": subset_classification(records)["false_authentic_count"],
        "reference_false_fake_count": subset_classification(records)["false_fake_count"],
        "stress_false_authentic_count": sum(bool(row["false_authentic"]) for row in failures),
        "stress_false_fake_count": sum(bool(row["false_fake"]) for row in failures),
        "final_test_samples_used": 0,
        "association_only": True,
    }


def screen_records(records: list[dict], test_sample_ids: set[str]) -> None:
    for row in records:
        if row.get("source_split") not in SOURCE_SPLITS:
            raise RuntimeError("quality-gate analysis encountered a final-test sample")
        if row.get("sample_id") in test_sample_ids:
            raise RuntimeError("quality-gate analysis encountered a final-test sample")


def screen_failures(failures: list[dict]) -> None:
    for row in failures:
        if row.get("source_split") not in SOURCE_SPLITS:
            raise RuntimeError("quality-gate analysis encountered a final-test sample")


def run_phase24(experiments_root: Path, manifest_path: Path) -> dict:
    root = Path(experiments_root)
    destination = _assert_destination(root / OUTPUT_DIR_NAME)
    if (destination / "quality_summary.json").is_file():
        raise RuntimeError("quality-gate analysis already completed; refusing to rerun")
    manifest_file = Path(manifest_path)
    checkpoint = root / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
    feature_csv = root / "image_quality_v1" / "quality_features_nontest.csv"
    linkage_csv = root / "image_quality_v1" / "quality_stress_linkage.csv"
    phase22_summary = root / "hard_negative_v1" / "challenge_summary.json"
    final_test_summary = root / "final_test_v1" / "final_test_summary.json"
    protected = {
        "checkpoint": file_sha256(checkpoint),
        "manifest_file": file_sha256(manifest_file),
        "features": file_sha256(feature_csv),
        "linkage": file_sha256(linkage_csv),
        "phase22": file_sha256(phase22_summary),
        "final_test_summary": file_sha256(final_test_summary),
    }
    if protected["checkpoint"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint hash does not match the Phase 21 checkpoint")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("membership_hash") != MEMBERSHIP_HASH:
        raise RuntimeError("Phase 16 membership hash does not match the authoritative hash")
    test_ids = _test_sample_ids(manifest)
    phase22 = json.loads(phase22_summary.read_text())
    if float(phase22["non_test_pool"]["accuracy"]) != 1.0 or int(phase22["non_test_distribution"]["misclassified_count"]) != 0:
        raise RuntimeError("Phase 22 non-test pool is not exact; refusing to invent predictions")
    if int(phase22["test_samples_used"]) != 0:
        raise RuntimeError("Phase 22 summary reports final-test samples")
    records = _load_feature_rows(feature_csv)
    screen_records(records, test_ids)
    failures = _load_failures(linkage_csv)
    screen_failures(failures)
    darker_observations = _load_condition_rows(linkage_csv, "darker")
    screen_failures(darker_observations)
    _attach_predictions(records)
    _attach_cached_probabilities(records, root)
    result = analyze(records, failures, darker_observations)
    result.update(
        {
            "phase": 24,
            "checkpoint_sha256": protected["checkpoint"],
            "membership_hash": MEMBERSHIP_HASH,
            "manifest_file_sha256": protected["manifest_file"],
            "challenge_manifest_sha256": phase22["challenge_manifest_sha256"],
            "temperature": FROZEN_TEMPERATURE,
            "temperature_changed": False,
            "checkpoint_changed": False,
            "training_performed": False,
            "ood_used": False,
            "final_test_opened": False,
            "final_test_samples_used": 0,
            "production_authenticity_threshold": None,
            "production_quality_gate": None,
            "source_splits": list(SOURCE_SPLITS),
            "prediction_source": "phase22_nontest_pool_exact_classification",
            "probability_source": "not_recomputed",
            "flag_definitions": {
                "LOW_RESOLUTION": "minimum side strictly below 512, the Phase 23 reference 5th percentile",
                "BLURRY": "Laplacian variance strictly below the Phase 23 5th percentile",
                "TOO_DARK": "mean luminance strictly below the Phase 23 5th percentile",
                "TOO_BRIGHT": "mean luminance strictly above the Phase 23 95th percentile",
                "LOW_CONTRAST": "luminance standard deviation strictly below the Phase 23 5th percentile",
                "EXTREME_ASPECT_RATIO": "width/height strictly outside the Phase 23 5th to 95th percentile",
                "EXCESSIVE_CLIPPING": "near-black or near-white fraction strictly above its Phase 23 95th percentile",
                "SUSPICIOUS_FRAMING": "foreground-occupancy proxy strictly below the Phase 23 5th percentile",
            },
            "level_definitions": LEVEL_RULES,
            "failure_linked_flags": list(FAILURE_LINKED_FLAGS),
        }
    )
    if file_sha256(checkpoint) != protected["checkpoint"]:
        raise RuntimeError("quality-gate analysis changed checkpoint bytes")
    if file_sha256(manifest_file) != protected["manifest_file"]:
        raise RuntimeError("quality-gate analysis changed the split manifest file")
    if file_sha256(final_test_summary) != protected["final_test_summary"]:
        raise RuntimeError("quality-gate analysis changed the final-test summary")
    if file_sha256(feature_csv) != protected["features"] or file_sha256(linkage_csv) != protected["linkage"]:
        raise RuntimeError("quality-gate analysis changed Phase 22 or Phase 23 inputs")
    _write_outputs(destination, result)
    return json.loads((destination / "quality_summary.json").read_text())


def _resolution_table(records: list[dict]) -> list[dict]:
    table = []
    cached = [row for row in records if row.get("calibrated_probability") is not None]
    for _lower, _upper, name in RESOLUTION_BINS:
        subset = [row for row in records if resolution_bin(row["min_dimension"]) == name]
        cached_subset = [row for row in subset if row.get("calibrated_probability") is not None]
        table.append(
            {
                "bin": name,
                **subset_classification(subset),
                "cached_probability_count": len(cached_subset),
                "mean_cached_calibrated_probability": _cached_mean(cached_subset),
                "mean_cached_calibrated_probability_authentic": _cached_mean(cached_subset, label=0),
                "mean_cached_calibrated_probability_fake": _cached_mean(cached_subset, label=1),
                "cached_probability_note": "Means use Phase 22 cached probabilities only. They are not full-bin means, and inference was not rerun.",
                "reference_cached_count": len(cached),
            }
        )
    return table


def _framing_study(records: list[dict], failures: list[dict]) -> dict:
    crop_failures = [row for row in failures if row["condition"] in {"mild_crop", "partial_crop"}]
    return {
        "proxy_note": "Foreground occupancy compares pixels with the border median. It is not watch detection.",
        "suspicious_framing_count": sum("SUSPICIOUS_FRAMING" in row["quality_flags"] for row in records),
        "extreme_aspect_count": sum("EXTREME_ASPECT_RATIO" in row["quality_flags"] for row in records),
        "crop_failure_count": len(crop_failures),
        "crop_failures_with_low_resolution_after": sum("LOW_RESOLUTION" in row["flags_after"] for row in crop_failures),
        "crop_failures_with_suspicious_framing_after": sum("SUSPICIOUS_FRAMING" in row["flags_after"] for row in crop_failures),
        "crop_failures_with_extreme_aspect_after": sum("EXTREME_ASPECT_RATIO" in row["flags_after"] for row in crop_failures),
        "crop_failure_mean_foreground_delta": _mean_or_none([row.get("delta_foreground_occupancy") for row in crop_failures]),
        "association_only": True,
    }


def _brightness_study(records: list[dict], failures: list[dict], darker_observations: list[dict]) -> dict:
    darker = [row for row in failures if row["condition"] == "darker" and row["false_fake"]]
    stable = [row for row in darker_observations if row["condition"] == "darker" and not row["false_fake"]]
    dark_flag = [row for row in records if "TOO_DARK" in row["quality_flags"]]
    contrast_flag = [row for row in records if "LOW_CONTRAST" in row["quality_flags"]]
    return {
        "too_dark_count": len(dark_flag),
        "low_contrast_count": len(contrast_flag),
        "too_dark_classification": subset_classification(dark_flag),
        "low_contrast_classification": subset_classification(contrast_flag),
        "reference_mean_luminance_median": float(np.median([float(row["mean_luminance"]) for row in records])) if records else None,
        "darker_false_fake_count": len(darker),
        "darker_stable_count": len(stable),
        "darker_false_fake_mean_luminance_before": _mean_or_none([row.get("before_mean_luminance") for row in darker]),
        "darker_false_fake_mean_luminance_after": _mean_or_none([row.get("after_mean_luminance") for row in darker]),
        "darker_stable_gained_low_contrast": sum(
            "LOW_CONTRAST" in row["flags_after"] and "LOW_CONTRAST" not in row["flags_before"] for row in stable
        ),
        "darker_stable_mean_luminance_delta": _mean_or_none(
            [None if row.get("before_mean_luminance") is None or row.get("after_mean_luminance") is None else float(row["after_mean_luminance"]) - float(row["before_mean_luminance"]) for row in stable]
        ),
        "darker_flip_gained_low_contrast": sum("LOW_CONTRAST" in row["flags_after"] and "LOW_CONTRAST" not in row["flags_before"] for row in darker),
        "darker_flip_too_dark_after": sum("TOO_DARK" in row["flags_after"] for row in darker),
        "association_only": True,
    }


def _compression_study(records: list[dict], failures: list[dict]) -> dict:
    recompression = [row for row in failures if row["condition"] == "resize_recompression" and row["false_fake"]]
    by_id = {row["sample_id"]: row for row in records}
    details = []
    reference_bytes = np.asarray(
        [float(row["bytes_per_pixel"]) for row in records if row.get("bytes_per_pixel") is not None],
        dtype=np.float64,
    )
    for row in recompression:
        source = by_id.get(row["sample_id"])
        bytes_per_pixel = None if source is None else source.get("bytes_per_pixel")
        rank = None
        if bytes_per_pixel is not None and reference_bytes.size:
            rank = float(np.mean(reference_bytes <= float(bytes_per_pixel)))
        details.append(
            {
                "sample_id": row["sample_id"],
                "before_min_dimension": row.get("before_min_dimension"),
                "after_min_dimension": row.get("after_min_dimension"),
                "bytes_per_pixel": bytes_per_pixel,
                "bytes_per_pixel_percentile_rank": rank,
                "file_bytes": None if source is None else source.get("file_bytes"),
                "probability_before": row["original_calibrated_probability"],
                "probability_after": row["transformed_calibrated_probability"],
                "jpeg_quality_estimated": False,
            }
        )
    return {
        "note": "Bytes per pixel is a file-size proxy. It is not a JPEG quality estimate, and no compression threshold is proposed from it.",
        "resize_recompression_false_fake_count": len(recompression),
        "cases": details,
        "compression_threshold": None,
        "association_only": True,
    }


def _failure_coverage(failures: list[dict], rules: list[dict]) -> list[dict]:
    by_id = {row["rule_id"]: row for row in rules}
    coverage = []
    for failure in failures:
        catching = [
            rule_id
            for rule_id in RULE_IDS
            if rule_matches(rule_id, failure["flags_after"], failure["features_after"])
        ]
        original_catching = [
            rule_id
            for rule_id in RULE_IDS
            if rule_matches(rule_id, failure["flags_before"], failure["features_before"])
        ]
        coverage.append(
            {
                "event_id": failure["event_id"],
                "sample_id": failure["sample_id"],
                "source_split": failure["source_split"],
                "condition": failure["condition"],
                "probability_before": failure["original_calibrated_probability"],
                "probability_after": failure["transformed_calibrated_probability"],
                "flags_before": list(failure["flags_before"]),
                "flags_after": list(failure["flags_after"]),
                "flags_added": [flag for flag in failure["flags_after"] if flag not in failure["flags_before"]],
                "original_quality_level": assign_quality_level(failure["flags_before"], failure["features_before"]),
                "degraded_quality_level": assign_quality_level(failure["flags_after"], failure["features_after"]),
                "rules_catching_degraded_image": catching,
                "rules_catching_original_image": original_catching,
                "low_resolution_catches_degraded": "low_resolution" in catching,
                "false_authentic": bool(failure["false_authentic"]),
                "false_fake": bool(failure["false_fake"]),
                "research_rank_context_rule_count": len(by_id),
            }
        )
    return coverage


def _load_feature_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_split": record["source_split"],
                    "brand": record["brand"],
                    "label": int(record["label"]),
                    "quality_flags": parse_flags(record.get("quality_flags")),
                    "min_dimension": float(record["min_dimension"]),
                    "aspect_ratio": float(record["aspect_ratio"]),
                    "mean_luminance": float(record["mean_luminance"]),
                    "luminance_std": float(record["luminance_std"]),
                    "foreground_occupancy": float(record["foreground_occupancy"]),
                    "bytes_per_pixel": float(record["bytes_per_pixel"]) if record.get("bytes_per_pixel") else None,
                    "file_bytes": int(float(record["file_bytes"])) if record.get("file_bytes") else None,
                    "sharpness_laplacian_variance": float(record["sharpness_laplacian_variance"]),
                    "calibrated_probability": None,
                }
            )
    return rows


def _load_failures(path: Path) -> list[dict]:
    return [row for row in _load_condition_rows(path, None) if row["false_fake"]]


def _load_condition_rows(path: Path, condition: str | None) -> list[dict]:
    failures = []
    with Path(path).open(newline="") as handle:
        for record in csv.DictReader(handle):
            if condition is not None and record["condition"] != condition:
                continue
            before_dimension = float(record["before_min_dimension"])
            after_dimension = float(record["after_min_dimension"])
            flags_before = parse_flags(record.get("flags_before"))
            flags_after = parse_flags(record.get("flags_after"))
            failures.append(
                {
                    "event_id": f"{record['sample_id']}|{record['condition']}",
                    "sample_id": record["sample_id"],
                    "source_split": record["source_split"],
                    "condition": record["condition"],
                    "false_authentic": record["false_authentic"] in {"true", "1"},
                    "false_fake": record["false_fake"] in {"true", "1"},
                    "original_calibrated_probability": float(record["original_calibrated_probability"]),
                    "transformed_calibrated_probability": float(record["transformed_calibrated_probability"]),
                    "flags_before": flags_before,
                    "flags_after": flags_after,
                    "features_before": {"min_dimension": before_dimension},
                    "features_after": {"min_dimension": after_dimension},
                    "before_min_dimension": before_dimension,
                    "after_min_dimension": after_dimension,
                    "before_mean_luminance": float(record["before_mean_luminance"]),
                    "after_mean_luminance": float(record["after_mean_luminance"]),
                    "delta_foreground_occupancy": float(record["delta_foreground_occupancy"])
                    if record.get("delta_foreground_occupancy")
                    else None,
                }
            )
    return failures


def _attach_cached_probabilities(records: list[dict], experiments_root: Path) -> None:
    """Join probabilities already stored by Phase 22 and Phase 23. Does not rescore."""
    found: dict[str, float] = {}
    challenge = experiments_root / "hard_negative_v1"
    for filename in ("low_margin.csv", "high_confidence.csv"):
        with (challenge / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                found[row["sample_id"]] = float(row["calibrated_probability"])
    linkage = experiments_root / "image_quality_v1" / "quality_stress_linkage.csv"
    with linkage.open(newline="") as handle:
        for row in csv.DictReader(handle):
            found.setdefault(row["sample_id"], float(row["original_calibrated_probability"]))
    for row in records:
        if row["sample_id"] in found:
            row["calibrated_probability"] = found[row["sample_id"]]


def _attach_predictions(records: list[dict]) -> None:
    for row in records:
        row["authenticity_prediction"] = "fake" if int(row["label"]) == 1 else "authentic"


def _test_sample_ids(manifest: dict) -> set[str]:
    test_paths = {str(Path(path).resolve()) for path in manifest["membership"]["test"]}
    identifiers = set()
    for row in manifest.get("samples", []):
        path = row.get("path")
        if isinstance(path, str) and str(Path(path).resolve()) in test_paths:
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str):
                identifiers.add(sample_id)
    return identifiers


def _cached_mean(rows: list[dict], label: int | None = None) -> float | None:
    chosen = [
        float(row["calibrated_probability"])
        for row in rows
        if row.get("calibrated_probability") is not None and (label is None or int(row["label"]) == label)
    ]
    if not chosen:
        return None
    return float(np.mean(chosen))


def _mean_or_none(values: list) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return float(np.mean(usable))


def _write_outputs(destination: Path, result: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "quality_operating_points.json").write_text(
        _dump(
            {
                "production_quality_gate": None,
                "recommended_research_candidate": result["recommended_research_candidate"],
                "review_budgets": result["review_budgets"],
                "rules": result["rules"],
            }
        )
        + "\n"
    )
    (destination / "quality_summary.json").write_text(_dump(result) + "\n")
    _write_dicts(
        destination / "quality_operating_points.csv",
        result["review_budgets"],
        (
            "review_budget",
            "rule_id",
            "review_rate",
            "absolute_gap",
            "within_budget",
            "stress_failures_caught",
            "stress_failures_missed",
            "false_authentic_count",
            "false_fake_count",
            "hidden_false_authentic_count",
            "label",
            "production_quality_gate",
            "production_optimal",
        ),
    )
    rule_rows = []
    for row in result["rules"]:
        rule_rows.append(
            {
                "research_rank": row["research_rank"],
                "rule_id": row["rule_id"],
                "review_rate": row["review_rate"],
                "clean_pass_rate": row["clean_pass_rate"],
                "affected_images": row["affected_images"],
                "false_authentic_count": row["false_authentic_count"],
                "false_fake_count": row["false_fake_count"],
                "hidden_false_authentic_count": row["hidden_false_authentic_count"],
                "stress_failures_caught": row["stress_failures_caught"],
                "failure_linked_catches": row["failure_linked_catches"],
                "stress_failures_missed": row["stress_failures_missed"],
                "label": row["label"],
                "production_quality_gate": None,
            }
        )
    _write_dicts(
        destination / "quality_rule_comparison.csv",
        rule_rows,
        (
            "research_rank",
            "rule_id",
            "review_rate",
            "clean_pass_rate",
            "affected_images",
            "false_authentic_count",
            "false_fake_count",
            "hidden_false_authentic_count",
            "stress_failures_caught",
            "failure_linked_catches",
            "stress_failures_missed",
            "label",
            "production_quality_gate",
        ),
    )
    coverage_rows = []
    for row in result["failure_coverage"]:
        coverage_rows.append(
            {
                **row,
                "flags_before": "|".join(row["flags_before"]),
                "flags_after": "|".join(row["flags_after"]),
                "flags_added": "|".join(row["flags_added"]),
                "rules_catching_degraded_image": "|".join(row["rules_catching_degraded_image"]),
                "rules_catching_original_image": "|".join(row["rules_catching_original_image"]),
            }
        )
    _write_dicts(
        destination / "quality_failure_coverage.csv",
        coverage_rows,
        (
            "event_id",
            "sample_id",
            "source_split",
            "condition",
            "probability_before",
            "probability_after",
            "flags_before",
            "flags_after",
            "flags_added",
            "original_quality_level",
            "degraded_quality_level",
            "low_resolution_catches_degraded",
            "false_authentic",
            "false_fake",
            "rules_catching_degraded_image",
            "rules_catching_original_image",
        ),
    )


def _write_dicts(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
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
    if value is None:
        return ""
    return value


def _dump(payload: dict) -> str:
    return json.dumps(_json_ready(payload), indent=2, allow_nan=False)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _assert_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != OUTPUT_DIR_NAME:
        raise RuntimeError(f"quality-gate output must be named {OUTPUT_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | set(SPLIT_NAMES) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        "final_test_v1",
        "final_model_freeze",
        "hard_negative_v1",
        "image_quality_v1",
        "v2_dinov2_cls_only",
        "v2_dinov2_cls_patch_attention",
        "v2_dinov3_cls_only",
        "v2_dinov3_cls_patch_attention",
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the quality gate into {resolved}")
    return resolved


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    summary = run_phase24(root, root / "dataset_audit" / "split_manifest_v2.json")
    candidate = summary["recommended_research_candidate"]
    print(
        json.dumps(
            {
                "any_flag_rate": summary["any_flag_rate"],
                "recommended_rule": None if candidate is None else candidate["rule_id"],
                "production_quality_gate": None,
                "final_test_samples_used": summary["final_test_samples_used"],
                "membership_hash": summary["membership_hash"],
                "manifest_file_sha256": summary["manifest_file_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
