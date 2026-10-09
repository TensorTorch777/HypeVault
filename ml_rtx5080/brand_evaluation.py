"""Diagnostic brand metrics for one already-scored evaluation pass.

Brand identity is the parent-directory name from dataset.brand_from_image_path,
the same field load_records stores. This module does not run the model, rewrite
a split, or choose a checkpoint. A brand FAR above the configured limit is an
analysis flag only.

Global metrics are micro metrics: one confusion count over every sample.
Macro metrics are the unweighted mean of per-brand metrics. A brand contributes
to a macro metric only when it has at least minimum_brand_samples and that
metric is defined. Undefined metrics are left out. They are not treated as zero.

A brand is low-sample when its image count is strictly below
minimum_brand_samples (default 20). Those brands stay in the report and out of
the macro averages.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import torch

from checkpoint_selection import (
    DEFAULT_MAX_FALSE_AUTHENTIC_RATE,
    THRESHOLD_POLICY,
    validate_max_false_authentic_rate,
)
from dataset import UNKNOWN, brand_from_image_path
from evaluation import evaluate_logits
from training_config import assert_experiment_output_dir

DEFAULT_MINIMUM_BRAND_SAMPLES = 20
KNOWN_BRAND = "KNOWN_BRAND"
UNKNOWN_BRAND = "UNKNOWN_BRAND"
FAR_DIAGNOSTIC_POLICY = (
    "brand_far_eligible compares a brand's validation false-authentic rate with "
    "max_false_authentic_rate. It is an analysis flag. It does not change "
    "checkpoint selection, early stopping, the FAR limit, or the decision threshold."
)
MACRO_DEFINITION = (
    "Unweighted mean across sample-eligible brands for which the metric is defined. "
    "Undefined metrics are omitted rather than filled with zero. "
    "This is not the micro/global metric."
)
LOW_SAMPLE_STATUS = "low_sample"
ELIGIBLE_STATUS = "eligible"

_BRAND_CSV_FIELDS = (
    "brand",
    "brand_status",
    "sample_status",
    "metrics_reliable",
    "sample_count",
    "authentic_count",
    "fake_count",
    "fake_rate",
    "zero_authentic",
    "zero_fake",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_authentic",
    "false_authentic_rate",
    "false_fake",
    "false_fake_rate",
    "brand_far_eligible",
    "far_gap_to_limit",
    "far_gap_versus_global",
)
_RISK_CSV_FIELDS = (
    "brand",
    "brand_status",
    "sample_status",
    "sample_count",
    "authentic_count",
    "fake_count",
    "false_authentic",
    "false_authentic_rate",
    "false_fake",
    "false_fake_rate",
    "recall",
    "f1",
    "brand_far_eligible",
    "far_gap_to_limit",
    "far_gap_versus_global",
)
_MACRO_METRICS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_authentic_rate",
    "false_fake_rate",
)


def validate_minimum_brand_samples(value: int) -> int:
    count = int(value)
    if count < 1:
        raise ValueError(f"minimum_brand_samples must be >= 1, got {value}")
    return count


def build_brand_report(
    rows: list[dict],
    *,
    minimum_brand_samples: int = DEFAULT_MINIMUM_BRAND_SAMPLES,
    max_false_authentic_rate: float = DEFAULT_MAX_FALSE_AUTHENTIC_RATE,
    metadata: dict | None = None,
) -> dict:
    """Score brands from cached logits. Does not call the model."""
    minimum = validate_minimum_brand_samples(minimum_brand_samples)
    limit = validate_max_false_authentic_rate(max_false_authentic_rate)
    if not rows:
        raise ValueError("brand evaluation received no prediction rows")
    threshold = _shared_threshold(rows)
    labeled = [_with_brand(row) for row in rows]
    logits = torch.tensor([row["logit"] for row in labeled], dtype=torch.float32)
    labels = torch.tensor([row["label"] for row in labeled], dtype=torch.long)
    global_metrics = evaluate_logits(logits, labels, threshold=threshold)
    grouped: dict[str, list[dict]] = {}
    for row in labeled:
        grouped.setdefault(row["brand"], []).append(row)
    brands = [
        _brand_record(
            brand,
            grouped[brand],
            threshold=threshold,
            minimum=minimum,
            limit=limit,
            global_far=global_metrics["false_authentic_rate"],
        )
        for brand in sorted(grouped)
    ]
    macro = _macro(brands)
    diagnostics = _diagnostics(brands, limit)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "threshold": threshold,
        "threshold_policy": THRESHOLD_POLICY,
        "max_false_authentic_rate": limit,
        "minimum_brand_samples": minimum,
        "low_sample_rule": f"sample_count < {minimum}",
        "far_diagnostic_policy": FAR_DIAGNOSTIC_POLICY,
        "far_gap_to_limit_definition": "max_false_authentic_rate - brand_false_authentic_rate",
        "far_gap_versus_global_definition": "brand_false_authentic_rate - global_false_authentic_rate",
        "macro_definition": MACRO_DEFINITION,
        "checkpoint_selection_affected": False,
        "threshold_affected": False,
        "analysis_only": True,
        "global_scope": "micro",
        "global": global_metrics,
        "macro": macro,
        "diagnostics": diagnostics,
        "rankings": _rankings(brands, reliable_only=True),
        "low_sample_rankings": _rankings(brands, reliable_only=False),
        "brands": brands,
    }
    if metadata:
        for key, value in metadata.items():
            report.setdefault(key, value)
    return report


def write_brand_evaluation(
    experiment_dir: Path,
    rows: list[dict],
    *,
    minimum_brand_samples: int = DEFAULT_MINIMUM_BRAND_SAMPLES,
    max_false_authentic_rate: float = DEFAULT_MAX_FALSE_AUTHENTIC_RATE,
    metadata: dict | None = None,
) -> Path:
    """Write the brand report under the experiment evaluation directory."""
    root = assert_experiment_output_dir(experiment_dir)
    destination = root / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    report = build_brand_report(
        rows,
        minimum_brand_samples=minimum_brand_samples,
        max_false_authentic_rate=max_false_authentic_rate,
        metadata=metadata,
    )
    report["plots"] = _write_plots(report, destination)
    (destination / "brand_metrics.json").write_text(_dump(report) + "\n")
    summary = {
        key: value
        for key, value in report.items()
        if key != "brands"
    }
    summary["brand_table"] = "brand_metrics.csv"
    (destination / "brand_summary.json").write_text(_dump(summary) + "\n")
    _write_csv(destination / "brand_metrics.csv", report["brands"], _BRAND_CSV_FIELDS)
    _write_csv(destination / "brand_risk.csv", report["brands"], _RISK_CSV_FIELDS)
    return destination


def _with_brand(row: dict) -> dict:
    brand = row.get("brand")
    if brand is None or brand == "":
        brand = brand_from_image_path(str(row.get("sample_id", "")))
    copied = dict(row)
    copied["brand"] = str(brand)
    return copied


def _shared_threshold(rows: list[dict]) -> float:
    thresholds = {float(row["threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise ValueError(f"brand evaluation expected one threshold, found {sorted(thresholds)}")
    return thresholds.pop()


def _brand_record(
    brand: str,
    rows: list[dict],
    *,
    threshold: float,
    minimum: int,
    limit: float,
    global_far: float | None,
) -> dict:
    logits = torch.tensor([float(row["logit"]) for row in rows], dtype=torch.float32)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    metrics = evaluate_logits(logits, labels, threshold=threshold)
    sample_count = int(metrics["sample_count"])
    authentic_count = int(metrics["authentic_count"])
    fake_count = int(metrics["fake_count"])
    reliable = sample_count >= minimum
    far = metrics["false_authentic_rate"]
    # One-class average precision is defined by the scorer but is not a
    # brand ROC/PR result. Both classes are required.
    one_class = authentic_count == 0 or fake_count == 0
    roc_auc = None if one_class else metrics["roc_auc"]
    pr_auc = None if one_class else metrics["pr_auc"]
    if not reliable or far is None:
        far_eligible = None
    else:
        far_eligible = bool(far <= limit)
    gap_to_limit = None if far is None else limit - float(far)
    gap_to_global = None if far is None or global_far is None else float(far) - float(global_far)
    return {
        "brand": brand,
        "brand_status": UNKNOWN_BRAND if brand == UNKNOWN else KNOWN_BRAND,
        "sample_status": ELIGIBLE_STATUS if reliable else LOW_SAMPLE_STATUS,
        "metrics_reliable": reliable,
        "sample_count": sample_count,
        "authentic_count": authentic_count,
        "fake_count": fake_count,
        "fake_rate": None if sample_count == 0 else fake_count / sample_count,
        "zero_authentic": authentic_count == 0,
        "zero_fake": fake_count == 0,
        "roc_auc": roc_auc,
        "roc_auc_status": "unavailable" if roc_auc is None else "ok",
        "pr_auc": pr_auc,
        "pr_auc_status": "unavailable" if pr_auc is None else "ok",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_authentic": int(metrics["false_authentic"]),
        "false_authentic_rate": far,
        "false_fake": int(metrics["false_fake"]),
        "false_fake_rate": metrics["false_fake_rate"],
        "brand_far_eligible": far_eligible,
        "far_gap_to_limit": gap_to_limit,
        "far_gap_versus_global": gap_to_global,
        "analysis_only": True,
    }


def _macro(brands: list[dict]) -> dict:
    reliable = [brand for brand in brands if brand["metrics_reliable"]]
    payload: dict = {
        "scope": "macro-brand",
        "definition": MACRO_DEFINITION,
        "eligible_brand_count": len(reliable),
    }
    for metric in _MACRO_METRICS:
        contributing = [brand for brand in reliable if brand[metric] is not None]
        values = [float(brand[metric]) for brand in contributing]
        payload[metric] = None if not values else sum(values) / len(values)
        payload[f"{metric}_macro_n"] = len(values)
        payload[f"{metric}_macro_brands"] = [brand["brand"] for brand in contributing]
    return payload


def _diagnostics(brands: list[dict], limit: float) -> dict:
    eligible = [brand for brand in brands if brand["metrics_reliable"]]
    low_sample = [brand for brand in brands if not brand["metrics_reliable"]]
    defined = [brand for brand in eligible if brand["false_authentic_rate"] is not None]
    rates = [float(brand["false_authentic_rate"]) for brand in defined]
    above = [brand for brand in defined if brand["brand_far_eligible"] is False]
    below = [brand for brand in defined if brand["brand_far_eligible"] is True]
    unknown = [brand for brand in brands if brand["brand_status"] == UNKNOWN_BRAND]
    return {
        "far_limit": limit,
        "far_limit_role": "diagnostic_only",
        "eligible_brands": len(eligible),
        "low_sample_brands": len(low_sample),
        "brands_above_far_limit": len(above),
        "brands_at_or_below_far_limit": len(below),
        "eligible_brands_with_undefined_far": len(eligible) - len(defined),
        "unknown_brand_groups": len(unknown),
        "unknown_brand_samples": sum(brand["sample_count"] for brand in unknown),
        "median_brand_far": None if not rates else float(statistics.median(rates)),
        "mean_brand_far": None if not rates else sum(rates) / len(rates),
        "max_eligible_brand_far": None if not rates else max(rates),
        "min_eligible_brand_far": None if not rates else min(rates),
        "far_spread": None if not rates else max(rates) - min(rates),
        "brand_far_std": None if len(rates) < 2 else float(statistics.pstdev(rates)),
        "brand_far_std_definition": (
            "Population standard deviation of false-authentic rate over eligible "
            "brands with a defined rate. Null when fewer than two such brands."
        ),
        "median_brand_far_definition": (
            "statistics.median of eligible-brand false-authentic rates. "
            "An even count averages the two central values."
        ),
    }


def _rankings(brands: list[dict], *, reliable_only: bool) -> dict:
    pool = [
        brand
        for brand in brands
        if brand["metrics_reliable"] is reliable_only
    ]
    return {
        "scope": "eligible" if reliable_only else "low_sample",
        "highest_false_authentic_rate": _rank(pool, "false_authentic_rate", descending=True),
        "highest_false_authentic_count": _rank(pool, "false_authentic", descending=True),
        "lowest_recall": _rank(pool, "recall", descending=False),
        "lowest_f1": _rank(pool, "f1", descending=False),
        "largest_far_gap_versus_global": _rank(pool, "far_gap_versus_global", descending=True),
    }


def _rank(brands: list[dict], metric: str, *, descending: bool) -> list[dict]:
    usable = [brand for brand in brands if brand[metric] is not None]
    def key(brand: dict) -> tuple:
        value = float(brand[metric])
        return (-value if descending else value, brand["brand"])
    ordered = sorted(usable, key=key)
    return [_ranking_entry(brand) for brand in ordered]


def _ranking_entry(brand: dict) -> dict:
    return {
        "brand": brand["brand"],
        "brand_status": brand["brand_status"],
        "sample_status": brand["sample_status"],
        "metrics_reliable": brand["metrics_reliable"],
        "sample_count": brand["sample_count"],
        "authentic_count": brand["authentic_count"],
        "fake_count": brand["fake_count"],
        "false_authentic": brand["false_authentic"],
        "false_authentic_rate": brand["false_authentic_rate"],
        "false_fake": brand["false_fake"],
        "false_fake_rate": brand["false_fake_rate"],
        "recall": brand["recall"],
        "f1": brand["f1"],
        "brand_far_eligible": brand["brand_far_eligible"],
        "far_gap_to_limit": brand["far_gap_to_limit"],
        "far_gap_versus_global": brand["far_gap_versus_global"],
    }


def _write_csv(path: Path, brands: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for brand in brands:
            writer.writerow(
                {field: "" if brand.get(field) is None else brand.get(field) for field in fields}
            )


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _write_plots(report: dict, destination: Path) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    try:
        written = {
            "far_by_brand": str(_plot_metric(report, destination, plt, "false_authentic_rate", "FAR by brand")),
            "f1_by_brand": str(_plot_metric(report, destination, plt, "f1", "F1 by brand")),
            "sample_count_by_brand": str(
                _plot_metric(report, destination, plt, "sample_count", "Sample count by brand")
            ),
        }
    except Exception as exc:
        return {"status": "failed", "reason": f"{exc.__class__.__name__}: {exc}"}
    return {"status": "written", **written}


def _plot_metric(report: dict, destination: Path, plt, metric: str, title: str) -> Path:
    brands = report["brands"]
    labels = [brand["brand"] for brand in brands]
    values = [brand[metric] if brand[metric] is not None else math.nan for brand in brands]
    colors = ["#00B4D8" if brand["metrics_reliable"] else "#9A9A9A" for brand in brands]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.7), 4))
    positions = list(range(len(labels)))
    axis.bar(positions, [0 if math.isnan(value) else value for value in values], color=colors)
    for position, value in zip(positions, values):
        if math.isnan(value):
            axis.text(position, 0, "n/a", ha="center", va="bottom", fontsize=8, color="#666666")
    if metric == "false_authentic_rate":
        axis.axhline(
            report["max_false_authentic_rate"],
            color="#FF4444",
            linestyle="--",
            linewidth=1.2,
            label=f"FAR limit {report['max_false_authentic_rate']:.2f}",
        )
        axis.legend()
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(f"{title} (gray = low-sample, diagnostic only)")
    axis.set_ylabel(metric)
    figure.tight_layout()
    filename = {
        "false_authentic_rate": "brand_far.png",
        "f1": "brand_f1.png",
        "sample_count": "brand_sample_counts.png",
    }[metric]
    path = destination / filename
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
