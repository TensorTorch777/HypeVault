"""Validate the Phase 14 OOD detector on an explicitly supplied unrelated corpus.

OOD stays a separate score from authenticity and from the research quality
gate. The ID reference, when a corpus is scored, is fit on the training
split only. This module does not download a public dataset, does not invent
images, and does not set a production OOD threshold.
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

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import ood
from quality_gate import rule_matches
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH, PHASE15_EXPERIMENT_DIRS, SPLIT_NAMES
from training_config import assert_experiment_output_dir

FROZEN_TEMPERATURE = 0.24038200410185356
EXPECTED_CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
MEMBERSHIP_HASH = AUTHORITATIVE_MEMBERSHIP_HASH
RESEARCH_QUALITY_CANDIDATE = "low_resolution_or_low_contrast"
PRODUCTION_QUALITY_GATE = None
OUTPUT_DIR_NAME = "ood_v1"
DEV_FRACTION = 0.80
MIN_OOD_IMAGES = 20
ID_FLAG_BUDGET = 0.05
STRONG_AUTHENTIC_P_FAKE = 0.10
DIAGNOSTIC_AUTHENTICITY_CUT = 0.50
SPLIT_TOKEN = "phase25-ood-split-v1"
CANDIDATE_SPECS = (
    ("id_flag_rate_0_01", "flag_rate", 0.01),
    ("id_flag_rate_0_02", "flag_rate", 0.02),
    ("id_flag_rate_0_05", "flag_rate", 0.05),
    ("id_percentile_95", "percentile", 0.95),
    ("id_percentile_99", "percentile", 0.99),
)
SCORE_FIELDS = (
    "sample_id",
    "path",
    "category",
    "dataset_type",
    "ood_split",
    "sha256",
    "method",
    "ood_score",
    "baseline_p_fake",
    "calibrated_p_fake",
    "authenticity_logit",
    "ood_flag",
    "quality_flags",
    "research_quality_candidate_match",
)
THRESHOLD_FIELDS = (
    "strategy",
    "threshold",
    "id_ood_flag_rate",
    "ood_detection_rate",
    "ood_false_accept_rate",
    "precision",
    "recall",
    "f1",
    "role",
    "production_ood_threshold",
)
CATEGORY_FIELDS = (
    "category",
    "sample_count",
    "mean_ood_score",
    "median_ood_score",
    "detection_rate",
    "false_accept_rate",
)
QUADRANT_FIELDS = ("dataset", "authenticity_prediction", "count", "percent")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrated_p_fake(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid of logit / frozen temperature. The temperature is not refit."""
    scaled = logits.detach().float().reshape(-1) / FROZEN_TEMPERATURE
    return torch.sigmoid(scaled)


def baseline_p_fake(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.detach().float().reshape(-1))


def assert_score_direction() -> None:
    """Fixed clusters: a farther CLS must outscore a nearer CLS before any corpus metric."""
    generator = torch.Generator().manual_seed(4)
    near = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(6, 4, generator=generator) * 0.01
    far = torch.tensor([[0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0]])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    reference = ood.fit_reference(near, labels)
    if ood.POSITIVE_CLASS != "ood" or ood.NEGATIVE_CLASS != "in_distribution":
        raise RuntimeError("OOD must be the positive class and ID the negative class")
    if ood.SCORE_DIRECTION != "larger_is_more_ood":
        raise RuntimeError("a larger OOD score must mean more OOD")
    for method in ood.METHODS:
        near_scores = ood.score_features(near, reference, method)
        far_scores = ood.score_features(far, reference, method)
        if float(far_scores.min()) <= float(near_scores.max()):
            raise RuntimeError(f"{method} failed the larger-is-more-OOD sanity check")


def id_ood_flag_rate(id_scores: torch.Tensor, threshold: float) -> float:
    vector = ood._score_vector(id_scores)
    if vector.numel() == 0:
        raise ValueError("ID OOD-flag rate needs in-distribution scores")
    return float((vector >= float(threshold)).double().mean())


def ood_false_accept_rate(ood_scores: torch.Tensor, threshold: float) -> float:
    vector = ood._score_vector(ood_scores)
    if vector.numel() == 0:
        raise ValueError("OOD false-accept rate needs OOD scores")
    return float((vector < float(threshold)).double().mean())


def development_operating_points(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> list[dict]:
    """Research cutoffs from development scores. Not a production threshold."""
    id_vector = ood._score_vector(id_scores)
    ood_vector = ood._score_vector(ood_scores)
    if id_vector.numel() == 0 or ood_vector.numel() == 0:
        return []
    rows = []
    for name, kind, value in CANDIDATE_SPECS:
        if kind == "percentile":
            cutoff = float(np.quantile(id_vector.numpy(), value, method="linear"))
        else:
            cutoff = ood._flag_rate_threshold(id_vector.numpy(), value)
        if not math.isfinite(cutoff):
            raise RuntimeError(f"threshold candidate {name} is not finite")
        point = ood.operating_point(id_vector, ood_vector, cutoff)
        rows.append(
            {
                "strategy": name,
                "threshold": point["threshold"],
                "id_ood_flag_rate": point["id_ood_flag_rate"],
                "ood_detection_rate": point["ood_detection_rate"],
                "ood_false_accept_rate": point["ood_false_accept_rate"],
                "precision": point["precision"],
                "recall": point["recall"],
                "f1": point["f1"],
                "role": "research_ood_threshold_candidate",
                "production_ood_threshold": None,
            }
        )
    return rows


def select_research_threshold(candidates: list[dict]) -> dict | None:
    """Freeze a research cutoff from development rows only.

    Prefer an ID flag rate at or below 5%, then a higher OOD detection rate,
    then a lower ID flag rate, then a lower threshold, then the strategy name.
    """
    eligible = [
        row
        for row in candidates
        if row.get("ood_detection_rate") is not None and float(row["id_ood_flag_rate"]) <= ID_FLAG_BUDGET + 1e-12
    ]
    if not eligible:
        return None
    chosen = min(
        eligible,
        key=lambda row: (
            -float(row["ood_detection_rate"]),
            float(row["id_ood_flag_rate"]),
            float(row["threshold"]),
            row["strategy"],
        ),
    )
    return {
        **chosen,
        "research_threshold_status": "frozen_before_ood_final",
        "production_ood_threshold": None,
        "selection_rule": "id_flag_rate<=0.05, then max OOD detection, then lower ID flag rate, then lower threshold, then strategy",
    }


def evaluate_held_out(frozen: dict, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
    """Score OOD_FINAL at a threshold that this function does not change."""
    if frozen is None:
        raise ValueError("held-out OOD evaluation requires a threshold frozen on development data")
    cutoff = float(frozen["threshold"])
    ranking = ood.ranking_metrics(id_scores, ood_scores)
    point = ood.operating_point(id_scores, ood_scores, cutoff)
    if float(frozen["threshold"]) != cutoff:
        raise RuntimeError("held-out evaluation changed the frozen threshold")
    point.update(ranking)
    point["threshold_changed_after_freeze"] = False
    point["used_for_threshold_selection"] = False
    point["production_ood_threshold"] = None
    point["confusion"] = {
        "true_positive_ood_flagged": point["ood_flagged"],
        "false_negative_ood_not_flagged": point["ood_count"] - point["ood_flagged"],
        "false_positive_id_flagged": point["id_flagged"],
        "true_negative_id_not_flagged": point["id_count"] - point["id_flagged"],
    }
    return point


def split_development_final(samples: list[dict]) -> tuple[list[dict], list[dict], str]:
    """Deterministic 80/20 split. A short corpus is not given a threshold."""
    ordered = sorted(samples, key=lambda row: row["sample_id"])
    if len(ordered) < MIN_OOD_IMAGES:
        return [], [], "corpus_too_small"
    keyed = sorted(ordered, key=lambda row: (_split_key(row["sample_id"]), row["sample_id"]))
    dev_count = int(len(keyed) * DEV_FRACTION)
    dev_count = min(max(dev_count, 1), len(keyed) - 1)
    development = [{**row, "ood_split": "OOD_DEVELOPMENT"} for row in keyed[:dev_count]]
    final = [{**row, "ood_split": "OOD_FINAL"} for row in keyed[dev_count:]]
    if {row["sample_id"] for row in development} & {row["sample_id"] for row in final}:
        raise RuntimeError("OOD development and OOD final overlap")
    return development, final, "split"


def screen_ood_paths(paths: list[str], membership: dict[str, set[str]]) -> None:
    """Watch-catalog paths, including the locked final test, cannot be OOD."""
    for split, blocked in membership.items():
        if split not in SPLIT_NAMES:
            raise ValueError(f"unexpected membership split {split!r}")
        overlap = [path for path in paths if path in blocked]
        if overlap:
            raise RuntimeError(f"OOD evaluation encountered a {split} image")


def collect_ood_images(
    root: Path,
    *,
    catalog_sha256: set[str],
    blocked_paths: set[str],
) -> tuple[list[dict], list[dict]]:
    """Hash every image. Exact catalog collisions and watch paths are dropped."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for path in ood.list_image_paths(root):
        resolved = str(path.resolve())
        relative = path.resolve().relative_to(Path(root).resolve()).as_posix()
        category = _category(relative)
        record = {
            "sample_id": relative,
            "path": resolved,
            "category": category,
            "dataset_type": "OOD",
            "sha256": file_sha256(path),
        }
        reason = _reject_reason(record, catalog_sha256=catalog_sha256, blocked_paths=blocked_paths, seen=seen)
        if reason is None:
            seen.add(record["sha256"])
            accepted.append(record)
        else:
            rejected.append({**record, "rejection_reason": reason})
    accepted.sort(key=lambda row: row["sample_id"])
    rejected.sort(key=lambda row: row["sample_id"])
    return accepted, rejected


def build_manifest(samples: list[dict], rejected: list[dict], *, available: bool) -> dict:
    payload = {
        "dataset_type": "OOD",
        "REAL_OOD_CORPUS_AVAILABLE": bool(available and bool(samples)),
        "sample_count": len(samples),
        "rejected_count": len(rejected),
        "samples": [_manifest_sample(row) for row in sorted(samples, key=lambda item: item["sample_id"])],
        "rejected": sorted(rejected, key=lambda item: item["sample_id"]),
    }
    return json.loads(_dump(payload))


def quality_interaction_rows(samples: list[dict], *, thresholds: dict, method: str) -> list[dict]:
    """Quality flags stay beside the OOD score. They are not one combined score."""
    from image_quality import apply_quality_flags

    rows = []
    for sample in samples:
        features = sample["quality_features"]
        flags = apply_quality_flags(features, thresholds)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "method": method,
                "ood_score": sample["ood_score"],
                "baseline_p_fake": sample["baseline_p_fake"],
                "calibrated_p_fake": sample["calibrated_p_fake"],
                "quality_flags": "|".join(flags),
                "research_quality_candidate": RESEARCH_QUALITY_CANDIDATE,
                "research_quality_candidate_match": rule_matches(RESEARCH_QUALITY_CANDIDATE, flags, features),
                "combined_quality_ood_score": None,
                "production_quality_gate": PRODUCTION_QUALITY_GATE,
            }
        )
    return rows


def category_metrics(samples: list[dict], threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["category"], []).append(sample)
    rows = []
    for category in sorted(grouped):
        block = grouped[category]
        scores = np.asarray([float(row["ood_score"]) for row in block], dtype=np.float64)
        detected = int(np.sum(scores >= float(threshold)))
        count = int(scores.size)
        detection = detected / count
        rows.append(
            {
                "category": category,
                "sample_count": count,
                "mean_ood_score": float(scores.mean()),
                "median_ood_score": float(np.median(scores)),
                "detection_rate": detection,
                "false_accept_rate": 1.0 - detection,
            }
        )
    return rows


def blind_spots(rows: list[dict]) -> dict:
    if not rows:
        return {"lowest_detection_category": None, "highest_false_accept_category": None}
    lowest = min(rows, key=lambda row: (float(row["detection_rate"]), row["category"]))
    highest = max(rows, key=lambda row: (float(row["false_accept_rate"]), row["category"]))
    return {
        "lowest_detection_category": lowest["category"],
        "highest_false_accept_category": highest["category"],
    }


def authenticity_quadrants(dataset_types: list[str], calibrated: torch.Tensor) -> list[dict]:
    """Ground-truth ID or OOD crossed with a diagnostic 0.50 cut on calibrated P(fake)."""
    probabilities = calibrated.detach().float().reshape(-1)
    if probabilities.numel() != len(dataset_types):
        raise ValueError("quadrant inputs must have the same length")
    predicted_fake = probabilities >= DIAGNOSTIC_AUTHENTICITY_CUT
    total = len(dataset_types)
    cells = []
    for dataset, prediction in (
        ("id", "authentic"),
        ("id", "fake"),
        ("ood", "authentic-looking"),
        ("ood", "fake-looking"),
    ):
        want_fake = prediction == "fake" or prediction == "fake-looking"
        count = sum(
            1
            for index, kind in enumerate(dataset_types)
            if kind == dataset and bool(predicted_fake[index]) == want_fake
        )
        cells.append(
            {
                "dataset": dataset,
                "authenticity_prediction": prediction,
                "count": count,
                "percent": None if total == 0 else count / total,
            }
        )
    return cells


def strong_authentic_ood_cases(samples: list[dict]) -> list[dict]:
    """OOD images the frozen head still calls strongly authentic. Not a verdict."""
    cases = []
    for sample in samples:
        probability = float(sample["calibrated_p_fake"])
        if probability <= STRONG_AUTHENTIC_P_FAKE:
            cases.append(
                {
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "calibrated_p_fake": probability,
                    "ood_score": float(sample["ood_score"]),
                    "definition": "ground-truth OOD and calibrated P(fake) <= 0.10",
                    "production_verdict": None,
                }
            )
    return cases


def brand_score_rows(samples: list[dict]) -> list[dict]:
    """Optional ID diagnostic. OOD rows do not receive a watch-brand label."""
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        if sample.get("dataset_type") != "ID":
            continue
        brand = sample.get("brand")
        if not brand:
            continue
        grouped.setdefault(str(brand), []).append(float(sample["ood_score"]))
    rows = []
    for brand in sorted(grouped):
        scores = np.asarray(grouped[brand], dtype=np.float64)
        rows.append(
            {
                "brand": brand,
                "sample_count": int(scores.size),
                "mean_ood_score": float(scores.mean()),
                "median_ood_score": float(np.median(scores)),
                "brand_defines_ood": False,
            }
        )
    return rows


def analyze_cached_scores(
    *,
    id_scores: torch.Tensor,
    development_scores: torch.Tensor,
    final_scores: torch.Tensor,
) -> dict:
    """Development selects the cutoff. OOD_FINAL is scored once afterward."""
    assert_score_direction()
    candidates = development_operating_points(id_scores, development_scores)
    frozen = select_research_threshold(candidates)
    held_out = None if frozen is None else evaluate_held_out(frozen, id_scores, final_scores)
    return {
        "positive_class": ood.POSITIVE_CLASS,
        "score_direction": ood.SCORE_DIRECTION,
        "threshold_candidates": candidates,
        "research_threshold": frozen,
        "production_ood_threshold": None,
        "held_out": held_out,
        "authenticity_combined_with_ood": False,
    }


def run_phase25(
    experiments_root: Path,
    manifest_path: Path,
    *,
    ood_root: Path | None = None,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    enforce_expected_sha: bool = True,
) -> dict:
    """Write the Phase 25 report. A missing corpus does not invent scores."""
    root = Path(experiments_root)
    destination = _assert_destination(Path(output_dir) if output_dir is not None else root / OUTPUT_DIR_NAME)
    if (destination / "ood_summary.json").is_file():
        raise RuntimeError("OOD validation already completed; refusing to rerun")
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
    ood.assert_reference_split("train")
    for split in ("validation", "calibration", "test"):
        try:
            ood.assert_reference_split(split)
        except ValueError:
            continue
        raise RuntimeError(f"reference fit accepted {split}")
    membership = {split: set(ood.resolve_membership(manifest, split)) for split in SPLIT_NAMES}
    catalog_sha = {
        str(row["sha256"])
        for row in manifest.get("samples", [])
        if isinstance(row, dict) and row.get("sha256")
    }
    blocked = set().union(*membership.values())
    corpus_root = None if ood_root is None else Path(ood_root)
    if corpus_root is not None and not corpus_root.is_dir():
        raise FileNotFoundError(f"explicit OOD directory does not exist: {corpus_root}")
    samples: list[dict] = []
    rejected: list[dict] = []
    if corpus_root is not None:
        samples, rejected = collect_ood_images(corpus_root, catalog_sha256=catalog_sha, blocked_paths=blocked)
        screen_ood_paths([row["path"] for row in samples], membership)
    available = bool(samples)
    development, final, split_status = split_development_final(samples) if available else ([], [], "corpus_unavailable")
    assigned = [*development, *final] if split_status == "split" else [{**row, "ood_split": None} for row in samples]
    manifest_payload = build_manifest(assigned, rejected, available=available)
    assert_score_direction()
    summary = _phase_summary(
        checkpoint_sha256=protected["checkpoint"],
        manifest_file_sha256=protected["manifest_file"],
        corpus_root=None if corpus_root is None else str(corpus_root.resolve()),
        available=available,
        split_status=split_status,
        sample_count=len(samples),
        rejected_count=len(rejected),
        development_count=len(development),
        final_count=len(final),
        categories=sorted({row["category"] for row in samples}),
    )
    if split_status == "split":
        summary.update(
            _score_explicit_corpus(
                load_batch=_model_loader(checkpoint_path),
                train_paths=sorted(membership["train"]),
                validation_paths=sorted(membership["validation"]),
                development=development,
                final=final,
                thresholds=_load_quality_thresholds(root / "image_quality_v1" / "quality_thresholds.json"),
            )
        )
    _write_outputs(destination, manifest_payload, summary)
    summary["ood_manifest_sha256"] = file_sha256(destination / "ood_manifest.json")
    (destination / "ood_summary.json").write_text(_dump(summary) + "\n")
    if file_sha256(checkpoint_path) != protected["checkpoint"]:
        raise RuntimeError("OOD validation changed checkpoint bytes")
    if file_sha256(manifest_file) != protected["manifest_file"]:
        raise RuntimeError("OOD validation changed the split manifest file")
    if protected["final_test_summary"] is not None and file_sha256(final_test_summary) != protected["final_test_summary"]:
        raise RuntimeError("OOD validation changed the final-test summary")
    return json.loads((destination / "ood_summary.json").read_text())


def _score_explicit_corpus(
    *,
    load_batch,
    train_paths: list[str],
    validation_paths: list[str],
    development: list[dict],
    final: list[dict],
    thresholds: dict | None,
) -> dict:
    """Fit on train, freeze on development, then score OOD_FINAL once."""
    stages: list[str] = []

    def _load(stage: str, paths: list[str], labeled: bool) -> dict:
        if stage in {"test", "calibration"}:
            raise RuntimeError(f"OOD scoring refused stage {stage}")
        stages.append(stage)
        return load_batch(stage, paths, labeled)

    reference_batch = _load("train", train_paths, True)
    reference = ood.fit_reference(reference_batch["features"], reference_batch["labels"])
    id_batch = _load("validation", validation_paths, True)
    dev_batch = _load("ood_development", [row["path"] for row in development], False)
    candidates: dict[str, list[dict]] = {}
    frozen: dict[str, dict | None] = {}
    id_scores: dict[str, torch.Tensor] = {}
    dev_scores: dict[str, torch.Tensor] = {}
    for method in ood.METHODS:
        id_scores[method] = ood.score_features(id_batch["features"], reference, method)
        dev_scores[method] = ood.score_features(dev_batch["features"], reference, method)
        candidates[method] = development_operating_points(id_scores[method], dev_scores[method])
        frozen[method] = select_research_threshold(candidates[method])
    final_batch = _load("ood_final", [row["path"] for row in final], False)
    if stages != ["train", "validation", "ood_development", "ood_final"]:
        raise RuntimeError(f"unexpected OOD scoring order: {stages}")
    methods = {}
    development_rows: list[dict] = []
    final_rows: list[dict] = []
    for method in ood.METHODS:
        final_scores = ood.score_features(final_batch["features"], reference, method)
        held = None if frozen[method] is None else evaluate_held_out(frozen[method], id_scores[method], final_scores)
        development_rows.extend(
            _score_table(development, dev_batch, dev_scores[method], method, frozen[method], thresholds)
        )
        final_rows.extend(_score_table(final, final_batch, final_scores, method, frozen[method], thresholds))
        methods[method] = {
            "method": method,
            "status": "scored",
            "development_ranking": ood.ranking_metrics(id_scores[method], dev_scores[method]),
            "threshold_candidates": candidates[method],
            "research_threshold": frozen[method],
            "held_out": _jsonable(held),
            "roc_auc": None if held is None else held["roc_auc"],
            "pr_auc": None if held is None else held["average_precision"],
            "ood_detection_rate": None if held is None else held["ood_detection_rate"],
            "ood_false_accept_rate": None if held is None else held["ood_false_accept_rate"],
            "id_ood_flag_rate": None if held is None else held["id_ood_flag_rate"],
            "precision": None if held is None else held["precision"],
            "recall": None if held is None else held["recall"],
            "f1": None if held is None else held["f1"],
        }
    primary = ood.PRIMARY_METHOD
    final_primary = [
        row for row in final_rows if row["method"] == primary
    ]
    cutoff = None if frozen[primary] is None else float(frozen[primary]["threshold"])
    categories = [] if cutoff is None else category_metrics(final_primary, cutoff)
    ood_samples = [row for row in [*development_rows, *final_rows] if row["method"] == primary]
    id_samples = _id_rows(id_batch, id_scores[primary], frozen[primary])
    quadrant_types = ["id"] * len(id_samples) + ["ood"] * len(ood_samples)
    quadrant_probabilities = torch.tensor(
        [float(row["calibrated_p_fake"]) for row in [*id_samples, *ood_samples]],
        dtype=torch.float32,
    )
    return {
        "methods": methods,
        "threshold_candidates": [*candidates[ood.METHODS[0]], *candidates[ood.METHODS[1]]],
        "research_ood_threshold": {method: frozen[method] for method in ood.METHODS},
        "research_threshold_status": "frozen_before_ood_final",
        "production_ood_threshold": None,
        "reference_fit_performed": True,
        "reference_fit_splits": ["train"],
        "reference_refit_on_ood": False,
        "scoring_stages": stages,
        "development_score_rows": development_rows,
        "final_score_rows": final_rows,
        "category_rows": categories,
        "quadrant_rows": authenticity_quadrants(quadrant_types, quadrant_probabilities) if ood_samples else [],
        "worst_categories": blind_spots(categories),
        "strong_authentic_ood_cases": strong_authentic_ood_cases(ood_samples),
        "strong_authentic_ood_count": len(strong_authentic_ood_cases(ood_samples)),
        "brand_score_rows": brand_score_rows(id_samples),
        "ood_performance_claimed": True,
        "validation_status": "scored_explicit_corpus_research_threshold_not_promoted",
        "final_test_samples_used": 0,
    }


def _model_loader(checkpoint: Path):
    from augmentations import CONSERVATIVE, build_transforms
    from evaluation import build_eval_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _image_size = build_eval_model("dinov3", "cls_patch_attention", checkpoint, device)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    transform = build_transforms("dinov3", train=False, profile=CONSERVATIVE)

    def load_batch(_stage: str, paths: list[str], labeled: bool) -> dict:
        batch = ood._load_split(model, paths, transform, device, 4, 0, labeled=labeled)
        ood._assert_weights_unchanged(model, before)
        return batch

    return load_batch


def _score_table(samples, batch, scores, method, frozen, thresholds) -> list[dict]:
    if list(batch["sample_ids"]) != [row["path"] for row in samples]:
        raise RuntimeError("OOD feature order does not match the manifest")
    logits = batch["logits"].detach().float().reshape(-1)
    baseline = baseline_p_fake(logits)
    calibrated = calibrated_p_fake(logits)
    cutoff = None if frozen is None else float(frozen["threshold"])
    rows = []
    for index, sample in enumerate(samples):
        features = _quality_features(sample["path"])
        flags: list[str] = []
        matched = False
        if thresholds is not None and features is not None:
            from image_quality import apply_quality_flags

            flags = apply_quality_flags(features, thresholds)
            matched = rule_matches(RESEARCH_QUALITY_CANDIDATE, flags, features)
        score = float(scores[index])
        rows.append(
            {
                **sample,
                "method": method,
                "ood_score": score,
                "baseline_p_fake": float(baseline[index]),
                "calibrated_p_fake": float(calibrated[index]),
                "authenticity_logit": float(logits[index]),
                "ood_flag": False if cutoff is None else score >= cutoff,
                "quality_flags": "|".join(flags),
                "research_quality_candidate_match": matched,
                "combined_quality_ood_score": None,
            }
        )
    return rows


def _id_rows(batch, scores, frozen) -> list[dict]:
    logits = batch["logits"].detach().float().reshape(-1)
    baseline = baseline_p_fake(logits)
    calibrated = calibrated_p_fake(logits)
    cutoff = None if frozen is None else float(frozen["threshold"])
    rows = []
    for index, sample_id in enumerate(batch["sample_ids"]):
        score = float(scores[index])
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_type": "ID",
                "brand": batch["brands"][index],
                "ood_score": score,
                "baseline_p_fake": float(baseline[index]),
                "calibrated_p_fake": float(calibrated[index]),
                "ood_flag": False if cutoff is None else score >= cutoff,
            }
        )
    return rows


def _quality_features(path: str) -> dict | None:
    from image_quality import extract_quality_features_from_path

    try:
        return extract_quality_features_from_path(Path(path))
    except (OSError, ValueError):
        return None


def _load_quality_thresholds(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _jsonable(payload):
    if payload is None:
        return None
    return json.loads(_dump(payload))


def _phase_summary(
    *,
    checkpoint_sha256: str,
    manifest_file_sha256: str,
    corpus_root: str | None,
    available: bool,
    split_status: str,
    sample_count: int,
    rejected_count: int,
    development_count: int,
    final_count: int,
    categories: list[str],
) -> dict:
    if not available:
        method_status = "not_evaluated_corpus_unavailable"
    elif split_status == "corpus_too_small":
        method_status = "corpus_too_small_no_threshold"
    else:
        method_status = "not_scored"
    methods = {
        method: {
            "method": method,
            "status": method_status,
            "roc_auc": None,
            "pr_auc": None,
            "ood_detection_rate": None,
            "ood_false_accept_rate": None,
            "id_ood_flag_rate": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
        for method in ood.METHODS
    }
    if not available:
        research_status = "not_selected_corpus_unavailable"
    elif split_status == "corpus_too_small":
        research_status = "not_selected_corpus_too_small"
    else:
        research_status = split_status
    return {
        "phase": 25,
        "REAL_OOD_CORPUS_AVAILABLE": available,
        "ood_performance_claimed": False,
        "validation_status": ood.NO_CORPUS_STATUS if not available else "corpus_present_threshold_not_promoted",
        "corpus_source": corpus_root,
        "corpus_sample_count": sample_count,
        "rejected_count": rejected_count,
        "categories": categories,
        "ood_split": split_status,
        "development_fraction": DEV_FRACTION,
        "ood_development_count": development_count,
        "ood_final_count": final_count,
        "minimum_images_for_split": MIN_OOD_IMAGES,
        "positive_class": ood.POSITIVE_CLASS,
        "negative_class": ood.NEGATIVE_CLASS,
        "score_direction": ood.SCORE_DIRECTION,
        "score_direction_sanity": "passed",
        "score_direction_sanity_uses_real_images": False,
        "methods": methods,
        "threshold_candidates": [],
        "research_ood_threshold": None,
        "research_threshold_status": research_status,
        "production_ood_threshold": None,
        "production_authenticity_threshold": None,
        "research_quality_candidate": RESEARCH_QUALITY_CANDIDATE,
        "production_quality_gate": PRODUCTION_QUALITY_GATE,
        "quality_combined_with_ood": False,
        "authenticity_combined_with_ood": False,
        "brand_defines_ood": False,
        "reference_fit_performed": False,
        "reference_fit_splits": [],
        "reference_excluded_splits": ["validation", "calibration", "test"],
        "reference_refit_on_ood": False,
        "checkpoint_sha256": checkpoint_sha256,
        "membership_hash": MEMBERSHIP_HASH,
        "manifest_file_sha256": manifest_file_sha256,
        "temperature": FROZEN_TEMPERATURE,
        "temperature_changed": False,
        "checkpoint_changed": False,
        "training_performed": False,
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "downloads_attempted": False,
        "worst_categories": {"lowest_detection_category": None, "highest_false_accept_category": None},
        "strong_authentic_ood_count": None,
        "strong_authentic_ood_cases": [],
        "development_score_rows": [],
        "final_score_rows": [],
        "category_rows": [],
        "quadrant_rows": [],
        "threshold_candidate_rows": [],
    }


def _reject_reason(record: dict, *, catalog_sha256: set[str], blocked_paths: set[str], seen: set[str]) -> str | None:
    if record["path"] in blocked_paths:
        return "split_membership_overlap"
    parts = [record["category"], *Path(record["path"]).parts]
    if any("watch" in part.casefold() for part in parts):
        return "watch_imagery_not_ood"
    if record["sha256"] in catalog_sha256:
        return "hypevault_sha256_collision"
    if record["sha256"] in seen:
        return "ood_internal_duplicate"
    return None


def _category(relative: str) -> str:
    parent = Path(relative).parent.as_posix()
    if parent in {"", "."}:
        return "uncategorized"
    return parent.split("/", 1)[0]


def _manifest_sample(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"],
        "path": row["path"],
        "category": row["category"],
        "dataset_type": "OOD",
        "sha256": row["sha256"],
        "ood_split": row.get("ood_split"),
    }


def _split_key(sample_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_TOKEN}:{sample_id}".encode()).hexdigest()


def _write_outputs(destination: Path, manifest_payload: dict, summary: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ood_manifest.json").write_text(_dump(manifest_payload) + "\n")
    (destination / "ood_metrics.json").write_text(
        _dump({"methods": summary["methods"], "production_ood_threshold": None, "final_test_samples_used": 0}) + "\n"
    )
    _write_csv(destination / "ood_development_scores.csv", summary.get("development_score_rows", []), SCORE_FIELDS)
    _write_csv(destination / "ood_final_scores.csv", summary.get("final_score_rows", []), SCORE_FIELDS)
    candidates = summary.get("threshold_candidates", [])
    _write_csv(destination / "ood_threshold_candidates.csv", candidates, THRESHOLD_FIELDS)
    _write_csv(destination / "ood_category_metrics.csv", summary.get("category_rows", []), CATEGORY_FIELDS)
    _write_csv(destination / "ood_quadrants.csv", summary.get("quadrant_rows", []), QUADRANT_FIELDS)


def _assert_destination(path: Path) -> Path:
    resolved = assert_experiment_output_dir(path)
    if resolved.name != OUTPUT_DIR_NAME:
        raise RuntimeError(f"OOD validation output must be named {OUTPUT_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | set(SPLIT_NAMES) | {
        "checkpoints",
        "checkpoints_watches",
        "dataset_audit",
        "final_test_v1",
        "final_model_freeze",
        "hard_negative_v1",
        "image_quality_v1",
        "image_quality_v2",
        "v2_dinov2_cls_only",
        "v2_dinov2_cls_patch_attention",
        "v2_dinov3_cls_only",
        "v2_dinov3_cls_patch_attention",
    }
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write OOD validation into {resolved}")
    return resolved


def _dump(payload: dict) -> str:
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


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    summary = run_phase25(root, root / "dataset_audit" / "split_manifest_v2.json", ood_root=None)
    print(
        json.dumps(
            {
                "REAL_OOD_CORPUS_AVAILABLE": summary["REAL_OOD_CORPUS_AVAILABLE"],
                "production_ood_threshold": summary["production_ood_threshold"],
                "research_threshold_status": summary["research_threshold_status"],
                "final_test_samples_used": summary["final_test_samples_used"],
                "membership_hash": summary["membership_hash"],
                "manifest_file_sha256": summary["manifest_file_sha256"],
                "ood_manifest_sha256": summary["ood_manifest_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
