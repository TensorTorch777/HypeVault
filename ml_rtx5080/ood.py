"""Feature-distance out-of-distribution scores for watch imagery.

The authenticity head still returns a raw logit. This module does not read
that logit to decide whether an image is unfamiliar. It reads the DINO CLS
token from ``DinoBackbone.get_cls_token`` (index 0). DINOv3 register tokens
stay out of that vector because they are not the CLS token.

Primary score: diagonal Mahalanobis distance on the raw CLS vector.
Baseline score: cosine distance from the centroid of L2-normalized CLS vectors.
A larger score is more OOD. OOD is the positive class for ROC-AUC and
average precision. Neither score is added to, multiplied by, or averaged with
the authenticity probability.

Reference mean and variance are fit on the weight-training split only.
Threshold candidates, when an ID development split and an OOD development
corpus both exist, are research rows. ``production_ood_threshold`` stays null
unless the caller passes a threshold.

Covariance is diagonal. Dimension ``j`` uses the unbiased sample variance
``sum((x_j - mean_j)^2) / (n - 1)``. Variances below ``VARIANCE_FLOOR``
(1e-6) are raised to that floor so a constant coordinate cannot make the
distance infinite. The floor is recorded in ``ood_config.json``. No full
768-by-768 covariance is inverted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augmentations import CONSERVATIVE, EVAL_TRANSFORM_VERSION, build_transforms
from dataset import IMAGE_SUFFIXES, brand_from_image_path
from evaluation import (
    POSITIVE_LABEL,
    _average_precision,
    _roc_auc,
    build_eval_model,
    label_from_dataset_path,
)
from training_config import assert_experiment_output_dir

REFERENCE_SPLIT = "train"
DEV_SPLITS = ("validation", "calibration")
PRIMARY_METHOD = "mahalanobis"
BASELINE_METHOD = "cosine"
METHODS = (PRIMARY_METHOD, BASELINE_METHOD)
VARIANCE_FLOOR = 1e-6
POSITIVE_CLASS = "ood"
NEGATIVE_CLASS = "in_distribution"
SCORE_DIRECTION = "larger_is_more_ood"
NO_CORPUS_STATUS = "Real OOD corpus not available; OOD performance not production-validated."
AUTHENTICITY_THRESHOLD = 0.50
STRONG_AUTHENTIC_FAKE_PROBABILITY = 0.10
FLAG_RULE = "ood_flag = ood_score >= threshold"

SCORE_FIELDS = (
    "sample_id",
    "label_if_available",
    "brand_if_available",
    "dataset_type",
    "method",
    "ood_score",
    "authenticity_logit_if_available",
    "authenticity_probability_if_available",
    "ood_flag",
)


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization. A zero vector raises instead of being nudged."""
    matrix = _feature_matrix(features)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not torch.isfinite(norms).all():
        raise ValueError("feature norms are not finite")
    if bool((norms == 0).any()):
        raise ValueError("refusing to L2-normalize a zero CLS feature")
    return matrix / norms.unsqueeze(1)


def fit_diagonal_gaussian(features: torch.Tensor) -> dict:
    """Unbiased per-dimension variance. Does not form a full covariance."""
    matrix = _feature_matrix(features).double()
    count, dimension = matrix.shape
    if count < 2:
        raise ValueError(f"diagonal covariance needs at least 2 samples, got {count}")
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    variance = (centered * centered).sum(dim=0) / (count - 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise RuntimeError("reference mean or variance is not finite")
    floored = variance < VARIANCE_FLOOR
    stabilized = torch.where(floored, torch.full_like(variance, VARIANCE_FLOOR), variance)
    inverse = 1.0 / stabilized
    if not torch.isfinite(inverse).all():
        raise RuntimeError("inverse diagonal variance is not finite")
    return {
        "sample_count": int(count),
        "feature_dim": int(dimension),
        "mean": mean,
        "variance": variance,
        "stabilized_variance": stabilized,
        "inverse_variance": inverse,
        "dimensions_floored": int(floored.sum()),
        "variance_floor": VARIANCE_FLOOR,
        "covariance": "diagonal",
        "variance_estimator": "unbiased_sample_variance",
    }


def mahalanobis_distance(features: torch.Tensor, reference: dict) -> torch.Tensor:
    """Sqrt of the diagonal Mahalanobis squared distance. Larger means more OOD."""
    matrix = _feature_matrix(features).double()
    _check_dimension(matrix, reference)
    delta = matrix - reference["mean"].double()
    squared = (delta * delta * reference["inverse_variance"].double()).sum(dim=1)
    if not torch.isfinite(squared).all() or bool((squared < 0).any()):
        raise RuntimeError("Mahalanobis distance is not finite")
    return torch.sqrt(squared)


def cosine_distance(features: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """``1 - cosine_similarity`` against an L2-normalized centroid."""
    matrix = l2_normalize(features).double()
    center = l2_normalize(centroid.reshape(1, -1)).reshape(-1).double()
    if matrix.shape[1] != center.shape[0]:
        raise ValueError(
            f"feature dimension {matrix.shape[1]} does not match centroid {center.shape[0]}"
        )
    similarity = matrix @ center
    distance = 1.0 - similarity
    if not torch.isfinite(distance).all():
        raise RuntimeError("cosine distance is not finite")
    return distance


def fit_reference(features: torch.Tensor, labels: torch.Tensor) -> dict:
    """Global raw-CLS Gaussian plus an L2 centroid. Class-conditional is diagnostic."""
    matrix = _feature_matrix(features)
    label_vector = _binary_labels(labels, matrix.shape[0])
    gaussian = fit_diagonal_gaussian(matrix)
    normalized = l2_normalize(matrix)
    centroid = l2_normalize(normalized.mean(dim=0, keepdim=True)).reshape(-1)
    authentic = int((label_vector == 0).sum())
    fake = int((label_vector == 1).sum())
    conditional = _class_conditional(matrix, label_vector)
    return {
        "gaussian": gaussian,
        "l2_centroid": centroid,
        "authentic_count": authentic,
        "fake_count": fake,
        "class_conditional": conditional,
        "active_reference": "global",
        "class_conditional_replaces_global": False,
    }


def score_features(features: torch.Tensor, reference: dict, method: str) -> torch.Tensor:
    """Score one cached feature matrix. The authenticity logit is not an input."""
    selected = _validate_method(method)
    if selected == PRIMARY_METHOD:
        return mahalanobis_distance(features, reference["gaussian"])
    return cosine_distance(features, reference["l2_centroid"])


def ranking_metrics(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
    """ROC-AUC and average precision with OOD as the positive class."""
    if POSITIVE_LABEL != 1:
        raise RuntimeError("OOD ranking expects the evaluator positive label to be 1")
    id_vector = _score_vector(id_scores)
    ood_vector = _score_vector(ood_scores)
    if id_vector.numel() == 0 or ood_vector.numel() == 0:
        return {
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
            "score_direction": SCORE_DIRECTION,
            "roc_auc": None,
            "average_precision": None,
            "status": "need both in-distribution and OOD samples",
        }
    labels = np.concatenate(
        [np.zeros(id_vector.numel(), dtype=np.int64), np.ones(ood_vector.numel(), dtype=np.int64)]
    )
    scores = np.concatenate([id_vector.numpy(), ood_vector.numpy()])
    return {
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "score_direction": SCORE_DIRECTION,
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "status": "ok",
    }


def operating_point(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    threshold: float,
) -> dict:
    """Flag rates at one threshold. This is not the authenticity false-authentic rate."""
    cutoff = _finite_threshold(threshold)
    id_vector = _score_vector(id_scores)
    ood_vector = _score_vector(ood_scores)
    if id_vector.numel() == 0:
        raise ValueError("ID OOD-flag rate needs in-distribution scores")
    if ood_vector.numel() == 0:
        raise ValueError("OOD false-accept rate needs OOD scores")
    id_flagged = int((id_vector >= cutoff).sum())
    ood_flagged = int((ood_vector >= cutoff).sum())
    id_count = int(id_vector.numel())
    ood_count = int(ood_vector.numel())
    id_flag_rate = id_flagged / id_count
    detection_rate = ood_flagged / ood_count
    false_accept = (ood_count - ood_flagged) / ood_count
    precision = None if (ood_flagged + id_flagged) == 0 else ood_flagged / (ood_flagged + id_flagged)
    recall = detection_rate
    if precision is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "threshold": cutoff,
        "flag_rule": FLAG_RULE,
        "id_ood_flag_rate": id_flag_rate,
        "id_flagged": id_flagged,
        "id_count": id_count,
        "ood_detection_rate": detection_rate,
        "ood_false_accept_rate": false_accept,
        "ood_flagged": ood_flagged,
        "ood_count": ood_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_candidates(id_scores: torch.Tensor, ood_scores: torch.Tensor | None) -> list[dict]:
    """Research thresholds from ID development scores. Not a production threshold."""
    id_vector = _score_vector(id_scores)
    if id_vector.numel() == 0:
        return []
    specs = (
        ("id_percentile_95", "percentile", 0.95),
        ("id_percentile_99", "percentile", 0.99),
        ("id_flag_rate_0_05", "flag_rate", 0.05),
        ("id_flag_rate_0_01", "flag_rate", 0.01),
    )
    rows = []
    ood_vector = None if ood_scores is None else _score_vector(ood_scores)
    for name, kind, value in specs:
        if kind == "percentile":
            cutoff = float(np.quantile(id_vector.numpy(), value, method="linear"))
        else:
            cutoff = _flag_rate_threshold(id_vector.numpy(), value)
        if not math.isfinite(cutoff):
            raise RuntimeError(f"threshold candidate {name} is not finite")
        row = {
            "strategy": name,
            "threshold": cutoff,
            "role": "research_candidate",
            "production_ood_threshold": None,
        }
        if ood_vector is None or ood_vector.numel() == 0:
            row["id_ood_flag_rate"] = float((id_vector >= cutoff).double().mean())
            row["ood_detection_rate"] = None
            row["ood_false_accept_rate"] = None
            row["precision"] = None
            row["recall"] = None
            row["f1"] = None
        else:
            point = operating_point(id_vector, ood_vector, cutoff)
            row.update(
                {
                    "id_ood_flag_rate": point["id_ood_flag_rate"],
                    "ood_detection_rate": point["ood_detection_rate"],
                    "ood_false_accept_rate": point["ood_false_accept_rate"],
                    "precision": point["precision"],
                    "recall": point["recall"],
                    "f1": point["f1"],
                }
            )
        rows.append(row)
    return rows


def signal_quadrants(
    dataset_types: list[str],
    scores: torch.Tensor,
    logits: torch.Tensor,
    threshold: float,
) -> dict:
    """Cross the OOD flag with the fixed 0.50 authenticity cut. Not a verdict."""
    cutoff = _finite_threshold(threshold)
    score_vector = _score_vector(scores)
    logit_vector = logits.detach().float().reshape(-1)
    if score_vector.numel() != logit_vector.numel() or score_vector.numel() != len(dataset_types):
        raise ValueError("quadrant inputs must have the same length")
    if not torch.isfinite(logit_vector).all():
        raise ValueError("authenticity logits are not finite")
    fake_probability = torch.sigmoid(logit_vector)
    predicted_fake = fake_probability >= AUTHENTICITY_THRESHOLD
    flagged = score_vector >= cutoff
    total = int(score_vector.numel())
    cells = []
    for is_flagged, prediction in (
        (False, "authentic"),
        (False, "fake"),
        (True, "authentic"),
        (True, "fake"),
    ):
        want_fake = prediction == "fake"
        mask = (flagged == is_flagged) & (predicted_fake == want_fake)
        count = int(mask.sum())
        cells.append(
            {
                "ood_flag": is_flagged,
                "group": "ood" if is_flagged else "id",
                "authenticity_prediction": prediction,
                "count": count,
                "percent": None if total == 0 else count / total,
            }
        )
    return {
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "authenticity_threshold_policy": "fixed_not_calibrated",
        "ood_threshold": cutoff,
        "basis": "ood_flag",
        "production_verdict": None,
        "sample_count": total,
        "cells": cells,
    }


def labeled_quadrants(dataset_types: list[str], logits: torch.Tensor) -> dict:
    """Ground-truth ID versus OOD crossed with the fixed 0.50 authenticity cut."""
    logit_vector = logits.detach().float().reshape(-1)
    if logit_vector.numel() != len(dataset_types):
        raise ValueError("quadrant inputs must have the same length")
    if not torch.isfinite(logit_vector).all():
        raise ValueError("authenticity logits are not finite")
    predicted_fake = torch.sigmoid(logit_vector) >= AUTHENTICITY_THRESHOLD
    total = len(dataset_types)
    cells = []
    for group, prediction in (
        ("id", "authentic"),
        ("id", "fake"),
        ("ood", "authentic"),
        ("ood", "fake"),
    ):
        want_fake = prediction == "fake"
        count = sum(
            1
            for index, kind in enumerate(dataset_types)
            if kind == group and bool(predicted_fake[index]) == want_fake
        )
        cells.append(
            {
                "dataset": group,
                "authenticity_prediction": prediction,
                "count": count,
                "percent": None if total == 0 else count / total,
            }
        )
    return {
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "authenticity_threshold_policy": "fixed_not_calibrated",
        "basis": "ground_truth_dataset",
        "production_verdict": None,
        "sample_count": total,
        "cells": cells,
    }


def strong_authentic_ood(logits: torch.Tensor) -> dict:
    """Ground-truth OOD images the authenticity head still calls strongly authentic."""
    logit_vector = logits.detach().float().reshape(-1)
    if logit_vector.numel() == 0:
        raise ValueError("strong-authentic diagnostic needs OOD logits")
    if not torch.isfinite(logit_vector).all():
        raise ValueError("authenticity logits are not finite")
    fake_probability = torch.sigmoid(logit_vector)
    count = int((fake_probability <= STRONG_AUTHENTIC_FAKE_PROBABILITY).sum())
    total = int(logit_vector.numel())
    return {
        "definition": "ground-truth OOD and P(fake) <= 0.10",
        "count": count,
        "ood_count": total,
        "percent_of_ood": count / total,
        "uses_ood_score": False,
    }


def feature_norm_stats(features: torch.Tensor) -> dict:
    matrix = _feature_matrix(features).double()
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not torch.isfinite(norms).all():
        raise RuntimeError("feature norms are not finite")
    return {
        "sample_count": int(norms.numel()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
    }


def histogram_rows(scores: torch.Tensor, *, method: str, dataset_type: str, bins: int = 20) -> list[dict]:
    vector = _score_vector(scores).numpy()
    if vector.size == 0:
        return []
    if int(bins) < 1:
        raise ValueError("histogram bins must be >= 1")
    if vector.size == 1 or float(vector.min()) == float(vector.max()):
        edge = float(vector[0])
        return [
            {
                "method": method,
                "dataset_type": dataset_type,
                "bin_left": edge,
                "bin_right": edge,
                "count": int(vector.size),
            }
        ]
    counts, edges = np.histogram(vector, bins=int(bins))
    rows = []
    for index, count in enumerate(counts.tolist()):
        rows.append(
            {
                "method": method,
                "dataset_type": dataset_type,
                "bin_left": float(edges[index]),
                "bin_right": float(edges[index + 1]),
                "count": int(count),
            }
        )
    return rows


def assert_reference_split(split: str) -> str:
    """The reference fit accepts the weight-training split and nothing else."""
    if split != REFERENCE_SPLIT:
        raise ValueError(
            "OOD reference statistics can be fit only on the weight-training split "
            f"'{REFERENCE_SPLIT}'. Refusing split {split!r}."
        )
    return split


def assert_development_split(split: str) -> str:
    if split not in DEV_SPLITS:
        raise ValueError(
            "OOD threshold development can use validation or calibration. "
            f"Refusing split {split!r}."
        )
    return split


def reject_ood_overlap(ood_paths: list[str], in_distribution_paths: set[str]) -> None:
    """Watch-catalog paths cannot be relabeled as the OOD corpus."""
    overlapped = [path for path in ood_paths if path in in_distribution_paths]
    if overlapped:
        raise ValueError("refusing to treat in-distribution manifest images as OOD")


def list_image_paths(root: Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"OOD directory does not exist: {directory}")
    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"OOD directory contains no images: {directory}")
    return sorted(paths)


def resolve_membership(manifest: dict, split: str) -> list[str]:
    membership = manifest.get("membership")
    if not isinstance(membership, dict):
        raise ValueError("split manifest has no membership object")
    paths = membership.get(split)
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"split manifest has no paths for {split}")
    resolved = [str(Path(path).resolve()) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"split {split} contains duplicate paths")
    return resolved


def collect_cls_and_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    """One eval forward per batch. Returns CLS, authenticity logit, ids, labels."""
    was_training = model.training
    model.eval()
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    sample_ids: list[str] = []
    labels: list[int] = []
    try:
        with torch.inference_mode():
            for images, batch_labels, batch_ids in loader:
                images = images.to(device, non_blocking=True)
                cls, logit = _cls_and_logit(model, images)
                if not torch.isfinite(cls).all() or not torch.isfinite(logit).all():
                    raise RuntimeError("feature extraction produced a non-finite value")
                features.append(cls.detach().float().cpu())
                logits.append(logit.detach().float().cpu().reshape(-1))
                sample_ids.extend(str(sample_id) for sample_id in batch_ids)
                labels.extend(int(label) for label in batch_labels)
    finally:
        model.train(was_training)
    if not features:
        raise ValueError("OOD loader produced no batches")
    return torch.cat(features), torch.cat(logits), sample_ids, labels


def run_ood_evaluation(
    *,
    checkpoint: Path,
    split_manifest: Path,
    output_dir: Path,
    family: str,
    classifier_arch: str,
    id_split: str = REFERENCE_SPLIT,
    ood_root: Path | None = None,
    ood_dev_root: Path | None = None,
    id_dev_split: str | None = None,
    report_final: bool = False,
    method: str = "both",
    threshold: float | None = None,
    batch_size: int = 4,
    device: str | None = None,
    num_workers: int = 0,
) -> dict:
    """Fit the training-split reference and score any explicitly supplied OOD roots."""
    assert_reference_split(id_split)
    methods = _method_list(method)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    explicit_threshold = None if threshold is None else _finite_threshold(threshold)
    manifest_path = Path(split_manifest)
    manifest_bytes = manifest_path.read_bytes()
    checkpoint_path = Path(checkpoint)
    checkpoint_bytes = checkpoint_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    train_paths = resolve_membership(manifest, REFERENCE_SPLIT)
    dev_paths: list[str] = []
    if id_dev_split is not None:
        assert_development_split(id_dev_split)
        dev_paths = resolve_membership(manifest, id_dev_split)
        if set(dev_paths) & set(train_paths):
            raise ValueError("ID development split overlaps the weight-training reference")
    final_paths: list[str] = []
    if report_final:
        final_paths = resolve_membership(manifest, "test")
        if set(final_paths) & set(train_paths):
            raise ValueError("final test split overlaps the weight-training reference")
    ood_dev_paths = [] if ood_dev_root is None else [str(path.resolve()) for path in list_image_paths(ood_dev_root)]
    ood_eval_paths = [] if ood_root is None else [str(path.resolve()) for path in list_image_paths(ood_root)]
    blocked = set(train_paths) | set(dev_paths) | set(final_paths)
    reject_ood_overlap([*ood_dev_paths, *ood_eval_paths], blocked)

    chosen_device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, image_size = build_eval_model(family, classifier_arch, checkpoint_path, chosen_device)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    transform = build_transforms(family, train=False, profile=CONSERVATIVE)
    reference_batch = _load_split(model, train_paths, transform, chosen_device, int(batch_size), int(num_workers), labeled=True)
    _assert_weights_unchanged(model, before)
    reference = fit_reference(reference_batch["features"], reference_batch["labels"])
    cached = {
        "id_reference": reference_batch,
    }
    if dev_paths:
        cached["id_development"] = _load_split(
            model, dev_paths, transform, chosen_device, int(batch_size), int(num_workers), labeled=True
        )
    if final_paths:
        cached["id_final"] = _load_split(
            model, final_paths, transform, chosen_device, int(batch_size), int(num_workers), labeled=True
        )
    if ood_dev_paths:
        cached["ood_development"] = _load_split(
            model, ood_dev_paths, transform, chosen_device, int(batch_size), int(num_workers), labeled=False
        )
    if ood_eval_paths:
        cached["ood_evaluation"] = _load_split(
            model, ood_eval_paths, transform, chosen_device, int(batch_size), int(num_workers), labeled=False
        )
    _assert_weights_unchanged(model, before)

    scored: dict[str, dict[str, torch.Tensor]] = {}
    for name, batch in cached.items():
        scored[name] = {
            selected: score_features(batch["features"], reference, selected) for selected in methods
        }
    _assert_weights_unchanged(model, before)

    real_ood = bool(ood_dev_paths or ood_eval_paths)
    summary = _build_summary(
        family=family,
        classifier_arch=classifier_arch,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint_bytes),
        methods=methods,
        reference=reference,
        reference_features=reference_batch["features"],
        scored=scored,
        cached=cached,
        explicit_threshold=explicit_threshold,
        real_ood=real_ood,
        id_dev_split=id_dev_split,
        report_final=bool(report_final),
    )
    destination = _write_artifacts(
        output_dir,
        summary=summary,
        reference=reference,
        family=family,
        classifier_arch=classifier_arch,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        cached=cached,
        scored=scored,
        methods=methods,
        explicit_threshold=explicit_threshold,
    )
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError(f"OOD evaluation rewrote {manifest_path}")
    if checkpoint_path.read_bytes() != checkpoint_bytes:
        raise RuntimeError(f"OOD evaluation rewrote {checkpoint_path}")
    summary["output_dir"] = str(destination)
    return summary


def _cls_and_logit(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "forward_features") or not hasattr(backbone, "get_cls_token"):
        raise TypeError("OOD feature extraction requires DinoBackbone.get_cls_token")
    tokens = backbone.forward_features(images)
    cls = backbone.get_cls_token(tokens)
    architecture = getattr(model, "classifier_arch", None)
    if architecture == "cls_only":
        hidden = cls
    elif architecture == "cls_patch_attention":
        patches = backbone.get_patch_tokens(tokens)
        pooled, _attention = model.attention_pool(patches)
        hidden = model.fusion(torch.cat((cls, pooled), dim=-1))
    else:
        raise ValueError(f"unsupported classifier head for OOD extraction: {architecture}")
    logits = model.trunk(hidden)
    return cls, logits


def _load_split(model, paths, transform, device, batch_size, num_workers, *, labeled: bool) -> dict:
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing image: {path}")
        label = label_from_dataset_path(path) if labeled else -1
        brand = brand_from_image_path(path) if labeled else ""
        rows.append((str(path.resolve()), path, label, brand))
    loader = DataLoader(
        _ImageRows(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    features, logits, sample_ids, labels = collect_cls_and_logits(model, loader, device)
    brands = [row[3] for row in rows]
    if sample_ids != [row[0] for row in rows]:
        raise RuntimeError("feature extraction reordered samples")
    return {
        "features": features,
        "logits": logits,
        "sample_ids": sample_ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "brands": brands,
    }


def _build_summary(
    *,
    family: str,
    classifier_arch: str,
    image_size: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    methods: tuple[str, ...],
    reference: dict,
    reference_features: torch.Tensor,
    scored: dict[str, dict[str, torch.Tensor]],
    cached: dict[str, dict],
    explicit_threshold: float | None,
    real_ood: bool,
    id_dev_split: str | None,
    report_final: bool,
) -> dict:
    method_reports = {}
    for selected in methods:
        method_reports[selected] = _method_report(
            selected,
            scored,
            cached,
            explicit_threshold,
        )
    danger = None
    if "ood_development" in cached or "ood_evaluation" in cached:
        blocks = [
            cached[name]["logits"]
            for name in ("ood_development", "ood_evaluation")
            if name in cached
        ]
        danger = strong_authentic_ood(torch.cat(blocks))
    reference_scores = {
        selected: _score_summary(scored["id_reference"][selected]) for selected in methods
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "score_direction": SCORE_DIRECTION,
        "primary_method": PRIMARY_METHOD,
        "production_detector_selected": False,
        "production_ood_threshold": explicit_threshold,
        "production_ood_threshold_source": None if explicit_threshold is None else "explicit",
        "checkpoint_selection_rerun": False,
        "calibration_temperature_used": False,
        "authenticity_threshold_changed": False,
        "scores_combined_with_authenticity": False,
        "real_ood_corpus_available": real_ood,
        "ood_performance_production_validated": False,
        "validation_status": "scored_explicit_corpus_not_production_validated" if real_ood else NO_CORPUS_STATUS,
        "reference_split": REFERENCE_SPLIT,
        "id_dev_split": id_dev_split,
        "final_evaluation_enabled": report_final,
        "model_family": family,
        "classifier_arch": classifier_arch,
        "input_size": image_size,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_version": EVAL_TRANSFORM_VERSION[CONSERVATIVE],
        "feature_norm": feature_norm_stats(reference_features),
        "in_sample_reference_scores": reference_scores,
        "in_sample_reference_scores_used_for_threshold": False,
        "class_conditional_replaces_global": False,
        "class_conditional_status": reference["class_conditional"]["status"],
        "methods": method_reports,
        "ground_truth_ood_and_strong_authentic": danger,
        "labeled_quadrants": _labeled_from_cache(cached),
        "brand_defines_ood": False,
    }


def _labeled_from_cache(cached: dict) -> dict:
    types: list[str] = []
    blocks = []
    for name, batch in cached.items():
        group = "ood" if name.startswith("ood") else "id"
        count = int(batch["logits"].reshape(-1).numel())
        types.extend([group] * count)
        blocks.append(batch["logits"])
    if not blocks:
        raise ValueError("quadrant diagnostic needs scored images")
    return labeled_quadrants(types, torch.cat(blocks))


def _method_report(method: str, scored, cached, explicit_threshold: float | None) -> dict:
    dev_id = scored.get("id_development", {}).get(method)
    dev_ood = scored.get("ood_development", {}).get(method)
    eval_id = scored.get("id_final", {}).get(method)
    eval_ood = scored.get("ood_evaluation", {}).get(method)
    candidates = []
    if dev_id is not None:
        candidates = threshold_candidates(dev_id, dev_ood)
    development = None
    if dev_id is not None and dev_ood is not None:
        development = ranking_metrics(dev_id, dev_ood)
    evaluation = None
    if eval_id is not None and eval_ood is not None:
        evaluation = ranking_metrics(eval_id, eval_ood)
        evaluation["used_for_threshold_selection"] = False
    cross = None
    if development is None and dev_id is not None and eval_ood is not None:
        cross = ranking_metrics(dev_id, eval_ood)
        cross["used_for_threshold_selection"] = False
    reported = development if development is not None else cross
    quadrants = None
    if explicit_threshold is not None:
        quadrants = _quadrants_for(method, scored, cached, explicit_threshold)
    return {
        "method": method,
        "development_ranking": development,
        "evaluation_ranking": evaluation,
        "held_out_id_versus_ood_root": cross,
        "reported_ranking": reported,
        "threshold_candidates": candidates,
        "explicit_threshold_quadrants": quadrants,
        "winner_selected": False,
    }


def _quadrants_for(method: str, scored, cached, threshold: float) -> dict:
    types: list[str] = []
    score_blocks = []
    logit_blocks = []
    for name in ("id_development", "id_final", "ood_development", "ood_evaluation"):
        if name not in scored or method not in scored[name]:
            continue
        count = int(scored[name][method].numel())
        group = "ood" if name.startswith("ood") else "id"
        types.extend([group] * count)
        score_blocks.append(scored[name][method])
        logit_blocks.append(cached[name]["logits"])
    if not score_blocks:
        return None
    return signal_quadrants(types, torch.cat(score_blocks), torch.cat(logit_blocks), threshold)


def _write_artifacts(
    output_dir: Path,
    *,
    summary: dict,
    reference: dict,
    family: str,
    classifier_arch: str,
    image_size: int,
    checkpoint_path: Path,
    cached: dict,
    scored: dict,
    methods: tuple[str, ...],
    explicit_threshold: float | None,
) -> Path:
    root = assert_experiment_output_dir(output_dir)
    destination = root / "ood"
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "cls_source": "DinoBackbone.get_cls_token",
        "cls_index": 0,
        "registers_excluded": True,
        "feature_dim": reference["gaussian"]["feature_dim"],
        "mahalanobis_representation": "raw_cls",
        "cosine_representation": "l2_normalized_cls",
        "covariance": "diagonal",
        "variance_estimator": "unbiased_sample_variance",
        "variance_floor": VARIANCE_FLOOR,
        "primary_method": PRIMARY_METHOD,
        "methods": list(methods),
        "positive_class": POSITIVE_CLASS,
        "score_direction": SCORE_DIRECTION,
        "flag_rule": FLAG_RULE,
        "reference_split": REFERENCE_SPLIT,
        "active_reference": "global",
        "class_conditional_replaces_global": False,
        "authenticity_probability": "sigmoid(logit) without temperature scaling",
        "scores_combined_with_authenticity": False,
        "model_family": family,
        "classifier_arch": classifier_arch,
        "input_size": image_size,
        "preprocessing_version": EVAL_TRANSFORM_VERSION[CONSERVATIVE],
        "checkpoint": str(checkpoint_path),
    }
    gaussian = reference["gaussian"]
    stats = {
        "feature_dim": gaussian["feature_dim"],
        "sample_count": gaussian["sample_count"],
        "authentic_count": reference["authentic_count"],
        "fake_count": reference["fake_count"],
        "mean": _vector(gaussian["mean"]),
        "variance": _vector(gaussian["variance"]),
        "stabilized_variance": _vector(gaussian["stabilized_variance"]),
        "inverse_variance": _vector(gaussian["inverse_variance"]),
        "dimensions_floored": gaussian["dimensions_floored"],
        "variance_floor": VARIANCE_FLOOR,
        "l2_centroid": _vector(reference["l2_centroid"]),
        "covariance": "diagonal",
        "normalization": {
            "mahalanobis": "raw_cls",
            "cosine": "l2_normalized_cls",
        },
        "model_family": family,
        "classifier_arch": classifier_arch,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "preprocessing_version": EVAL_TRANSFORM_VERSION[CONSERVATIVE],
        "reference_split": REFERENCE_SPLIT,
        "class_conditional": {
            "status": reference["class_conditional"]["status"],
            "replaces_global": False,
        },
    }
    (destination / "ood_config.json").write_text(_dump(config) + "\n")
    (destination / "reference_stats.json").write_text(_dump(stats) + "\n")
    (destination / "ood_summary.json").write_text(_dump(summary) + "\n")
    (destination / "ood_metrics.json").write_text(
        _dump(
            {
                "positive_class": POSITIVE_CLASS,
                "score_direction": SCORE_DIRECTION,
                "methods": summary["methods"],
                "ground_truth_ood_and_strong_authentic": summary["ground_truth_ood_and_strong_authentic"],
                "labeled_quadrants": summary["labeled_quadrants"],
            }
        )
        + "\n"
    )
    id_rows = []
    ood_rows = []
    histograms = []
    for dataset_type, batch in cached.items():
        for selected in methods:
            scores = scored[dataset_type][selected]
            histograms.extend(histogram_rows(scores, method=selected, dataset_type=dataset_type))
            target = ood_rows if dataset_type.startswith("ood") else id_rows
            target.extend(
                _score_rows(
                    batch,
                    scores,
                    dataset_type=dataset_type,
                    method=selected,
                    threshold=explicit_threshold,
                )
            )
    _write_csv(destination / "id_scores.csv", id_rows)
    if ood_rows:
        _write_csv(destination / "ood_scores.csv", ood_rows)
    candidate_rows = []
    for selected in methods:
        for row in summary["methods"][selected]["threshold_candidates"]:
            candidate_rows.append({"method": selected, **row})
    _write_csv(
        destination / "threshold_candidates.csv",
        candidate_rows,
        (
            "method",
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
        ),
    )
    _write_csv(
        destination / "score_histograms.csv",
        histograms,
        ("method", "dataset_type", "bin_left", "bin_right", "count"),
    )
    _write_histogram_plot(destination, histograms)
    brand_rows = _brand_rows(cached, scored, methods)
    if brand_rows:
        _write_csv(
            destination / "brand_score_diagnostics.csv",
            brand_rows,
            ("method", "dataset_type", "brand", "sample_count", "mean_ood_score", "affects_ood_decision"),
        )
    return destination


def _score_rows(batch: dict, scores: torch.Tensor, *, dataset_type: str, method: str, threshold: float | None) -> list[dict]:
    vector = scores.detach().double().reshape(-1).cpu()
    logit_vector = batch["logits"].detach().float().reshape(-1).cpu()
    labels = batch["labels"].detach().reshape(-1).cpu()
    rows = []
    for index, sample_id in enumerate(batch["sample_ids"]):
        label = int(labels[index])
        logit = float(logit_vector[index])
        score = float(vector[index])
        flagged = "" if threshold is None else str(bool(score >= threshold))
        rows.append(
            {
                "sample_id": sample_id,
                "label_if_available": "" if label < 0 else label,
                "brand_if_available": batch["brands"][index],
                "dataset_type": dataset_type,
                "method": method,
                "ood_score": score,
                "authenticity_logit_if_available": logit,
                "authenticity_probability_if_available": float(torch.sigmoid(torch.tensor(logit))),
                "ood_flag": flagged,
            }
        )
    return rows


def _brand_rows(cached: dict, scored: dict, methods: tuple[str, ...]) -> list[dict]:
    rows = []
    for dataset_type, batch in cached.items():
        if dataset_type.startswith("ood"):
            continue
        for selected in methods:
            grouped: dict[str, list[float]] = {}
            scores = scored[dataset_type][selected].detach().double().reshape(-1).cpu()
            for index, brand in enumerate(batch["brands"]):
                if not brand:
                    continue
                grouped.setdefault(brand, []).append(float(scores[index]))
            for brand in sorted(grouped):
                values = grouped[brand]
                rows.append(
                    {
                        "method": selected,
                        "dataset_type": dataset_type,
                        "brand": brand,
                        "sample_count": len(values),
                        "mean_ood_score": sum(values) / len(values),
                        "affects_ood_decision": False,
                    }
                )
    return rows


def _class_conditional(features: torch.Tensor, labels: torch.Tensor) -> dict:
    available = {}
    for label, name in ((0, "authentic"), (1, "fake")):
        subset = features[labels == label]
        if subset.shape[0] < 2:
            continue
        available[name] = fit_diagonal_gaussian(subset)
    if len(available) < 2:
        return {"status": "unavailable", "replaces_global": False}
    return {"status": "diagnostic_only", "replaces_global": False, "classes": sorted(available)}


def _method_list(method: str) -> tuple[str, ...]:
    if method == "both":
        return METHODS
    return (_validate_method(method),)


def _validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"OOD method must be one of {METHODS} or 'both', got {method}")
    return method


def _flag_rate_threshold(scores: np.ndarray, rate: float) -> float:
    if not 0.0 < float(rate) < 1.0:
        raise ValueError(f"flag rate must be in (0, 1), got {rate}")
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    count = int(ordered.size)
    flagged = max(1, int(math.ceil(float(rate) * count - 1e-12)))
    return float(ordered[count - flagged])


def _score_summary(scores: torch.Tensor) -> dict:
    vector = _score_vector(scores)
    return {
        "sample_count": int(vector.numel()),
        "min": float(vector.min()),
        "mean": float(vector.mean()),
        "max": float(vector.max()),
    }


def _feature_matrix(features: torch.Tensor) -> torch.Tensor:
    matrix = features.detach().float().cpu()
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"expected a non-empty feature matrix [N, C], got {tuple(features.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("features contain NaN or Inf")
    return matrix


def _check_dimension(features: torch.Tensor, reference: dict) -> None:
    if features.shape[1] != int(reference["feature_dim"]):
        raise ValueError(
            f"feature dimension {features.shape[1]} does not match reference {reference['feature_dim']}"
        )


def _binary_labels(labels: torch.Tensor, expected: int) -> torch.Tensor:
    vector = labels.detach().reshape(-1).cpu().long()
    if vector.numel() != expected:
        raise ValueError(f"features and labels differ in length: {expected} vs {vector.numel()}")
    if bool(((vector != 0) & (vector != 1)).any()):
        raise ValueError("reference labels must be 0 (authentic) or 1 (fake)")
    return vector


def _score_vector(scores: torch.Tensor) -> torch.Tensor:
    vector = scores.detach().double().reshape(-1).cpu()
    if vector.numel() and not torch.isfinite(vector).all():
        raise ValueError("OOD scores are not finite")
    return vector


def _finite_threshold(threshold: float) -> float:
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError(f"OOD threshold must be finite, got {threshold}")
    return value


def _vector(values: torch.Tensor) -> list[float]:
    return [float(item) for item in values.detach().double().reshape(-1).cpu()]


def _assert_weights_unchanged(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key, value in after.items():
        if not torch.equal(value, before[key]):
            raise RuntimeError(f"OOD extraction changed model parameter {key}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    chosen = SCORE_FIELDS if fields is None else fields
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in chosen})


def _write_histogram_plot(destination: Path, rows: list[dict]) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["dataset_type"]), []).append(row)
    figure, axis = plt.subplots(figsize=(8, 4))
    for (method, dataset_type), bins in sorted(grouped.items()):
        centers = [(bin_row["bin_left"] + bin_row["bin_right"]) / 2 for bin_row in bins]
        heights = [bin_row["count"] for bin_row in bins]
        axis.plot(centers, heights, label=f"{method} {dataset_type}")
    axis.set_xlabel("OOD score")
    axis.set_ylabel("count")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination / "score_histograms.png", dpi=80)
    plt.close(figure)


class _ImageRows(Dataset):
    def __init__(self, rows: list[tuple[str, Path, int, str]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        sample_id, path, label, _brand = self.rows[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, sample_id
