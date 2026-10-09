"""Read-only evaluation of raw authenticity logits.

The positive class is fake (label 1). Authentic is label 0. A false authentic
is a fake image predicted authentic. That rate is counted only over actual
fakes. Sigmoid is applied here, not inside the model.

The default threshold is 0.50. Passing another threshold changes predictions.
Nothing in this module searches for a threshold, writes a split, or steps an
optimizer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augmentations import CONSERVATIVE, build_transforms
from classifier import build_authenticity_classifier
from dataset import brand_from_image_path
from training_config import (
    assert_experiment_output_dir,
    resolve_family_image_size,
    validate_classifier_arch,
)

POSITIVE_CLASS = "fake"
NEGATIVE_CLASS = "authentic"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
DEFAULT_THRESHOLD = 0.50
EVAL_SPLITS = ("validation", "test")
PREDICTION_FIELDS = (
    "sample_id",
    "brand",
    "label",
    "logit",
    "probability",
    "prediction",
    "threshold",
)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map raw logits to P(fake). This is the only sigmoid on the eval path."""
    return torch.sigmoid(logits.detach().float().reshape(-1))


def predictions_from_probabilities(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
    cutoff = _validate_threshold(threshold)
    return (probabilities >= cutoff).to(dtype=torch.long)


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Score one fixed threshold. Does not search for a better one."""
    cutoff = _validate_threshold(threshold)
    logit_vector = logits.detach().float().reshape(-1).cpu()
    label_vector = labels.detach().reshape(-1).cpu().to(dtype=torch.long)
    if logit_vector.numel() != label_vector.numel():
        raise ValueError(
            f"logits and labels differ in length: {logit_vector.numel()} vs {label_vector.numel()}"
        )
    if logit_vector.numel() == 0:
        raise ValueError("evaluation received no samples")
    if not torch.isfinite(logit_vector).all():
        raise ValueError("evaluation logits contain NaN or Inf")
    unknown = ~((label_vector == NEGATIVE_LABEL) | (label_vector == POSITIVE_LABEL))
    if bool(unknown.any()):
        raise ValueError("labels must be 0 (authentic) or 1 (fake)")

    probabilities = probabilities_from_logits(logit_vector)
    predictions = predictions_from_probabilities(probabilities, cutoff)
    authentic = label_vector == NEGATIVE_LABEL
    fake = label_vector == POSITIVE_LABEL
    predicted_authentic = predictions == NEGATIVE_LABEL
    predicted_fake = predictions == POSITIVE_LABEL

    true_negative = int((authentic & predicted_authentic).sum())
    false_fake = int((authentic & predicted_fake).sum())
    false_authentic = int((fake & predicted_authentic).sum())
    true_positive = int((fake & predicted_fake).sum())
    authentic_count = int(authentic.sum())
    fake_count = int(fake.sum())
    total = int(label_vector.numel())

    accuracy = (true_negative + true_positive) / total
    precision = _ratio(true_positive, true_positive + false_fake)
    recall = _ratio(true_positive, true_positive + false_authentic)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "threshold": cutoff,
        "threshold_policy": "fixed_not_calibrated",
        "sample_count": total,
        "authentic_count": authentic_count,
        "fake_count": fake_count,
        "roc_auc": _roc_auc(label_vector.numpy(), probabilities.numpy()),
        "pr_auc": _average_precision(label_vector.numpy(), probabilities.numpy()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_authentic": false_authentic,
        "false_fake": false_fake,
        "false_authentic_rate": _ratio(false_authentic, fake_count),
        "false_fake_rate": _ratio(false_fake, authentic_count),
        "confusion_matrix": {
            "labels": [NEGATIVE_CLASS, POSITIVE_CLASS],
            "matrix": [[true_negative, false_fake], [false_authentic, true_positive]],
            "true_negative": true_negative,
            "false_fake": false_fake,
            "false_authentic": false_authentic,
            "true_positive": true_positive,
        },
    }


def threshold_sweep(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: list[float] | tuple[float, ...],
) -> list[dict]:
    """Report every requested threshold. Does not mark or return a winner."""
    if not thresholds:
        raise ValueError("Pass thresholds explicitly. This sweep does not search.")
    return [evaluate_logits(logits, labels, threshold=value) for value in thresholds]


def prediction_rows(
    sample_ids: list[str],
    labels: torch.Tensor,
    logits: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    cutoff = _validate_threshold(threshold)
    logit_vector = logits.detach().float().reshape(-1).cpu()
    label_vector = labels.detach().reshape(-1).cpu().to(dtype=torch.long)
    if len(sample_ids) != int(logit_vector.numel()):
        raise ValueError("sample_ids, logits, and labels must have the same length")
    probabilities = probabilities_from_logits(logit_vector)
    predictions = predictions_from_probabilities(probabilities, cutoff)
    rows = []
    for index, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "brand": brand_from_image_path(sample_id),
                "label": int(label_vector[index]),
                "logit": float(logit_vector[index]),
                "probability": float(probabilities[index]),
                "prediction": int(predictions[index]),
                "threshold": cutoff,
            }
        )
    return rows


def write_evaluation_artifacts(
    experiment_dir: Path,
    metrics: dict,
    rows: list[dict],
) -> Path:
    """Write metrics, predictions, and the confusion matrix under the experiment only."""
    root = assert_experiment_output_dir(experiment_dir)
    destination = root / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (destination / "confusion_matrix.json").write_text(
        json.dumps(metrics["confusion_matrix"], indent=2) + "\n"
    )
    with (destination / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def load_split_membership(split_manifest: Path, split: str) -> list[tuple[str, Path, int]]:
    """Read one split. Does not rewrite the manifest or rebuild membership."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split}")
    manifest_path = Path(split_manifest)
    payload = json.loads(manifest_path.read_text())
    membership = payload.get("membership", {}).get(split)
    if not isinstance(membership, list):
        raise ValueError(f"{manifest_path} has no membership list for split '{split}'")
    rows: list[tuple[str, Path, int]] = []
    for raw_path in membership:
        path = Path(raw_path)
        rows.append((str(path), path, label_from_dataset_path(path)))
    return rows


def label_from_dataset_path(path: Path) -> int:
    parts = set(Path(path).parts)
    if "Label_0_Watches" in parts and "Label_1_Watches" in parts:
        raise ValueError(f"Path contains both label folders: {path}")
    if "Label_0_Watches" in parts:
        return NEGATIVE_LABEL
    if "Label_1_Watches" in parts:
        return POSITIVE_LABEL
    raise ValueError(f"Cannot read a binary label from {path}")


class _EvalDataset(Dataset):
    def __init__(self, rows: list[tuple[str, Path, int]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        sample_id, path, label = self.rows[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, sample_id


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Eval-mode forward. Does not step an optimizer or update normalization stats."""
    was_training = model.training
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sample_ids: list[str] = []
    try:
        with torch.inference_mode():
            for images, batch_labels, batch_ids in loader:
                images = images.to(device, non_blocking=True)
                output = model(images)
                if isinstance(output, tuple):
                    output = output[0]
                logits.append(output.detach().float().cpu().reshape(-1))
                labels.append(torch.as_tensor(batch_labels).reshape(-1).cpu())
                sample_ids.extend(str(sample_id) for sample_id in batch_ids)
    finally:
        model.train(was_training)
    if not logits:
        raise ValueError("evaluation loader produced no batches")
    return torch.cat(logits), torch.cat(labels), sample_ids


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    metadata: dict | None = None,
) -> tuple[dict, list[dict]]:
    logits, labels, sample_ids = collect_logits(model, loader, device)
    metrics = evaluate_logits(logits, labels, threshold=threshold)
    if metadata:
        metrics = {**metadata, **metrics}
    rows = prediction_rows(sample_ids, labels, logits, threshold=threshold)
    return metrics, rows


def build_eval_model(
    family: str,
    classifier_arch: str,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, int]:
    """Load a checkpoint for inference. Image size comes from the family, not the file name."""
    arch = validate_classifier_arch(classifier_arch)
    image_size = resolve_family_image_size(family, 0)
    model = build_authenticity_classifier(
        family,
        arch,
        image_size=image_size,
        pretrained=False,
    )
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, image_size


def run_checkpoint_evaluation(
    *,
    checkpoint: Path,
    split_manifest: Path,
    output_dir: Path,
    family: str,
    classifier_arch: str,
    split: str = "validation",
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 4,
    device: str | None = None,
    experiment_name: str | None = None,
    num_workers: int = 0,
    brand_eval: bool = False,
    minimum_brand_samples: int = 20,
    max_false_authentic_rate: float = 0.02,
) -> dict:
    """Measure one existing split. Does not refit, retune, or rewrite the split."""
    cutoff = _validate_threshold(threshold)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    manifest_path = Path(split_manifest)
    manifest_bytes = manifest_path.read_bytes()
    rows = load_split_membership(manifest_path, split)
    if not rows:
        raise ValueError(f"Split '{split}' in {manifest_path} is empty")
    chosen_device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, image_size = build_eval_model(family, classifier_arch, checkpoint, chosen_device)
    transform = build_transforms(family, train=False, profile=CONSERVATIVE)
    loader = DataLoader(
        _EvalDataset(rows, transform),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    checkpoint_path = Path(checkpoint)
    metadata = {
        "experiment_name": experiment_name or Path(output_dir).name,
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "model_family": family,
        "model_architecture": classifier_arch,
        "input_size": image_size,
        "split": split,
        "augmentation_profile": CONSERVATIVE,
        "evaluation_transform": "resize_pad_square_eval_v1",
    }
    metrics, prediction_table = evaluate_loader(
        model,
        loader,
        chosen_device,
        threshold=cutoff,
        metadata=metadata,
    )
    write_evaluation_artifacts(output_dir, metrics, prediction_table)
    if brand_eval:
        # Diagnostic only. Brand rates do not select a checkpoint or a threshold.
        from brand_evaluation import write_brand_evaluation

        write_brand_evaluation(
            output_dir,
            prediction_table,
            minimum_brand_samples=int(minimum_brand_samples),
            max_false_authentic_rate=float(max_false_authentic_rate),
            metadata=metadata,
        )
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError(f"Evaluation rewrote {manifest_path}")
    return metrics


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == POSITIVE_LABEL))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    true_positives = np.cumsum(ordered_labels == POSITIVE_LABEL)
    false_positives = np.cumsum(ordered_labels == NEGATIVE_LABEL)
    ends = _tie_ends(ordered_scores)
    true_rate = np.r_[0.0, true_positives[ends] / positives]
    false_rate = np.r_[0.0, false_positives[ends] / negatives]
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(true_rate, false_rate))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == POSITIVE_LABEL))
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    true_positives = np.cumsum(ordered_labels == POSITIVE_LABEL)
    false_positives = np.cumsum(ordered_labels == NEGATIVE_LABEL)
    ends = _tie_ends(ordered_scores)
    recall = np.r_[0.0, true_positives[ends] / positives]
    precision = true_positives[ends] / (true_positives[ends] + false_positives[ends])
    return float(np.sum(np.diff(recall) * precision))


def _tie_ends(ordered_scores: np.ndarray) -> np.ndarray:
    if ordered_scores.size == 0:
        return np.array([], dtype=int)
    changes = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    return np.flatnonzero(changes)
