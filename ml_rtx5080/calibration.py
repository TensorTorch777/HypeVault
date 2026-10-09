"""Post-hoc temperature scaling and threshold analysis.

Temperature is a single positive scalar fit on the calibration split only.
Model weights are not updated. The checkpoint-selection validation split and
the final test split are not used to fit temperature or to choose a threshold.

p = sigmoid(logit / T). T is parameterized as exp(log T) so it stays positive.
The fit minimizes binary negative log-likelihood. T = 1 is the baseline
sigmoid. A lower NLL is not a claim that classification got better.

Expected calibration error uses predicted-class confidence. The predicted
class is fake when P(fake) >= 0.50 and authentic otherwise. Confidence is the
probability of that class. Accuracy is whether the class matches the label.
Bins are equal width on [0, 1], and a confidence of 1.0 falls in the last bin.
Empty bins do not contribute. ECE is the sample-weighted mean absolute gap.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import (
    CALIBRATION_SPLIT,
    DEFAULT_CALIBRATION_FRACTION,
    UNKNOWN,
    SampleRecord,
    brand_from_image_path,
    calibration_split_record,
    carve_calibration_records,
    membership_hash,
    product_group_key,
)
from evaluation import (
    DEFAULT_THRESHOLD,
    build_eval_model,
    build_transforms,
    collect_logits,
    evaluate_logits,
    label_from_dataset_path,
)
from augmentations import CONSERVATIVE
from training_config import assert_experiment_output_dir
from torch.utils.data import DataLoader

FITTING_METHOD = "lbfgs_log_temperature_nll"
ECE_DEFINITION = (
    "Predicted-class confidence. Fake is predicted when P(fake) >= 0.50; otherwise "
    "authentic. Confidence is the probability of that predicted class. Bins are "
    "equal width on [0, 1], and confidence 1.0 falls in the last bin. ECE is the "
    "sample-weighted mean of |accuracy - mean confidence| over non-empty bins."
)
THRESHOLD_GRID = tuple(step / 100 for step in range(5, 100, 5))
DEFAULT_BINS = 10
DEFAULT_FAR_LIMIT = 0.02


def scaled_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    value = _require_temperature(temperature)
    vector = logits.detach().float().reshape(-1)
    if vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ValueError("calibration logits must be non-empty and finite")
    return vector / value


def probabilities_from_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(scaled_logits(logits, temperature))


def binary_nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> float:
    """Mean binary NLL from logits. Probabilities are not rounded."""
    scaled = scaled_logits(logits, temperature)
    target = _labels(labels, scaled.numel())
    loss = F.binary_cross_entropy_with_logits(scaled, target, reduction="mean")
    if not torch.isfinite(loss):
        raise RuntimeError("calibration NLL is not finite")
    return float(loss)


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean squared error between P(fake) and the 0/1 label. Lower is better."""
    probability = probabilities.detach().float().reshape(-1)
    target = _labels(labels, probability.numel())
    if not torch.isfinite(probability).all():
        raise ValueError("Brier score received a non-finite probability")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("Brier score probabilities must be in [0, 1]")
    return float(torch.mean((probability - target) ** 2))


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = DEFAULT_BINS,
) -> tuple[float, list[dict]]:
    """Predicted-class confidence ECE. See the module docstring for the bins."""
    bins = int(n_bins)
    if bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    probability = probabilities.detach().float().reshape(-1).cpu()
    target = _labels(labels, probability.numel()).cpu()
    if not torch.isfinite(probability).all():
        raise ValueError("ECE received a non-finite probability")
    predicted = (probability >= DEFAULT_THRESHOLD).float()
    confidence = torch.where(predicted == 1, probability, 1 - probability)
    correct = (predicted == target).float()
    bin_ids = torch.clamp((confidence * bins).long(), max=bins - 1)
    total = int(probability.numel())
    rows: list[dict] = []
    total_error = 0.0
    for index in range(bins):
        mask = bin_ids == index
        count = int(mask.sum())
        lower = index / bins
        upper = (index + 1) / bins
        if count == 0:
            rows.append(
                {
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "sample_count": 0,
                    "confidence_mean": None,
                    "observed_accuracy": None,
                    "absolute_gap": None,
                }
            )
            continue
        confidence_mean = float(confidence[mask].mean())
        observed = float(correct[mask].mean())
        gap = abs(observed - confidence_mean)
        total_error += (count / total) * gap
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "sample_count": count,
                "confidence_mean": confidence_mean,
                "observed_accuracy": observed,
                "absolute_gap": gap,
            }
        )
    if not math.isfinite(total_error):
        raise RuntimeError("ECE is not finite")
    return total_error, rows


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Fit one global T. Does not modify logits or any model parameter."""
    source = logits.detach().float().reshape(-1).cpu()
    target = _labels(labels, source.numel()).cpu()
    unchanged = source.clone()
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    source64 = source.double()
    target64 = target.double()
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.5,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(source64 / log_temperature.exp(), target64)
        if not torch.isfinite(loss):
            raise RuntimeError("calibration loss is not finite")
        loss.backward()
        return loss

    optimizer.step(closure)
    optimized = float(log_temperature.detach().exp())
    _require_temperature(optimized)
    if not torch.equal(source, unchanged):
        raise RuntimeError("temperature fitting modified the logits")
    baseline_nll = binary_nll(source, target, 1.0)
    optimized_nll = binary_nll(source, target, optimized)
    # Keep the baseline when the search does not improve the calibration NLL.
    temperature = optimized if optimized_nll <= baseline_nll + 1e-8 else 1.0
    chosen_nll = binary_nll(source, target, temperature)
    if not math.isfinite(chosen_nll):
        raise RuntimeError("fitted calibration NLL is not finite")
    return {
        "temperature": temperature,
        "optimizer_temperature": optimized,
        "fitting_method": FITTING_METHOD,
        "baseline_nll": baseline_nll,
        "fitted_nll": chosen_nll,
        "fitting_loss": chosen_nll,
    }


def classification_at_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    temperature: float = 1.0,
) -> dict:
    metrics = evaluate_logits(scaled_logits(logits, temperature), labels, threshold=threshold)
    return {
        "threshold": metrics["threshold"],
        "false_authentic_rate": metrics["false_authentic_rate"],
        "false_authentic": metrics["false_authentic"],
        "false_fake_rate": metrics["false_fake_rate"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }


def threshold_sweep(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    grid: tuple[float, ...] = THRESHOLD_GRID,
) -> list[dict]:
    return [
        classification_at_threshold(logits, labels, threshold, temperature)
        for threshold in grid
    ]


def calibration_threshold_candidates(
    rows: list[dict],
    *,
    max_false_authentic_rate: float = DEFAULT_FAR_LIMIT,
) -> dict:
    """Research candidates on the calibration sweep. Not a production threshold."""
    limit = float(max_false_authentic_rate)
    if not 0.0 <= limit <= 1.0:
        raise ValueError(f"max_false_authentic_rate must be in [0, 1], got {limit}")
    safe = [
        row
        for row in rows
        if row["false_authentic_rate"] is not None and float(row["false_authentic_rate"]) <= limit
    ]
    return {
        "role": "calibration_candidates",
        "max_false_authentic_rate": limit,
        "production_threshold": None,
        "safe_thresholds": [row["threshold"] for row in safe],
        "highest_f1": _pick(safe, "f1", prefer_high=True),
        "highest_recall": _pick(safe, "recall", prefer_high=True),
        "lowest_far": _pick(safe, "false_authentic_rate", prefer_high=False),
        "closest_to_0_50": _closest_to_half(safe),
    }


def threshold_sensitivity(logits: torch.Tensor, labels: torch.Tensor, temperature: float, candidate: dict | None) -> list[dict]:
    """Report the candidate and its neighbors. Does not choose a new threshold."""
    if candidate is None:
        return []
    center = float(candidate["threshold"])
    neighbors = [center - 0.02, center, center + 0.02]
    reported = []
    for threshold in neighbors:
        if not 0.0 <= threshold <= 1.0:
            continue
        row = classification_at_threshold(logits, labels, threshold, temperature)
        row["relation"] = "candidate" if threshold == center else "neighbor"
        reported.append(row)
    return reported


def score_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    n_bins: int,
) -> dict:
    probability = probabilities_from_temperature(logits, temperature)
    nll = binary_nll(logits, labels, temperature)
    brier = brier_score(probability, labels)
    ece, bins = expected_calibration_error(probability, labels, n_bins)
    operating = classification_at_threshold(logits, labels, DEFAULT_THRESHOLD, temperature)
    return {
        "temperature": _require_temperature(temperature),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "ece_bins": n_bins,
        "reliability_bins": bins,
        "roc_auc": operating["roc_auc"],
        "pr_auc": operating["pr_auc"],
        "threshold_0_50": {
            "threshold": operating["threshold"],
            "threshold_policy": "fixed_not_calibrated" if temperature == 1.0 else "temperature_scaled_not_a_production_threshold",
            "accuracy": operating["accuracy"],
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "false_authentic_rate": operating["false_authentic_rate"],
            "false_fake_rate": operating["false_fake_rate"],
        },
    }


def compare_calibration(baseline: dict, calibrated: dict) -> dict:
    baseline_nll = float(baseline["nll"])
    calibrated_nll = float(calibrated["nll"])
    absolute = baseline_nll - calibrated_nll
    relative = None if baseline_nll == 0 else absolute / baseline_nll
    return {
        "nll_absolute_improvement": absolute,
        "nll_relative_improvement": relative,
        "brier_baseline": baseline["brier"],
        "brier_calibrated": calibrated["brier"],
        "ece_baseline": baseline["ece"],
        "ece_calibrated": calibrated["ece"],
        "roc_auc_baseline": baseline["roc_auc"],
        "roc_auc_calibrated": calibrated["roc_auc"],
        "pr_auc_baseline": baseline["pr_auc"],
        "pr_auc_calibrated": calibrated["pr_auc"],
        "roc_auc_unchanged": _same_optional(baseline["roc_auc"], calibrated["roc_auc"]),
        "pr_auc_unchanged": _same_optional(baseline["pr_auc"], calibrated["pr_auc"]),
        "classification_change_claimed": False,
    }


def brand_calibration_summaries(
    rows: list[dict],
    temperature: float,
    *,
    minimum_brand_samples: int = 20,
    n_bins: int = DEFAULT_BINS,
) -> list[dict]:
    """One global temperature. Brand rows are diagnostic and do not refit T."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["brand"]), []).append(row)
    summaries = []
    for brand in sorted(grouped):
        subset = grouped[brand]
        logits = torch.tensor([float(row["logit"]) for row in subset])
        labels = torch.tensor([int(row["label"]) for row in subset])
        reliable = len(subset) >= int(minimum_brand_samples)
        if not reliable:
            summaries.append(
                {
                    "brand": brand,
                    "sample_count": len(subset),
                    "sample_status": "low_sample",
                    "nll": None,
                    "brier": None,
                    "ece": None,
                    "temperature": temperature,
                }
            )
            continue
        scored = score_temperature(logits, labels, temperature, n_bins)
        summaries.append(
            {
                "brand": brand,
                "sample_count": len(subset),
                "sample_status": "eligible",
                "nll": scored["nll"],
                "brier": scored["brier"],
                "ece": scored["ece"],
                "temperature": temperature,
            }
        )
    return summaries


def resolve_calibration_membership(
    manifest: dict,
    *,
    fraction: float = DEFAULT_CALIBRATION_FRACTION,
    seed: int,
    debug_nonproduction_split: str | None = None,
) -> dict:
    """Choose calibration paths. The normal path never reads validation or test labels."""
    membership = manifest.get("membership")
    if not isinstance(membership, dict):
        raise ValueError("split manifest has no membership object")
    if debug_nonproduction_split is not None:
        if debug_nonproduction_split not in {"validation", "test"}:
            raise ValueError("debug_nonproduction_split must be validation or test")
        paths = list(membership.get(debug_nonproduction_split) or [])
        if not paths:
            raise ValueError(f"debug split {debug_nonproduction_split} is empty")
        return {
            "paths": [str(Path(path).resolve()) for path in paths],
            "source": debug_nonproduction_split,
            "non_production": True,
            "weight_train_paths": [],
            "sample_origin": (
                "debug mode read "
                f"{debug_nonproduction_split}; this run is not a production calibration"
            ),
        }
    validation_paths = {str(Path(path).resolve()) for path in membership.get("validation") or []}
    test_paths = {str(Path(path).resolve()) for path in membership.get("test") or []}
    existing = list(membership.get(CALIBRATION_SPLIT) or [])
    if existing:
        chosen = [str(Path(path).resolve()) for path in existing]
        weight_paths = [str(Path(path).resolve()) for path in membership.get("train") or []]
        source = "manifest_calibration"
        sample_origin = (
            "manifest calibration membership; these paths are not in the weight-training list"
        )
    else:
        train_paths = list(membership.get("train") or [])
        if not train_paths:
            raise ValueError(
                "Refusing to calibrate on the validation or test split. "
                "The manifest has no training-side membership to carve from."
            )
        records = _records_from_paths(train_paths)
        weight_records, calibration_records = carve_calibration_records(records, fraction, seed)
        chosen = [str(record.image_path.resolve()) for record in calibration_records]
        weight_paths = [str(record.image_path.resolve()) for record in weight_records]
        source = "carved_from_train"
        sample_origin = (
            "carved from training-side parent directories after the validation and test "
            "membership was fixed. product_id is not a split key. A manifest written "
            "before this carve may already have used those images for gradients."
        )
    chosen_set = set(chosen)
    if chosen_set & validation_paths:
        raise ValueError("calibration membership overlaps checkpoint-selection validation")
    if chosen_set & test_paths:
        raise ValueError("calibration membership overlaps the final test split")
    if chosen_set & set(weight_paths):
        raise ValueError("calibration membership overlaps the weight-training split")
    calibration_groups = {str(Path(path).resolve().parent) for path in chosen}
    blocked_groups = {
        str(Path(path).resolve().parent)
        for path in (*weight_paths, *validation_paths, *test_paths)
    }
    if calibration_groups & blocked_groups:
        raise ValueError("calibration parent directories overlap train, validation, or test")
    if not chosen:
        raise ValueError("calibration split is empty")
    return {
        "paths": chosen,
        "source": source,
        "non_production": False,
        "weight_train_paths": weight_paths,
        "validation_path_count": len(validation_paths),
        "test_path_count": len(test_paths),
        "sample_origin": sample_origin,
    }


def run_calibration(
    *,
    checkpoint: Path,
    split_manifest: Path,
    experiment_dir: Path,
    family: str,
    classifier_arch: str,
    bins: int = DEFAULT_BINS,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    seed: int = 42,
    max_false_authentic_rate: float = DEFAULT_FAR_LIMIT,
    minimum_brand_samples: int = 20,
    batch_size: int = 4,
    device: str | None = None,
    num_workers: int = 0,
    debug_nonproduction_split: str | None = None,
) -> dict:
    """Collect logits once, fit one temperature, and write calibration artifacts."""
    if int(bins) < 1:
        raise ValueError("bins must be >= 1")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    manifest_path = Path(split_manifest)
    manifest_bytes = manifest_path.read_bytes()
    checkpoint_path = Path(checkpoint)
    checkpoint_bytes = checkpoint_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    resolved = resolve_calibration_membership(
        manifest,
        fraction=calibration_fraction,
        seed=seed,
        debug_nonproduction_split=debug_nonproduction_split,
    )
    records = _records_from_paths(resolved["paths"])
    chosen_device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, image_size = build_eval_model(family, classifier_arch, checkpoint_path, chosen_device)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    loader = DataLoader(
        _CalibrationDataset(records, build_transforms(family, train=False, profile=CONSERVATIVE)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    logits, labels, sample_ids = collect_logits(model, loader, chosen_device)
    after = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key, value in after.items():
        if not torch.equal(value, before[key]):
            raise RuntimeError(f"calibration changed model parameter {key}")
    fit = fit_temperature(logits, labels)
    temperature = float(fit["temperature"])
    baseline = score_temperature(logits, labels, 1.0, int(bins))
    calibrated = score_temperature(logits, labels, temperature, int(bins))
    comparison = compare_calibration(baseline, calibrated)
    sweep = threshold_sweep(logits, labels, temperature)
    candidates = calibration_threshold_candidates(
        sweep, max_false_authentic_rate=max_false_authentic_rate
    )
    sensitivity = {
        name: threshold_sensitivity(logits, labels, temperature, candidates[name])
        for name in ("highest_f1", "highest_recall", "lowest_far", "closest_to_0_50")
    }
    prediction_rows = _prediction_rows(sample_ids, labels, logits, temperature)
    brand_rows = brand_calibration_summaries(
        prediction_rows,
        temperature,
        minimum_brand_samples=minimum_brand_samples,
        n_bins=int(bins),
    )
    data_hash = manifest.get("dataset_hash") or membership_hash(resolved["paths"])
    split_record = calibration_split_record(
        records,
        seed=int(seed),
        requested_fraction=float(calibration_fraction),
        parent_sample_count=len(resolved["weight_train_paths"]) + len(records),
        parent_group_count=len({record.split_group for record in records})
        + len({str(Path(path).resolve().parent) for path in resolved["weight_train_paths"]}),
        dataset_hash=str(data_hash),
    )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "calibration_succeeded": True,
        "valid_temperature": True,
        "production_threshold": None,
        "production_threshold_selected": False,
        "checkpoint_selection_completed": True,
        "calibration_fitted_after_selection": True,
        "checkpoint_selection_rerun": False,
        "non_production": bool(resolved["non_production"]),
        "calibration_source": resolved["source"],
        "sample_origin": resolved["sample_origin"],
        "temperature": temperature,
        "comparison": comparison,
        "threshold_candidates": candidates,
        "threshold_sensitivity": sensitivity,
        "brand_calibration_affects_global_temperature": False,
        "model_family": family,
        "model_architecture": classifier_arch,
        "input_size": image_size,
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
    }
    _write_artifacts(
        experiment_dir,
        prediction_rows=prediction_rows,
        fit=fit,
        baseline=baseline,
        calibrated=calibrated,
        comparison=comparison,
        sweep=sweep,
        candidates=candidates,
        split_record=split_record,
        summary=summary,
        brand_rows=brand_rows,
        family=family,
        classifier_arch=classifier_arch,
        checkpoint_path=checkpoint_path,
        labels=labels,
    )
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError(f"calibration rewrote {manifest_path}")
    if checkpoint_path.read_bytes() != checkpoint_bytes:
        raise RuntimeError(f"calibration rewrote {checkpoint_path}")
    return summary


def _records_from_paths(paths: list[str]) -> list[SampleRecord]:
    records = []
    for raw_path in paths:
        path = Path(raw_path)
        records.append(
            SampleRecord(
                image_path=path,
                label=label_from_dataset_path(path),
                brand=brand_from_image_path(path),
                product_id=UNKNOWN,
                marketplace=UNKNOWN,
                source=UNKNOWN,
                split_group=product_group_key(path),
            )
        )
    return records


def _prediction_rows(
    sample_ids: list[str],
    labels: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
) -> list[dict]:
    baseline = probabilities_from_temperature(logits, 1.0)
    calibrated = probabilities_from_temperature(logits, temperature)
    label_vector = labels.detach().reshape(-1).cpu()
    logit_vector = logits.detach().float().reshape(-1).cpu()
    rows = []
    for index, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "brand": brand_from_image_path(sample_id),
                "label": int(label_vector[index]),
                "logit": float(logit_vector[index]),
                "probability_at_0_5": float(baseline[index]),
                "calibrated_probability": float(calibrated[index]),
            }
        )
    return rows


def _write_artifacts(
    experiment_dir: Path,
    *,
    prediction_rows: list[dict],
    fit: dict,
    baseline: dict,
    calibrated: dict,
    comparison: dict,
    sweep: list[dict],
    candidates: dict,
    split_record: dict,
    summary: dict,
    brand_rows: list[dict],
    family: str,
    classifier_arch: str,
    checkpoint_path: Path,
    labels: torch.Tensor,
) -> Path:
    root = assert_experiment_output_dir(experiment_dir)
    destination = root / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    label_vector = labels.detach().reshape(-1).cpu()
    authentic = int((label_vector == 0).sum())
    fake = int((label_vector == 1).sum())
    temperature_payload = {
        "temperature": fit["temperature"],
        "optimizer_temperature": fit["optimizer_temperature"],
        "fitting_method": fit["fitting_method"],
        "fitting_loss": fit["fitting_loss"],
        "calibration_sample_count": int(label_vector.numel()),
        "authentic_count": authentic,
        "fake_count": fake,
        "generated_at": summary["generated_at"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "model_family": family,
        "classifier_arch": classifier_arch,
        "checkpoint_selection_completed": True,
        "calibration_fitted_after_selection": True,
        "non_production": bool(summary["non_production"]),
        "production_threshold": None,
    }
    (destination / "temperature.json").write_text(_dump(temperature_payload) + "\n")
    (destination / "calibration_metrics.json").write_text(
        _dump(
            {
                "baseline": _metrics_without_bins(baseline),
                "calibrated": _metrics_without_bins(calibrated),
                "comparison": comparison,
                "ece_definition": ECE_DEFINITION,
                "ece_bins": int(baseline["ece_bins"]),
                "brand_summaries": brand_rows,
                "classification_change_claimed": False,
            }
        )
        + "\n"
    )
    (destination / "calibration_summary.json").write_text(_dump(summary) + "\n")
    (destination / "calibration_manifest.json").write_text(_dump(split_record) + "\n")
    _write_prediction_csv(destination / "calibration_predictions.csv", prediction_rows)
    _write_dict_csv(destination / "threshold_candidates.csv", _threshold_csv_rows(sweep, candidates))
    bin_rows = []
    for kind, scored in (("baseline", baseline), ("calibrated", calibrated)):
        for row in scored["reliability_bins"]:
            bin_rows.append({"probabilities": kind, **row})
    _write_dict_csv(destination / "reliability_bins.csv", bin_rows)
    return destination


def _metrics_without_bins(scored: dict) -> dict:
    return {key: value for key, value in scored.items() if key != "reliability_bins"}


def _threshold_csv_rows(sweep: list[dict], candidates: dict) -> list[dict]:
    roles = {
        "highest_f1": _threshold_of(candidates.get("highest_f1")),
        "highest_recall": _threshold_of(candidates.get("highest_recall")),
        "lowest_far": _threshold_of(candidates.get("lowest_far")),
        "closest_to_0_50": _threshold_of(candidates.get("closest_to_0_50")),
    }
    safe = set(candidates["safe_thresholds"])
    rows = []
    for row in sweep:
        copied = dict(row)
        copied["safe"] = row["threshold"] in safe
        for name, threshold in roles.items():
            copied[name] = threshold is not None and row["threshold"] == threshold
        rows.append(copied)
    return rows


def _threshold_of(candidate: dict | None) -> float | None:
    if candidate is None:
        return None
    return float(candidate["threshold"])


def _pick(rows: list[dict], metric: str, *, prefer_high: bool) -> dict | None:
    usable = [row for row in rows if row.get(metric) is not None]
    if not usable:
        return None

    def key(row: dict) -> tuple:
        value = float(row[metric])
        far = float(row["false_authentic_rate"])
        distance = abs(float(row["threshold"]) - 0.5)
        primary = -value if prefer_high else value
        return (primary, far, distance, float(row["threshold"]))

    return min(usable, key=key)


def _closest_to_half(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    def key(row: dict) -> tuple:
        return (
            abs(float(row["threshold"]) - 0.5),
            float(row["false_authentic_rate"]),
            float(row["threshold"]),
        )

    return min(rows, key=key)


def _labels(labels: torch.Tensor, expected: int) -> torch.Tensor:
    target = labels.detach().float().reshape(-1)
    if target.numel() != expected:
        raise ValueError(f"logits and labels differ in length: {expected} vs {target.numel()}")
    if not torch.isfinite(target).all():
        raise ValueError("calibration labels must be finite")
    if bool(((target != 0) & (target != 1)).any()):
        raise ValueError("calibration labels must be 0 or 1")
    return target


def _require_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    return value


def _same_optional(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= 1e-12


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _write_prediction_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "brand",
        "label",
        "logit",
        "probability_at_0_5",
        "calibrated_probability",
    ]
    _write_dict_csv(path, rows, fields)


def _write_dict_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


class _CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[SampleRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        from PIL import Image

        image = Image.open(record.image_path).convert("RGB")
        return self.transform(image), record.label, str(record.image_path.resolve())
