"""Validation-only summary of the Phase 17 runs. Does not read the test set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from baseline_comparison import NO_WINNER, WINNER_RULE, _rank

EXPERIMENTS = (
    {
        "family": "dinov2",
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "head": "cls_only",
        "input_size": 504,
        "directory": "v2_dinov2_cls_only",
        "phase15_directory": "dinov2_cls_only",
    },
    {
        "family": "dinov2",
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "head": "cls_patch_attention",
        "input_size": 504,
        "directory": "v2_dinov2_cls_patch_attention",
        "phase15_directory": "dinov2_cls_patch_attention",
    },
    {
        "family": "dinov3",
        "backbone": "vit_base_patch16_dinov3.lvd1689m",
        "head": "cls_only",
        "input_size": 512,
        "directory": "v2_dinov3_cls_only",
        "phase15_directory": "dinov3_cls_only",
    },
    {
        "family": "dinov3",
        "backbone": "vit_base_patch16_dinov3.lvd1689m",
        "head": "cls_patch_attention",
        "input_size": 512,
        "directory": "v2_dinov3_cls_patch_attention",
        "phase15_directory": "dinov3_cls_patch_attention",
    },
)


def write_baseline_comparison_v2(experiments_root: Path) -> dict:
    root = Path(experiments_root)
    phase15 = _phase15_index(root / "baseline_comparison" / "baseline_comparison.json")
    rows = [_experiment_row(root / spec["directory"], spec, phase15.get(spec["phase15_directory"])) for spec in EXPERIMENTS]
    winner, status = _rank(rows)
    payload = {
        "winner": None if winner is None else winner["directory"],
        "winner_selection_rule": WINNER_RULE,
        "winner_status": status,
        "ranking_source": "phase17_validation_only",
        "phase15_used_for_ranking": False,
        "test_set_used": False,
        "production_threshold": None,
        "temperature": "not_fitted",
        "ood_used_for_selection": False,
        "experiments": rows,
    }
    destination = root / "baseline_comparison_v2"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baseline_comparison_v2.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(destination / "baseline_comparison_v2.csv", rows, winner)
    return payload


def _experiment_row(directory: Path, spec: dict, phase15: dict | None) -> dict:
    selection_path = directory / "selection" / "checkpoint_selection.json"
    config_path = directory / "config.json"
    resource_path = directory / "resource.json"
    brand_path = directory / "evaluation" / "brand_metrics.json"
    row = {
        "directory": spec["directory"],
        "family": spec["family"],
        "backbone": spec["backbone"],
        "head": spec["head"],
        "input_size": spec["input_size"],
        "completed": selection_path.is_file(),
        "safety_validated": False,
        "checkpoint": None,
        "epoch": None,
        "pr_auc": None,
        "f1": None,
        "false_authentic_rate": None,
        "roc_auc": None,
        "validation_loss": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "runtime_seconds": None,
        "brand_summary": _brand_diagnostics(brand_path),
        "test_set_used_for_selection": False,
        "production_threshold": None,
        "phase15": phase15,
        "performance_delta": _delta(None, phase15),
    }
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        row["hyperparameters"] = {
            "manifest_path": config.get("fixed_split_manifest"),
            "manifest_membership_hash": config.get("manifest_membership_hash"),
            "split_strategy": config.get("split_strategy"),
            "calibration_membership_source": config.get("calibration_membership_source"),
            "dataset_counts": config.get("dataset_counts"),
            "seed": config.get("seed"),
            "family": config.get("family"),
            "backbone": config.get("model_name"),
            "head": config.get("classifier_arch"),
            "input_size": config.get("input_size"),
            "precision": config.get("amp_dtype"),
            "batch_size": config.get("batch_size"),
            "accumulate_grad": config.get("accumulate_grad"),
            "effective_batch": config.get("effective_batch"),
            "backbone_lr": config.get("backbone_lr"),
            "head_lr": config.get("head_lr"),
            "layer_decay": config.get("layer_decay"),
            "weight_decay": config.get("weight_decay"),
            "warmup_epochs": config.get("warmup_epochs"),
            "num_epochs": config.get("num_epochs"),
            "early_stop_patience": config.get("early_stop_patience"),
            "grad_clip": config.get("grad_clip"),
            "label_smoothing": config.get("label_smoothing"),
            "mixup_alpha": config.get("mixup_alpha"),
            "validation_threshold": config.get("validation_threshold"),
            "max_false_authentic_rate": config.get("max_false_authentic_rate"),
            "preprocessing_version": config.get("preprocessing_version"),
            "deterministic_algorithms": config.get("deterministic_algorithms"),
            "cudnn_benchmark": config.get("cudnn_benchmark"),
        }
        row["test_root"] = config.get("test_root")
    if resource_path.is_file():
        resource = json.loads(resource_path.read_text())
        row["peak_allocated_bytes"] = resource.get("peak_allocated_bytes")
        row["peak_reserved_bytes"] = resource.get("peak_reserved_bytes")
        row["gpu_name"] = resource.get("gpu_name")
        row["test_set_used_for_selection"] = bool(resource.get("test_set_used_for_selection"))
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        summary = selection.get("summary", selection)
        metrics = summary.get("selected_metrics") or {}
        row["safety_validated"] = bool(summary.get("safety_validated"))
        row["checkpoint"] = summary.get("selected_checkpoint")
        row["epoch"] = metrics.get("epoch")
        row["pr_auc"] = metrics.get("pr_auc")
        row["f1"] = metrics.get("f1")
        row["false_authentic_rate"] = metrics.get("false_authentic_rate")
        row["roc_auc"] = metrics.get("roc_auc")
        row["validation_loss"] = metrics.get("validation_loss")
        row["accuracy"] = metrics.get("accuracy")
        row["precision"] = metrics.get("precision")
        row["recall"] = metrics.get("recall")
        row["false_fake_rate"] = metrics.get("false_fake_rate")
        row["threshold"] = metrics.get("threshold")
        row["threshold_policy"] = metrics.get("threshold_policy")
    history_path = directory / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        epochs = history.get("epochs", [])
        row["runtime_seconds"] = sum(float(item.get("epoch_seconds") or 0) for item in epochs)
        row["history_contains_test"] = "test" in history
        selected = next((item for item in epochs if item.get("epoch") == row["epoch"]), None)
        if selected is not None:
            row["false_authentic"] = selected.get("val_false_authentic")
            row["false_fake"] = selected.get("val_false_fake")
    row["performance_delta"] = _delta(row, phase15)
    return row


def _brand_diagnostics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    brands = []
    rates = []
    for brand in report.get("brands") or []:
        rate = brand.get("false_authentic_rate")
        if rate is not None:
            rates.append(float(rate))
        brands.append(
            {
                "brand": brand.get("brand"),
                "authentic_count": brand.get("authentic_count"),
                "fake_count": brand.get("fake_count"),
                "false_authentic_rate": rate,
                "f1": brand.get("f1"),
                "recall": brand.get("recall"),
                "pr_auc": brand.get("pr_auc"),
            }
        )
    macro = report.get("macro") or {}
    global_metrics = report.get("global") or {}
    return {
        "global_false_authentic_rate": global_metrics.get("false_authentic_rate"),
        "global_pr_auc": global_metrics.get("pr_auc"),
        "macro_false_authentic_rate": macro.get("false_authentic_rate"),
        "macro_pr_auc": macro.get("pr_auc"),
        "macro_f1": macro.get("f1"),
        "macro_recall": macro.get("recall"),
        "worst_brand_far": None if not rates else max(rates),
        "brand_far_spread": None if not rates else max(rates) - min(rates),
        "checkpoint_selection_affected": report.get("checkpoint_selection_affected"),
        "brands": brands,
    }


def _phase15_index(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {
        row["directory"]: {
            "pr_auc": row.get("pr_auc"),
            "f1": row.get("f1"),
            "false_authentic_rate": row.get("false_authentic_rate"),
            "roc_auc": row.get("roc_auc"),
        }
        for row in payload.get("experiments") or []
    }


def _delta(current: dict | None, phase15: dict | None) -> dict:
    if not current or not phase15:
        return {"pr_auc": None, "f1": None, "false_authentic_rate": None}

    def subtract(name: str):
        left = current.get(name)
        right = phase15.get(name)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    return {
        "pr_auc": subtract("pr_auc"),
        "f1": subtract("f1"),
        "false_authentic_rate": subtract("false_authentic_rate"),
    }


def _write_csv(path: Path, rows: list[dict], winner: dict | None) -> None:
    fields = [
        "directory",
        "family",
        "head",
        "input_size",
        "safety_validated",
        "checkpoint",
        "epoch",
        "pr_auc",
        "f1",
        "false_authentic_rate",
        "roc_auc",
        "worst_brand_far",
        "peak_allocated_bytes",
        "runtime_seconds",
        "phase15_pr_auc",
        "phase15_false_authentic_rate",
        "delta_pr_auc",
        "delta_f1",
        "delta_false_authentic_rate",
        "winner",
    ]
    winner_name = None if winner is None else winner["directory"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            phase15 = row.get("phase15") or {}
            delta = row.get("performance_delta") or {}
            brand = row.get("brand_summary") or {}
            writer.writerow(
                {
                    "directory": row["directory"],
                    "family": row["family"],
                    "head": row["head"],
                    "input_size": row["input_size"],
                    "safety_validated": row["safety_validated"],
                    "checkpoint": row["checkpoint"],
                    "epoch": row["epoch"],
                    "pr_auc": row["pr_auc"],
                    "f1": row["f1"],
                    "false_authentic_rate": row["false_authentic_rate"],
                    "roc_auc": row["roc_auc"],
                    "worst_brand_far": brand.get("worst_brand_far"),
                    "peak_allocated_bytes": row["peak_allocated_bytes"],
                    "runtime_seconds": row["runtime_seconds"],
                    "phase15_pr_auc": phase15.get("pr_auc"),
                    "phase15_false_authentic_rate": phase15.get("false_authentic_rate"),
                    "delta_pr_auc": delta.get("pr_auc"),
                    "delta_f1": delta.get("f1"),
                    "delta_false_authentic_rate": delta.get("false_authentic_rate"),
                    "winner": row["directory"] == winner_name,
                }
            )


_ = NO_WINNER
