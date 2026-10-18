"""Summarize the four Phase 15 validation runs. Does not train or read the test set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

EXPERIMENTS = (
    {
        "family": "dinov2",
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "head": "cls_only",
        "input_size": 504,
        "directory": "dinov2_cls_only",
    },
    {
        "family": "dinov2",
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "head": "cls_patch_attention",
        "input_size": 504,
        "directory": "dinov2_cls_patch_attention",
    },
    {
        "family": "dinov3",
        "backbone": "vit_base_patch16_dinov3.lvd1689m",
        "head": "cls_only",
        "input_size": 512,
        "directory": "dinov3_cls_only",
    },
    {
        "family": "dinov3",
        "backbone": "vit_base_patch16_dinov3.lvd1689m",
        "head": "cls_patch_attention",
        "input_size": 512,
        "directory": "dinov3_cls_patch_attention",
    },
)

WINNER_RULE = (
    "Among safety-qualified validation checkpoints, highest PR-AUC, then higher F1, "
    "then lower false-authentic rate. The final test set is not used."
)
NO_WINNER = "NO ARCHITECTURE PRODUCED A SAFETY-VALIDATED CHECKPOINT"


def write_baseline_comparison(experiments_root: Path) -> dict:
    root = Path(experiments_root)
    rows = [_experiment_row(root / spec["directory"], spec) for spec in EXPERIMENTS]
    winner, status = _rank(rows)
    payload = {
        "winner": None if winner is None else winner["directory"],
        "winner_selection_rule": WINNER_RULE,
        "winner_status": status,
        "test_set_used": False,
        "production_threshold": None,
        "ood_used_for_selection": False,
        "experiments": rows,
    }
    destination = root / "baseline_comparison"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baseline_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(destination / "baseline_comparison.csv", rows, winner)
    return payload


def _experiment_row(directory: Path, spec: dict) -> dict:
    selection_path = directory / "selection" / "checkpoint_selection.json"
    config_path = directory / "config.json"
    resource_path = directory / "resource.json"
    dataset_path = directory / "dataset_report.json"
    manifest_path = directory / "split_manifest.json"
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
        "seed": None,
        "split_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "brand_summary": _brand_summary(brand_path),
        "test_set_used_for_selection": False,
        "production_threshold": None,
    }
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        row["hyperparameters"] = {
            "batch_size": config.get("batch_size"),
            "accumulate_grad": config.get("accumulate_grad"),
            "effective_batch": config.get("effective_batch"),
            "amp_dtype": config.get("amp_dtype"),
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
            "weighted_sampler": config.get("weighted_sampler"),
            "max_false_authentic_rate": config.get("max_false_authentic_rate"),
            "validation_threshold": config.get("validation_threshold"),
            "augmentation_profile": config.get("augmentation_profile"),
            "split_strategy": config.get("split_strategy"),
            "seed": config.get("seed"),
        }
        row["seed"] = config.get("seed")
        row["test_root"] = config.get("test_root")
    if dataset_path.is_file():
        row["dataset"] = json.loads(dataset_path.read_text())
    if resource_path.is_file():
        resource = json.loads(resource_path.read_text())
        row["peak_allocated_bytes"] = resource.get("peak_allocated_bytes")
        row["peak_reserved_bytes"] = resource.get("peak_reserved_bytes")
        row["gpu_name"] = resource.get("gpu_name")
        row["test_set_used_for_selection"] = bool(resource.get("test_set_used_for_selection"))
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        summary = selection.get("summary", selection)
        row["safety_validated"] = bool(summary.get("safety_validated"))
        metrics = summary.get("selected_metrics") or {}
        row["checkpoint"] = summary.get("selected_checkpoint")
        row["epoch"] = metrics.get("epoch")
        row["pr_auc"] = metrics.get("pr_auc")
        row["f1"] = metrics.get("f1")
        row["false_authentic_rate"] = metrics.get("false_authentic_rate")
        row["roc_auc"] = metrics.get("roc_auc")
        row["validation_loss"] = metrics.get("validation_loss")
        row["threshold"] = metrics.get("threshold")
        row["threshold_policy"] = metrics.get("threshold_policy")
    history_path = directory / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        epochs = history.get("epochs", [])
        row["runtime_seconds"] = sum(float(item.get("epoch_seconds") or 0) for item in epochs)
        if epochs and row["peak_allocated_bytes"] is None:
            row["peak_allocated_bytes"] = epochs[-1].get("peak_allocated_bytes")
            row["peak_reserved_bytes"] = epochs[-1].get("peak_reserved_bytes")
        losses = [item.get("val_loss") for item in epochs if item.get("val_loss") is not None]
        if losses:
            row["best_validation_loss"] = min(losses)
        row["history_contains_test"] = "test" in history
    return row


def _brand_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    diagnostics = report.get("diagnostics") or {}
    brands = []
    for brand in report.get("brands") or []:
        brands.append(
            {
                "brand": brand.get("brand"),
                "sample_count": brand.get("sample_count"),
                "sample_status": brand.get("sample_status"),
                "false_authentic_rate": brand.get("false_authentic_rate"),
                "brand_far_eligible": brand.get("brand_far_eligible"),
            }
        )
    above = [
        brand["brand"]
        for brand in brands
        if brand["false_authentic_rate"] is not None and brand["brand_far_eligible"] is False
    ]
    return {
        "global_false_authentic_rate": (report.get("global") or {}).get("false_authentic_rate"),
        "macro_false_authentic_rate": (report.get("macro") or {}).get("false_authentic_rate"),
        "brands_above_far_limit": above,
        "low_sample_brands": diagnostics.get("low_sample_brands"),
        "unknown_brand_groups": diagnostics.get("unknown_brand_groups"),
        "unknown_brand_samples": diagnostics.get("unknown_brand_samples"),
        "checkpoint_selection_affected": report.get("checkpoint_selection_affected"),
        "brands": brands,
    }


def _rank(rows: list[dict]) -> tuple[dict | None, str]:
    qualified = [
        row
        for row in rows
        if row["safety_validated"]
        and row["false_authentic_rate"] is not None
        and float(row["false_authentic_rate"]) <= 0.02
        and row["pr_auc"] is not None
    ]
    if not qualified:
        return None, NO_WINNER

    def key(row: dict) -> tuple:
        f1 = row["f1"]
        return (
            -float(row["pr_auc"]),
            -(float(f1) if f1 is not None else -1.0),
            float(row["false_authentic_rate"]),
        )

    ordered = sorted(qualified, key=key)
    if len(ordered) > 1 and key(ordered[0]) == key(ordered[1]):
        return None, "Safety-qualified models tied on PR-AUC, F1, and false-authentic rate."
    return ordered[0], "safety-qualified validation checkpoint selected"


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
        "peak_allocated_bytes",
        "runtime_seconds",
        "winner",
    ]
    winner_name = None if winner is None else winner["directory"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
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
                    "peak_allocated_bytes": row["peak_allocated_bytes"],
                    "runtime_seconds": row["runtime_seconds"],
                    "winner": row["directory"] == winner_name,
                }
            )
