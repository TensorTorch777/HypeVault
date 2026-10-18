"""Select Phase 17 checkpoints only when the epoch file is a real artifact.

This module reads saved validation metrics and epoch_*.pt files. It does not
train, rewrite Phase 17 history, copy best_model.pt, fit a temperature, or
open the final test split. Missing weights are not reconstructed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from checkpoint_policy_v2 import PHASE17_EXPERIMENTS
from checkpoint_selection import (
    SELECTION_RULE,
    SELECTION_VERSION,
    epoch_artifact_path,
    select_checkpoints,
)
from model import DINOV2_IMAGE_SIZE, DINOV3_IMAGE_SIZE
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH, PHASE15_EXPERIMENT_DIRS

POLICY_DIR_NAME = "checkpoint_policy_v3"
PREPROCESSING_VERSION = "resize_pad_square_eval_v1"
MISSING_ARTIFACT_REASON = "no checkpoint artifact"
HISTORICAL_NOTE = (
    "Phase 17 saved epoch_*.pt only when epoch % save_every == 0, and save_every was 2. "
    "Odd epochs have validation metrics and no weight file. Those epochs could not have "
    "been selected online, and this phase does not rewrite that history or regenerate training."
)
ARTIFACT_GATE = (
    "A candidate is eligible only when its validation metrics exist, its metadata names "
    "the expected model family and head, its epoch_*.pt file exists and loads, the state "
    "dict contains the expected weights, and the checkpoint belongs to this experiment "
    "with a matching manifest hash, input size, and preprocessing version. "
    "best_model.pt, latest.pt, final_model.pt, and ONNX files are not substitutes."
)


def build_policy_report(experiments_root: Path, *, validate_selected: bool = True) -> tuple[dict, dict]:
    """Re-rank real epoch files. Returns the policy report and the artifact audit."""
    root = Path(experiments_root)
    experiments = []
    audits = []
    for spec in PHASE17_EXPERIMENTS:
        row, audit = _rank_experiment(root / spec["directory"], spec, validate_selected=validate_selected)
        experiments.append(row)
        audits.append(audit)
    report = {
        "phase": 19,
        "selection_version": SELECTION_VERSION,
        "selection_rule": SELECTION_RULE,
        "artifact_gate": ARTIFACT_GATE,
        "ranking_order": (
            "Filter to real checkpoint artifacts first, then apply the Phase 18 ranking unchanged."
        ),
        "metric_tolerance": 1e-6,
        "max_false_authentic_rate": 0.02,
        "historical_early_stopping_note": HISTORICAL_NOTE,
        "architecture_winner": None,
        "training_performed": False,
        "test_images_loaded": False,
        "temperature": "not_fitted",
        "production_threshold": None,
        "ood_used": False,
        "phase17_checkpoint_bytes_modified": False,
        "experiments": experiments,
    }
    audit = {
        "phase": 19,
        "missing_artifact_reason": MISSING_ARTIFACT_REASON,
        "experiments": audits,
    }
    return report, audit


def write_policy_report(report: dict, audit: dict, destination_dir: Path) -> Path:
    """Write Phase 19 artifacts. Refuses Phase 15, Phase 17, and checkpoint directories."""
    destination = _assert_policy_destination(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoint_policy_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "checkpoint_artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fieldnames = [
        "directory",
        "family",
        "head",
        "history_selected_epoch",
        "artifact_selected_epoch",
        "history_candidate_excluded",
        "excluded_candidate_reason",
        "selected_checkpoint_path",
        "selected_checkpoint_exists",
        "checkpoint_load_valid",
        "manifest_hash",
        "model_family",
        "classifier_head",
        "pr_auc",
        "f1",
        "false_authentic_rate",
        "validation_loss",
        "calibration_checkpoint_path",
    ]
    with (destination / "checkpoint_policy_v3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report["experiments"]:
            writer.writerow(row)
    return destination


def _rank_experiment(directory: Path, spec: dict, *, validate_selected: bool) -> tuple[dict, dict]:
    selection_path = directory / "selection" / "checkpoint_selection.json"
    config = json.loads((directory / "config.json").read_text())
    payload = json.loads(selection_path.read_text())
    history_summary = select_checkpoints(
        list(payload.get("candidates") or []),
        max_false_authentic_rate=payload["max_false_authentic_rate"],
        metric_tolerance=payload["metric_tolerance"],
        expected_family=spec["family"],
        expected_classifier_arch=spec["head"],
    )
    history_chosen = history_summary["selected_metrics"] or {}
    history_epoch = history_chosen.get("epoch")
    artifact_candidates = [
        _with_artifact(candidate, directory) for candidate in list(payload.get("candidates") or [])
    ]
    expectations = None
    if validate_selected:
        expectations = _expectations(directory, spec, config)
    artifact_summary = select_checkpoints(
        artifact_candidates,
        max_false_authentic_rate=payload["max_false_authentic_rate"],
        metric_tolerance=payload["metric_tolerance"],
        expected_family=spec["family"],
        expected_classifier_arch=spec["head"],
        artifact_expectations=expectations,
    )
    chosen = artifact_summary["selected_metrics"] or {}
    selected_path = chosen.get("checkpoint_path")
    exists = isinstance(selected_path, str) and Path(selected_path).is_file()
    validation = chosen.get("checkpoint_validation") or {}
    load_valid = validation.get("checkpoint_load_valid")
    if not validate_selected:
        load_valid = None
    history_path = epoch_artifact_path(directory, history_epoch) if isinstance(history_epoch, int) else None
    history_missing = history_path is None or not history_path.is_file()
    audit_rows = _audit_rows(directory, payload.get("candidates") or [], artifact_summary["candidates"])
    return (
        {
            "directory": spec["directory"],
            "family": spec["family"],
            "head": spec["head"],
            "save_every": config.get("save_every"),
            "history_selected_epoch": history_epoch,
            "artifact_selected_epoch": chosen.get("epoch"),
            "history_candidate_excluded": history_missing,
            "excluded_candidate_reason": MISSING_ARTIFACT_REASON if history_missing else None,
            "excluded_metric_epochs": [
                row["epoch"] for row in audit_rows if row["metrics_available"] and not row["checkpoint_available"]
            ],
            "selected_checkpoint_path": selected_path if exists else None,
            "selected_checkpoint_exists": exists,
            "checkpoint_load_valid": load_valid,
            "manifest_hash": validation.get("manifest_membership_hash") or config.get("manifest_membership_hash"),
            "model_family": validation.get("model_family") or spec["family"],
            "classifier_head": validation.get("classifier_arch") or spec["head"],
            "input_size": validation.get("input_size"),
            "preprocessing_version": validation.get("preprocessing_version"),
            "pr_auc": chosen.get("pr_auc"),
            "f1": chosen.get("f1"),
            "false_authentic_rate": chosen.get("false_authentic_rate"),
            "validation_loss": chosen.get("validation_loss"),
            "calibration_checkpoint_path": selected_path if exists and load_valid else None,
            "best_model_pt_replaced": False,
        },
        {"directory": spec["directory"], "rows": audit_rows},
    )


def _with_artifact(candidate: dict, directory: Path) -> dict:
    row = dict(candidate)
    epoch = row.get("epoch")
    if isinstance(epoch, int) and not isinstance(epoch, bool):
        path = epoch_artifact_path(directory, epoch)
        row["checkpoint_path"] = str(path)
        row["checkpoint_available"] = path.is_file()
    else:
        row["checkpoint_path"] = None
        row["checkpoint_available"] = False
    return row


def _expectations(directory: Path, spec: dict, config: dict) -> dict:
    expected_size = DINOV2_IMAGE_SIZE if spec["family"] == "dinov2" else DINOV3_IMAGE_SIZE
    if config.get("family") != spec["family"] or config.get("classifier_arch") != spec["head"]:
        raise ValueError(f"{directory} config does not match the Phase 17 experiment")
    if config.get("input_size") != expected_size or config.get("img_size") != expected_size:
        raise ValueError(f"{directory} input size does not match {spec['family']}")
    if config.get("manifest_membership_hash") != AUTHORITATIVE_MEMBERSHIP_HASH:
        raise ValueError(f"{directory} manifest hash does not match Phase 16")
    if config.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError(f"{directory} preprocessing version is not {PREPROCESSING_VERSION}")
    if Path(config.get("output_dir", "")).resolve() != directory.resolve():
        raise ValueError(f"{directory} output_dir does not match the experiment directory")
    return {
        "expected_input_size": expected_size,
        "expected_membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "expected_preprocessing_version": PREPROCESSING_VERSION,
        "expected_experiment_dir": directory,
    }


def _audit_rows(directory: Path, history_candidates: list[dict], ranked: list[dict]) -> list[dict]:
    ranked_by_id = {row.get("checkpoint_id"): row for row in ranked}
    seen: set[str] = set()
    rows = []
    for candidate in history_candidates:
        epoch = candidate.get("epoch")
        checkpoint_id = candidate.get("checkpoint_id")
        path = epoch_artifact_path(directory, epoch) if isinstance(epoch, int) else None
        available = bool(path is not None and path.is_file())
        ranked_row = ranked_by_id.get(checkpoint_id) or {}
        reason = ranked_row.get("rejection_reason")
        if not available:
            reason = MISSING_ARTIFACT_REASON
        rows.append(
            {
                "epoch": epoch,
                "checkpoint_id": checkpoint_id,
                "metrics_available": True,
                "checkpoint_available": available,
                "checkpoint_path": None if path is None else str(path),
                "eligible": bool(available and ranked_row.get("far_eligible")),
                "reason": reason,
            }
        )
        if isinstance(checkpoint_id, str):
            seen.add(checkpoint_id)
    for path in sorted(directory.glob("epoch_*.pt")):
        checkpoint_id = path.stem
        if checkpoint_id in seen:
            continue
        rows.append(
            {
                "epoch": None,
                "checkpoint_id": checkpoint_id,
                "metrics_available": False,
                "checkpoint_available": True,
                "checkpoint_path": str(path),
                "eligible": False,
                "reason": "checkpoint has no validation metrics",
            }
        )
    return rows


def _assert_policy_destination(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.name != POLICY_DIR_NAME:
        raise RuntimeError(f"Phase 19 output must be named {POLICY_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {spec["directory"] for spec in PHASE17_EXPERIMENTS}
    blocked.update({"checkpoints", "checkpoints_watches", "dataset_audit"})
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the Phase 19 report into {resolved}")
    return resolved


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    report, audit = build_policy_report(root, validate_selected=True)
    destination = write_policy_report(report, audit, root / POLICY_DIR_NAME)
    summary = {
        "destination": str(destination),
        "calibration_checkpoints": {
            row["directory"]: row["calibration_checkpoint_path"] for row in report["experiments"]
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
