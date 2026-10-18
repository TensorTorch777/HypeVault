"""Re-rank saved Phase 17 validation histories. Does not train or open images.

The revised selector lives in checkpoint_selection. This module only reads the
Phase 17 selection reports and the Phase 16 manifest, then writes a new report
under experiments/checkpoint_policy_v2/. It does not copy checkpoints, fit a
temperature, or read the final test split as images.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from checkpoint_selection import (
    EARLY_STOPPING_POLICY,
    SELECTION_RULE,
    SELECTION_VERSION,
    new_selection_state,
    select_checkpoints,
    update_selection,
)
from split_v2 import (
    AUTHORITATIVE_COUNTS,
    AUTHORITATIVE_MEMBERSHIP_HASH,
    CATALOG_BRANDS,
    PHASE15_EXPERIMENT_DIRS,
    SPLIT_NAMES,
    STRATEGY,
    membership_hash,
)

PHASE17_EXPERIMENTS = (
    {"directory": "v2_dinov2_cls_only", "family": "dinov2", "head": "cls_only"},
    {"directory": "v2_dinov2_cls_patch_attention", "family": "dinov2", "head": "cls_patch_attention"},
    {"directory": "v2_dinov3_cls_only", "family": "dinov3", "head": "cls_only"},
    {"directory": "v2_dinov3_cls_patch_attention", "family": "dinov3", "head": "cls_patch_attention"},
)
POLICY_DIR_NAME = "checkpoint_policy_v2"
EARLY_STOP_PATIENCE = 6
TIE_BEHAVIOR = (
    "Ranking differences inside metric_tolerance are ties. "
    "The false-authentic safety limit stays exact. "
    "Among eligible checkpoints, higher PR-AUC wins, then higher F1, then lower "
    "false-authentic rate, then lower validation loss. Remaining ties keep the "
    "earlier epoch, then the lower global step, then the lexicographically smaller "
    "checkpoint id. A later checkpoint does not replace an earlier one when the "
    "PR-AUC, F1, false-authentic rate, and validation-loss bands match."
)


def build_policy_report(experiments_root: Path, manifest_path: Path | None = None) -> dict:
    """Read saved metrics and manifest metadata. Does not open images or checkpoints."""
    root = Path(experiments_root)
    manifest_file = Path(manifest_path) if manifest_path is not None else (
        root / "dataset_audit" / "split_manifest_v2.json"
    )
    manifest = json.loads(manifest_file.read_text())
    experiments = []
    for spec in PHASE17_EXPERIMENTS:
        experiments.append(_rerank_experiment(root / spec["directory"], spec))
    return {
        "phase": 18,
        "selection_version": SELECTION_VERSION,
        "selection_rule": SELECTION_RULE,
        "early_stopping_policy": EARLY_STOPPING_POLICY,
        "tie_behavior": TIE_BEHAVIOR,
        "metric_tolerance": 1e-6,
        "max_false_authentic_rate": 0.02,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "architecture_winner": None,
        "architecture_ranking_performed": False,
        "training_performed": False,
        "test_images_loaded": False,
        "temperature": "not_fitted",
        "production_threshold": None,
        "ood_used": False,
        "phase17_checkpoint_bytes_modified": False,
        "experiments": experiments,
        "calibration": assess_calibration_readiness(manifest),
    }


def assess_calibration_readiness(manifest: dict) -> dict:
    """Structural calibration checks from manifest records. Does not open image files."""
    failures: list[str] = []
    samples = manifest.get("samples")
    membership = manifest.get("membership") if isinstance(manifest.get("membership"), dict) else {}
    if manifest.get("strategy") != STRATEGY:
        failures.append("strategy is not brand_stratified_duplicate_safe")
    if not isinstance(samples, list) or not samples:
        failures.append("manifest has no samples")
        samples = []
    recorded_membership = {
        name: [str(path) for path in list(membership.get(name) or [])] for name in SPLIT_NAMES
    }
    digest = membership_hash(recorded_membership) if all(name in membership for name in SPLIT_NAMES) else None
    if digest != manifest.get("membership_hash"):
        failures.append("stored membership hash does not match the membership lists")
    if digest != AUTHORITATIVE_MEMBERSHIP_HASH:
        failures.append("membership hash does not match the Phase 16 manifest")
    policy = manifest.get("calibration_policy") or {}
    if policy.get("weight_updates_allowed") is not False:
        failures.append("calibration weight_updates_allowed is not false")

    by_split: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for row in samples:
        split = row.get("split")
        if split in by_split:
            by_split[split].append(row)
    paths = {name: {_resolved(row.get("path")) for row in rows} for name, rows in by_split.items()}
    calibration_rows = by_split["calibration"]
    calibration_membership = {_resolved(path) for path in recorded_membership.get("calibration", [])}
    if not calibration_rows or not calibration_membership:
        failures.append("calibration split is missing")
    if paths["calibration"] != calibration_membership:
        failures.append("calibration samples do not match membership.calibration")
    for name, expected in AUTHORITATIVE_COUNTS.items():
        rows = by_split[name]
        authentic = sum(1 for row in rows if row.get("label") == 0)
        fake = sum(1 for row in rows if row.get("label") == 1)
        if (len(rows), authentic, fake) != (expected["samples"], expected["authentic"], expected["fake"]):
            failures.append(f"{name} counts do not match the Phase 16 manifest")

    authentic = sum(1 for row in calibration_rows if row.get("label") == 0)
    fake = sum(1 for row in calibration_rows if row.get("label") == 1)
    brand_rows = []
    for brand in CATALOG_BRANDS:
        brand_authentic = sum(1 for row in calibration_rows if row.get("brand") == brand and row.get("label") == 0)
        brand_fake = sum(1 for row in calibration_rows if row.get("brand") == brand and row.get("label") == 1)
        brand_rows.append(
            {
                "brand": brand,
                "authentic": brand_authentic,
                "fake": brand_fake,
                "total": brand_authentic + brand_fake,
            }
        )
        if brand_authentic == 0 or brand_fake == 0:
            failures.append(f"calibration is missing a class for {brand}")
    present_brands = {row.get("brand") for row in calibration_rows}
    if any(brand not in present_brands for brand in CATALOG_BRANDS):
        failures.append("calibration is missing a catalog brand")

    for other in ("train", "validation", "test"):
        overlap = paths["calibration"] & paths[other]
        if overlap:
            failures.append(f"calibration overlaps {other} ({len(overlap)} paths)")
    group_owners: dict[str, str] = {}
    crossing: list[str] = []
    for name in SPLIT_NAMES:
        for row in by_split[name]:
            group_id = row.get("duplicate_group_id")
            if not isinstance(group_id, str) or not group_id:
                failures.append("sample is missing duplicate_group_id")
                continue
            previous = group_owners.get(group_id)
            if previous is not None and previous != name:
                crossing.append(group_id)
                group_owners[group_id] = previous
            else:
                group_owners[group_id] = name
    if crossing:
        failures.append(f"{len(set(crossing))} duplicate groups cross splits")

    calibration_groups: dict[str, int] = {}
    for row in calibration_rows:
        group_id = row.get("duplicate_group_id")
        if isinstance(group_id, str):
            calibration_groups[group_id] = calibration_groups.get(group_id, 0) + 1
    total = len(calibration_rows)
    return {
        "calibration_ready": not failures,
        "failures": failures,
        "membership_hash": digest,
        "expected_membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "calibration_membership_source": "split_manifest_v2",
        "weight_updates_allowed": False,
        "samples": total,
        "authentic": authentic,
        "fake": fake,
        "authentic_fraction": None if total == 0 else authentic / total,
        "fake_fraction": None if total == 0 else fake / total,
        "brands": brand_rows,
        "all_five_brands_in_both_classes": all(
            row["authentic"] > 0 and row["fake"] > 0 for row in brand_rows
        ) and len(brand_rows) == len(CATALOG_BRANDS),
        "duplicate_groups": len(calibration_groups),
        "duplicate_groups_larger_than_one": sum(1 for size in calibration_groups.values() if size > 1),
        "largest_calibration_duplicate_group": max(calibration_groups.values(), default=0),
        "duplicate_groups_cross_splits": len(set(crossing)),
        "overlaps": {
            other: len(paths["calibration"] & paths[other]) for other in ("train", "validation", "test")
        },
        "statistical_sufficiency": "not_declared",
        "statistical_sufficiency_note": (
            "Calibration size and balance are reported. A non-empty split is not "
            "treated as proof that temperature fitting is stable."
        ),
        "temperature": "not_fitted",
        "test_images_loaded": False,
        "test_metrics_computed": False,
    }


def write_policy_report(report: dict, destination_dir: Path) -> Path:
    """Write the Phase 18 report. Refuses Phase 15, Phase 17, and checkpoint directories."""
    destination = _assert_policy_destination(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoint_policy_v2.json").write_text(json.dumps(report, indent=2) + "\n")
    fieldnames = [
        "directory",
        "family",
        "head",
        "old_selected_epoch",
        "new_selected_epoch",
        "old_selected_PR_AUC",
        "new_selected_PR_AUC",
        "old_selected_F1",
        "new_selected_F1",
        "old_selected_FAR",
        "new_selected_FAR",
        "old_selected_loss",
        "new_selected_loss",
        "policy_v2_selected_checkpoint",
        "weights_file_present",
    ]
    with (destination / "checkpoint_policy_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report["experiments"]:
            writer.writerow(row)
    return destination


def _rerank_experiment(directory: Path, spec: dict) -> dict:
    selection_path = directory / "selection" / "checkpoint_selection.json"
    payload = json.loads(selection_path.read_text())
    old = payload.get("selected_metrics") or {}
    summary = select_checkpoints(
        list(payload.get("candidates") or []),
        max_false_authentic_rate=payload["max_false_authentic_rate"],
        metric_tolerance=payload["metric_tolerance"],
        expected_family=spec["family"],
        expected_classifier_arch=spec["head"],
    )
    chosen = summary["selected_metrics"] or {}
    epoch = chosen.get("epoch")
    checkpoint_id = chosen.get("checkpoint_id")
    weights = directory / f"epoch_{int(epoch):03d}.pt" if isinstance(epoch, int) else None
    present = bool(weights is not None and weights.is_file())
    replay = _replay_early_stopping(
        list(payload.get("candidates") or []),
        family=spec["family"],
        head=spec["head"],
        max_false_authentic_rate=payload["max_false_authentic_rate"],
        metric_tolerance=payload["metric_tolerance"],
    )
    return {
        "directory": spec["directory"],
        "family": spec["family"],
        "head": spec["head"],
        "old_selection_version": payload.get("selection_version"),
        "old_selected_checkpoint": payload.get("selected_checkpoint"),
        "old_selected_epoch": old.get("epoch"),
        "new_selected_checkpoint": checkpoint_id,
        "new_selected_epoch": epoch,
        "policy_v2_selected_checkpoint": str(weights) if present else None,
        "policy_v2_selected_checkpoint_id": checkpoint_id,
        "weights_file_present": present,
        "weights_note": None if present else (
            "The revised checkpoint id is recorded from saved validation metrics. "
            "Its epoch file is not in the experiment directory, and best_model.pt was not replaced."
        ),
        "old_selected_PR_AUC": old.get("pr_auc"),
        "new_selected_PR_AUC": chosen.get("pr_auc"),
        "old_selected_F1": old.get("f1"),
        "new_selected_F1": chosen.get("f1"),
        "old_selected_FAR": old.get("false_authentic_rate"),
        "new_selected_FAR": chosen.get("false_authentic_rate"),
        "old_selected_loss": old.get("validation_loss"),
        "new_selected_loss": chosen.get("validation_loss"),
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_halt_epoch": replay["halt_epoch"],
        "checkpoint_selected_when_patience_exhausted": replay["selected_checkpoint"],
        "best_model_pt_replaced": False,
    }


def _replay_early_stopping(
    candidates: list[dict],
    *,
    family: str,
    head: str,
    max_false_authentic_rate: float,
    metric_tolerance: float,
) -> dict:
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row["epoch"]),
            -1 if row.get("global_step") is None else int(row["global_step"]),
            str(row["checkpoint_id"]),
        ),
    )
    state = new_selection_state(
        max_false_authentic_rate=max_false_authentic_rate,
        metric_tolerance=metric_tolerance,
        validation_threshold=0.5,
        model_family=family,
        classifier_arch=head,
    )
    halt_epoch = None
    for candidate in ordered:
        state = update_selection(state, candidate)
        if state["safety_validated"] and state["epochs_without_improvement"] >= EARLY_STOP_PATIENCE:
            halt_epoch = int(candidate["epoch"])
            break
    return {
        "halt_epoch": halt_epoch,
        "selected_checkpoint": state["selected_checkpoint"],
    }


def _assert_policy_destination(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.name != POLICY_DIR_NAME:
        raise RuntimeError(f"Phase 18 output must be named {POLICY_DIR_NAME}")
    blocked = set(PHASE15_EXPERIMENT_DIRS) | {spec["directory"] for spec in PHASE17_EXPERIMENTS}
    blocked.update({"checkpoints", "checkpoints_watches", "dataset_audit"})
    if any(part in blocked for part in resolved.parts):
        raise RuntimeError(f"refusing to write the Phase 18 report into {resolved}")
    return resolved


def _resolved(path: object) -> str:
    if not isinstance(path, str) or not path:
        return ""
    return str(Path(path).resolve())


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
    report = build_policy_report(root)
    destination = write_policy_report(report, root / POLICY_DIR_NAME)
    ready = report["calibration"]["calibration_ready"]
    epochs = {
        row["directory"]: {
            "old": row["old_selected_epoch"],
            "new": row["new_selected_epoch"],
        }
        for row in report["experiments"]
    }
    print(json.dumps({"destination": str(destination), "calibration_ready": ready, "epochs": epochs}, indent=2))


if __name__ == "__main__":
    main()
