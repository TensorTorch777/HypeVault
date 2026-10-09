"""Safety-constrained validation checkpoint selection.

The positive class is fake. false_authentic_rate is false authentics divided by
actual fakes, the same rate evaluation.evaluate_logits already computes.

A checkpoint is production-eligible only when that rate is at or below
max_false_authentic_rate. Among those checkpoints the winner is the highest
PR-AUC. Ties use higher F1, then a lower false-authentic rate, then a lower
validation loss, then the earlier epoch, the lower global step, and finally
the lexicographically smaller checkpoint id.

The false-authentic constraint is exact. Tie bands apply only to ranking
comparisons: PR-AUC, F1, false-authentic rate, and validation loss are each
rounded to the nearest multiple of metric_tolerance (Python's banker's round).
A gap larger than the tolerance never ties. A later checkpoint does not replace
an earlier one when those ranking bands match. The final test split is not an
input.

When a candidate names a checkpoint path or sets checkpoint availability, the
file must already exist. A metric row with no artifact is not selectable, and
weights are not reconstructed. That gate does not change the ranking order
among candidates whose artifacts exist.

If no checkpoint satisfies the constraint, nothing is selected and
best_model.pt must not be written as a production checkpoint. Early stopping
does not consume patience until a safety-validated checkpoint exists. After
that, patience advances whenever the selected checkpoint stays the same,
including a later epoch whose PR-AUC, F1, false-authentic rate, and validation
loss are inside the tie bands. Only a ranking win outside those bands resets
patience.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from evaluation import DEFAULT_THRESHOLD
from training_config import assert_experiment_output_dir

DEFAULT_MAX_FALSE_AUTHENTIC_RATE = 0.02
DEFAULT_METRIC_TOLERANCE = 1e-6
DEFAULT_VALIDATION_THRESHOLD = DEFAULT_THRESHOLD
THRESHOLD_POLICY = "fixed_not_calibrated"
SELECTION_VERSION = "phase18-far-v2"
SELECTION_RULE = (
    "max PR-AUC subject to false_authentic_rate <= max_false_authentic_rate; "
    "tie-break higher F1; then lowest false-authentic rate; "
    "then lowest validation loss; then earlier epoch, lower global_step, "
    "then lexicographically smaller checkpoint id"
)
EARLY_STOPPING_POLICY = (
    "Patience advances only after a safety-validated checkpoint exists, and only "
    "when that checkpoint stays selected. A later epoch resets patience only when "
    "it wins on PR-AUC, F1, false-authentic rate, or validation loss outside "
    "metric_tolerance. A tie on those bands does not replace the earlier "
    "checkpoint and counts toward patience. An epoch that misses the "
    "false-authentic constraint does not become best_model.pt. Before the first "
    "qualified checkpoint, a miss does not consume patience."
)
UNQUALIFIED_STATUS = "no_safety_validated_checkpoint"
QUALIFIED_STATUS = "safety_validated"
UNQUALIFIED_MESSAGE = "NO SAFETY-VALIDATED CHECKPOINT"

_METRIC_FIELDS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_authentic_rate",
    "false_fake_rate",
    "threshold",
)
_CSV_FIELDS = (
    "checkpoint_id",
    "epoch",
    "global_step",
    "validation_loss",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_authentic_rate",
    "false_fake_rate",
    "threshold",
    "threshold_policy",
    "far_eligible",
    "selected",
    "eligible",
    "rejection_reason",
    "model_family",
    "classifier_arch",
)


def validate_max_false_authentic_rate(value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"max_false_authentic_rate must be in [0, 1], got {value}")
    return rate


def validate_metric_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"metric_tolerance must be >= 0, got {value}")
    return tolerance


def validate_validation_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"validation_threshold must be in [0, 1], got {value}")
    return threshold


def metric_band(value: float, tolerance: float) -> float:
    """Round a ranking metric into a tie band. The FAR limit does not use this."""
    number = float(value)
    if tolerance == 0.0:
        return number
    return round(number / tolerance) * tolerance


def candidate_from_validation(
    *,
    checkpoint_id: str,
    epoch: int,
    global_step: int | None,
    validation_loss: float,
    model_family: str,
    classifier_arch: str,
    metrics: dict,
    eligible: bool = True,
    rejection_reason: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_available: bool | None = None,
) -> dict:
    """Copy validation metrics only. Test-split fields are discarded."""
    copied = {field: metrics.get(field) for field in _METRIC_FIELDS}
    row = {
        "checkpoint_id": str(checkpoint_id),
        "epoch": int(epoch),
        "global_step": None if global_step is None else int(global_step),
        "validation_loss": float(validation_loss),
        "model_family": str(model_family),
        "classifier_arch": str(classifier_arch),
        "eligible": bool(eligible),
        "rejection_reason": rejection_reason,
        "threshold_policy": THRESHOLD_POLICY,
        **copied,
    }
    if checkpoint_path is not None or checkpoint_available is not None:
        row["checkpoint_path"] = None if checkpoint_path is None else str(checkpoint_path)
        row["checkpoint_available"] = (
            None if checkpoint_available is None else bool(checkpoint_available)
        )
    return row


def checkpoint_compatibility(
    payload: dict | None,
    *,
    expected_family: str,
    expected_classifier_arch: str,
) -> tuple[bool, str]:
    """Accept only a same-experiment validation checkpoint with model weights."""
    if not isinstance(payload, dict):
        return False, "checkpoint is not a dict"
    if payload.get("split") == "test":
        return False, "test split cannot be used for selection"
    weights = payload.get("model_state")
    if not isinstance(weights, dict) or not weights:
        return False, "missing model weights"
    config = payload.get("config")
    if not isinstance(config, dict):
        return False, "missing experiment config"
    family = config.get("family")
    architecture = config.get("classifier_arch")
    if family != expected_family:
        return False, f"model family {family!r} does not match {expected_family!r}"
    if architecture != expected_classifier_arch:
        return False, f"classifier head {architecture!r} does not match {expected_classifier_arch!r}"
    if "epoch" not in payload:
        return False, "missing epoch"
    return True, ""


def inspect_checkpoint_file(
    path: Path,
    *,
    expected_family: str,
    expected_classifier_arch: str,
) -> dict:
    """Read one checkpoint. Unreadable and incompatible files stay unselected."""
    checkpoint_id = Path(path).name
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return _rejected_file(checkpoint_id, f"unreadable checkpoint: {exc.__class__.__name__}")
    eligible, reason = checkpoint_compatibility(
        payload,
        expected_family=expected_family,
        expected_classifier_arch=expected_classifier_arch,
    )
    if not eligible:
        return _rejected_file(checkpoint_id, reason)
    return {
        "checkpoint_id": checkpoint_id,
        "eligible": True,
        "rejection_reason": None,
        "epoch": int(payload["epoch"]),
        "global_step": None if payload.get("global_step") is None else int(payload["global_step"]),
        "model_family": str(payload["config"]["family"]),
        "classifier_arch": str(payload["config"]["classifier_arch"]),
    }


def epoch_artifact_path(experiment_dir: Path, epoch: int) -> Path:
    """Path of the epoch file. best_model.pt is not a substitute."""
    return Path(experiment_dir) / f"epoch_{int(epoch):03d}.pt"


def epoch_artifact_is_saved(epoch: int, save_every: int) -> bool:
    interval = int(save_every)
    if interval < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")
    return int(epoch) % interval == 0


def validate_checkpoint_artifact(
    path: Path,
    *,
    expected_family: str,
    expected_classifier_arch: str,
    expected_epoch: int,
    expected_input_size: int,
    expected_membership_hash: str,
    expected_preprocessing_version: str,
    expected_experiment_dir: Path,
) -> dict:
    """Load one checkpoint and check identity. Does not run the dataset."""
    artifact = Path(path)
    result = {
        "checkpoint_path": str(artifact),
        "checkpoint_exists": artifact.is_file(),
        "checkpoint_load_valid": False,
        "reasons": [],
        "model_family": None,
        "classifier_arch": None,
        "input_size": None,
        "manifest_membership_hash": None,
        "preprocessing_version": None,
        "epoch": None,
        "parameter_keys_valid": False,
    }
    if not artifact.is_file():
        result["reasons"].append("no checkpoint artifact")
        return result
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
    except Exception as exc:
        result["reasons"].append(f"unreadable checkpoint: {exc.__class__.__name__}")
        return result
    try:
        compatible, reason = checkpoint_compatibility(
            payload,
            expected_family=expected_family,
            expected_classifier_arch=expected_classifier_arch,
        )
        if not compatible:
            result["reasons"].append(reason)
        config = payload.get("config") if isinstance(payload, dict) else None
        weights = payload.get("model_state") if isinstance(payload, dict) else None
        result["reasons"].extend(_weight_key_reasons(weights, expected_classifier_arch))
        if isinstance(config, dict):
            result["model_family"] = config.get("family")
            result["classifier_arch"] = config.get("classifier_arch")
            result["manifest_membership_hash"] = config.get("manifest_membership_hash")
            result["preprocessing_version"] = config.get("preprocessing_version")
            recorded_size = config.get("input_size", config.get("img_size"))
            result["input_size"] = recorded_size
            if recorded_size != expected_input_size:
                result["reasons"].append(
                    f"input size {recorded_size!r} does not match {expected_input_size}"
                )
            if config.get("manifest_membership_hash") != expected_membership_hash:
                result["reasons"].append("manifest membership hash does not match")
            if config.get("preprocessing_version") != expected_preprocessing_version:
                result["reasons"].append("preprocessing version does not match")
            recorded_dir = config.get("output_dir")
            if not isinstance(recorded_dir, str) or Path(recorded_dir).resolve() != Path(expected_experiment_dir).resolve():
                result["reasons"].append("checkpoint does not belong to this experiment")
        if isinstance(payload, dict) and payload.get("epoch") != expected_epoch:
            result["reasons"].append("checkpoint epoch does not match the selected epoch")
            result["epoch"] = payload.get("epoch") if isinstance(payload, dict) else None
        elif isinstance(payload, dict):
            result["epoch"] = payload.get("epoch")
        result["parameter_keys_valid"] = not any(
            reason.startswith("missing parameter") or reason.startswith("missing backbone") or "patch-attention" in reason
            for reason in result["reasons"]
        )
        result["checkpoint_load_valid"] = not result["reasons"]
        return result
    finally:
        del payload


def _weight_key_reasons(weights: object, classifier_arch: str) -> list[str]:
    if not isinstance(weights, dict) or not weights:
        return []
    reasons: list[str] = []
    for key in ("trunk.norm.weight", "trunk.fc1.weight", "trunk.fc2.weight"):
        if key not in weights:
            reasons.append(f"missing parameter {key}")
    if not any(str(key).startswith("backbone.") for key in weights):
        reasons.append("missing backbone parameters")
    has_fusion = "fusion.weight" in weights
    if classifier_arch == "cls_patch_attention" and not has_fusion:
        reasons.append("missing patch-attention parameters")
    if classifier_arch == "cls_only" and has_fusion:
        reasons.append("checkpoint weights include a patch-attention head")
    return reasons


def select_checkpoints(
    candidates: list[dict],
    *,
    max_false_authentic_rate: float,
    metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    expected_family: str | None = None,
    expected_classifier_arch: str | None = None,
    artifact_expectations: dict | None = None,
) -> dict:
    """Rank validation candidates. Does not read test metrics.

    Filesystem reads happen only for candidates that name a checkpoint path.
    Checkpoint contents are loaded only when artifact_expectations is set, and
    only for the current ranking winner. A failed load removes that winner and
    the Phase 18 ranking is applied again to the remaining artifacts.
    """
    limit = validate_max_false_authentic_rate(max_false_authentic_rate)
    tolerance = validate_metric_tolerance(metric_tolerance)
    prepared = [
        _prepare_candidate(
            candidate,
            limit,
            expected_family=expected_family,
            expected_classifier_arch=expected_classifier_arch,
        )
        for candidate in candidates
    ]
    identities = [row["checkpoint_id"] for row in prepared if isinstance(row.get("checkpoint_id"), str)]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate checkpoint id")
    if artifact_expectations is not None:
        _reject_invalid_artifacts(
            prepared,
            tolerance=tolerance,
            expected_family=expected_family,
            expected_classifier_arch=expected_classifier_arch,
            expectations=artifact_expectations,
        )
    eligible = [row for row in prepared if row["far_eligible"]]
    selected_id = None
    if eligible:
        winner = min(eligible, key=lambda row: _rank_key(row, tolerance))
        selected_id = winner["checkpoint_id"]
    for row in prepared:
        row["selected"] = bool(row["far_eligible"] and row["checkpoint_id"] == selected_id)
    prepared.sort(key=_report_order)
    selected_metrics = next((row for row in prepared if row["selected"]), None)
    return {
        "selection_version": SELECTION_VERSION,
        "selection_rule": SELECTION_RULE,
        "early_stopping_policy": EARLY_STOPPING_POLICY,
        "max_false_authentic_rate": limit,
        "metric_tolerance": tolerance,
        "threshold_policy": THRESHOLD_POLICY,
        "checkpoints_evaluated": len(prepared),
        "far_eligible_checkpoints": len(eligible),
        "selected_checkpoint": selected_id,
        "selected_metrics": selected_metrics,
        "safety_validated": selected_id is not None,
        "safety_status": QUALIFIED_STATUS if selected_id is not None else UNQUALIFIED_STATUS,
        "status_message": None if selected_id is not None else UNQUALIFIED_MESSAGE,
        "candidates": prepared,
    }


def new_selection_state(
    *,
    max_false_authentic_rate: float,
    metric_tolerance: float,
    validation_threshold: float,
    model_family: str,
    classifier_arch: str,
) -> dict:
    summary = select_checkpoints(
        [],
        max_false_authentic_rate=max_false_authentic_rate,
        metric_tolerance=metric_tolerance,
        expected_family=model_family,
        expected_classifier_arch=classifier_arch,
    )
    return {
        "max_false_authentic_rate": summary["max_false_authentic_rate"],
        "metric_tolerance": summary["metric_tolerance"],
        "validation_threshold": validate_validation_threshold(validation_threshold),
        "model_family": str(model_family),
        "classifier_arch": str(classifier_arch),
        "candidates": [],
        "selected_checkpoint": None,
        "safety_validated": False,
        "epochs_without_improvement": 0,
        "action": "hold",
        "summary": summary,
    }


def update_selection(state: dict, candidate: dict) -> dict:
    """Add one validation result and decide whether it wins the safety ranking."""
    checkpoint_id = str(candidate["checkpoint_id"])
    retained = [row for row in state["candidates"] if row["checkpoint_id"] != checkpoint_id]
    retained.append(candidate)
    summary = select_checkpoints(
        retained,
        max_false_authentic_rate=state["max_false_authentic_rate"],
        metric_tolerance=state["metric_tolerance"],
        expected_family=state["model_family"],
        expected_classifier_arch=state["classifier_arch"],
    )
    new_id = summary["selected_checkpoint"]
    previous_id = state["selected_checkpoint"]
    if new_id is not None and new_id != previous_id and new_id == checkpoint_id:
        action = "reset"
        patience = 0
    elif new_id is None:
        action = "hold"
        patience = int(state["epochs_without_improvement"])
    else:
        action = "increment"
        patience = int(state["epochs_without_improvement"]) + 1
    return {
        "max_false_authentic_rate": summary["max_false_authentic_rate"],
        "metric_tolerance": summary["metric_tolerance"],
        "validation_threshold": state["validation_threshold"],
        "model_family": state["model_family"],
        "classifier_arch": state["classifier_arch"],
        "candidates": summary["candidates"],
        "selected_checkpoint": new_id,
        "safety_validated": summary["safety_validated"],
        "epochs_without_improvement": patience,
        "action": action,
        "summary": summary,
    }


def should_save_production_checkpoint(state: dict) -> bool:
    """True only when the epoch just evaluated became the safety-validated winner."""
    return state["action"] == "reset" and bool(state["safety_validated"])


def write_selection_report(experiment_dir: Path, state: dict) -> Path:
    """Write the authoritative selection report under the experiment directory."""
    root = assert_experiment_output_dir(experiment_dir)
    destination = root / "selection"
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(state["summary"])
    summary["model_family"] = state["model_family"]
    summary["classifier_arch"] = state["classifier_arch"]
    summary["validation_threshold"] = validate_validation_threshold(state["validation_threshold"])
    summary["epochs_without_improvement"] = int(state["epochs_without_improvement"])
    summary["early_stopping_policy"] = EARLY_STOPPING_POLICY
    (destination / "checkpoint_selection.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (destination / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in summary["candidates"]:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in _CSV_FIELDS})
    return destination


def load_selection_state(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    candidates = list(payload.get("candidates") or [])
    family = payload.get("model_family") or (candidates[0]["model_family"] if candidates else None)
    architecture = payload.get("classifier_arch") or (
        candidates[0]["classifier_arch"] if candidates else None
    )
    if not family or not architecture:
        raise ValueError(f"{path} has no model family or classifier head")
    summary = select_checkpoints(
        candidates,
        max_false_authentic_rate=payload["max_false_authentic_rate"],
        metric_tolerance=payload["metric_tolerance"],
        expected_family=family,
        expected_classifier_arch=architecture,
    )
    threshold = payload.get("validation_threshold", DEFAULT_VALIDATION_THRESHOLD)
    return {
        "max_false_authentic_rate": summary["max_false_authentic_rate"],
        "metric_tolerance": summary["metric_tolerance"],
        "validation_threshold": validate_validation_threshold(threshold),
        "model_family": str(family),
        "classifier_arch": str(architecture),
        "candidates": summary["candidates"],
        "selected_checkpoint": summary["selected_checkpoint"],
        "safety_validated": summary["safety_validated"],
        "epochs_without_improvement": int(payload.get("epochs_without_improvement", 0)),
        "action": "hold",
        "summary": summary,
    }


def _reject_invalid_artifacts(
    prepared: list[dict],
    *,
    tolerance: float,
    expected_family: str | None,
    expected_classifier_arch: str | None,
    expectations: dict,
) -> None:
    """Drop ranking winners whose files do not match the experiment. Does not rewrite files."""
    if not expected_family or not expected_classifier_arch:
        raise ValueError("artifact validation requires the expected model family and head")
    checked: set[str] = set()
    while True:
        eligible = [
            row for row in prepared if row["far_eligible"] and row["checkpoint_id"] not in checked
        ]
        if not eligible:
            return
        winner = min(eligible, key=lambda row: _rank_key(row, tolerance))
        path = winner.get("checkpoint_path")
        if not isinstance(path, str) or not path:
            reasons = ["no checkpoint artifact"]
            valid = False
            details = {}
        else:
            details = validate_checkpoint_artifact(
                Path(path),
                expected_family=expected_family,
                expected_classifier_arch=expected_classifier_arch,
                expected_epoch=int(winner["epoch"]),
                expected_input_size=int(expectations["expected_input_size"]),
                expected_membership_hash=str(expectations["expected_membership_hash"]),
                expected_preprocessing_version=str(expectations["expected_preprocessing_version"]),
                expected_experiment_dir=Path(expectations["expected_experiment_dir"]),
            )
            valid = bool(details["checkpoint_load_valid"])
            reasons = list(details["reasons"])
        winner["checkpoint_validation"] = {
            "checkpoint_load_valid": valid,
            "reasons": reasons,
            "manifest_membership_hash": details.get("manifest_membership_hash"),
            "model_family": details.get("model_family"),
            "classifier_arch": details.get("classifier_arch"),
            "input_size": details.get("input_size"),
            "preprocessing_version": details.get("preprocessing_version"),
        }
        if valid:
            return
        checked.add(str(winner["checkpoint_id"]))
        winner["far_eligible"] = False
        winner["eligible"] = False
        extra = "; ".join(reasons) if reasons else "checkpoint validation failed"
        previous = winner.get("rejection_reason")
        winner["rejection_reason"] = extra if not previous else f"{previous}; {extra}"


def _rejected_file(checkpoint_id: str, reason: str) -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "eligible": False,
        "far_eligible": False,
        "selected": False,
        "rejection_reason": reason,
        "epoch": None,
        "global_step": None,
        "model_family": None,
        "classifier_arch": None,
    }


def _prepare_candidate(
    candidate: dict,
    max_false_authentic_rate: float,
    *,
    expected_family: str | None,
    expected_classifier_arch: str | None,
) -> dict:
    """Keep the validation whitelist. Ignore test metrics and discovery order."""
    row = {field: candidate.get(field) for field in _CSV_FIELDS}
    row["threshold_policy"] = THRESHOLD_POLICY
    row["selected"] = False
    row["far_eligible"] = False
    reasons: list[str] = []
    if candidate.get("eligible") is False:
        reasons.append(str(candidate.get("rejection_reason") or "ineligible checkpoint"))
    checkpoint_id = candidate.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        reasons.append("missing checkpoint id")
    else:
        row["checkpoint_id"] = checkpoint_id
    if candidate.get("split") == "test":
        reasons.append("test split cannot be used for selection")
    family = candidate.get("model_family")
    architecture = candidate.get("classifier_arch")
    if expected_family is not None and family != expected_family:
        reasons.append(f"model family {family!r} does not match {expected_family!r}")
    if expected_classifier_arch is not None and architecture != expected_classifier_arch:
        reasons.append(
            f"classifier head {architecture!r} does not match {expected_classifier_arch!r}"
        )
    epoch = candidate.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        reasons.append("missing epoch")
    else:
        row["epoch"] = epoch
    step = candidate.get("global_step")
    if step is not None and (not isinstance(step, int) or isinstance(step, bool)):
        reasons.append("global_step must be an integer")
    else:
        row["global_step"] = step
    for field in ("validation_loss", *_METRIC_FIELDS):
        value = candidate.get(field)
        if field == "threshold":
            continue
        if value is None:
            continue
        try:
            row[field] = float(value)
        except (TypeError, ValueError):
            row[field] = None
            reasons.append(f"{field} is not a number")
    threshold = candidate.get("threshold")
    if threshold is None:
        reasons.append("missing threshold")
    else:
        try:
            row["threshold"] = validate_validation_threshold(threshold)
        except ValueError:
            reasons.append("threshold is outside [0, 1]")
    if row.get("validation_loss") is None:
        reasons.append("missing validation loss")
    if "checkpoint_path" in candidate or "checkpoint_available" in candidate:
        path_value = candidate.get("checkpoint_path")
        path = Path(path_value) if isinstance(path_value, str) and path_value else None
        marked_available = candidate.get("checkpoint_available", True)
        exists = bool(path is not None and path.is_file())
        if marked_available is False or not exists:
            reasons.append("no checkpoint artifact")
            row["checkpoint_available"] = False
            row["checkpoint_path"] = None if path is None else str(path)
        else:
            row["checkpoint_available"] = True
            row["checkpoint_path"] = str(path)
    rate = row.get("false_authentic_rate")
    blockers: list[str] = []
    if rate is None:
        blockers.append("false_authentic_rate is undefined")
    elif float(rate) > max_false_authentic_rate:
        blockers.append("false_authentic_rate above max_false_authentic_rate")
    if row.get("pr_auc") is None:
        blockers.append("missing PR-AUC")
    if row.get("f1") is None:
        blockers.append("missing F1")
    # Structural problems reject the file. A FAR or metric miss is a real
    # validation checkpoint that simply cannot be the production selection.
    row["eligible"] = not reasons
    row["far_eligible"] = not reasons and not blockers
    combined = reasons + blockers
    row["rejection_reason"] = "; ".join(combined) if combined else None
    row["selected"] = False
    return row


def _rank_key(row: dict, tolerance: float) -> tuple:
    """Lower is better. Missing global_step sorts after every recorded step."""
    step = math.inf if row["global_step"] is None else int(row["global_step"])
    return (
        -metric_band(row["pr_auc"], tolerance),
        -metric_band(row["f1"], tolerance),
        metric_band(row["false_authentic_rate"], tolerance),
        metric_band(row["validation_loss"], tolerance),
        int(row["epoch"]),
        step,
        str(row["checkpoint_id"]),
    )


def _report_order(row: dict) -> tuple:
    epoch = row["epoch"] if isinstance(row["epoch"], int) else -1
    step = row["global_step"] if isinstance(row["global_step"], int) else -1
    return (epoch, step, str(row["checkpoint_id"]))
