"""Train the four architectures on the frozen Phase 16 manifest.

Validation selects the checkpoint. The final test membership is not scored.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from baseline_comparison_v2 import EXPERIMENTS, write_baseline_comparison_v2
from split_v2 import (
    AUTHORITATIVE_COUNTS,
    AUTHORITATIVE_MEMBERSHIP_HASH,
    CATALOG_BRANDS,
    PHASE15_EXPERIMENT_DIRS,
    STRATEGY,
    membership_hash,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "ml_rtx5080" / "experiments"
MANIFEST = EXPERIMENTS_ROOT / "dataset_audit" / "split_manifest_v2.json"
PYTHON = ROOT / ".venv" / "bin" / "python"
FORBIDDEN = {ROOT / "ml_rtx5080" / "checkpoints", ROOT / "ml_rtx5080" / "checkpoints_watches"}

SHARED = [
    "--data_root",
    str(ROOT),
    "--fixed_split_manifest",
    str(MANIFEST),
    "--batch_size",
    "16",
    "--accumulate_grad",
    "2",
    "--num_epochs",
    "30",
    "--backbone_lr",
    "1e-5",
    "--head_lr",
    "1e-4",
    "--layer_decay",
    "0.75",
    "--weight_decay",
    "0.05",
    "--warmup_epochs",
    "2",
    "--label_smoothing",
    "0.02",
    "--mixup_alpha",
    "0.0",
    "--weighted_sampler",
    "0",
    "--grad_clip",
    "1.0",
    "--amp_dtype",
    "bf16",
    "--early_stop_patience",
    "6",
    "--max_false_authentic_rate",
    "0.02",
    "--validation_threshold",
    "0.5",
    "--seed",
    "42",
    "--augmentation_profile",
    "conservative",
    "--img_size",
    "0",
]


def main() -> None:
    _preflight()
    for spec in EXPERIMENTS:
        directory = EXPERIMENTS_ROOT / spec["directory"]
        if directory.name in PHASE15_EXPERIMENT_DIRS:
            raise RuntimeError(f"refusing to write into Phase 15 directory {directory}")
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing experiment directory {directory}")
        _verify_manifest()
        command = [
            str(PYTHON),
            "-u",
            "-m",
            "ml_rtx5080.train",
            "--family",
            spec["family"],
            "--classifier_arch",
            spec["head"],
            "--output_dir",
            str(directory),
            *SHARED,
        ]
        if "--test_root" in command:
            raise RuntimeError("v2 comparison must not pass --test_root")
        log_path = EXPERIMENTS_ROOT / f"{spec['directory']}.log"
        print(f"Starting {spec['directory']}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            print(f"{spec['directory']} exited {completed.returncode}. Stopping the comparison.", flush=True)
            write_baseline_comparison_v2(EXPERIMENTS_ROOT)
            raise SystemExit(completed.returncode)
        _confirm_experiment_manifest(directory)
        _maybe_brand_eval(directory, spec)
    payload = write_baseline_comparison_v2(EXPERIMENTS_ROOT)
    print(json.dumps({"winner": payload["winner"], "winner_status": payload["winner_status"]}, indent=2))


def _verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise RuntimeError(f"Phase 16 manifest is missing: {MANIFEST}")
    payload = json.loads(MANIFEST.read_text())
    membership = payload.get("membership") or {}
    digest = membership_hash({name: list(membership.get(name) or []) for name in AUTHORITATIVE_COUNTS})
    if payload.get("strategy") != STRATEGY or digest != AUTHORITATIVE_MEMBERSHIP_HASH:
        raise RuntimeError(
            "Phase 16 manifest hash does not match "
            f"{AUTHORITATIVE_MEMBERSHIP_HASH}. Refusing to train."
        )
    if digest != payload.get("membership_hash"):
        raise RuntimeError("Phase 16 manifest stored hash does not match its membership")
    samples = payload.get("samples") or []
    for name, expected in AUTHORITATIVE_COUNTS.items():
        rows = [row for row in samples if row.get("split") == name]
        authentic = sum(1 for row in rows if row.get("label") == 0)
        fake = sum(1 for row in rows if row.get("label") == 1)
        if (len(rows), authentic, fake) != (expected["samples"], expected["authentic"], expected["fake"]):
            raise RuntimeError(f"{name} counts do not match the Phase 16 manifest")
        if len(membership.get(name) or []) != expected["samples"]:
            raise RuntimeError(f"{name} membership length does not match the Phase 16 manifest")
        for label in (0, 1):
            present = {row.get("brand") for row in rows if row.get("label") == label}
            missing = [brand for brand in CATALOG_BRANDS if brand not in present]
            if missing:
                raise RuntimeError(f"{name} label {label} is missing brands {missing}")


def _preflight() -> None:
    _verify_manifest()
    for spec in EXPERIMENTS:
        directory = (EXPERIMENTS_ROOT / spec["directory"]).resolve()
        if directory.name in PHASE15_EXPERIMENT_DIRS:
            raise RuntimeError(f"refusing Phase 15 output path {directory}")
        if any(directory == forbidden or forbidden in directory.parents for forbidden in FORBIDDEN):
            raise RuntimeError(f"experiment path collides with a checkpoint directory: {directory}")


def _confirm_experiment_manifest(directory: Path) -> None:
    payload = json.loads((directory / "split_manifest.json").read_text())
    digest = membership_hash(payload["membership"])
    if digest != AUTHORITATIVE_MEMBERSHIP_HASH:
        raise RuntimeError(f"{directory.name} did not keep the Phase 16 membership hash")
    if payload.get("calibration_membership_source") != "split_manifest_v2":
        raise RuntimeError(f"{directory.name} did not record the fixed calibration membership")
    if payload.get("test_set_used_for_selection") is not False:
        raise RuntimeError(f"{directory.name} recorded the test set as used for selection")


def _maybe_brand_eval(directory: Path, spec: dict) -> None:
    selection_path = directory / "selection" / "checkpoint_selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection.get("safety_validated"):
        print(f"{spec['directory']}: no safety-validated checkpoint, brand eval skipped", flush=True)
        return
    checkpoint = directory / "best_model.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"{directory} is safety-validated but best_model.pt is missing")
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "ml_rtx5080.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--split_manifest",
        str(directory / "split_manifest.json"),
        "--output_dir",
        str(directory),
        "--model",
        spec["family"],
        "--head",
        spec["head"],
        "--split",
        "validation",
        "--threshold",
        "0.50",
        "--brand_eval",
        "1",
        "--batch_size",
        "16",
        "--device",
        "cuda",
        "--num_workers",
        "4",
    ]
    log_path = directory / "brand_eval.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"brand evaluation failed for {spec['directory']}")
    _require_brand_coverage(directory / "evaluation" / "brand_metrics.json")


def _require_brand_coverage(path: Path) -> None:
    report = json.loads(path.read_text())
    if report.get("checkpoint_selection_affected") is not False:
        raise RuntimeError(f"{path} changed checkpoint selection")
    if report.get("split") != "validation":
        raise RuntimeError(f"{path} is not a validation diagnostic")
    rows = report.get("brands") or []
    by_name = {row.get("brand"): row for row in rows}
    missing = [brand for brand in CATALOG_BRANDS if brand not in by_name]
    if missing:
        raise RuntimeError(f"brand diagnostic omitted {missing}")
    for brand in CATALOG_BRANDS:
        row = by_name[brand]
        if not row.get("authentic_count") or not row.get("fake_count"):
            raise RuntimeError(f"{brand} is missing a class in the brand diagnostic")
        if row.get("false_authentic_rate") is None:
            raise RuntimeError(f"{brand} has no false-authentic rate")


if __name__ == "__main__":
    main()
