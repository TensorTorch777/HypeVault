"""Run the four Phase 15 architectures with one shared training configuration.

The final test split is not loaded. Calibration and OOD fitting are not run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from baseline_comparison import EXPERIMENTS, write_baseline_comparison

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "ml_rtx5080" / "experiments"
PYTHON = ROOT / ".venv" / "bin" / "python"
FORBIDDEN = {ROOT / "ml_rtx5080" / "checkpoints", ROOT / "ml_rtx5080" / "checkpoints_watches"}

SHARED = [
    "--data_root",
    str(ROOT),
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
    "--split_strategy",
    "product_disjoint",
    "--calibration_fraction",
    "0.10",
    "--val_split",
    "0.1",
    "--augmentation_profile",
    "conservative",
    "--img_size",
    "0",
]


def main() -> None:
    _preflight()
    reference_hash = None
    for spec in EXPERIMENTS:
        directory = EXPERIMENTS_ROOT / spec["directory"]
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing experiment directory {directory}")
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
            raise RuntimeError("baseline comparison must not pass --test_root")
        log_path = EXPERIMENTS_ROOT / f"{spec['directory']}.log"
        print(f"Starting {spec['directory']}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            print(f"{spec['directory']} exited {completed.returncode}. Stopping the comparison.", flush=True)
            write_baseline_comparison(EXPERIMENTS_ROOT)
            raise SystemExit(completed.returncode)
        membership = _membership_hash(directory / "split_manifest.json")
        if reference_hash is None:
            reference_hash = membership
        elif membership != reference_hash:
            raise RuntimeError(f"{spec['directory']} split membership does not match the first experiment")
        _maybe_brand_eval(directory, spec)
    payload = write_baseline_comparison(EXPERIMENTS_ROOT)
    print(json.dumps({"winner": payload["winner"], "winner_status": payload["winner_status"]}, indent=2))


def _preflight() -> None:
    sys.path.insert(0, str(ROOT / "ml_rtx5080"))
    import train as trainer

    defaults = trainer.DEFAULTS
    expected = {
        "batch_size": 16,
        "accumulate_grad": 2,
        "amp_dtype": "bf16",
        "max_false_authentic_rate": 0.02,
        "num_epochs": 30,
        "early_stop_patience": 6,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "layer_decay": 0.75,
        "weight_decay": 0.05,
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "label_smoothing": 0.02,
        "mixup_alpha": 0.0,
        "weighted_sampler": 0,
        "seed": 42,
        "validation_threshold": 0.5,
        "augmentation_profile": "conservative",
        "split_strategy": "product_disjoint",
    }
    for key, value in expected.items():
        if defaults[key] != value:
            raise RuntimeError(f"trainer default {key}={defaults[key]!r} does not match the baseline {value!r}")
    if int(defaults["batch_size"]) * int(defaults["accumulate_grad"]) != 32:
        raise RuntimeError("effective batch is not 32")
    for spec in EXPERIMENTS:
        directory = (EXPERIMENTS_ROOT / spec["directory"]).resolve()
        if any(directory == forbidden or forbidden in directory.parents for forbidden in FORBIDDEN):
            raise RuntimeError(f"experiment path collides with a checkpoint directory: {directory}")
        if "checkpoints" in directory.parts or "checkpoints_watches" in directory.parts:
            raise RuntimeError(f"refusing checkpoint output path {directory}")


def _membership_hash(manifest_path: Path) -> str:
    payload = json.loads(manifest_path.read_text())
    membership = payload["membership"]
    digest = hashlib.sha256()
    for split in ("train", "calibration", "validation", "test"):
        digest.update(split.encode())
        for path in sorted(membership.get(split) or []):
            digest.update(path.encode())
            digest.update(b"\n")
    return digest.hexdigest()


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


if __name__ == "__main__":
    main()
