"""Phase 56: rerun the Phase 45 leave-one-brand-out protocol on surface-normalized images.

Training, selection, calibration, and scoring are ``cross_brand_train.train_fold``
unchanged. Only the image files differ (see ``normalize_surface.py``). The
CLS-only ablation is not run. Phase 45 numbers are read back for comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from cross_brand import FOLDS, classify_generalization
from cross_brand_train import train_fold
from normalize_surface import OUT as ROOT
from run_cross_brand import ROOT as PHASE45_ROOT
from run_cross_brand import _public_result, _write

MANIFEST = ROOT / "fold_manifest.json"
SHM_ROOT = Path("/dev/shm/hypevault_normalized_v1")
ARCH = "cls_patch_attention"


def _run_rows(rows: list[dict], weight_name: str, roles: tuple[str, ...]) -> dict:
    result = train_fold(rows, arch=ARCH, weight_path=SHM_ROOT / weight_name, eval_roles=roles)
    bundles = result.pop("bundles")
    for role in ("cross_brand_test", "random_test"):
        if role + "_probabilities" in bundles:
            return _public_result(
                result,
                probabilities=bundles[role + "_probabilities"],
                labels=bundles[role + "_labels"],
                paths=bundles[role + "_paths"],
            )
    return _public_result(result)


def _load_finished(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("in_domain") and payload.get("unseen") and payload.get("temperature") is not None:
        print(f"keeping finished metrics {path.relative_to(ROOT)}", flush=True)
        return payload
    return None


def _phase45(directory: str) -> dict | None:
    path = PHASE45_ROOT / directory / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    primary = {}
    for directory, _brand in FOLDS:
        destination = ROOT / directory / "metrics.json"
        finished = _load_finished(destination)
        if finished is not None:
            primary[directory] = finished
            continue
        print(f"primary {directory}", flush=True)
        primary[directory] = _run_rows(
            payload["folds"][directory]["rows"],
            f"{directory}_{ARCH}.pt",
            ("validation", "cross_brand_test", "train"),
        )
        _write(destination, primary[directory])
    random_path = ROOT / "random_split_control.json"
    random_result = _load_finished(random_path)
    if random_result is None:
        print("random split control", flush=True)
        random_result = _run_rows(
            payload["random_split_control"]["rows"],
            f"random_split_{ARCH}.pt",
            ("validation", "random_test"),
        )
        _write(random_path, random_result)
    _finalize(primary, random_result)


def _finalize(primary: dict, random_result: dict) -> None:
    fold_rows = []
    comparison = []
    for directory, brand in FOLDS:
        row = primary[directory]
        fold_rows.append(
            {
                "fold": directory,
                "held_out_brand": brand,
                "in_domain": row["in_domain"],
                "unseen": row["unseen"],
                "gap": row["gap"],
                "temperature": row["temperature"],
                "calibration": row["calibration"],
                "confidence": row.get("confidence"),
                "features": row.get("features"),
                "selected_epoch": row["selected_epoch"],
            }
        )
        before = _phase45(directory)
        comparison.append(
            {
                "held_out_brand": brand,
                "phase45_unseen": None if before is None else {
                    key: before["unseen"][key] for key in ("pr_auc", "roc_auc", "f1", "false_authentic_rate", "false_fake_rate")
                },
                "normalized_unseen": {
                    key: row["unseen"][key] for key in ("pr_auc", "roc_auc", "f1", "false_authentic_rate", "false_fake_rate")
                },
            }
        )
    summary = {
        "generalization": classify_generalization(fold_rows),
        "mean_unseen_pr_auc": float(np.mean([row["unseen"]["pr_auc"] for row in fold_rows])),
        "mean_unseen_f1": float(np.mean([row["unseen"]["f1"] for row in fold_rows])),
        "mean_unseen_false_authentic_rate": float(np.mean([row["unseen"]["false_authentic_rate"] for row in fold_rows])),
        "mean_unseen_false_fake_rate": float(np.mean([row["unseen"]["false_fake_rate"] for row in fold_rows])),
        "mean_generalization_gap": float(np.mean([row["gap"]["pr_auc"] for row in fold_rows])),
        "random_split_in_domain_pr_auc": random_result["in_domain"]["pr_auc"],
        "random_split_test_pr_auc": random_result["unseen"]["pr_auc"],
        "random_split_test_false_authentic_rate": random_result["unseen"]["false_authentic_rate"],
        "random_split_test_false_fake_rate": random_result["unseen"]["false_fake_rate"],
        "phase45_comparison": comparison,
        "folds": fold_rows,
        "loaded_frozen_checkpoint": False,
        "final_test_touched": False,
    }
    _write(ROOT / "cross_brand_metrics.json", summary)
    lines = ["held_out_brand,phase45_unseen_pr_auc,normalized_unseen_pr_auc,phase45_unseen_far,normalized_unseen_far,normalized_unseen_ffr,normalized_unseen_f1"]
    for row in comparison:
        before = row["phase45_unseen"] or {}
        after = row["normalized_unseen"]
        lines.append(
            ",".join(
                [
                    row["held_out_brand"],
                    f"{before.get('pr_auc', float('nan')):.6f}",
                    f"{after['pr_auc']:.6f}",
                    f"{before.get('false_authentic_rate', float('nan')):.6f}",
                    f"{after['false_authentic_rate']:.6f}",
                    f"{after['false_fake_rate']:.6f}",
                    f"{after['f1']:.6f}",
                ]
            )
        )
    (ROOT / "cross_brand_summary.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key not in {"folds", "phase45_comparison"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
