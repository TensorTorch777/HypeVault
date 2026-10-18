"""Run the locked cross-brand protocol and write the research artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from cross_brand import FOLDS, classify_generalization, generalization_gap
from cross_brand_train import SHM_ROOT, train_fold

ROOT = _PKG / "experiments" / "v3_cross_brand"
MANIFEST = ROOT / "fold_manifest.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def _confidence(probabilities: np.ndarray, labels: np.ndarray, paths: list[str]) -> dict:
    order = (0, 5, 25, 50, 75, 95, 100)
    names = ("minimum", "p5", "p25", "median", "p75", "p95", "maximum")

    def block(mask: np.ndarray) -> dict:
        chosen = probabilities[mask]
        return {name: float(value) for name, value in zip(names, np.percentile(chosen, order))}

    authentic = labels == 0
    fake = labels == 1
    wrong = []
    if fake.any():
        ranking = np.argsort(probabilities[fake])[:15]
        fake_paths = [paths[index] for index in np.flatnonzero(fake)]
        for index in ranking:
            wrong.append(
                {
                    "path": fake_paths[int(index)],
                    "label": 1,
                    "p_fake": float(probabilities[fake][int(index)]),
                    "error": "fake image received a low P(fake)",
                }
            )
    return {
        "all_images_p_fake": block(np.ones(len(labels), dtype=bool)),
        "authentic_images_p_fake": block(authentic),
        "fake_images_p_fake": block(fake),
        "mean_p_fake": float(probabilities.mean()),
        "median_p_fake": float(np.median(probabilities)),
        "most_confident_false_authentic_examples": wrong,
    }


def _public_result(result: dict, *, probabilities=None, labels=None, paths=None) -> dict:
    public = {
        "initialization": result["initialization"],
        "loaded_frozen_checkpoint": False,
        "classifier_arch": result["classifier_arch"],
        "selected_epoch": result["selected_epoch"],
        "selected_checkpoint": result["selected_checkpoint"],
        "temperature": result["temperature"],
        "calibration": result["calibration"],
        "in_domain": result["reports"].get("in_domain"),
        "unseen": result["reports"].get("unseen") or result["reports"].get("random_test"),
        "history": result["history"],
        "features": result["features"],
    }
    if probabilities is not None:
        public["confidence"] = _confidence(probabilities, labels, paths)
    if public["in_domain"] and public["unseen"]:
        public["gap"] = generalization_gap(public["in_domain"], public["unseen"])
    return public


def _run_rows(rows: list[dict], arch: str, weight_name: str, roles: tuple[str, ...]) -> dict:
    weight_path = SHM_ROOT / weight_name
    result = train_fold(rows, arch=arch, weight_path=weight_path, eval_roles=roles)
    bundles = result.pop("bundles")
    if "cross_brand_test_probabilities" in bundles:
        public = _public_result(
            result,
            probabilities=bundles["cross_brand_test_probabilities"],
            labels=bundles["cross_brand_test_labels"],
            paths=bundles["cross_brand_test_paths"],
        )
    elif "random_test_probabilities" in bundles:
        public = _public_result(
            result,
            probabilities=bundles["random_test_probabilities"],
            labels=bundles["random_test_labels"],
            paths=bundles["random_test_paths"],
        )
    else:
        public = _public_result(result)
    return public


def _load_finished(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("in_domain") and payload.get("unseen") and payload.get("temperature") is not None:
        print(f"keeping finished metrics {path.relative_to(ROOT)}", flush=True)
        return payload
    return None


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    primary = {}
    ablation = {}
    for directory, _brand in FOLDS:
        destination = ROOT / directory / "metrics.json"
        finished = _load_finished(destination)
        if finished is not None:
            primary[directory] = finished
            continue
        fold = payload["folds"][directory]
        print(f"primary {directory}", flush=True)
        primary[directory] = _run_rows(
            fold["rows"],
            "cls_patch_attention",
            f"{directory}_cls_patch_attention.pt",
            ("validation", "cross_brand_test", "train"),
        )
        _write(destination, primary[directory])
        _write(ROOT / "primary_partial.json", primary)
    for directory, _brand in FOLDS:
        destination = ROOT / directory / "cls_only_metrics.json"
        finished = _load_finished(destination)
        if finished is not None:
            ablation[directory] = finished
            continue
        fold = payload["folds"][directory]
        print(f"ablation {directory}", flush=True)
        ablation[directory] = _run_rows(
            fold["rows"],
            "cls_only",
            f"{directory}_cls_only.pt",
            ("validation", "cross_brand_test"),
        )
        _write(destination, ablation[directory])
    random_path = ROOT / "random_split_control.json"
    finished_random = _load_finished(random_path)
    if finished_random is not None:
        random_result = finished_random
    else:
        print("random split control", flush=True)
        random_result = _run_rows(
            payload["random_split_control"]["rows"],
            "cls_patch_attention",
            "random_split_cls_patch_attention.pt",
            ("validation", "random_test"),
        )
        _write(random_path, random_result)
    _finalize(primary, ablation, random_result)


def _finalize(primary: dict, ablation: dict, random_result: dict) -> None:
    fold_rows = []
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
    label = classify_generalization(fold_rows)
    ranking = sorted(
        fold_rows,
        key=lambda row: (
            float(row["unseen"]["false_authentic_rate"]),
            -float(row["unseen"]["pr_auc"]),
            -float(row["unseen"]["f1"] or 0.0),
            float(row["unseen"]["ece"]),
        ),
    )
    summary = {
        "generalization": label,
        "mean_unseen_pr_auc": float(np.mean([row["unseen"]["pr_auc"] for row in fold_rows])),
        "mean_unseen_f1": float(np.mean([row["unseen"]["f1"] for row in fold_rows])),
        "mean_unseen_false_authentic_rate": float(np.mean([row["unseen"]["false_authentic_rate"] for row in fold_rows])),
        "mean_generalization_gap": float(np.mean([row["gap"]["pr_auc"] for row in fold_rows])),
        "best_held_out_brand": ranking[0]["held_out_brand"],
        "worst_held_out_brand": ranking[-1]["held_out_brand"],
        "ranking_low_false_authentic_first": [row["held_out_brand"] for row in ranking],
        "random_split_in_domain_pr_auc": random_result["in_domain"]["pr_auc"],
        "random_split_test_pr_auc": random_result["unseen"]["pr_auc"],
        "random_split_test_false_authentic_rate": random_result["unseen"]["false_authentic_rate"],
        "folds": fold_rows,
        "cls_only": {
            directory: {
                "unseen_pr_auc": ablation[directory]["unseen"]["pr_auc"],
                "unseen_false_authentic_rate": ablation[directory]["unseen"]["false_authentic_rate"],
                "unseen_f1": ablation[directory]["unseen"]["f1"],
                "gap_pr_auc": ablation[directory]["gap"]["pr_auc"],
            }
            for directory, _brand in FOLDS
        },
        "loaded_frozen_checkpoint": False,
        "final_test_touched": False,
        "production_ood_threshold": None,
    }
    _write(ROOT / "cross_brand_metrics.json", summary)
    _write(ROOT / "generalization_gap.json", {"definition": "in_domain_metric - unseen_brand_metric", "folds": [
        {"fold": row["fold"], "held_out_brand": row["held_out_brand"], "gap": row["gap"]} for row in fold_rows
    ]})
    _write(ROOT / "confidence_analysis.json", {
        row["held_out_brand"]: row["confidence"] for row in fold_rows
    })
    _write(ROOT / "feature_space_analysis.json", {
        row["held_out_brand"]: row["features"] for row in fold_rows
    })
    lines = ["held_out_brand,in_domain_pr_auc,unseen_pr_auc,in_domain_far,unseen_far,in_domain_f1,unseen_f1"]
    for row in fold_rows:
        lines.append(
            ",".join(
                [
                    row["held_out_brand"],
                    f"{row['in_domain']['pr_auc']:.6f}",
                    f"{row['unseen']['pr_auc']:.6f}",
                    f"{row['in_domain']['false_authentic_rate']:.6f}",
                    f"{row['unseen']['false_authentic_rate']:.6f}",
                    f"{row['in_domain']['f1']:.6f}",
                    f"{row['unseen']['f1']:.6f}",
                ]
            )
        )
    (ROOT / "cross_brand_summary.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key not in {"folds", "cls_only"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
