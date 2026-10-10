"""Metadata-only readiness audit for the frozen DINOv2 and DINOv3 checkpoints.

This module does not load a model, run inference, or open locked final-test or
excluded OOD image files. Directory labels stay historical.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL_NAME = "phase54_data_readiness_audit_v1"
DATA_READINESS_NO_GO = "NO_GO_FOR_AUTHENTICITY_CLAIMS"
CATALOG_DIRS = ("Label_0_Watches", "Label_1_Watches", "Label_0_Sneakers", "Label_1_Sneakers")
LOCKED_SPLITS = frozenset({"test"})
HISTORICAL_LABEL = {0: "authentic_labeled", 1: "fake_labeled"}


class AuditStop(RuntimeError):
    """Membership or identity cannot be established."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str | None, label: str) -> dict:
    if not path.is_file():
        raise AuditStop(f"missing manifest or artifact: {label}")
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise AuditStop(f"{label} hash {actual} != {expected_sha256}")
    return json.loads(path.read_text()) if path.suffix == ".json" else {"sha256": actual}


def canonical_id(path: str) -> str:
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts):
        if part in CATALOG_DIRS:
            return "/".join(parts[index:])
    raise AuditStop(f"path is outside the known catalog roots: {path!r}")


def folder_key(canonical: str) -> str:
    parts = canonical.split("/")
    if len(parts) < 2:
        raise AuditStop(f"catalog id has no brand folder: {canonical}")
    return f"{parts[0]}/{parts[1]}"


def historical_label(canonical: str) -> str:
    label_dir = canonical.split("/", 1)[0]
    if label_dir == "Label_0_Watches":
        return HISTORICAL_LABEL[0]
    if label_dir == "Label_1_Watches":
        return HISTORICAL_LABEL[1]
    raise AuditStop(f"no watch directory label for {canonical}")


def brand_name(canonical: str) -> str:
    return canonical.split("/", 2)[1]


def folder_sets(manifest: dict) -> dict[str, set[str]]:
    """Map a product-folder manifest onto canonical label/brand keys."""
    mapping = {
        "train": manifest.get("train_product_folders") or [],
        "validation": manifest.get("val_product_folders") or manifest.get("validation_product_folders") or [],
        "calibration": manifest.get("calibration_product_folders") or [],
    }
    sets = {name: {folder_key(canonical_id(path)) for path in paths} for name, paths in mapping.items()}
    overlap = sets["train"] & sets["validation"] | sets["train"] & sets["calibration"] | sets["validation"] & sets["calibration"]
    if overlap:
        raise AuditStop("DINOv2 folder manifest lists one folder in more than one split")
    return sets


def index_samples(samples: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    multi = []
    for sample in samples:
        canonical = canonical_id(sample["path"])
        split = sample["split"]
        previous = indexed.get(canonical)
        if previous is not None and previous["split"] != split:
            multi.append(canonical)
            continue
        if previous is not None:
            raise AuditStop("duplicate sample row")
        indexed[canonical] = {
            "split": split,
            "label": historical_label(canonical),
            "brand": sample.get("brand") or brand_name(canonical),
            "sha256": sample.get("sha256"),
            "duplicate_group_id": sample.get("duplicate_group_id"),
            "excluded_ood": bool(sample.get("excluded_ood")),
            "manual_review": bool(sample.get("manual_review")),
        }
    if multi:
        raise AuditStop(f"{len(multi)} samples are listed in more than one split")
    return indexed


def assign_rows(samples: dict[str, dict], folders: dict[str, set[str]]) -> list[dict]:
    rows = []
    for canonical in sorted(samples):
        sample = samples[canonical]
        folder = folder_key(canonical)
        if folder in folders["validation"]:
            dinov2_split = "validation"
        elif folder in folders["train"]:
            dinov2_split = "train"
        elif folder in folders["calibration"]:
            dinov2_split = "calibration"
        else:
            dinov2_split = "outside_manifest"
        rows.append(
            {
                "canonical_id": canonical,
                "dinov2_split": dinov2_split,
                "dinov3_split": sample["split"],
                "historical_label": sample["label"],
                "brand": sample["brand"],
                "image_sha256": None if sample["split"] in LOCKED_SPLITS or sample["excluded_ood"] else sample["sha256"],
                "sha256_status": (
                    "NOT_OPENED_LOCKED_FINAL_TEST"
                    if sample["split"] in LOCKED_SPLITS
                    else "NOT_OPENED_EXCLUDED_OOD"
                    if sample["excluded_ood"]
                    else "RECORDED_IN_MANIFEST"
                    if sample["sha256"]
                    else "NOT_RECORDED"
                ),
                "duplicate_group_id": sample["duplicate_group_id"],
                "excluded_ood": sample["excluded_ood"],
                "manual_review": sample["manual_review"],
            }
        )
    return rows


def intersection_table(rows: list[dict]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        table[row["dinov2_split"]][row["dinov3_split"]] += 1
    return {left: dict(sorted(right.items())) for left, right in sorted(table.items())}


def primary_cohort(rows: list[dict]) -> list[dict]:
    """Validation intersection minus either model's train or calibration membership."""
    eligible = []
    for row in rows:
        if row["excluded_ood"] or row["dinov3_split"] in LOCKED_SPLITS:
            continue
        if row["dinov2_split"] != "validation" or row["dinov3_split"] != "validation":
            continue
        if row["dinov2_split"] in {"train", "calibration"} or row["dinov3_split"] in {"train", "calibration"}:
            continue
        eligible.append(row)
    return eligible


def exact_duplicate_groups(rows: list[dict]) -> dict:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["sha256_status"] != "RECORDED_IN_MANIFEST" or not row["image_sha256"]:
            continue
        if row["dinov3_split"] in LOCKED_SPLITS or row["excluded_ood"]:
            continue
        groups[row["image_sha256"]].append(row["canonical_id"])
    multi = {digest: members for digest, members in groups.items() if len(members) > 1}
    return {
        "non_test_images_with_recorded_sha256": sum(len(members) for members in groups.values()),
        "exact_duplicate_groups": len(multi),
        "images_in_exact_duplicate_groups": sum(len(members) for members in multi.values()),
    }


def provenance_status(row: dict) -> dict:
    """Directory metadata only. Nothing here is an independent verification."""
    return {
        "sample_key": hashlib.sha256(row["canonical_id"].encode()).hexdigest(),
        "image_sha256_status": row["sha256_status"],
        "label_source": "historical_directory",
        "label_semantics": "HISTORICAL_DIRECTORY_LABEL",
        "historical_label": row["historical_label"],
        "verified_label": "NOT_RECORDED",
        "independent_verification_status": "NOT_RECORDED",
        "evidence_type": "NOT_RECORDED",
        "verifier": "NOT_RECORDED",
        "verification_date": "NOT_RECORDED",
        "adjudication_status": "NOT_RECORDED",
        "product_id": "UNKNOWN",
        "image_group_id": row["duplicate_group_id"] or "NOT_RECORDED",
        "image_group_meaning": "manifest_duplicate_group_not_a_physical_product",
        "source": "UNKNOWN",
        "seller": "UNKNOWN",
        "marketplace": "UNKNOWN",
        "brand": row["brand"],
        "brand_source": "parent_directory",
        "license_consent": "NOT_RECORDED",
        "permitted_evaluation_use": "NOT_RECORDED",
        "dinov2_split": row["dinov2_split"],
        "dinov3_split": row["dinov3_split"],
        "exclusion_reason": (
            "locked_final_test"
            if row["dinov3_split"] in LOCKED_SPLITS
            else "excluded_ood"
            if row["excluded_ood"]
            else None
        ),
    }


def decide_readiness(
    *,
    primary: list[dict],
    independently_verified: int,
    product_ids_known: bool,
    source_disjoint: bool,
    product_count: int,
    required_products: int,
    membership_established: bool,
) -> dict:
    """Return GO only when every pre-registered condition holds. Directory labels do not count."""
    reasons = []
    if not membership_established:
        reasons.append("membership_not_established")
    labels = {row["historical_label"] for row in primary}
    if "fake_labeled" not in labels or "authentic_labeled" not in labels:
        reasons.append("primary_cohort_missing_a_binary_class")
    if independently_verified != len(primary) or independently_verified == 0:
        reasons.append("no_independently_verified_labels")
    if not product_ids_known:
        reasons.append("product_id_unknown")
    if not source_disjoint:
        reasons.append("source_or_product_disjointness_not_established")
    if product_count < required_products:
        reasons.append("independent_product_count_below_preregistered_minimum")
    status = DATA_READINESS_NO_GO if reasons else "GO"
    if status == "GO" and independently_verified == 0:
        raise AuditStop("refusing to emit GO without verified labels")
    return {"data_readiness": status, "reasons": reasons}


def assert_not_opened(path: str, blocked: set[str], opener) -> None:
    canonical = canonical_id(path)
    if canonical in blocked or folder_key(canonical) in blocked:
        raise AuditStop("refusing to open a locked final-test or excluded OOD image")
    return opener(path)


def cohort_summary(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "historical_labels": dict(Counter(row["historical_label"] for row in rows)),
        "brands": dict(Counter(row["brand"] for row in rows)),
        "single_brand": len({row["brand"] for row in rows}) == 1,
        "single_historical_label": len({row["historical_label"] for row in rows}) == 1,
    }


def build_audit(samples: list[dict], folder_manifest: dict, *, required_products: int = 300) -> dict:
    indexed = index_samples(samples)
    folders = folder_sets(folder_manifest)
    rows = assign_rows(indexed, folders)
    primary = primary_cohort(rows)
    verified = 0
    decision = decide_readiness(
        primary=primary,
        independently_verified=verified,
        product_ids_known=False,
        source_disjoint=False,
        product_count=0,
        required_products=required_products,
        membership_established=True,
    )
    non_test = [row for row in rows if row["dinov3_split"] not in LOCKED_SPLITS and not row["excluded_ood"]]
    return {
        "rows": rows,
        "primary": primary,
        "intersection": intersection_table(rows),
        "primary_summary": cohort_summary(primary),
        "exact_duplicates": exact_duplicate_groups(non_test),
        "decision": decision,
        "provenance_example_fields": sorted(provenance_status(primary[0]).keys()) if primary else [],
        "independently_verified": verified,
    }


def _check_phase53(repo: Path, protocol: dict) -> None:
    pinned = protocol["phase53_inputs"]
    metrics = repo / "ml_rtx5080/experiments/dinov2_vs_dinov3_full_scale/metrics.json"
    predictions = repo / "ml_rtx5080/experiments/dinov2_vs_dinov3_full_scale/local_results/predictions.jsonl"
    if file_sha256(metrics) != pinned["metrics_sha256"]:
        raise AuditStop("Phase 53 metrics.json hash changed")
    if predictions.is_file() and file_sha256(predictions) != pinned["predictions_sha256"]:
        raise AuditStop("Phase 53 predictions hash changed")


def _load_real(repo: Path, protocol: dict) -> tuple[list[dict], dict]:
    dinov2 = protocol["checkpoints"]["dinov2_legacy"]
    dinov3 = protocol["checkpoints"]["dinov3_experimental"]
    other = protocol["checkpoints"]["not_the_live_dinov2_checkpoint"]
    require_file(repo / dinov2["path"], dinov2["sha256"], "DINOv2 checkpoint")
    require_file(repo / dinov3["path"], dinov3["sha256"], "DINOv3 checkpoint")
    require_file(repo / other["path"], other["sha256"], "non-live watches checkpoint")
    require_file(repo / dinov2["training_config"], dinov2["training_config_sha256"], "DINOv2 training config")
    require_file(repo / dinov3["training_config"], dinov3["training_config_sha256"], "DINOv3 training config")
    calibration = require_file(repo / dinov3["calibration_file"], dinov3["calibration_file_sha256"], "DINOv3 calibration file")
    if calibration.get("manifest_hash") != dinov3["membership_hash"]:
        raise AuditStop("DINOv3 calibration file does not name the training membership hash")
    if not str(calibration.get("checkpoint", "")).endswith("epoch_018.pt"):
        raise AuditStop("DINOv3 calibration file does not name epoch_018.pt")
    folder_manifest = require_file(repo / dinov2["split_manifest"], dinov2["split_manifest_sha256"], "DINOv2 split manifest")
    image_manifest = require_file(repo / dinov3["split_manifest"], dinov3["split_manifest_sha256"], "DINOv3 split manifest")
    if image_manifest.get("membership_hash") != dinov3["membership_hash"]:
        raise AuditStop("DINOv3 membership hash drifted")
    if int(folder_manifest.get("seed")) != 42 or float(folder_manifest.get("val_split")) != 0.1:
        raise AuditStop("DINOv2 split manifest does not match the training config seed and val_split")
    return image_manifest["samples"], folder_manifest


def _public_metrics(protocol: dict, audit: dict, folder_manifest: dict) -> dict:
    expected = protocol["expected_if_manifests_match"]
    summary = audit["primary_summary"]
    if summary["count"] != expected["primary_count"]:
        raise AuditStop("recomputed primary count does not match Phase 53")
    if summary["historical_labels"].get("authentic_labeled") != expected["primary_authentic_labeled"]:
        raise AuditStop("recomputed authentic-labeled count does not match Phase 53")
    if summary["historical_labels"].get("fake_labeled", 0) != expected["primary_fake_labeled"]:
        raise AuditStop("recomputed fake-labeled count does not match Phase 53")
    if sorted(summary["brands"]) != expected["primary_brands"]:
        raise AuditStop("recomputed primary brands do not match Phase 53")
    if audit["decision"]["data_readiness"] != DATA_READINESS_NO_GO:
        raise AuditStop("directory-label audit emitted GO")
    watch_rows = [row for row in audit["rows"] if row["canonical_id"].startswith("Label_")]
    locked = sum(1 for row in watch_rows if row["dinov3_split"] in LOCKED_SPLITS)
    sneaker_folders = sum(
        1
        for folder in set(folder_manifest["train_product_folders"]) | set(folder_manifest["val_product_folders"])
        if "Sneakers" in canonical_id(folder)
    )
    return {
        "protocol": PROTOCOL_NAME,
        "inference_run": False,
        "final_test_images_opened": 0,
        "excluded_ood_images_opened": 0,
        "data_readiness": audit["decision"]["data_readiness"],
        "readiness_reasons": audit["decision"]["reasons"],
        "winner": None,
        "phase53": {
            "protocol_commit": protocol["phase53_inputs"]["protocol_commit"],
            "results_commit": protocol["phase53_inputs"]["results_commit"],
            "metrics_sha256": protocol["phase53_inputs"]["metrics_sha256"],
            "primary_count_reproduced": summary["count"] == protocol["phase53_inputs"]["reported_primary_count"],
        },
        "checkpoints": {
            "dinov2_sha256": protocol["checkpoints"]["dinov2_legacy"]["sha256"],
            "dinov3_sha256": protocol["checkpoints"]["dinov3_experimental"]["sha256"],
            "unrelated_watches_checkpoint_sha256": protocol["checkpoints"]["not_the_live_dinov2_checkpoint"]["sha256"],
        },
        "watch_samples": len(watch_rows),
        "locked_final_test": locked,
        "dinov2_calibration_split": "NOT_RECORDED",
        "dinov2_non_watch_folders": sneaker_folders,
        "intersection": audit["intersection"],
        "primary": {
            "count": summary["count"],
            "historical_labels": summary["historical_labels"],
            "brands": summary["brands"],
            "single_brand": summary["single_brand"],
            "single_historical_label": summary["single_historical_label"],
            "independently_verified": 0,
            "false_authentic_rate": None,
            "roc_auc": None,
            "pr_auc": None,
            "reason_metrics_undefined": "no fake-labeled sample in the shared held-out cohort",
        },
        "exact_duplicates_non_test": audit["exact_duplicates"],
        "provenance": {
            "product_id": "UNKNOWN",
            "source": "UNKNOWN",
            "seller": "UNKNOWN",
            "marketplace": "UNKNOWN",
            "independent_verification_status": "NOT_RECORDED",
            "license_consent": "NOT_RECORDED",
            "directory_labels_are_ground_truth": False,
        },
        "contradictions": [
            {
                "document": "docs/ML_PRODUCTION_AUDIT.md",
                "claim": "Validation folders are Label_0_Watches/Vacheron Constantin and Label_1_Watches/Richard Mille.",
                "artifact": "ml_rtx5080/checkpoints_watches/split_manifest.json",
                "artifact_checkpoint_sha256": protocol["checkpoints"]["not_the_live_dinov2_checkpoint"]["sha256"],
                "live_checkpoint_sha256": protocol["checkpoints"]["dinov2_legacy"]["sha256"],
                "resolution": "These are different checkpoints. The VC/RM folder split is not the live DINOv2 validation set and was not applied to Phase 53.",
            }
        ],
    }


def _report(metrics: dict) -> str:
    primary = metrics["primary"]
    labels = primary["historical_labels"]
    return f"""# Phase 54 data-readiness audit

DATA_READINESS = {metrics["data_readiness"]}

No confirmatory model comparison was run. Phase 53 commits `{metrics["phase53"]["protocol_commit"]}` and `{metrics["phase53"]["results_commit"]}` were checked and not rewritten. The shared held-out cohort is {primary["count"]} images, {labels.get("authentic_labeled", 0)} authentic-labeled and {labels.get("fake_labeled", 0)} fake-labeled, all from {", ".join(primary["brands"])}. False-authentic rate, ROC-AUC, and PR-AUC are undefined. Independently verified labels: {primary["independently_verified"]}.

## Membership

DINOv2 `{metrics["checkpoints"]["dinov2_sha256"]}` is tied to `ml_rtx5080/checkpoints/split_manifest.json` by the Triton identity record and by the training config's seed and validation fraction. That manifest has no calibration split (`NOT_RECORDED`) and includes {metrics["dinov2_non_watch_folders"]} sneaker folders outside the watch corpus. DINOv3 `{metrics["checkpoints"]["dinov3_sha256"]}` is tied to `split_manifest_v2.json` by the training config's membership hash and by the calibration file that names `epoch_018.pt`.

The watch manifest contains {metrics["watch_samples"]} samples, of which {metrics["locked_final_test"]} are the locked final test. Those image files were not opened. The primary count matches Phase 53: {str(metrics["phase53"]["primary_count_reproduced"]).lower()}.

DINOv2 rows by DINOv3 columns:

| DINOv2 \\ DINOv3 | calibration | test | train | validation |
| --- | ---: | ---: | ---: | ---: |
| train | {metrics["intersection"]["train"].get("calibration", 0)} | {metrics["intersection"]["train"].get("test", 0)} | {metrics["intersection"]["train"].get("train", 0)} | {metrics["intersection"]["train"].get("validation", 0)} |
| validation | {metrics["intersection"]["validation"].get("calibration", 0)} | {metrics["intersection"]["validation"].get("test", 0)} | {metrics["intersection"]["validation"].get("train", 0)} | {metrics["intersection"]["validation"].get("validation", 0)} |

The validation/validation cell is the primary cohort. The validation/test cell stays inside the locked split and was not opened. The train/validation cell is DINOv3 validation that the live DINOv2 checkpoint already used for training.

## Contradiction that was not smoothed over

`docs/ML_PRODUCTION_AUDIT.md` says validation is Vacheron Constantin authentic-labeled and Richard Mille fake-labeled. That description matches `checkpoints_watches/split_manifest.json`. The checkpoint beside that manifest hashes to `{metrics["checkpoints"]["unrelated_watches_checkpoint_sha256"]}`, which is not the live DINOv2 checkpoint. Phase 53's Lange-only intersection stands. The older VC/RM statement does not describe the live model.

## Provenance

Product, seller, source, and marketplace identifiers are `UNKNOWN`. Verification, evidence, adjudication, license, and consent are `NOT_RECORDED`. Duplicate-group ids in the manifest are hash groups, not physical watches. Directory labels are not ground truth.

Exact duplicate groups among non-test images with a manifest SHA-256: {metrics["exact_duplicates_non_test"]["exact_duplicate_groups"]}.

## Readiness reasons

{chr(10).join(f"- {reason}" for reason in metrics["readiness_reasons"])}

The future paired benchmark in `phase54_label_acquisition_plan.md` is not authorized to start.
"""


def write_outputs(repo: Path) -> dict:
    protocol_path = repo / "ml_rtx5080/experiments/dinov2_vs_dinov3_full_scale/phase54_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol"] != PROTOCOL_NAME or protocol["inference_allowed"] is not False:
        raise AuditStop("phase 54 protocol drifted")
    _check_phase53(repo, protocol)
    samples, folder_manifest = _load_real(repo, protocol)
    audit = build_audit(samples, folder_manifest)
    metrics = _public_metrics(protocol, audit, folder_manifest)
    destination = repo / "ml_rtx5080/experiments/dinov2_vs_dinov3_full_scale"
    (destination / "phase54_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (destination / "phase54_report.md").write_text(_report(metrics))
    local = destination / "local_results"
    local.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in audit["rows"]:
        if row["dinov3_split"] in LOCKED_SPLITS or row["excluded_ood"]:
            continue
        record = provenance_status(row)
        record["relative_path"] = row["canonical_id"]
        lines.append(json.dumps(record, sort_keys=True))
    (local / "phase54_provenance.jsonl").write_text("\n".join(lines) + "\n")
    return metrics


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    metrics = write_outputs(repo)
    print(json.dumps({"data_readiness": metrics["data_readiness"], "primary": metrics["primary"]["count"], "inference_run": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditStop as exc:
        print(f"STOP {exc}")
        raise SystemExit(2)
