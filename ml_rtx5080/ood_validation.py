"""Phase 35 real open-set corpus and frozen-model evaluation.

The repository had no OOD images. This module records that audit, admits only
files whose provenance can be stated, and scores the frozen FP32 checkpoint.
It does not train, refit temperature, open final-test images, or promote a
threshold.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

import ood
from policy import AUTHENTICITY_THRESHOLD, FROZEN_TEMPERATURE, calibrated_probability_from_logit
from quality_fastpath import fast_quality_gate
from reference_inference import file_sha256, load_reference_model, preprocess_image
from split_v2 import dhash_image, hamming_distance

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
_OUTPUT = _PKG / "experiments" / "ood_validation_v1"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_FINAL_TEST_SUMMARY = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
METHODOLOGY_VERSION = "phase35_real_ood_v1"
POLICY_VERSION = "shadow_v1"
NEAR_DUP_HAMMING = 4
MIN_SCORED = 80
USER_AGENT = "HypeVaultOODValidation/1.0 (local research audit; contact: local workstation)"
TRAINED_BRAND_TOKENS = (
    "lange",
    "audemars",
    "patek",
    "richard mille",
    "vacheron",
)
TAXONOMY = (
    "A_unrelated",
    "B_product_non_watch",
    "C_watch_outside_scope",
    "D_unseen_luxury_brand",
    "E_structurally_different",
)
REQUIRED_ROW_FIELDS = (
    "image_path",
    "ood_category",
    "taxonomy",
    "source",
    "source_url",
    "source_type",
    "brand",
    "is_watch",
    "is_trained_brand",
    "license_or_usage_note",
    "sha256",
    "width",
    "height",
    "aspect_ratio",
    "duplicate_group",
    "split",
    "collection_method",
)
OOD_PASS = "OOD_PASS"
OOD_PARTIAL = "OOD_PARTIAL"
OOD_FAIL = "OOD_FAIL"
OOD_BLOCKED = "OOD_BLOCKED"
GATE_SPECS = (
    ("none", None),
    ("conservative", 0.99),
    ("moderate", 0.95),
    ("aggressive", 0.80),
)
COMMONS_CATEGORIES = (
    {"category": "Pizzas", "taxonomy": "A_unrelated", "ood_category": "food", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Portrait photographs", "taxonomy": "A_unrelated", "ood_category": "people", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Skyscrapers", "taxonomy": "A_unrelated", "ood_category": "buildings", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Landscapes", "taxonomy": "A_unrelated", "ood_category": "landscapes", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Laptop computers", "taxonomy": "A_unrelated", "ood_category": "electronics", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Automobiles", "taxonomy": "A_unrelated", "ood_category": "vehicles", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Domestic cats", "taxonomy": "A_unrelated", "ood_category": "animals", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Washing machines", "taxonomy": "A_unrelated", "ood_category": "consumer_products", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Handbags", "taxonomy": "B_product_non_watch", "ood_category": "handbags", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Necklaces", "taxonomy": "B_product_non_watch", "ood_category": "jewelry", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Eyeglasses", "taxonomy": "B_product_non_watch", "ood_category": "glasses", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Smartphones", "taxonomy": "B_product_non_watch", "ood_category": "phones", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Belts", "taxonomy": "B_product_non_watch", "ood_category": "accessories", "is_watch": False, "brand": "UNKNOWN", "limit": 10},
    {"category": "Sneakers", "taxonomy": "B_product_non_watch", "ood_category": "shoes", "is_watch": False, "brand": "UNKNOWN", "limit": 12},
    {"category": "Smartwatches", "taxonomy": "C_watch_outside_scope", "ood_category": "smartwatches", "is_watch": True, "brand": "UNKNOWN", "limit": 15},
    {"category": "Digital watches", "taxonomy": "C_watch_outside_scope", "ood_category": "digital_watches", "is_watch": True, "brand": "UNKNOWN", "limit": 12},
    {"category": "Watch movements", "taxonomy": "C_watch_outside_scope", "ood_category": "watch_movements", "is_watch": True, "brand": "UNKNOWN", "limit": 12},
    {"category": "Casio watches", "taxonomy": "C_watch_outside_scope", "ood_category": "inexpensive_or_fashion_watches", "is_watch": True, "brand": "Casio", "limit": 12},
    {"category": "Swatch watches", "taxonomy": "C_watch_outside_scope", "ood_category": "fashion_watches", "is_watch": True, "brand": "Swatch", "limit": 12},
    {"category": "Omega watches", "taxonomy": "D_unseen_luxury_brand", "ood_category": "unseen_luxury_brand", "is_watch": True, "brand": "Omega", "limit": 12},
    {"category": "Cartier watches", "taxonomy": "D_unseen_luxury_brand", "ood_category": "unseen_luxury_brand", "is_watch": True, "brand": "Cartier", "limit": 12},
    {"category": "Jaeger-LeCoultre watches", "taxonomy": "D_unseen_luxury_brand", "ood_category": "unseen_luxury_brand", "is_watch": True, "brand": "Jaeger-LeCoultre", "limit": 10},
    {"category": "People wearing wristwatches", "taxonomy": "E_structurally_different", "ood_category": "wrist_shots", "is_watch": True, "brand": "UNKNOWN", "limit": 12},
    {"category": "Watch shops", "taxonomy": "E_structurally_different", "ood_category": "store_photographs", "is_watch": True, "brand": "UNKNOWN", "limit": 10},
    {"category": "Watch boxes", "taxonomy": "E_structurally_different", "ood_category": "boxed_watches", "is_watch": True, "brand": "UNKNOWN", "limit": 10},
)
ROLEX_DATASETS = (
    {
        "root": Path("/home/tensortorch26/Downloads/rolex-day-date-watches.v4i.yolo26"),
        "source_url": "https://universe.roboflow.com/bbi-tech/rolex-day-date-watches-dje46/dataset/4",
        "license": "CC BY 4.0",
    },
    {
        "root": Path("/home/tensortorch26/Downloads/Rolex Watches.v1i.yolo26"),
        "source_url": "https://universe.roboflow.com/modeltime/rolex-watches/dataset/1",
        "license": "CC BY 4.0",
    },
    {
        "root": Path("/home/tensortorch26/Downloads/rolex-submariner.v6i.yolo26"),
        "source_url": "https://universe.roboflow.com/bbitech2/rolex-submariner-vfzmv/dataset/6",
        "license": "CC BY 4.0",
    },
)
SNEAKER_ROOT = Path("/home/tensortorch26/Desktop/scraper/Label_0_Sneakers")
REJECTED_LOCAL = (
    {
        "path": "/home/tensortorch26/Downloads/archive (2)",
        "reason": "numeric filenames, augmentation suffixes, no license file, and no source URL; category and brand would have to be invented",
    },
    {
        "path": str(_REPO / "synthetic_deepfakes"),
        "reason": "synthetic images are not real open-set evidence",
    },
    {
        "path": str(_REPO / "dataset_sample_20_per_class"),
        "reason": "sample of the five trained watch brands",
    },
    {
        "path": str(_REPO / "Label_0_Watches"),
        "reason": "trained-brand catalog",
    },
    {
        "path": str(_REPO / "Label_1_Watches"),
        "reason": "trained-brand catalog",
    },
)


class OODArtifactError(ValueError):
    """An OOD manifest or score file is incomplete. Scoring must stop."""


def classify_ood_status(facts: dict) -> dict:
    """Map measured facts to one status. The status string is not preset."""
    reasons = []
    counts = facts.get("group_counts") or {}
    if not facts.get("real_corpus") or int(facts.get("scored_count") or 0) < MIN_SCORED:
        reasons.append("a real scored corpus below the usefulness minimum is not validation")
        return {"status": OOD_BLOCKED, "reasons": reasons}
    covered = [name for name in TAXONOMY if int(counts.get(name, 0)) >= 15]
    if len(covered) < 3:
        reasons.append("fewer than three taxonomy groups have 15 scored images")
        return {"status": OOD_BLOCKED, "reasons": reasons}
    if facts.get("provenance_failures"):
        reasons.append("a scored row is missing source or sha256")
        return {"status": OOD_BLOCKED, "reasons": reasons}
    unsafe_groups = ("B_product_non_watch", "C_watch_outside_scope", "D_unseen_luxury_brand")
    no_gate = facts.get("no_gate_authentic_rate") or {}
    unsafe = [
        name
        for name in unsafe_groups
        if int(counts.get(name, 0)) >= 15 and float(no_gate.get(name, 0.0)) >= 0.15
    ]
    gates = list(facts.get("gates") or [])

    def contained(gate: dict) -> bool:
        if float(gate.get("calibration_review_increase", 1.0)) > 0.10:
            return False
        for name in unsafe_groups:
            if int(counts.get(name, 0)) >= 15 and float((gate.get("authentic_escape") or {}).get(name, 1.0)) > 0.05:
                return False
        return True

    if unsafe and not any(contained(gate) for gate in gates):
        reasons.append(
            "authentic escape stays at or above 0.15 on "
            + ", ".join(unsafe)
            + ", and no research gate brings those groups to 0.05 without raising calibration review by more than 0.10"
        )
        return {"status": OOD_FAIL, "reasons": reasons}
    holdout = facts.get("holdout_counts") or {}
    moderate = next((gate for gate in gates if gate.get("name") == "moderate"), None)
    moderate_holdout = (facts.get("holdout_authentic_escape") or {}).get("moderate") or {}
    pass_ok = (
        all(int(counts.get(name, 0)) >= 30 for name in TAXONOMY)
        and all(int(holdout.get(name, 0)) >= 8 for name in TAXONOMY)
        and int(facts.get("scored_license_unknown_count") or 0) == 0
        and facts.get("independent_photo_count_known") is True
        and moderate is not None
        and contained(moderate)
        and all(float(moderate_holdout.get(name, 1.0)) <= 0.05 for name in TAXONOMY)
        and float(facts.get("unseen_luxury_mahalanobis_auroc") or 0.0) >= 0.90
        and not unsafe
    )
    if pass_ok:
        reasons.append("coverage, holdout escape, detector separation, license, and calibration review all meet the pass bar")
        return {"status": OOD_PASS, "reasons": reasons}
    reasons.append("real images were scored, and at least one pass condition is unmet")
    if unsafe:
        reasons.append("authentic escape remains material on " + ", ".join(unsafe))
    if not all(int(counts.get(name, 0)) >= 30 for name in TAXONOMY):
        reasons.append("at least one taxonomy group has fewer than 30 scored images")
    if int(facts.get("scored_license_unknown_count") or 0) != 0:
        reasons.append("at least one scored image has license UNKNOWN")
    if facts.get("independent_photo_count_known") is not True:
        reasons.append("the number of independent photographs is not known")
    if not all(int(holdout.get(name, 0)) >= 8 for name in TAXONOMY):
        reasons.append("at least one taxonomy group has fewer than 8 holdout images")
    return {"status": OOD_PARTIAL, "reasons": reasons}


def decide_case(probability: float, flags: tuple[str, ...] | list[str], ood_score: float, ood_threshold: float | None) -> str:
    """Quality review, then an optional OOD review, then the frozen 0.50 rule."""
    flag_set = set(flags)
    if "LOW_RESOLUTION" in flag_set or "LOW_CONTRAST" in flag_set:
        return "REVIEW"
    if ood_threshold is not None and float(ood_score) >= float(ood_threshold):
        return "REVIEW"
    if float(probability) >= AUTHENTICITY_THRESHOLD:
        return "FAKE"
    return "AUTHENTIC"


def assign_split(identifier: str) -> str:
    """Stable 50/25/25 split from a sha256. Holdout is one residue class."""
    bucket = int(identifier[:8], 16) % 4
    if bucket == 0:
        return "holdout"
    if bucket == 1:
        return "validation"
    return "development"


def fpr_at_tpr(id_scores, ood_scores, target_tpr: float = 0.95) -> dict:
    """FPR at the OOD quantile that targets the given true-positive rate."""
    id_vector = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_vector = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if id_vector.size < 20 or ood_vector.size < 20:
        return {"status": "too_few_samples", "fpr_at_95_tpr": None, "threshold": None, "achieved_tpr": None}
    threshold = float(np.quantile(ood_vector, 1.0 - target_tpr, method="linear"))
    return {
        "status": "ok",
        "threshold": threshold,
        "achieved_tpr": float(np.mean(ood_vector >= threshold)),
        "fpr_at_95_tpr": float(np.mean(id_vector >= threshold)),
    }


def validate_row(row: dict) -> None:
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise OODArtifactError("manifest row is missing " + ", ".join(missing))
    if not row["source"] or row["source"] == "UNKNOWN":
        raise OODArtifactError("source cannot be empty or UNKNOWN")
    if not row["collection_method"] or row["collection_method"] == "UNKNOWN":
        raise OODArtifactError("collection_method cannot be empty or UNKNOWN")
    if not row["sha256"] or len(str(row["sha256"])) != 64:
        raise OODArtifactError("sha256 must be a 64-character digest")
    if row["taxonomy"] not in TAXONOMY:
        raise OODArtifactError(f"unknown taxonomy {row['taxonomy']}")
    for field in ("source_url", "brand", "license_or_usage_note"):
        if row[field] in ("", None):
            raise OODArtifactError(f"{field} must be present; use UNKNOWN when it is not known")
    for field in ("is_watch", "is_trained_brand"):
        if row[field] not in (True, False, "UNKNOWN"):
            raise OODArtifactError(f"{field} must be true, false, or UNKNOWN")
    if row["is_trained_brand"] is True:
        raise OODArtifactError("a trained-brand image cannot enter the OOD corpus")


def open_rgb(path: Path, blocked: set[str]) -> Image.Image:
    """Open one OOD or in-distribution image. Final-test paths raise first."""
    resolved = str(path.resolve())
    if resolved in blocked or "final_test_v1" in Path(resolved).parts:
        raise OODArtifactError(f"refusing to open a locked final-test path: {resolved}")
    try:
        image = Image.open(path)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise OODArtifactError(f"unreadable image: {path}") from exc
    return image.convert("RGB")


def screen_against_catalog(rows: list[dict], catalog_sha: set[str], catalog_hash: np.ndarray, blocked: set[str]) -> dict:
    """Drop exact and near-duplicate leakage. This compares hashes, not test pixels."""
    admitted = []
    rejected = []
    for row in rows:
        path = Path(row["image_path"])
        resolved = str(path.resolve())
        if resolved in blocked:
            rejected.append({**row, "rejection": "final_test_path"})
            continue
        if row["sha256"] in catalog_sha:
            rejected.append({**row, "rejection": "exact_sha256_match_to_catalog"})
            continue
        if any(token in resolved.lower() for token in ("label_0_watches", "label_1_watches")):
            rejected.append({**row, "rejection": "trained_brand_catalog_path"})
            continue
        try:
            image = open_rgb(path, blocked)
        except OODArtifactError as exc:
            rejected.append({**row, "rejection": f"unreadable:{exc}"})
            continue
        perceptual = dhash_image(image)
        row = dict(row)
        row["perceptual_hash"] = f"{perceptual:016x}"
        row["width"], row["height"] = image.size
        row["aspect_ratio"] = image.size[0] / image.size[1]
        nearest = _nearest_hamming(catalog_hash, perceptual)
        row["nearest_catalog_hamming"] = nearest
        if nearest <= NEAR_DUP_HAMMING:
            rejected.append({**row, "rejection": f"near_duplicate_hamming_{nearest}"})
            continue
        admitted.append(row)
    groups: dict[str, list[dict]] = {}
    for row in admitted:
        groups.setdefault(row["perceptual_hash"], []).append(row)
    kept = []
    for perceptual, members in groups.items():
        members = sorted(members, key=lambda item: item["sha256"])
        leader = members[0]["sha256"]
        for index, member in enumerate(members):
            member["duplicate_group"] = leader
            if index == 0:
                kept.append(member)
            else:
                rejected.append({**member, "rejection": "exact_perceptual_duplicate_in_ood_corpus"})
    for row in kept:
        row["split"] = assign_split(row["sha256"])
        validate_row(row)
    return {"admitted": kept, "rejected": rejected}


def threshold_from_percentile(scores, quantile: float) -> float:
    vector = np.asarray(scores, dtype=np.float64).reshape(-1)
    if vector.size < 20:
        raise OODArtifactError("refusing to freeze an OOD threshold on fewer than 20 in-domain scores")
    cutoff = float(np.quantile(vector, quantile, method="linear"))
    if not np.isfinite(cutoff):
        raise OODArtifactError("OOD threshold quantile is not finite")
    return cutoff


def audit_local_sources() -> dict:
    """State what already existed before any download."""
    rolex_files = []
    for dataset in ROLEX_DATASETS:
        images = _image_files(dataset["root"]) if dataset["root"].is_dir() else []
        rolex_files.append(
            {
                "root": str(dataset["root"]),
                "exists": dataset["root"].is_dir(),
                "image_count": len(images),
                "roboflow_suffix_count": sum(1 for path in images if ".rf." in path.name),
                "source_url": dataset["source_url"],
                "license_or_usage_note": dataset["license"],
                "admitted_basis": "data.yaml records CC BY 4.0 and a Roboflow dataset URL",
            }
        )
    sneaker_count = len(_image_files(SNEAKER_ROOT)) if SNEAKER_ROOT.is_dir() else 0
    return {
        "repository_real_ood_corpus": False,
        "phase25_corpus_sample_count": 0,
        "statement": "No usable real OOD corpus exists inside the HypeVault repository.",
        "rolex_roboflow_exports": rolex_files,
        "sneaker_scraper": {
            "path": str(SNEAKER_ROOT),
            "image_count": sneaker_count,
            "source_url": "UNKNOWN",
            "license_or_usage_note": "UNKNOWN",
            "decision": "admitted_only_as_a_bounded_sample_with_unknown_license",
        },
        "rejected_without_scoring": list(REJECTED_LOCAL),
        "downloads_after_this_audit": "wikimedia_commons_category_api",
    }


def collect_local_rows(limit_per_sneaker_brand: int = 8) -> list[dict]:
    rows = []
    for dataset in ROLEX_DATASETS:
        for path in _image_files(dataset["root"]):
            rows.append(
                _file_row(
                    path,
                    taxonomy="D_unseen_luxury_brand",
                    ood_category="unseen_luxury_brand",
                    source="roboflow",
                    source_url=dataset["source_url"],
                    source_type="roboflow_resized_or_augmented_derivative",
                    brand="Rolex",
                    is_watch=True,
                    is_trained_brand=False,
                    license_note=dataset["license"],
                    collection_method="preexisting_local_roboflow_export",
                )
            )
    if SNEAKER_ROOT.is_dir():
        for brand_dir in sorted(path for path in SNEAKER_ROOT.iterdir() if path.is_dir()):
            files = _image_files(brand_dir)
            if not files:
                continue
            stride = max(1, len(files) // limit_per_sneaker_brand)
            chosen = files[::stride][:limit_per_sneaker_brand]
            for path in chosen:
                rows.append(
                    _file_row(
                        path,
                        taxonomy="B_product_non_watch",
                        ood_category="shoes",
                        source="local_scraper_directory",
                        source_url="UNKNOWN",
                        source_type="preexisting_local_directory",
                        brand=brand_dir.name,
                        is_watch=False,
                        is_trained_brand=False,
                        license_note="UNKNOWN",
                        collection_method="preexisting_local_directory_bounded_sample",
                    )
                )
    return rows


def download_commons(destination: Path, categories=COMMONS_CATEGORIES) -> tuple[list[dict], list[dict]]:
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    seen_sha: set[str] = set()
    for spec in categories:
        try:
            pages = _commons_pages(spec["category"], int(spec["limit"]))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            failures.append({"category": spec["category"], "error": str(exc)})
            continue
        for page in pages:
            title = str(page.get("title") or "")
            info = (page.get("imageinfo") or [{}])[0]
            mime = str(info.get("mime") or "")
            if not mime.startswith("image/") or mime in {"image/svg+xml", "image/tiff"}:
                failures.append({"category": spec["category"], "title": title, "error": f"skipped mime {mime}"})
                continue
            meta = info.get("extmetadata") or {}
            license_name = _meta_text(meta, "LicenseShortName")
            description = " ".join(
                part for part in (title, _meta_text(meta, "ImageDescription"), _meta_text(meta, "ObjectName")) if part
            )
            if license_name == "UNKNOWN":
                failures.append({"category": spec["category"], "title": title, "error": "license missing"})
                continue
            if _mentions_trained_brand(description):
                failures.append({"category": spec["category"], "title": title, "error": "trained-brand text"})
                continue
            download_url = info.get("thumburl") or info.get("url")
            if not download_url:
                failures.append({"category": spec["category"], "title": title, "error": "no download url"})
                continue
            try:
                payload = _http_bytes(download_url)
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                failures.append({"category": spec["category"], "title": title, "error": str(exc)})
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen_sha:
                failures.append({"category": spec["category"], "title": title, "error": "duplicate download"})
                continue
            seen_sha.add(digest)
            stored = destination / f"{digest[:16]}.jpg"
            stored.write_bytes(payload)
            try:
                with Image.open(stored) as image:
                    image.load()
                    width, height = image.size
            except (UnidentifiedImageError, OSError) as exc:
                stored.unlink(missing_ok=True)
                failures.append({"category": spec["category"], "title": title, "error": f"unreadable download: {exc}"})
                continue
            page_url = info.get("descriptionurl") or "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            rows.append(
                {
                    "image_path": str(stored.resolve()),
                    "ood_category": spec["ood_category"],
                    "taxonomy": spec["taxonomy"],
                    "source": "wikimedia_commons",
                    "source_url": page_url,
                    "download_url": download_url,
                    "source_type": "wikimedia_commons_category_member",
                    "brand": spec["brand"],
                    "is_watch": spec["is_watch"],
                    "is_trained_brand": False,
                    "license_or_usage_note": license_name,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "duplicate_group": "",
                    "split": "",
                    "collection_method": "wikimedia_commons_category_api",
                    "commons_category": spec["category"],
                    "visual_verification": "not_performed",
                    "is_watch_basis": "commons_category_assignment",
                }
            )
            time.sleep(0.15)
    return rows, failures


def run_phase35() -> dict:
    status_path = _OUTPUT / "ood_status.json"
    if status_path.is_file():
        raise RuntimeError("ood_validation_v1 already exists; refusing to rewrite it")
    if not torch.cuda.is_available():
        raise OODArtifactError("the frozen FP32 shadow path requires CUDA")
    before = _final_test_stat()
    checkpoint_sha = file_sha256(_CHECKPOINT)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise OODArtifactError("checkpoint sha256 does not match the frozen digest")
    catalog = json.loads(_MANIFEST.read_text())
    blocked = {str(Path(path).resolve()) for path in catalog["membership"]["test"]}
    catalog_sha = {row["sha256"] for row in catalog["samples"]}
    catalog_hash = np.array([int(row["perceptual_hash"], 16) for row in catalog["samples"]], dtype=np.uint64)
    audit = audit_local_sources()
    local_rows = collect_local_rows()
    commons_rows, commons_failures = download_commons(_OUTPUT / "corpus" / "wikimedia")
    screened = screen_against_catalog(local_rows + commons_rows, catalog_sha, catalog_hash, blocked)
    admitted = screened["admitted"]
    if _final_test_stat() != before:
        raise OODArtifactError("final-test summary changed during corpus construction")
    protocol = {
        "methodology_version": METHODOLOGY_VERSION,
        "frozen_before_holdout_scoring": True,
        "threshold_source": "validation split mahalanobis percentiles only",
        "holdout_used_for_threshold": False,
        "final_test_used": False,
        "gates": [
            {"name": "none", "ood_threshold": None},
            {"name": "conservative", "id_quantile": 0.99},
            {"name": "moderate", "id_quantile": 0.95},
            {"name": "aggressive", "id_quantile": 0.80},
        ],
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "temperature": FROZEN_TEMPERATURE,
        "quality_rule": "LOW_RESOLUTION OR LOW_CONTRAST",
        "primary_ood_score": "mahalanobis",
        "production_ood_threshold": None,
        "threshold_status": "research_candidate_only",
    }
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "ood_protocol_frozen.json").write_text(_dump(protocol) + "\n")
    device = torch.device("cuda")
    model, digest = load_reference_model(_CHECKPOINT, device)
    if digest != CHECKPOINT_SHA256:
        raise OODArtifactError("loaded checkpoint digest changed")
    weight_mark = _weight_mark(model)
    print("scoring train reference", flush=True)
    train = _score_membership(model, device, catalog["membership"]["train"], blocked, "train")
    reference = ood.fit_reference(train["features"], train["labels"])
    _assert_mark(model, weight_mark)
    print("scoring validation for threshold freeze", flush=True)
    validation = _score_membership(model, device, catalog["membership"]["validation"], blocked, "validation")
    validation_scores = ood.score_features(validation["features"], reference, "mahalanobis")
    frozen_thresholds = {
        "conservative": threshold_from_percentile(validation_scores, 0.99),
        "moderate": threshold_from_percentile(validation_scores, 0.95),
        "aggressive": threshold_from_percentile(validation_scores, 0.80),
    }
    protocol["frozen_mahalanobis_thresholds"] = frozen_thresholds
    protocol["threshold_frozen_before_ood_and_holdout_metrics"] = True
    (_OUTPUT / "ood_protocol_frozen.json").write_text(_dump(protocol) + "\n")
    print("scoring calibration", flush=True)
    calibration = _score_membership(model, device, catalog["membership"]["calibration"], blocked, "calibration")
    admitted, capped = cap_rows(admitted, "roboflow_resized_or_augmented_derivative", 60)
    screened["admitted"] = admitted
    screened["rejected"].extend(capped)
    print("scoring ood", len(admitted), flush=True)
    ood_scored = _score_ood(model, device, admitted, blocked)
    _patch_score_attachment(ood_scored, reference)
    _assert_mark(model, weight_mark)
    if file_sha256(_CHECKPOINT) != CHECKPOINT_SHA256 or _final_test_stat() != before:
        raise OODArtifactError("checkpoint or final-test summary changed during scoring")
    report = _compile(
        audit=audit,
        commons_failures=commons_failures,
        screened=screened,
        protocol=protocol,
        reference=reference,
        train_count=len(train["paths"]),
        validation=validation,
        validation_scores=validation_scores,
        calibration=calibration,
        reference_features=reference,
        ood_scored=ood_scored,
        checkpoint_sha=checkpoint_sha,
    )
    _write_outputs(report)
    return report


def _compile(*, audit, commons_failures, screened, protocol, reference, train_count, validation, validation_scores, calibration, reference_features, ood_scored, checkpoint_sha) -> dict:
    calibration_scores = ood.score_features(calibration["features"], reference_features, "mahalanobis")
    calibration_cosine = ood.score_features(calibration["features"], reference_features, "cosine")
    validation_cosine = ood.score_features(validation["features"], reference_features, "cosine")
    ood_maha = ood_scored["mahalanobis"]
    ood_cosine = ood_scored["cosine"]
    rows = []
    for index, row in enumerate(ood_scored["rows"]):
        probability = calibrated_probability_from_logit(float(ood_scored["logits"][index]))
        flags = tuple(ood_scored["flags"][index])
        record = {
            "image_path": row["image_path"],
            "ood_category": row["ood_category"],
            "taxonomy": row["taxonomy"],
            "brand": row["brand"],
            "split": row["split"],
            "source": row["source"],
            "source_url": row["source_url"],
            "license_or_usage_note": row["license_or_usage_note"],
            "sha256": row["sha256"],
            "raw_logit": float(ood_scored["logits"][index]),
            "p_fake": probability,
            "predicted_class_at_0_50": "FAKE" if probability >= AUTHENTICITY_THRESHOLD else "AUTHENTIC",
            "quality_flags": list(flags),
            "mahalanobis": float(ood_maha[index]),
            "cosine": float(ood_cosine[index]),
            "is_watch": row["is_watch"],
            "is_trained_brand": row["is_trained_brand"],
            "source_type": row["source_type"],
        }
        for name, threshold in (("none", None), *protocol["frozen_mahalanobis_thresholds"].items()):
            record[f"decision_{name}"] = decide_case(probability, flags, float(ood_maha[index]), threshold)
        rows.append(record)
    group_metrics = {}
    for taxonomy in TAXONOMY:
        selected = [row for row in rows if row["taxonomy"] == taxonomy]
        group_metrics[taxonomy] = _distribution(selected)
        group_metrics[taxonomy]["categories"] = _category_distributions(selected)
        group_metrics[taxonomy]["detectors"] = _detector_block(
            calibration_scores, calibration_cosine, selected, id_name="calibration"
        )
    sweep = []
    id_probabilities = [calibrated_probability_from_logit(float(value)) for value in calibration["logits"]]
    id_flags = calibration["flags"]
    none_review = _review_rate(id_probabilities, id_flags, calibration_scores, None)
    for name, quantile in GATE_SPECS:
        threshold = None if quantile is None else protocol["frozen_mahalanobis_thresholds"][name]
        escape = {}
        caught = {}
        authentic_count = {}
        for taxonomy in TAXONOMY:
            selected = [row for row in rows if row["taxonomy"] == taxonomy]
            decisions = [row[f"decision_{name}"] for row in selected]
            authentic_count[taxonomy] = decisions.count("AUTHENTIC")
            escape[taxonomy] = _rate(decisions, "AUTHENTIC")
            caught[taxonomy] = 1.0 - escape[taxonomy] if selected else None
        review = _review_rate(id_probabilities, id_flags, calibration_scores, threshold)
        id_decisions = [
            decide_case(probability, flags, float(score), threshold)
            for probability, flags, score in zip(id_probabilities, id_flags, calibration_scores.tolist())
        ]
        sweep.append(
            {
                "name": name,
                "role": "research_candidate_only" if name != "none" else "current_shadow_without_ood",
                "production_ood_threshold": None,
                "mahalanobis_threshold": threshold,
                "ood_authentic_escape": escape,
                "ood_caught_rate": caught,
                "ood_authentic_count": authentic_count,
                "calibration_review_rate": review,
                "calibration_review_increase": review - none_review,
                "calibration_decision_counts": _counts(id_decisions),
                "calibration_authentic_rate": _rate(id_decisions, "AUTHENTIC"),
                "calibration_fake_rate": _rate(id_decisions, "FAKE"),
            }
        )
    failures = _failure_table(rows)
    facts = _facts(rows, sweep, group_metrics)
    status = classify_ood_status(facts)
    return {
        "audit": audit,
        "commons_failures": commons_failures,
        "rejected": [
            {"image_path": row.get("image_path"), "rejection": row.get("rejection"), "sha256": row.get("sha256")}
            for row in screened["rejected"]
        ],
        "manifest_rows": screened["admitted"],
        "predictions": rows,
        "protocol": protocol,
        "train_reference_count": train_count,
        "reference_sample_count": int(reference["gaussian"]["sample_count"]),
        "validation_count": len(validation["paths"]),
        "calibration_count": len(calibration["paths"]),
        "group_metrics": group_metrics,
        "threshold_sweep": sweep,
        "failures": failures,
        "facts": facts,
        "status": status,
        "checkpoint_sha256": checkpoint_sha,
        "validation_mahalanobis_mean": float(validation_scores.mean()),
        "production_ood_threshold": None,
        "final_test_samples_used": 0,
    }


def _facts(rows: list[dict], sweep: list[dict], group_metrics: dict) -> dict:
    counts = {taxonomy: sum(1 for row in rows if row["taxonomy"] == taxonomy) for taxonomy in TAXONOMY}
    holdout = {taxonomy: sum(1 for row in rows if row["taxonomy"] == taxonomy and row["split"] == "holdout") for taxonomy in TAXONOMY}
    no_gate = next(item for item in sweep if item["name"] == "none")
    luxury = group_metrics["D_unseen_luxury_brand"]["detectors"]["mahalanobis"]
    return {
        "real_corpus": len(rows) > 0,
        "scored_count": len(rows),
        "group_counts": counts,
        "holdout_counts": holdout,
        "provenance_failures": 0,
        "no_gate_authentic_rate": no_gate["ood_authentic_escape"],
        "gates": [
            {
                "name": item["name"],
                "calibration_review_increase": item["calibration_review_increase"],
                "authentic_escape": item["ood_authentic_escape"],
            }
            for item in sweep
            if item["name"] != "none"
        ],
        "holdout_authentic_escape": {
            item["name"]: {
                taxonomy: _rate(
                    [row[f"decision_{item['name']}"] for row in rows if row["taxonomy"] == taxonomy and row["split"] == "holdout"],
                    "AUTHENTIC",
                )
                for taxonomy in TAXONOMY
            }
            for item in sweep
        },
        "scored_license_unknown_count": sum(1 for row in rows if row["license_or_usage_note"] == "UNKNOWN"),
        "independent_photo_count_known": not any(
            row.get("source_type") == "roboflow_resized_or_augmented_derivative" for row in rows
        ),
        "unseen_luxury_mahalanobis_auroc": luxury.get("roc_auc"),
    }


def _distribution(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "mean_p_fake": None, "median_p_fake": None, "percentiles": {}, "authentic_fraction": None, "fake_fraction": None}
    probabilities = np.array([row["p_fake"] for row in rows], dtype=np.float64)
    classes = [row["predicted_class_at_0_50"] for row in rows]
    return {
        "count": len(rows),
        "mean_p_fake": float(probabilities.mean()),
        "median_p_fake": float(np.median(probabilities)),
        "percentiles": {str(percentile): float(np.quantile(probabilities, percentile / 100, method="linear")) for percentile in (5, 25, 50, 75, 95)},
        "authentic_fraction_at_0_50": classes.count("AUTHENTIC") / len(rows),
        "fake_fraction_at_0_50": classes.count("FAKE") / len(rows),
        "split_counts": _counts([row["split"] for row in rows]),
    }


def _category_distributions(rows: list[dict]) -> dict:
    names = sorted({row["ood_category"] for row in rows})
    return {name: _distribution([row for row in rows if row["ood_category"] == name]) for name in names}


def _detector_block(id_maha, id_cosine, rows: list[dict], id_name: str) -> dict:
    block = {}
    for method, id_scores, key in (
        ("mahalanobis", id_maha, "mahalanobis"),
        ("cosine", id_cosine, "cosine"),
    ):
        ood_scores = [row[key] for row in rows]
        ranking = ood.ranking_metrics(id_scores, torch.tensor(ood_scores, dtype=torch.float64) if ood_scores else torch.empty(0))
        point = None
        if ood_scores:
            point = fpr_at_tpr(id_scores.detach().cpu().numpy(), ood_scores)
        block[method] = {
            "id_reference": id_name,
            "roc_auc": ranking.get("roc_auc"),
            "average_precision": ranking.get("average_precision"),
            "fpr_at_95_tpr": None if point is None else point["fpr_at_95_tpr"],
            "fpr_status": None if point is None else point["status"],
            "ood_score_mean": None if not ood_scores else float(np.mean(ood_scores)),
            "ood_score_median": None if not ood_scores else float(np.median(ood_scores)),
            "id_score_mean": float(id_scores.mean()),
            "id_score_median": float(id_scores.median()),
        }
    return block


def _failure_table(rows: list[dict]) -> list[dict]:
    mistakes = [row for row in rows if row["decision_none"] == "AUTHENTIC"]
    mistakes.sort(key=lambda row: row["p_fake"])
    table = []
    for row in mistakes[:40]:
        table.append(
            {
                "image_path": row["image_path"],
                "taxonomy": row["taxonomy"],
                "ood_category": row["ood_category"],
                "brand": row["brand"],
                "p_fake": row["p_fake"],
                "mahalanobis": row["mahalanobis"],
                "cosine": row["cosine"],
                "quality_flags": row["quality_flags"],
                "resemblance_note": _resemblance(row),
            }
        )
    return table


def cap_rows(rows: list[dict], source_type: str, per_source: int) -> tuple[list[dict], list[dict]]:
    """Keep a bounded number of non-independent derivatives. The cap is not a threshold."""
    kept = []
    capped = []
    grouped: dict[str, list[dict]] = {}
    others = []
    for row in rows:
        if row.get("source_type") == source_type:
            grouped.setdefault(row["source_url"], []).append(row)
        else:
            others.append(row)
    for members in grouped.values():
        ordered = sorted(members, key=lambda item: item["sha256"])
        kept.extend(ordered[:per_source])
        for member in ordered[per_source:]:
            capped.append({**member, "rejection": "roboflow_derivative_not_scored_cap"})
    return others + kept, capped


def _resemblance(row: dict) -> str:
    if row["taxonomy"] == "D_unseen_luxury_brand":
        return "unseen luxury-watch photograph; resemblance to the trained watch-photo distribution is a hypothesis, not a measured mechanism"
    if row["taxonomy"] == "C_watch_outside_scope":
        return "watch-shaped object outside the five trained brands"
    if row["taxonomy"] == "B_product_non_watch":
        return "product photograph that is not a watch"
    if row["taxonomy"] == "E_structurally_different":
        return "watch context with a different composition from a typical isolated catalog photo"
    return "category is visually unrelated to a luxury-watch catalog photo"


def _review_rate(probabilities, flags, scores, threshold) -> float:
    decisions = [decide_case(probability, flag, float(score), threshold) for probability, flag, score in zip(probabilities, flags, scores.tolist())]
    if not decisions:
        raise OODArtifactError("calibration review rate has no rows")
    return decisions.count("REVIEW") / len(decisions)


def _score_membership(model, device, paths: list[str], blocked: set[str], name: str) -> dict:
    cache = _OUTPUT / "cache" / f"{name}.pt"
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_sha256") == CHECKPOINT_SHA256 and payload.get("count") == len(paths):
            return payload
    features, logits, flags, kept = _forward_paths(model, device, paths, blocked)
    labels = []
    for path in kept:
        lowered = path.lower()
        if "/label_1_watches/" in lowered or lowered.startswith("label_1"):
            labels.append(1)
        else:
            labels.append(0 if "/label_0_watches/" in lowered else _label_from_manifest_path(path))
    payload = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "count": len(paths),
        "paths": kept,
        "features": features,
        "logits": logits,
        "flags": flags,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def _label_from_manifest_path(path: str) -> int:
    lowered = path.lower()
    if "label_1" in lowered:
        return 1
    if "label_0" in lowered:
        return 0
    raise OODArtifactError(f"cannot read an in-domain label without opening a final-test image: {path}")


def _score_ood(model, device, rows: list[dict], blocked: set[str]) -> dict:
    paths = [row["image_path"] for row in rows]
    features, logits, flags, kept = _forward_paths(model, device, paths, blocked)
    if kept != paths:
        raise OODArtifactError("OOD scoring dropped or reordered a row")
    return {
        "rows": rows,
        "features": features,
        "logits": logits,
        "flags": flags,
        "mahalanobis": None,
        "cosine": None,
    }


def _forward_paths(model, device, paths: list[str], blocked: set[str], batch_size: int = 8):
    features = []
    logits = []
    flags = []
    kept = []
    batch_images = []
    batch_flags = []
    batch_paths = []

    def flush() -> None:
        if not batch_images:
            return
        tensor = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            cls, logit = ood._cls_and_logit(model, tensor)
        if cls.dtype != torch.float32 or logit.dtype != torch.float32:
            raise OODArtifactError("authoritative forward is not FP32")
        features.append(cls.detach().float().cpu())
        logits.append(logit.detach().float().cpu().reshape(-1))
        flags.extend(batch_flags)
        kept.extend(batch_paths)
        batch_images.clear()
        batch_flags.clear()
        batch_paths.clear()

    for index, raw in enumerate(paths):
        path = Path(raw)
        image = open_rgb(path, blocked)
        _features, quality_flags = fast_quality_gate(image)
        batch_images.append(preprocess_image(image))
        batch_flags.append(list(quality_flags))
        batch_paths.append(str(path.resolve()))
        if len(batch_images) == batch_size:
            flush()
        if index and index % 500 == 0:
            print(f"scored {index}/{len(paths)}", flush=True)
    flush()
    if not features:
        raise OODArtifactError("scoring produced no features")
    return torch.cat(features), torch.cat(logits), flags, kept


def _write_outputs(report: dict) -> None:
    rows = report["manifest_rows"]
    manifest = {
        "methodology_version": METHODOLOGY_VERSION,
        "samples": rows,
        "sample_count": len(rows),
    }
    manifest_bytes = (_dump(manifest) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "ood_manifest.json").write_bytes(manifest_bytes)
    (_OUTPUT / "ood_manifest.sha256").write_text(manifest_sha + "\n")
    audit = {
        "preexisting": report["audit"],
        "commons_download_failures": report["commons_failures"],
        "rejected_count": len(report["rejected"]),
        "rejected": report["rejected"],
        "admitted_count": len(rows),
        "admitted_by_taxonomy": report["facts"]["group_counts"],
        "exact_catalog_rejections": sum(1 for row in report["rejected"] if row["rejection"] == "exact_sha256_match_to_catalog"),
        "near_duplicate_rejections": sum(1 for row in report["rejected"] if str(row["rejection"]).startswith("near_duplicate")),
        "final_test_samples_used": 0,
        "final_test_images_opened": False,
    }
    (_OUTPUT / "ood_data_audit.json").write_text(_dump(audit) + "\n")
    with (_OUTPUT / "ood_predictions.jsonl").open("w") as handle:
        for row in report["predictions"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (_OUTPUT / "ood_metrics.json").write_text(_dump({"groups": report["group_metrics"], "facts": report["facts"]}) + "\n")
    (_OUTPUT / "ood_threshold_sweep.json").write_text(
        _dump({"production_ood_threshold": None, "threshold_status": "research_candidate_only", "gates": report["threshold_sweep"]}) + "\n"
    )
    (_OUTPUT / "ood_failure_analysis.json").write_text(_dump({"authentic_predictions": report["failures"], "count": len(report["failures"])}) + "\n")
    status = {
        "REAL_OOD_CORPUS_AVAILABLE": True,
        "OOD_VALIDATED": report["status"]["status"] == OOD_PASS,
        "OOD_STATUS": report["status"]["status"],
        "status_reasons": report["status"]["reasons"],
        "production_ood_threshold": None,
        "threshold_status": "research_candidate_only",
        "ood_escape_rate": report["threshold_sweep"][0]["ood_authentic_escape"],
        "ood_review_rate_impact": {
            item["name"]: item["calibration_review_increase"] for item in report["threshold_sweep"] if item["name"] != "none"
        },
        "corpus_counts_by_taxonomy": report["facts"]["group_counts"],
        "provenance_summary": _provenance_summary(rows),
        "frozen_checkpoint_sha256": report["checkpoint_sha256"],
        "frozen_temperature": FROZEN_TEMPERATURE,
        "policy_version": POLICY_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methodology_version": METHODOLOGY_VERSION,
        "final_test_samples_used": 0,
        "model_changed": False,
        "calibration_changed": False,
        "policy_changed": False,
        "promoted": False,
    }
    (_OUTPUT / "ood_status.json").write_text(_dump(status) + "\n")
    (_OUTPUT / "ood_report.md").write_text(_markdown(report, status))
    # Attach scores after the protocol freeze. The compile step needs them.
    report["status_document"] = status


def _provenance_summary(rows: list[dict]) -> dict:
    sources = {}
    licenses = {}
    for row in rows:
        sources[row["source"]] = sources.get(row["source"], 0) + 1
        licenses[row["license_or_usage_note"]] = licenses.get(row["license_or_usage_note"], 0) + 1
    return {"sources": sources, "licenses": licenses, "unknown_license_count": licenses.get("UNKNOWN", 0)}


def _markdown(report: dict, status: dict) -> str:
    lines = [
        "# Phase 35 real OOD validation",
        "",
        status["OOD_STATUS"],
        "",
        "The HypeVault repository did not contain a real OOD corpus. Phase 25 recorded zero samples.",
        "Images were then taken from CC BY 4.0 Rolex Roboflow exports already on disk, a bounded local sneaker sample with license UNKNOWN, and Wikimedia Commons category downloads.",
        "No threshold is promoted. production_ood_threshold is null.",
        "",
        "## Status reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in status["status_reasons"])
    lines.extend(["", "## Corpus counts", ""])
    for name, count in status["corpus_counts_by_taxonomy"].items():
        metrics = report["group_metrics"][name]
        if metrics["count"] == 0:
            lines.append(f"- {name}: 0 scored")
            continue
        lines.append(
            f"- {name}: {count} scored; mean P(fake) {metrics['mean_p_fake']}; "
            f"fraction AUTHENTIC at 0.50 {metrics['authentic_fraction_at_0_50']}; "
            f"fraction FAKE at 0.50 {metrics['fake_fraction_at_0_50']}"
        )
    lines.extend(["", "## Threshold sweep", ""])
    for gate in report["threshold_sweep"]:
        lines.append(
            f"- {gate['name']}: calibration review increase {gate['calibration_review_increase']}; "
            f"authentic escape {json.dumps(gate['ood_authentic_escape'])}"
        )
    lines.extend(["", "## Strongest authentic OOD predictions", ""])
    if not report["failures"]:
        lines.append("- None. No scored OOD image was labeled AUTHENTIC by the current shadow rule without an OOD gate.")
    for row in report["failures"][:15]:
        lines.append(
            f"- {row['taxonomy']} / {row['ood_category']} / {row['brand']}: P(fake) {row['p_fake']}, "
            f"Mahalanobis {row['mahalanobis']}, flags {row['quality_flags']}"
        )
    lines.extend(
        [
            "",
            "Rolex files are Roboflow derivatives. dHash does not undo flips, so the independent photograph count is unknown.",
            "The frozen checkpoint, temperature, policy, and final test were not modified.",
            "",
        ]
    )
    return "\n".join(lines)


def _file_row(path, *, taxonomy, ood_category, source, source_url, source_type, brand, is_watch, is_trained_brand, license_note, collection_method) -> dict:
    payload = path.read_bytes()
    return {
        "image_path": str(path.resolve()),
        "ood_category": ood_category,
        "taxonomy": taxonomy,
        "source": source,
        "source_url": source_url,
        "source_type": source_type,
        "brand": brand,
        "is_watch": is_watch,
        "is_trained_brand": is_trained_brand,
        "license_or_usage_note": license_note,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "width": 0,
        "height": 0,
        "aspect_ratio": 0.0,
        "duplicate_group": "",
        "split": "",
        "collection_method": collection_method,
        "visual_verification": "not_performed",
    }


def _image_files(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _commons_pages(category: str, limit: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "640",
        }
    )
    payload = json.loads(_http_bytes("https://commons.wikimedia.org/w/api.php?" + query))
    pages = (payload.get("query") or {}).get("pages") or {}
    return list(pages.values())


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _meta_text(meta: dict, key: str) -> str:
    value = (meta.get(key) or {}).get("value")
    if not value:
        return "UNKNOWN" if key == "LicenseShortName" else ""
    text = str(value).replace("<", " <")
    cleaned = "".join(part.split(">", 1)[-1] for part in text.split("<"))
    return " ".join(cleaned.split())


def _mentions_trained_brand(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in TRAINED_BRAND_TOKENS)


def _nearest_hamming(catalog: np.ndarray, perceptual: int) -> int:
    if catalog.size == 0:
        return 64
    nearest = 64
    query = np.uint64(perceptual)
    for start in range(0, catalog.size, 4096):
        chunk = catalog[start : start + 4096]
        distance = _popcount(np.bitwise_xor(chunk, query))
        nearest = min(nearest, int(distance.min()))
        if nearest == 0:
            return 0
    return nearest


def _popcount(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.uint64, copy=True)
    masks = (
        (np.uint64(0x5555555555555555), 1),
        (np.uint64(0x3333333333333333), 2),
        (np.uint64(0x0F0F0F0F0F0F0F0F), 4),
    )
    for mask, shift in masks:
        x = (x & mask) + ((x >> np.uint64(shift)) & mask)
    x = (x + (x >> np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x + (x >> np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x + (x >> np.uint64(32))) & np.uint64(0x00000000000000FF)
    return x.astype(np.int16)


def _weight_mark(model) -> tuple:
    parameter = next(model.parameters())
    return (str(parameter.dtype), float(parameter.detach().float().sum().cpu()), tuple(parameter.shape))


def _assert_mark(model, mark) -> None:
    if _weight_mark(model) != mark:
        raise OODArtifactError("frozen weights changed during OOD scoring")


def _final_test_stat() -> tuple:
    stat = _FINAL_TEST_SUMMARY.stat()
    return (stat.st_size, stat.st_mtime_ns)


def _rate(values: list[str], target: str) -> float:
    if not values:
        return 0.0
    return values.count(target) / len(values)


def _counts(values: list[str]) -> dict:
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _attach_detector_scores(report_unused: dict) -> None:
    """Scores are attached in run_phase35 before compile. This marker keeps the call order explicit."""


def _patch_score_attachment(ood_scored: dict, reference: dict) -> None:
    ood_scored["mahalanobis"] = ood.score_features(ood_scored["features"], reference, "mahalanobis")
    ood_scored["cosine"] = ood.score_features(ood_scored["features"], reference, "cosine")
