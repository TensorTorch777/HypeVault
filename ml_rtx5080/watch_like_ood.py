"""Phase 36 independent watch-like open-set evaluation.

The protocol object below is frozen before any holdout image is scored.
No threshold is fit on this corpus. Phase 35 Roboflow derivatives are excluded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
from ood_validation import (
    OODArtifactError,
    _forward_paths,
    _nearest_hamming,
    decide_case,
    fpr_at_tpr,
    open_rgb,
)
from policy import AUTHENTICITY_THRESHOLD, FROZEN_TEMPERATURE, calibrated_probability_from_logit
from reference_inference import file_sha256, load_reference_model
from split_v2 import dhash_image

_PKG = Path(__file__).resolve().parent
_OUTPUT = _PKG / "experiments" / "watch_like_ood_v1"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_FINAL_TEST_SUMMARY = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
_PHASE35_PREDICTIONS = _PKG / "experiments" / "ood_validation_v1" / "ood_predictions.jsonl"
_PHASE35_MANIFEST = _PKG / "experiments" / "ood_validation_v1" / "ood_manifest.json"
_TRAIN_CACHE = _PKG / "experiments" / "ood_validation_v1" / "cache" / "train.pt"
_CALIBRATION_CACHE = _PKG / "experiments" / "ood_validation_v1" / "cache" / "calibration.pt"
CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
PHASE35_UNSEEN_ESCAPE = 0.8814432989690721
SUBSTANTIAL_DROP = 0.25
EXISTING_THRESHOLDS = {
    "conservative": 37.69066974303325,
    "moderate": 35.753779125719554,
    "aggressive": 33.5842585568423,
}
USER_AGENT = "HypeVaultWatchOOD/1.0 (local research audit)"
ORIGINAL_BYTE_CAP = 1_500_000
BRAND_CAP = 40
NEAR_DUP_HAMMING = 4
TRAINED_BRAND_TOKENS = ("lange", "audemars", "patek", "richard mille", "vacheron")
DERIVATIVE_TOKENS = ("derivative", "cropped", "flipped", "flip", "color-jitter", "recolored", "collage", "resized copy")
EXCLUSION_TOKENS = (
    "movement",
    "tram",
    "tramway",
    "advertisement",
    "annonce",
    "newspaper",
    "recueil",
    "logo",
    "wordmark",
    "icon",
    "diagram",
    "schematic",
    "poster",
    "headquarters",
    "factory",
    "calibre",
    "caliber",
)
CATEGORIES = (
    ("Rolex watches", "Rolex", "unseen_luxury"),
    ("Rolex Datejust", "Rolex", "unseen_luxury"),
    ("Rolex Submariner", "Rolex", "unseen_luxury"),
    ("Rolex Daytona", "Rolex", "unseen_luxury"),
    ("Omega watches", "Omega", "unseen_luxury"),
    ("Cartier watches", "Cartier", "unseen_luxury"),
    ("Jaeger-LeCoultre wristwatches", "Jaeger-LeCoultre", "unseen_luxury"),
    ("IWC watches", "IWC", "unseen_luxury"),
    ("Tudor watches", "Tudor", "unseen_luxury"),
    ("Breitling watches", "Breitling", "unseen_luxury"),
    ("Panerai watches", "Panerai", "unseen_luxury"),
    ("TAG Heuer watches", "TAG Heuer", "unseen_luxury"),
    ("Hublot watches", "Hublot", "unseen_luxury"),
    ("Chopard watches", "Chopard", "unseen_luxury"),
    ("Longines watches", "Longines", "unseen_luxury"),
    ("Zenith (watch)", "Zenith", "unseen_luxury"),
    ("Breguet (watch)", "Breguet", "unseen_luxury"),
    ("Casio watches", "Casio", "ordinary_watch"),
    ("Swatch watches", "Swatch", "ordinary_watch"),
    ("Seiko watches", "Seiko", "ordinary_watch"),
    ("Citizen watches", "Citizen", "ordinary_watch"),
    ("Timex Watches", "Timex", "ordinary_watch"),
    ("Tissot wristwatches", "Tissot", "ordinary_watch"),
    ("Smartwatches", "UNKNOWN", "smart_digital"),
    ("Apple Watch", "Apple", "smart_digital"),
    ("Samsung Galaxy Watch", "Samsung", "smart_digital"),
    ("Garmin watches", "Garmin", "smart_digital"),
    ("Fitbit", "Fitbit", "smart_digital"),
    ("Digital watches", "UNKNOWN", "smart_digital"),
    ("People wearing wristwatches", "UNKNOWN", "wrist_store"),
    ("Watch shops", "UNKNOWN", "wrist_store"),
)
GROUPS = ("unseen_luxury", "ordinary_watch", "smart_digital", "wrist_store")
TARGETS = {"unseen_luxury": 300, "ordinary_watch": 200, "smart_digital": 100, "wrist_store": 50}
ROW_FIELDS = (
    "image_path",
    "sha256",
    "source",
    "source_url",
    "brand",
    "watch_category",
    "collection_method",
    "is_original_or_derivative",
    "license_or_usage_note",
    "width",
    "height",
    "aspect_ratio",
)
CONFIRMED_FAIL = "WATCH_OOD_CONFIRMED_FAIL"
PARTIAL = "WATCH_OOD_PARTIAL"
UNRESOLVED = "WATCH_OOD_UNRESOLVED"
IMPROVED = "WATCH_OOD_IMPROVED_BUT_NOT_SAFE"
PASSED = "WATCH_OOD_PASS"


def protocol_document() -> dict:
    """Rules fixed before holdout scoring. This document must not gain results."""
    return {
        "methodology_version": "phase36_watch_like_ood_v1",
        "inclusion": [
            "Commons original file bytes at or below the byte cap",
            "Category is one of the pre-listed watch categories",
            "Brand is the category brand, or UNKNOWN when the category is not brand-specific",
            "LicenseShortName is present",
            "MIME is jpeg, png, or webp",
        ],
        "exclusion": [
            "Five trained brands in the title or description",
            "Roboflow names, .rf. suffixes, and the Phase 35 corpus hashes and source URLs",
            "Derivative tokens and the pre-listed non-watch title tokens",
            "Thumbnail renditions are not admitted",
            "Originals above the byte cap are skipped rather than resized",
            "Exact SHA-256 or dHash Hamming <= 4 against the Phase 16 catalog or Phase 35 scored files",
            "Duplicate source URLs and exact perceptual copies inside this corpus",
        ],
        "categories": [{"commons_category": item[0], "brand": item[1], "watch_category": item[2]} for item in CATEGORIES],
        "targets": TARGETS,
        "brand_cap": BRAND_CAP,
        "brand_cap_meaning": "At most 40 admitted files per brand, or per Commons category when brand is UNKNOWN. The cap is a diversity rule.",
        "original_byte_cap": ORIGINAL_BYTE_CAP,
        "split_rule": "sha256 first 8 hex digits modulo 5 equals 0 means WATCH_OOD_DEV; every other residue is WATCH_OOD_HOLDOUT",
        "holdout_uses": "evaluation only",
        "development_uses": "descriptive counts only",
        "threshold_search": "forbidden",
        "existing_phase35_thresholds_reported_only": EXISTING_THRESHOLDS,
        "authenticity_threshold": AUTHENTICITY_THRESHOLD,
        "temperature": FROZEN_TEMPERATURE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "primary_metric": "WATCH_OOD_HOLDOUT fraction whose current shadow decision is AUTHENTIC",
        "shadow_decision": "LOW_RESOLUTION or LOW_CONTRAST -> REVIEW; else P(fake) >= 0.50 -> FAKE; else AUTHENTIC",
        "meaningful_brand_count": 8,
        "minimum_for_a_conclusion": {
            "holdout_unseen_luxury": 60,
            "unseen_brands_with_at_least_8_holdout_images": 4,
            "holdout_ordinary_watch": 40,
            "holdout_smart_digital": 25,
            "roboflow_count": 0,
            "phase35_sha_overlap": 0,
            "max_single_brand_fraction_of_holdout_unseen": 0.40,
        },
        "wrist_store_shortfall_alone_does_not_block": True,
        "status_rules": [
            "UNRESOLVED when the minimum is not met",
            "PASS only when unseen, ordinary, and smart/digital holdout authentic escape are all <= 0.05",
            "CONFIRMED_FAIL when unseen-luxury holdout escape is >= 0.50",
            "IMPROVED_BUT_NOT_SAFE when unseen-luxury escape is below 0.50, more than 0.25 below the Phase 35 rate, and still above 0.05",
            "PARTIAL when the remaining pattern is mixed",
        ],
        "phase35_unseen_luxury_escape": PHASE35_UNSEEN_ESCAPE,
        "substantial_drop": SUBSTANTIAL_DROP,
        "phase35_group_map": {
            "unseen_luxury": "taxonomy D_unseen_luxury_brand",
            "ordinary_watch": "ood_category fashion_watches or inexpensive_or_fashion_watches",
            "smart_digital": "ood_category smartwatches or digital_watches",
            "wrist_store": "taxonomy E_structurally_different",
        },
        "id_geometry_reference": "Phase 35 calibration CLS cache for this checkpoint, authentic label 0 reported separately",
        "production_ood_threshold": None,
        "final_test_used": False,
    }


def classify_watch_ood(facts: dict) -> dict:
    """Apply the pre-registered status rules. The label is not preset."""
    reasons = []
    if not facts.get("minimum_met"):
        reasons.append("the independent holdout does not meet the pre-registered minimum")
        for item in facts.get("minimum_failures") or []:
            reasons.append(item)
        return {"status": UNRESOLVED, "reasons": reasons}
    escape = facts["holdout_escape"]
    unseen = float(escape["unseen_luxury"])
    ordinary = float(escape["ordinary_watch"])
    smart = float(escape["smart_digital"])
    if unseen <= 0.05 and ordinary <= 0.05 and smart <= 0.05:
        reasons.append("holdout authentic escape is at or below 0.05 for unseen luxury, ordinary, and smart/digital watches")
        return {"status": PASSED, "reasons": reasons}
    if unseen >= 0.50:
        reasons.append("unseen-luxury holdout escape remains at or above 0.50, so the Phase 35 failure is reproduced")
        return {"status": CONFIRMED_FAIL, "reasons": reasons}
    improved_ceiling = PHASE35_UNSEEN_ESCAPE - SUBSTANTIAL_DROP
    if 0.05 < unseen <= improved_ceiling:
        reasons.append("unseen-luxury escape fell by more than 0.25 from Phase 35 and is still above 0.05")
        return {"status": IMPROVED, "reasons": reasons}
    rates = (unseen, ordinary, smart)
    if min(rates) <= 0.05 and max(rates) > 0.15:
        reasons.append("at least one primary group is near the safe line and another remains above 0.15")
        return {"status": PARTIAL, "reasons": reasons}
    reasons.append("the holdout does not reproduce the Phase 35 unseen-luxury failure at the 0.50 line and does not clear the other status bars")
    return {"status": PARTIAL, "reasons": reasons}


def assign_split(digest: str) -> str:
    if int(digest[:8], 16) % 5 == 0:
        return "WATCH_OOD_DEV"
    return "WATCH_OOD_HOLDOUT"


def validate_watch_row(row: dict) -> None:
    missing = [field for field in ROW_FIELDS if field not in row or row[field] in ("", None)]
    if missing:
        raise OODArtifactError("watch row is missing " + ", ".join(missing))
    if row["watch_category"] not in GROUPS:
        raise OODArtifactError(f"unknown watch category {row['watch_category']}")
    if row["source"] == "UNKNOWN" or row["collection_method"] == "UNKNOWN":
        raise OODArtifactError("source and collection_method must be known")
    if len(str(row["sha256"])) != 64:
        raise OODArtifactError("sha256 must be 64 hex characters")
    if row.get("is_trained_brand") is True:
        raise OODArtifactError("trained-brand image")
    lowered = f"{row['brand']} {row.get('source_type', '')}".lower()
    if any(token in lowered for token in TRAINED_BRAND_TOKENS):
        raise OODArtifactError("trained brand in row")
    if ".rf." in str(row["image_path"]) or row.get("source") == "roboflow":
        raise OODArtifactError("roboflow derivative")


def escape_rate(rows: list[dict], decision_key: str = "decision_current") -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[decision_key] == "AUTHENTIC") / len(rows)


def run_phase36() -> dict:
    status_path = _OUTPUT / "watch_like_ood_status.json"
    if status_path.is_file():
        raise RuntimeError("watch_like_ood_v1 already exists; refusing to rewrite it")
    if not torch.cuda.is_available():
        raise OODArtifactError("the frozen FP32 path requires CUDA")
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = _OUTPUT / "watch_like_ood_protocol_v1.json"
    protocol_bytes = _canonical(protocol_document())
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_path.is_file() and protocol_path.read_bytes() != protocol_bytes:
        raise OODArtifactError("protocol file no longer matches the pre-registered document")
    protocol_path.write_bytes(protocol_bytes)
    (_OUTPUT / "watch_like_ood_protocol_v1.sha256").write_text(protocol_sha + "\n")
    before = _final_test_stat()
    if file_sha256(_CHECKPOINT) != CHECKPOINT_SHA256:
        raise OODArtifactError("checkpoint sha256 mismatch")
    catalog = json.loads(_MANIFEST.read_text())
    blocked = {str(Path(path).resolve()) for path in catalog["membership"]["test"]}
    catalog_sha = {row["sha256"] for row in catalog["samples"]}
    catalog_hash = np.array([int(row["perceptual_hash"], 16) for row in catalog["samples"]], dtype=np.uint64)
    phase35 = json.loads(_PHASE35_MANIFEST.read_text())["samples"]
    phase35_sha = {row["sha256"] for row in phase35}
    phase35_urls = {row["source_url"] for row in phase35}
    phase35_hash = np.array([int(row["perceptual_hash"], 16) for row in phase35], dtype=np.uint64)
    print("downloading commons originals", flush=True)
    download_cache = _OUTPUT / "downloaded_rows.json"
    if download_cache.is_file():
        cached_download = json.loads(download_cache.read_text())
        downloaded = cached_download["rows"]
        failures = cached_download["failures"]
        print("reusing downloaded row cache", len(downloaded), flush=True)
    else:
        downloaded, failures = download_commons(_OUTPUT / "corpus" / "commons", phase35_sha, phase35_urls)
        download_cache.write_bytes(_canonical({"rows": downloaded, "failures": failures}))
    screened = screen_rows(downloaded, catalog_sha, catalog_hash, phase35_sha, phase35_hash, blocked)
    admitted = cap_diverse(screened["admitted"])
    screened["rejected"].extend(admitted["capped"])
    rows = admitted["kept"]
    for row in rows:
        row["split"] = assign_split(row["sha256"])
        validate_watch_row(row)
    if protocol_path.read_bytes() != protocol_bytes:
        raise OODArtifactError("protocol changed before scoring")
    print("scoring", len(rows), flush=True)
    device = torch.device("cuda")
    model, digest = load_reference_model(_CHECKPOINT, device)
    if digest != CHECKPOINT_SHA256:
        raise OODArtifactError("loaded digest mismatch")
    reference = _reference_from_cache()
    scored = _score_rows(model, device, rows, blocked, reference)
    if file_sha256(_CHECKPOINT) != CHECKPOINT_SHA256 or _final_test_stat() != before:
        raise OODArtifactError("checkpoint or final-test summary changed")
    if protocol_path.read_bytes() != protocol_bytes:
        raise OODArtifactError("protocol changed after scoring")
    report = _compile(scored, screened, failures, reference, protocol_sha)
    _write(report, protocol_sha)
    return report


def download_commons(destination: Path, phase35_sha: set[str], phase35_urls: set[str]) -> tuple[list[dict], list[dict]]:
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    seen = set(phase35_sha)
    for category, brand, watch_category in CATEGORIES:
        print(f"category {category} admitted {len(rows)}", flush=True)
        try:
            pages = _category_pages(category)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            failures.append({"category": category, "error": str(exc)})
            continue
        candidates = []
        for page in pages:
            title = str(page.get("title") or "")
            info = (page.get("imageinfo") or [{}])[0]
            title_reason = _title_reject(title, watch_category)
            if title_reason:
                failures.append({"category": category, "title": title, "error": title_reason})
                continue
            size = int(info.get("size") or 0)
            mime = str(info.get("mime") or "")
            if mime not in {"image/jpeg", "image/png", "image/webp"} or size <= 0 or size > ORIGINAL_BYTE_CAP or not info.get("url"):
                failures.append({"category": category, "title": title, "error": "original_outside_byte_cap_or_mime"})
                continue
            page_id = page.get("pageid")
            if not isinstance(page_id, int):
                failures.append({"category": category, "title": title, "error": "missing_page_id"})
                continue
            candidates.append((title, info, page_id))
        for start in range(0, len(candidates), 20):
            chunk = candidates[start : start + 20]
            try:
                metadata = _extmetadata_batch([page_id for _title, _info, page_id in chunk])
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"metadata failed {category}: {exc}", flush=True)
                failures.append({"category": category, "error": str(exc)})
                continue
            for title, info, page_id in chunk:
                if shutil.disk_usage(destination).free < 400_000_000:
                    failures.append({"category": category, "error": "stopped because free disk fell below 400MB"})
                    return rows, failures
                info = {**info, **metadata.get(page_id, {})}
                reason = _reject_page(title, info, brand, watch_category)
                if reason:
                    failures.append({"category": category, "title": title, "error": reason})
                    continue
                page_url = info.get("descriptionurl") or ""
                if page_url in phase35_urls:
                    failures.append({"category": category, "title": title, "error": "phase35_source_url"})
                    continue
                try:
                    payload = _http_bytes(info["url"])
                except (OSError, urllib.error.URLError, TimeoutError) as exc:
                    failures.append({"category": category, "title": title, "error": str(exc)})
                    continue
                if len(payload) != int(info.get("size") or -1):
                    failures.append({"category": category, "title": title, "error": "download_size_mismatch"})
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                if digest in seen:
                    failures.append({"category": category, "title": title, "error": "duplicate_or_phase35_sha"})
                    continue
                seen.add(digest)
                stored = destination / f"{digest[:16]}.jpg"
                stored.write_bytes(payload)
                try:
                    with Image.open(stored) as image:
                        image.load()
                        width, height = image.size
                except (UnidentifiedImageError, OSError) as exc:
                    stored.unlink(missing_ok=True)
                    failures.append({"category": category, "title": title, "error": f"unreadable:{exc}"})
                    continue
                rows.append(
                    {
                        "image_path": str(stored.resolve()),
                        "sha256": digest,
                        "source": "wikimedia_commons",
                        "source_url": page_url,
                        "brand": brand,
                        "watch_category": watch_category,
                        "collection_method": "wikimedia_commons_original_file",
                        "is_original_or_derivative": "commons_original_upload",
                        "license_or_usage_note": _meta_text(info.get("extmetadata") or {}, "LicenseShortName"),
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height if height else 0.0,
                        "commons_category": category,
                        "source_type": "commons_original_upload",
                        "is_trained_brand": False,
                        "visual_verification": "not_performed",
                        "category_basis": "commons_category_assignment",
                    }
                )
                time.sleep(0.05)
        print(f"finished {category} total {len(rows)}", flush=True)
    return rows, failures


def screen_rows(rows, catalog_sha, catalog_hash, phase35_sha, phase35_hash, blocked) -> dict:
    admitted = []
    rejected = []
    for row in rows:
        path = Path(row["image_path"])
        if str(path.resolve()) in blocked:
            rejected.append({**row, "rejection": "final_test_path"})
            continue
        if row["sha256"] in catalog_sha or row["sha256"] in phase35_sha:
            rejected.append({**row, "rejection": "exact_sha256_overlap"})
            continue
        try:
            image = open_rgb(path, blocked)
        except OODArtifactError as exc:
            rejected.append({**row, "rejection": f"unreadable:{exc}"})
            continue
        perceptual = dhash_image(image)
        row = dict(row)
        row["perceptual_hash"] = f"{perceptual:016x}"
        nearest_catalog = _nearest_hamming(catalog_hash, perceptual)
        nearest_phase35 = _nearest_hamming(phase35_hash, perceptual)
        row["nearest_catalog_hamming"] = nearest_catalog
        row["nearest_phase35_hamming"] = nearest_phase35
        if nearest_catalog <= NEAR_DUP_HAMMING or nearest_phase35 <= NEAR_DUP_HAMMING:
            rejected.append({**row, "rejection": "perceptual_near_duplicate"})
            continue
        admitted.append(row)
    return {"admitted": _collapse_duplicates(admitted, rejected), "rejected": rejected}


def _collapse_duplicates(admitted: list[dict], rejected: list[dict]) -> list[dict]:
    by_url: dict[str, list[dict]] = {}
    for row in admitted:
        by_url.setdefault(row["source_url"], []).append(row)
    unique = []
    for members in by_url.values():
        ordered = sorted(members, key=lambda item: item["sha256"])
        unique.append(ordered[0])
        for extra in ordered[1:]:
            rejected.append({**extra, "rejection": "duplicate_source_url"})
    by_hash: dict[str, list[dict]] = {}
    for row in unique:
        by_hash.setdefault(row["perceptual_hash"], []).append(row)
    kept = []
    for members in by_hash.values():
        ordered = sorted(members, key=lambda item: item["sha256"])
        leader = ordered[0]["sha256"]
        for index, member in enumerate(ordered):
            member["duplicate_group"] = leader
            if index == 0:
                kept.append(member)
            else:
                rejected.append({**member, "rejection": "exact_perceptual_duplicate"})
    return kept


def cap_diverse(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["watch_category"], row["commons_category"] if row["brand"] == "UNKNOWN" else row["brand"])
        grouped.setdefault(key, []).append(row)
    kept = []
    capped = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda item: item["sha256"])
        kept.extend(ordered[:BRAND_CAP])
        for extra in ordered[BRAND_CAP:]:
            capped.append({**extra, "rejection": "diversity_cap"})
    return {"kept": kept, "capped": capped}


def _compile(scored, screened, failures, reference, protocol_sha) -> dict:
    rows = scored["records"]
    holdout = [row for row in rows if row["split"] == "WATCH_OOD_HOLDOUT"]
    development = [row for row in rows if row["split"] == "WATCH_OOD_DEV"]
    calibration = _calibration_scores(reference)
    group_metrics = {name: _group_metrics(holdout, name, calibration) for name in GROUPS}
    group_metrics["overall_watch_like"] = _summary(holdout, calibration)
    brands = _brand_table(holdout)
    phase35 = _phase35_comparison(holdout)
    facts = _facts(holdout, rows)
    status = classify_watch_ood(facts)
    existing = {
        name: _threshold_view(holdout, threshold)
        for name, threshold in (("current", None), *EXISTING_THRESHOLDS.items())
    }
    return {
        "records": rows,
        "manifest_rows": screened_manifest(rows),
        "rejected": screened["rejected"],
        "download_failures": failures,
        "holdout_count": len(holdout),
        "development_count": len(development),
        "development_escape": {name: escape_rate([row for row in development if row["watch_category"] == name]) for name in GROUPS},
        "group_metrics": group_metrics,
        "brands": brands,
        "phase35_comparison": phase35,
        "facts": facts,
        "status": status,
        "existing_thresholds_not_used_for_selection": existing,
        "geometry": _geometry(holdout, calibration),
        "failures": _failure_table(holdout),
        "protocol_sha256": protocol_sha,
        "targets": _target_report(rows),
    }


def _facts(holdout: list[dict], all_rows: list[dict]) -> dict:
    counts = {name: sum(1 for row in holdout if row["watch_category"] == name) for name in GROUPS}
    brand_counts = {}
    for row in holdout:
        if row["watch_category"] == "unseen_luxury":
            brand_counts[row["brand"]] = brand_counts.get(row["brand"], 0) + 1
    unseen_total = max(1, counts["unseen_luxury"])
    largest = max(brand_counts.values()) if brand_counts else 0
    failures = []
    if counts["unseen_luxury"] < 60:
        failures.append(f"holdout unseen luxury is {counts['unseen_luxury']}, below 60")
    if sum(1 for count in brand_counts.values() if count >= 8) < 4:
        failures.append("fewer than 4 unseen brands have 8 holdout images")
    if counts["ordinary_watch"] < 40:
        failures.append(f"holdout ordinary watches are {counts['ordinary_watch']}, below 40")
    if counts["smart_digital"] < 25:
        failures.append(f"holdout smart/digital watches are {counts['smart_digital']}, below 25")
    if any(row.get("source") == "roboflow" for row in all_rows):
        failures.append("roboflow image present")
    if largest / unseen_total > 0.40 and counts["unseen_luxury"] >= 60:
        failures.append("one unseen brand exceeds 40 percent of the holdout unseen group")
    return {
        "minimum_met": not failures,
        "minimum_failures": failures,
        "holdout_escape": {name: escape_rate([row for row in holdout if row["watch_category"] == name]) for name in GROUPS},
        "holdout_counts": counts,
        "overall_escape": escape_rate(holdout),
        "brand_counts": brand_counts,
    }


def _group_metrics(holdout, name, calibration) -> dict:
    selected = [row for row in holdout if row["watch_category"] == name]
    summary = _summary(selected, calibration)
    summary["brands"] = _brand_table(selected)
    return summary


def _summary(rows: list[dict], calibration: dict) -> dict:
    if not rows:
        return {"count": 0, "authentic": 0, "fake": 0, "review": 0, "authentic_escape_rate": None}
    probabilities = np.array([row["p_fake"] for row in rows], dtype=np.float64)
    decisions = [row["decision_current"] for row in rows]
    maha = np.array([row["mahalanobis"] for row in rows], dtype=np.float64)
    cosine = np.array([row["cosine"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "authentic": decisions.count("AUTHENTIC"),
        "fake": decisions.count("FAKE"),
        "review": decisions.count("REVIEW"),
        "authentic_escape_rate": decisions.count("AUTHENTIC") / len(rows),
        "predicted_authentic_at_0_50": sum(1 for row in rows if row["predicted_class_at_0_50"] == "AUTHENTIC") / len(rows),
        "mean_p_fake": float(probabilities.mean()),
        "median_p_fake": float(np.median(probabilities)),
        "p95_p_fake": float(np.quantile(probabilities, 0.95, method="linear")),
        "max_p_fake": float(probabilities.max()),
        "mahalanobis": _dist(maha),
        "cosine": _dist(cosine),
        "detectors": {
            "mahalanobis": _detector(calibration["mahalanobis_authentic"], maha),
            "cosine": _detector(calibration["cosine_authentic"], cosine),
            "p_fake": _detector(calibration["p_fake_authentic"], probabilities),
        },
    }


def _brand_table(rows: list[dict]) -> list[dict]:
    brands = sorted({row["brand"] for row in rows})
    table = []
    for brand in brands:
        selected = [row for row in rows if row["brand"] == brand]
        if len(selected) < 8:
            continue
        decisions = [row["decision_current"] for row in selected]
        table.append(
            {
                "brand": brand,
                "count": len(selected),
                "authentic_escape_rate": decisions.count("AUTHENTIC") / len(selected),
                "mean_p_fake": float(np.mean([row["p_fake"] for row in selected])),
                "median_mahalanobis": float(np.median([row["mahalanobis"] for row in selected])),
            }
        )
    return table


def _phase35_comparison(holdout: list[dict]) -> dict:
    previous_rows = [json.loads(line) for line in _PHASE35_PREDICTIONS.read_text().splitlines()]
    mapping = {
        "unseen_luxury": lambda row: row["taxonomy"] == "D_unseen_luxury_brand",
        "ordinary_watch": lambda row: row["ood_category"] in {"fashion_watches", "inexpensive_or_fashion_watches"},
        "smart_digital": lambda row: row["ood_category"] in {"smartwatches", "digital_watches"},
        "wrist_store": lambda row: row["taxonomy"] == "E_structurally_different",
    }
    rows = []
    for name, predicate in mapping.items():
        old = [row for row in previous_rows if predicate(row)]
        new = [row for row in holdout if row["watch_category"] == name]
        old_rate = sum(1 for row in old if row["decision_none"] == "AUTHENTIC") / len(old) if old else None
        new_rate = escape_rate(new)
        rows.append(
            {
                "category": name,
                "phase35_count": len(old),
                "phase35_escape": old_rate,
                "phase36_count": len(new),
                "phase36_escape": new_rate if new else None,
                "difference": None if old_rate is None or not new else new_rate - old_rate,
            }
        )
    old_watch = [row for row in previous_rows if row["taxonomy"] in {"C_watch_outside_scope", "D_unseen_luxury_brand", "E_structurally_different"}]
    old_rate = sum(1 for row in old_watch if row["decision_none"] == "AUTHENTIC") / len(old_watch)
    new_rate = escape_rate(holdout) if holdout else None
    rows.append(
        {
            "category": "overall_watch_like",
            "phase35_count": len(old_watch),
            "phase35_escape": old_rate,
            "phase36_count": len(holdout),
            "phase36_escape": new_rate,
            "difference": None if new_rate is None else new_rate - old_rate,
        }
    )
    return {"rows": rows, "independence_note": "Phase 36 excludes Phase 35 hashes, source URLs, and Roboflow derivatives."}


def _geometry(holdout, calibration) -> dict:
    if not holdout:
        return {"status": "no_holdout"}
    maha = [row["mahalanobis"] for row in holdout]
    cosine = [row["cosine"] for row in holdout]
    probability = [row["p_fake"] for row in holdout]
    return {
        "id_reference": "calibration authentic images from the frozen checkpoint cache",
        "holdout_count": len(holdout),
        "mahalanobis": _detector(calibration["mahalanobis_authentic"], maha),
        "cosine": _detector(calibration["cosine_authentic"], cosine),
        "authenticity_probability": _detector(calibration["p_fake_authentic"], probability),
        "score_direction": "larger score is treated as more OOD for all three columns",
        "threshold_selected": False,
    }


def _failure_table(holdout: list[dict]) -> dict:
    mistakes = [row for row in holdout if row["decision_current"] == "AUTHENTIC"]
    mistakes.sort(key=lambda row: (row["p_fake"], row["sha256"]))
    table = [
        {
            "image": row["image_path"],
            "brand": row["brand"],
            "category": row["watch_category"],
            "p_fake": row["p_fake"],
            "mahalanobis": row["mahalanobis"],
            "cosine": row["cosine"],
            "quality_flags": row["quality_flags"],
            "decision": row["decision_current"],
            "source": row["source_url"],
        }
        for row in mistakes[:40]
    ]
    return {"authentic_count": len(mistakes), "lowest_p_fake": table}


def _threshold_view(holdout: list[dict], threshold) -> dict:
    view = {}
    for name in GROUPS:
        selected = [row for row in holdout if row["watch_category"] == name]
        decisions = [
            decide_case(row["p_fake"], row["quality_flags"], row["mahalanobis"], threshold)
            for row in selected
        ]
        view[name] = {
            "count": len(selected),
            "authentic_escape_rate": decisions.count("AUTHENTIC") / len(selected) if selected else None,
        }
    return {"threshold": threshold, "used_for_selection": False, "groups": view}


def _score_rows(model, device, rows, blocked, reference) -> dict:
    if not rows:
        return {"records": []}
    paths = [str(Path(row["image_path"]).resolve()) for row in rows]
    features, logits, flags, kept = _forward_paths(model, device, paths, blocked)
    if kept != paths:
        raise OODArtifactError("scoring reordered watch images")
    mahalanobis = ood.score_features(features, reference, "mahalanobis").tolist()
    cosine = ood.score_features(features, reference, "cosine").tolist()
    records = []
    for index, row in enumerate(rows):
        probability = calibrated_probability_from_logit(float(logits[index]))
        decision = decide_case(probability, flags[index], mahalanobis[index], None)
        records.append(
            {
                **row,
                "raw_logit": float(logits[index]),
                "p_fake": probability,
                "predicted_class_at_0_50": "FAKE" if probability >= AUTHENTICITY_THRESHOLD else "AUTHENTIC",
                "quality_flags": flags[index],
                "mahalanobis": mahalanobis[index],
                "cosine": cosine[index],
                "decision_current": decision,
            }
        )
    return {"records": records}


def _reference_from_cache() -> dict:
    train = torch.load(_TRAIN_CACHE, map_location="cpu", weights_only=False)
    if train["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise OODArtifactError("training-feature cache is not the frozen checkpoint")
    return ood.fit_reference(train["features"], train["labels"])


def _calibration_scores(reference) -> dict:
    payload = torch.load(_CALIBRATION_CACHE, map_location="cpu", weights_only=False)
    if payload["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise OODArtifactError("calibration-feature cache is not the frozen checkpoint")
    authentic = payload["labels"] == 0
    features = payload["features"][authentic]
    logits = payload["logits"][authentic]
    return {
        "mahalanobis_authentic": ood.score_features(features, reference, "mahalanobis"),
        "cosine_authentic": ood.score_features(features, reference, "cosine"),
        "p_fake_authentic": torch.tensor([calibrated_probability_from_logit(float(value)) for value in logits], dtype=torch.float64),
    }


def _detector(id_scores, ood_scores) -> dict:
    id_vector = torch.as_tensor(id_scores, dtype=torch.float64).reshape(-1)
    ood_vector = torch.as_tensor(ood_scores, dtype=torch.float64).reshape(-1)
    ranking = ood.ranking_metrics(id_vector, ood_vector)
    point = fpr_at_tpr(id_vector.numpy(), ood_vector.numpy())
    return {
        "roc_auc": ranking.get("roc_auc"),
        "average_precision": ranking.get("average_precision"),
        "fpr_at_95_tpr": point["fpr_at_95_tpr"],
        "fpr_status": point["status"],
        "ood_mean": None if ood_vector.numel() == 0 else float(ood_vector.mean()),
        "ood_median": None if ood_vector.numel() == 0 else float(ood_vector.median()),
        "id_mean": float(id_vector.mean()),
        "id_median": float(id_vector.median()),
    }


def _extmetadata_batch(page_ids: list[int]) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "pageids": "|".join(str(page_id) for page_id in page_ids),
        "prop": "imageinfo",
        "iiprop": "extmetadata|url",
    }
    payload = json.loads(_http_bytes("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
    found = {}
    for page in ((payload.get("query") or {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        found[int(page.get("pageid"))] = {
            "extmetadata": info.get("extmetadata") or {},
            "descriptionurl": info.get("descriptionurl") or "",
        }
    return found


def _title_reject(title: str, watch_category: str) -> str | None:
    lowered = title.lower()
    if any(token in lowered for token in TRAINED_BRAND_TOKENS):
        return "trained_brand_text"
    if any(token in lowered for token in DERIVATIVE_TOKENS):
        return "derivative_marker"
    if any(token in lowered for token in EXCLUSION_TOKENS):
        return "non_watch_or_part_marker"
    if watch_category != "wrist_store" and "boutique" in lowered:
        return "boutique_exterior"
    return None


def _category_pages(category: str) -> list[dict]:
    pages = []
    continuation = None
    for _ in range(2):
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
        }
        if continuation:
            params["gcmcontinue"] = continuation
        payload = json.loads(_http_bytes("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))
        pages.extend((payload.get("query") or {}).get("pages", {}).values())
        continuation = (payload.get("continue") or {}).get("gcmcontinue")
        if not continuation:
            break
        time.sleep(0.2)
    return pages


def _reject_page(title: str, info: dict, brand: str, watch_category: str) -> str | None:
    mime = str(info.get("mime") or "")
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        return f"mime {mime or 'missing'}"
    size = int(info.get("size") or 0)
    if size <= 0 or size > ORIGINAL_BYTE_CAP:
        return "original_outside_byte_cap"
    if not info.get("url") or not info.get("descriptionurl"):
        return "missing_original_url"
    meta = info.get("extmetadata") or {}
    description = " ".join((title, _meta_text(meta, "ImageDescription"), _meta_text(meta, "ObjectName")))
    lowered = description.lower()
    if any(token in lowered for token in TRAINED_BRAND_TOKENS):
        return "trained_brand_text"
    if any(token in lowered for token in DERIVATIVE_TOKENS):
        return "derivative_marker"
    if any(token in lowered for token in EXCLUSION_TOKENS):
        return "non_watch_or_part_marker"
    if watch_category != "wrist_store" and "boutique" in lowered:
        return "boutique_exterior"
    license_name = _meta_text(meta, "LicenseShortName")
    if license_name == "UNKNOWN":
        return "license_missing"
    if brand != "UNKNOWN" and _mentions_other_trained(lowered):
        return "trained_brand_text"
    return None


def _mentions_other_trained(text: str) -> bool:
    return any(token in text for token in TRAINED_BRAND_TOKENS)


def _target_report(rows: list[dict]) -> dict:
    achieved = {name: sum(1 for row in rows if row["watch_category"] == name) for name in GROUPS}
    return {
        "target": TARGETS,
        "achieved": achieved,
        "shortfall": {name: max(0, TARGETS[name] - achieved[name]) for name in GROUPS},
        "distinct_source_urls": len({row["source_url"] for row in rows}),
        "confirmed_original_uploads": sum(1 for row in rows if row["is_original_or_derivative"] == "commons_original_upload"),
        "unknown_original_status": sum(1 for row in rows if row["is_original_or_derivative"] == "UNKNOWN"),
        "known_independent_photograph_count": sum(1 for row in rows if row["is_original_or_derivative"] == "commons_original_upload"),
    }


def _write(report: dict, protocol_sha: str) -> None:
    manifest = {"samples": report["manifest_rows"], "sample_count": len(report["manifest_rows"])}
    manifest_bytes = _canonical(manifest)
    (_OUTPUT / "watch_like_ood_manifest.json").write_bytes(manifest_bytes)
    (_OUTPUT / "watch_like_ood_manifest.sha256").write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n")
    audit = {
        "download_failure_count": len(report["download_failures"]),
        "download_failures_sample": report["download_failures"][:80],
        "rejected_count": len(report["rejected"]),
        "rejection_counts": _counts(row.get("rejection", "unknown") for row in report["rejected"]),
        "admitted_count": len(report["manifest_rows"]),
        "targets": report["targets"],
        "final_test_images_opened": False,
        "final_test_samples_used": 0,
        "roboflow_admitted": 0,
        "protocol_sha256": protocol_sha,
    }
    (_OUTPUT / "data_audit.json").write_bytes(_canonical(audit))
    with (_OUTPUT / "predictions.jsonl").open("w") as handle:
        for row in report["records"]:
            handle.write(json.dumps(_public_prediction(row), sort_keys=True) + "\n")
    (_OUTPUT / "metrics.json").write_bytes(_canonical({"holdout": report["group_metrics"], "brands": report["brands"], "development_escape_descriptive_only": report["development_escape"]}))
    (_OUTPUT / "phase35_comparison.json").write_bytes(_canonical(report["phase35_comparison"]))
    (_OUTPUT / "failure_analysis.json").write_bytes(_canonical(report["failures"]))
    (_OUTPUT / "ood_geometry_analysis.json").write_bytes(_canonical({"geometry": report["geometry"], "existing_thresholds_not_selected": report["existing_thresholds_not_used_for_selection"]}))
    status = {
        "WATCH_OOD_STATUS": report["status"]["status"],
        "status_reasons": report["status"]["reasons"],
        "INDEPENDENT_CORPUS_AVAILABLE": len(report["records"]) > 0,
        "WATCH_LIKE_OOD_VALIDATED": report["status"]["status"] == PASSED,
        "corpus_counts": report["facts"]["holdout_counts"],
        "admitted_counts": report["targets"]["achieved"],
        "targets": report["targets"],
        "independent_source_count": report["targets"]["distinct_source_urls"],
        "categories": list(GROUPS),
        "duplicate_counts": audit["rejection_counts"],
        "phase35_comparison": report["phase35_comparison"]["rows"],
        "watch_like_ood_escape_rate": report["facts"]["overall_escape"],
        "category_escape_rates": report["facts"]["holdout_escape"],
        "frozen_checkpoint_sha256": CHECKPOINT_SHA256,
        "frozen_temperature": FROZEN_TEMPERATURE,
        "policy_version": "shadow_v1",
        "final_test_untouched": True,
        "final_test_samples_used": 0,
        "model_changed": False,
        "calibration_changed": False,
        "threshold_changed": False,
        "production_ood_threshold": None,
        "promoted": False,
        "protocol_sha256": protocol_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (_OUTPUT / "watch_like_ood_status.json").write_bytes(_canonical(status))
    (_OUTPUT / "watch_like_ood_report.md").write_text(_markdown(report, status))


def _markdown(report: dict, status: dict) -> str:
    lines = [
        "# Phase 36 watch-like open-set evaluation",
        "",
        status["WATCH_OOD_STATUS"],
        "",
        "Phase 35 Roboflow derivatives were excluded. Commons original files were scored with the frozen FP32 checkpoint.",
        "No threshold was selected on this holdout.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in status["status_reasons"])
    lines.extend(["", "## Holdout escape", ""])
    for name, rate in status["category_escape_rates"].items():
        count = status["corpus_counts"][name]
        lines.append(f"- {name}: {count} images, authentic escape {rate}")
    lines.append(f"- overall: {status['watch_like_ood_escape_rate']}")
    lines.extend(["", "## Phase 35 comparison", ""])
    for row in status["phase35_comparison"]:
        lines.append(
            f"- {row['category']}: phase35 {row['phase35_escape']} ({row['phase35_count']}), "
            f"phase36 {row['phase36_escape']} ({row['phase36_count']}), difference {row['difference']}"
        )
    lines.extend(["", "The final test was not opened. The checkpoint, temperature, and policy were not changed.", ""])
    return "\n".join(lines)


def screened_manifest(rows: list[dict]) -> list[dict]:
    public = []
    for row in rows:
        public.append({field: row[field] for field in ROW_FIELDS} | {"split": row["split"], "commons_category": row["commons_category"], "sha_group": row.get("duplicate_group")})
    return public


def _public_prediction(row: dict) -> dict:
    return {
        "image_path": row["image_path"],
        "sha256": row["sha256"],
        "source_url": row["source_url"],
        "brand": row["brand"],
        "watch_category": row["watch_category"],
        "split": row["split"],
        "raw_logit": row["raw_logit"],
        "p_fake": row["p_fake"],
        "predicted_class_at_0_50": row["predicted_class_at_0_50"],
        "quality_flags": row["quality_flags"],
        "mahalanobis": row["mahalanobis"],
        "cosine": row["cosine"],
        "decision_current": row["decision_current"],
        "is_original_or_derivative": row["is_original_or_derivative"],
        "license_or_usage_note": row["license_or_usage_note"],
    }


def _dist(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
    }


def _counts(values) -> dict:
    result = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def _meta_text(meta: dict, key: str) -> str:
    value = (meta.get(key) or {}).get("value")
    if not value:
        return "UNKNOWN" if key == "LicenseShortName" else ""
    text = str(value).replace("<", " <")
    cleaned = "".join(part.split(">", 1)[-1] for part in text.split("<"))
    return " ".join(cleaned.split())


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code != 429:
            raise
        time.sleep(3)
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read()


def _final_test_stat() -> tuple:
    stat = _FINAL_TEST_SUMMARY.stat()
    return (stat.st_size, stat.st_mtime_ns)


def _canonical(payload) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
