"""Phase 37 watch-corpus acquisition and audit.

This module does not load the authenticity model, does not score images,
and does not rewrite the Phase 36 protocol.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROTOCOL_PATH = (
    Path(__file__).resolve().parent
    / "experiments"
    / "watch_like_ood_v1"
    / "watch_like_ood_protocol_v1.json"
)
PROTOCOL_SHA256 = "babaaf2b5c30d252c39765886ee5db032c09c4e3f25cf0e9eacc63f17128dde9"
ADMITTED_ORIGINALITY = frozenset({"original_upload", "original_photo", "dataset_original"})
CATEGORIES = frozenset({"unseen_luxury", "ordinary_watch", "smart_digital", "wrist_store"})
EXCLUDED_BRANDS = frozenset(
    {
        "a. lange & sohne",
        "a. lange & söhne",
        "audemars piguet",
        "patek philippe",
        "richard mille",
        "vacheron constantin",
    }
)
DERIVATIVE_TOKENS = (
    ".rf.",
    "/thumb/",
    "thumbnail",
    "roboflow",
    "_aug",
    "screenshot",
)
FLICKR_DERIVATIVE_SUFFIXES = ("_b.", "_c.", "_z.", "_m.", "_n.", "_t.", "_s.", "_q.", "_w.", "_k.")
MANIFEST_FIELDS = (
    "image_path",
    "sha256",
    "source",
    "source_url",
    "source_id",
    "brand",
    "category",
    "subcategory",
    "license",
    "license_url",
    "collection_timestamp",
    "image_width",
    "image_height",
    "file_size",
    "format",
    "is_original_or_derivative",
    "independent_source_group",
    "phase",
    "admission_status",
    "rejection_reason",
)
NEAR_DUP_HAMMING = 4
DHASH_SIZE = 8
WRIST_HOLDOUT_TARGET = 25
SPLIT_DEV = "WATCH_OOD_DEV"
SPLIT_HOLDOUT = "WATCH_OOD_HOLDOUT"


class AcquisitionError(ValueError):
    """A Phase 37 row or manifest violated a recorded rule."""


def protocol_sha256(path: Path = PROTOCOL_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    digest = protocol_sha256(path)
    if digest != PROTOCOL_SHA256:
        raise AcquisitionError("Phase 36 protocol hash changed")
    return json.loads(path.read_text())


def dhash_image(image: Image.Image, hash_size: int = DHASH_SIZE) -> int:
    """Same difference hash as split_v2.dhash_image. No model is involved."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BOX)
    pixels = list(gray.getdata())
    bits = 0
    width = hash_size + 1
    for row in range(hash_size):
        base = row * width
        for col in range(hash_size):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


def _norm(value: object) -> str:
    return str(value or "").strip()


def _lower(value: object) -> str:
    return _norm(value).lower()


def brand_excluded(brand: object) -> bool:
    return _lower(brand) in EXCLUDED_BRANDS


def derivative_reason(row: dict) -> str:
    originality = _lower(row.get("is_original_or_derivative"))
    if originality == "derivative":
        return "derivative_originality_value"
    blob = " ".join(
        _lower(row.get(key))
        for key in ("image_path", "source_url", "source", "source_id", "format")
    )
    if any(token in blob for token in DERIVATIVE_TOKENS):
        return "derivative_marker"
    if "staticflickr.com" in blob and any(suffix in blob for suffix in FLICKR_DERIVATIVE_SUFFIXES):
        return "flickr_resized_rendition"
    return ""


def provenance_reason(row: dict) -> str:
    if _norm(row.get("source")).upper() == "UNKNOWN" or not _norm(row.get("source")):
        return "missing_source"
    if _norm(row.get("source_url")).upper() == "UNKNOWN" or not _norm(row.get("source_url")):
        return "missing_source_url"
    if _norm(row.get("license")).upper() == "UNKNOWN" or not _norm(row.get("license")):
        return "missing_license"
    originality = _norm(row.get("is_original_or_derivative"))
    if originality not in ADMITTED_ORIGINALITY:
        return "originality_not_admitted"
    if _norm(row.get("brand_verification")) not in {"source_verified", "unknown"}:
        return "brand_verification_not_enumerated"
    return ""


def category_reason(row: dict) -> str:
    category = _norm(row.get("category"))
    if category not in CATEGORIES:
        return "ambiguous_category"
    if category in {"unseen_luxury", "ordinary_watch", "smart_digital"}:
        brand = _norm(row.get("brand"))
        if brand.upper() == "UNKNOWN" or not brand:
            return "brand_not_source_verified"
        if _norm(row.get("brand_verification")) != "source_verified":
            return "brand_not_source_verified"
    return ""


def rejection_reason(row: dict) -> str:
    """Return the first admission failure, or an empty string when the row may be admitted."""
    for check in (derivative_reason, provenance_reason):
        reason = check(row)
        if reason:
            return reason
    if brand_excluded(row.get("brand")):
        return "excluded_trained_brand"
    return category_reason(row)


def validate_manifest_row(row: dict) -> None:
    missing = [field for field in MANIFEST_FIELDS if field not in row]
    if missing:
        raise AcquisitionError(f"manifest row missing {missing}")
    status = row["admission_status"]
    if status not in {"admitted", "rejected"}:
        raise AcquisitionError(f"bad admission_status {status}")
    if status == "admitted" and rejection_reason(row):
        raise AcquisitionError(f"admitted row failed {rejection_reason(row)}")
    if status == "rejected" and not _norm(row.get("rejection_reason")):
        raise AcquisitionError("rejected row needs a rejection_reason")
    if "p_fake" in row or "logit" in row or "mahalanobis" in row:
        raise AcquisitionError("model scores are not part of an acquisition row")


def assign_source_group_split(group_id: str) -> str:
    """Deterministic split for a whole source family. Model scores are not an input."""
    digest = hashlib.sha256(f"phase37-source-group:{group_id}".encode()).hexdigest()
    if int(digest[:8], 16) % 5 == 0:
        return SPLIT_DEV
    return SPLIT_HOLDOUT


def assign_holdout(rows: list[dict]) -> list[dict]:
    """Assign dev or holdout after admission. Every member of a source group shares a split."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("admission_status") != "admitted":
            continue
        if any(key in row for key in ("p_fake", "logit", "mahalanobis")):
            raise AcquisitionError("model scores cannot assign holdout membership")
        group = _norm(row.get("independent_source_group"))
        if not group or group.upper() == "UNKNOWN":
            raise AcquisitionError("admitted row needs an independent source group")
        grouped.setdefault(group, []).append(row)
    assigned: list[dict] = []
    for group_id in sorted(grouped):
        split = assign_source_group_split(group_id)
        for row in grouped[group_id]:
            item = dict(row)
            item["split"] = split
            assigned.append(item)
    return assigned


def coverage(rows: list[dict], protocol: dict) -> dict:
    admitted = [row for row in rows if row.get("admission_status") == "admitted"]
    assigned = assign_holdout(admitted)
    targets = protocol["targets"]
    minimum = protocol["minimum_for_a_conclusion"]
    groups = ("unseen_luxury", "ordinary_watch", "smart_digital", "wrist_store")
    by_group = {}
    for group in groups:
        candidates = [row for row in rows if row.get("category") == group]
        kept = [row for row in assigned if row.get("category") == group]
        holdout = [row for row in kept if row.get("split") == SPLIT_HOLDOUT]
        holdout_target = {
            "unseen_luxury": minimum["holdout_unseen_luxury"],
            "ordinary_watch": minimum["holdout_ordinary_watch"],
            "smart_digital": minimum["holdout_smart_digital"],
            "wrist_store": WRIST_HOLDOUT_TARGET,
        }[group]
        by_group[group] = {
            "target": targets[group],
            "candidate": len(candidates),
            "admitted": len(kept),
            "holdout": len(holdout),
            "holdout_target": holdout_target,
            "ready": len(kept) >= targets[group] and len(holdout) >= holdout_target,
        }
    brand_holdout: dict[str, int] = {}
    for row in assigned:
        if row.get("category") == "unseen_luxury" and row.get("split") == SPLIT_HOLDOUT:
            brand_holdout[row["brand"]] = brand_holdout.get(row["brand"], 0) + 1
    qualified_brands = sorted(brand for brand, count in brand_holdout.items() if count >= 8)
    unseen_holdout = by_group["unseen_luxury"]["holdout"]
    largest = max(brand_holdout.values()) if brand_holdout else 0
    fraction_ok = unseen_holdout == 0 or (largest / unseen_holdout) <= minimum[
        "max_single_brand_fraction_of_holdout_unseen"
    ]
    protocol_minimum_met = (
        by_group["unseen_luxury"]["holdout"] >= minimum["holdout_unseen_luxury"]
        and by_group["ordinary_watch"]["holdout"] >= minimum["holdout_ordinary_watch"]
        and by_group["smart_digital"]["holdout"] >= minimum["holdout_smart_digital"]
        and by_group["unseen_luxury"]["admitted"] >= targets["unseen_luxury"]
        and by_group["ordinary_watch"]["admitted"] >= targets["ordinary_watch"]
        and by_group["smart_digital"]["admitted"] >= targets["smart_digital"]
        and len(qualified_brands) >= minimum["unseen_brands_with_at_least_8_holdout_images"]
        and fraction_ok
    )
    return {
        "groups": by_group,
        "unseen_luxury_brands": qualified_brands,
        "brand_holdout_counts": brand_holdout,
        "independent_source_count": len({row["independent_source_group"] for row in assigned}),
        "protocol_minimum_met": protocol_minimum_met,
        "corpus_ready": protocol_minimum_met and all(item["ready"] for item in by_group.values()),
    }


def corpus_ready(rows: list[dict], protocol: dict | None = None) -> bool:
    document = protocol if protocol is not None else load_protocol()
    return bool(coverage(rows, document)["corpus_ready"])


def manifest_sha256(payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse_final_test(path: Path) -> None:
    if "final_test_v1" in Path(path).parts:
        raise AcquisitionError("refusing to open a locked final-test path")


def perceptual_hash(path: Path) -> int:
    _refuse_final_test(path)
    with Image.open(path) as image:
        return dhash_image(image)


class IdentityIndex:
    """Hashes and URLs from earlier corpora. This index does not open final-test images."""

    def __init__(self) -> None:
        self.sha256: set[str] = set()
        self.source_urls: set[str] = set()
        self.source_ids: set[str] = set()
        self.filenames: set[str] = set()
        self.perceptual: list[int] = []

    def add_record(self, record: dict) -> None:
        sha = _lower(record.get("sha256"))
        if len(sha) == 64:
            self.sha256.add(sha)
        for key in ("source_url", "url"):
            url = _norm(record.get(key))
            if url and url.upper() != "UNKNOWN":
                self.source_urls.add(url)
        source_id = _norm(record.get("source_id"))
        if source_id and source_id.upper() != "UNKNOWN":
            self.source_ids.add(source_id)
        path = _norm(record.get("image_path") or record.get("path"))
        if path:
            self.filenames.add(Path(path).name.lower())
        perceptual = _norm(record.get("perceptual_hash"))
        if perceptual:
            self.perceptual.append(int(perceptual, 16))


def duplicate_reason(row: dict, index: IdentityIndex, perceptual: int | None) -> str:
    sha = _lower(row.get("sha256"))
    if sha in index.sha256:
        return "exact_sha256"
    url = _norm(row.get("source_url"))
    if url and url in index.source_urls:
        return "source_url"
    source_id = _norm(row.get("source_id"))
    if source_id and source_id.upper() != "UNKNOWN" and source_id in index.source_ids:
        return "source_id"
    name = Path(_norm(row.get("image_path"))).name.lower()
    if name and name in index.filenames:
        return "filename"
    if perceptual is not None and index.perceptual:
        nearest = min(hamming_distance(perceptual, other) for other in index.perceptual)
        if nearest <= NEAR_DUP_HAMMING:
            return "near_duplicate"
    return ""


def audit_rows(rows: list[dict], index: IdentityIndex, perceptual_by_sha: dict[str, int]) -> dict:
    counts = {
        "candidate_count": len(rows),
        "admitted_count": 0,
        "exact_duplicates_rejected": 0,
        "near_duplicates_rejected": 0,
        "derivatives_rejected": 0,
        "provenance_failures_rejected": 0,
        "ambiguous_category_rejected": 0,
        "brand_exclusions_rejected": 0,
    }
    audited: list[dict] = []
    for row in rows:
        item = dict(row)
        perceptual = perceptual_by_sha.get(_lower(item.get("sha256")))
        reason = derivative_reason(item)
        kind = "derivative"
        if not reason:
            reason = duplicate_reason(item, index, perceptual)
            kind = "near" if reason == "near_duplicate" else "exact"
        if not reason:
            reason = provenance_reason(item)
            kind = "provenance"
        if not reason and brand_excluded(item.get("brand")):
            reason = "excluded_trained_brand"
            kind = "brand"
        if not reason:
            reason = category_reason(item)
            kind = "category"
        if reason:
            item["admission_status"] = "rejected"
            item["rejection_reason"] = reason
            if kind == "derivative":
                counts["derivatives_rejected"] += 1
            elif kind == "near":
                counts["near_duplicates_rejected"] += 1
            elif kind == "exact":
                counts["exact_duplicates_rejected"] += 1
            elif kind == "provenance":
                counts["provenance_failures_rejected"] += 1
            elif kind == "brand":
                counts["brand_exclusions_rejected"] += 1
            else:
                counts["ambiguous_category_rejected"] += 1
        else:
            item["admission_status"] = "admitted"
            item["rejection_reason"] = ""
            counts["admitted_count"] += 1
        validate_manifest_row(item)
        audited.append(item)
    return {"counts": counts, "rows": audited}


def _records_from_json(path: Path, list_keys: tuple[str, ...]) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    records: list[dict] = []
    for key in list_keys:
        value = payload.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def load_prior_index(repo: Path) -> IdentityIndex:
    index = IdentityIndex()
    experiments = repo / "ml_rtx5080" / "experiments"
    catalogs = (
        (experiments / "dataset_audit" / "split_manifest_v2.json", ("samples",)),
        (experiments / "ood_validation_v1" / "ood_manifest.json", ("samples",)),
        (experiments / "ood_v1" / "ood_manifest.json", ("samples",)),
        (experiments / "watch_like_ood_v1" / "watch_like_ood_manifest.json", ("samples",)),
    )
    for path, keys in catalogs:
        for record in _records_from_json(path, keys):
            index.add_record(record)
    phase36 = experiments / "watch_like_ood_v1"
    downloaded = _records_from_json(phase36 / "downloaded_rows.json", ("rows",))
    for record in downloaded:
        index.add_record(record)
        image_path = _norm(record.get("image_path"))
        if not image_path:
            continue
        path = Path(image_path)
        if "final_test_v1" in path.parts or not path.exists():
            continue
        if not _norm(record.get("perceptual_hash")):
            index.perceptual.append(perceptual_hash(path))
    return index


def commons_rate_limit_ceiling(repo: Path) -> dict[str, int]:
    protocol = load_protocol()
    category_to_group = {
        item["commons_category"]: item["watch_category"] for item in protocol["categories"]
    }
    path = repo / "ml_rtx5080" / "experiments" / "watch_like_ood_v1" / "downloaded_rows.json"
    payload = json.loads(path.read_text())
    counts = {name: 0 for name in CATEGORIES}
    for item in payload["failures"]:
        if not str(item.get("error", "")).startswith("HTTP Error 429"):
            continue
        group = category_to_group.get(item.get("category"))
        if group in counts:
            counts[group] += 1
    return counts


def qualification_sources() -> list[dict]:
    """Recorded pilots. These facts come from qualification requests, not from model scores."""
    return [
        {
            "source": "Pixnio",
            "source_url": "https://pixnio.com/objects/clocks/wristwatch-bracelet-metal-leather-hour-minute-hand",
            "access_method": "HTTPS original JPEG whose filename has no pixel-size suffix",
            "original_bytes_downloadable": True,
            "typical_dimensions": "5400x3600",
            "typical_file_size": 972467,
            "license": "CC0 1.0",
            "provenance": "Public CC0 photo page names an author and a camera. It does not name a watch brand.",
            "stable_ids": True,
            "duplicate_detection_possible": True,
            "derivatives_identifiable": True,
            "expected_independent_photographs": "No brand catalog. Search pages for Rolex, Omega, Casio, and Seiko did not list source-verified brand sets.",
            "automation_permitted": "Public HTTP pages are readable. Bulk collection cannot satisfy brand verification.",
            "rate_limits": "UNKNOWN",
            "originates_from_another_dataset": False,
            "pilot_result": "rejected",
            "rejection_reason": "ambiguous_category",
        },
        {
            "source": "Openverse",
            "source_url": "https://api.openverse.org/v1/images/",
            "access_method": "Public JSON API",
            "original_bytes_downloadable": False,
            "typical_dimensions": "1024px Flickr large rendition",
            "typical_file_size": "UNKNOWN",
            "license": "License of the upstream photo is present. The delivered URL is not the original file.",
            "provenance": "Aggregator. The sample URLs were Flickr _b.jpg renditions, not _o originals.",
            "stable_ids": True,
            "duplicate_detection_possible": True,
            "derivatives_identifiable": True,
            "expected_independent_photographs": 49,
            "automation_permitted": "The API is public. It does not supply the original bytes required here.",
            "rate_limits": "UNKNOWN",
            "originates_from_another_dataset": True,
            "pilot_result": "rejected",
            "rejection_reason": "resized_derivative_url",
        },
        {
            "source": "Flickr",
            "source_url": "https://www.flickr.com/photos/8869879@N05/5424986061",
            "access_method": "Public sizes page exposes an _o original URL. No API key is configured.",
            "original_bytes_downloadable": True,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": 2475875,
            "license": "CC BY-SA 2.0",
            "provenance": "Uploader title names an Omega Speedmaster. One photograph is not a corpus.",
            "stable_ids": True,
            "duplicate_detection_possible": True,
            "derivatives_identifiable": True,
            "expected_independent_photographs": "UNKNOWN",
            "automation_permitted": "Bulk search automation is not available without an API key and is not used.",
            "rate_limits": "UNKNOWN",
            "originates_from_another_dataset": False,
            "pilot_result": "rejected",
            "rejection_reason": "original_above_byte_cap",
        },
        {
            "source": "Wikimedia Commons",
            "source_url": "https://commons.wikimedia.org/",
            "access_method": "MediaWiki API metadata, then original upload URL",
            "original_bytes_downloadable": False,
            "typical_dimensions": "One probed original was 1600x1200",
            "typical_file_size": 662475,
            "license": "Per-file LicenseShortName when a download succeeds",
            "provenance": "Category membership can verify a brand. Phase 36 bulk original downloads received HTTP 429.",
            "stable_ids": True,
            "duplicate_detection_possible": True,
            "derivatives_identifiable": True,
            "expected_independent_photographs": "Rate-limited size-eligible files still on the server: unseen 258, ordinary 155, smart/digital 104, wrist/store 14.",
            "automation_permitted": "API listing is allowed. Original-byte delivery was rate-limited in Phase 36 and is not a dependable collection method.",
            "rate_limits": "HTTP 429 on bulk original downloads. Thumbnails are forbidden by the frozen protocol.",
            "originates_from_another_dataset": False,
            "pilot_result": "rejected",
            "rejection_reason": "bulk_original_bytes_not_dependable_and_below_target_ceiling",
        },
        {
            "source": "Internet Archive",
            "source_url": "https://archive.org/advancedsearch.php",
            "access_method": "Public advanced search API",
            "original_bytes_downloadable": False,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": "UNKNOWN",
            "license": "UNKNOWN",
            "provenance": "52 image hits for wristwatch. The sample records were an Instagram dump and a matchbox label, not brand-verified watch photographs.",
            "stable_ids": True,
            "duplicate_detection_possible": False,
            "derivatives_identifiable": False,
            "expected_independent_photographs": 52,
            "automation_permitted": "Search API is public.",
            "rate_limits": "UNKNOWN",
            "originates_from_another_dataset": True,
            "pilot_result": "rejected",
            "rejection_reason": "not_a_brand_verified_watch_photo_corpus",
        },
        {
            "source": "Metropolitan Museum of Art",
            "source_url": "https://collectionapi.metmuseum.org/public/collection/v1/search",
            "access_method": "Public collection API",
            "original_bytes_downloadable": False,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": "UNKNOWN",
            "license": "UNKNOWN",
            "provenance": "The search endpoint returned HTTP 410.",
            "stable_ids": "UNKNOWN",
            "duplicate_detection_possible": False,
            "derivatives_identifiable": False,
            "expected_independent_photographs": "UNKNOWN",
            "automation_permitted": "UNKNOWN",
            "rate_limits": "Endpoint gone",
            "originates_from_another_dataset": False,
            "pilot_result": "rejected",
            "rejection_reason": "source_unavailable",
        },
        {
            "source": "Smithsonian Open Access",
            "source_url": "https://api.si.edu/openaccess/api/v1.0/search",
            "access_method": "Open Access API",
            "original_bytes_downloadable": False,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": "UNKNOWN",
            "license": "UNKNOWN",
            "provenance": "The search endpoint returned HTTP 403.",
            "stable_ids": "UNKNOWN",
            "duplicate_detection_possible": False,
            "derivatives_identifiable": False,
            "expected_independent_photographs": "UNKNOWN",
            "automation_permitted": "UNKNOWN",
            "rate_limits": "HTTP 403",
            "originates_from_another_dataset": False,
            "pilot_result": "rejected",
            "rejection_reason": "source_unavailable",
        },
        {
            "source": "Europeana",
            "source_url": "https://api.europeana.eu/record/v2/search.json",
            "access_method": "Search API",
            "original_bytes_downloadable": False,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": "UNKNOWN",
            "license": "UNKNOWN",
            "provenance": "The search endpoint returned HTTP 401.",
            "stable_ids": "UNKNOWN",
            "duplicate_detection_possible": False,
            "derivatives_identifiable": False,
            "expected_independent_photographs": "UNKNOWN",
            "automation_permitted": "UNKNOWN",
            "rate_limits": "API key required",
            "originates_from_another_dataset": False,
            "pilot_result": "rejected",
            "rejection_reason": "source_unavailable",
        },
        {
            "source": "Hippopx",
            "source_url": "https://www.hippopx.com/en/free-photo-ajzsb",
            "access_method": "Public photo page",
            "original_bytes_downloadable": False,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": "UNKNOWN",
            "license": "UNKNOWN",
            "provenance": "The page returned HTTP 403, so brand labels on third-party search snippets were not treated as source metadata.",
            "stable_ids": "UNKNOWN",
            "duplicate_detection_possible": False,
            "derivatives_identifiable": False,
            "expected_independent_photographs": "UNKNOWN",
            "automation_permitted": "UNKNOWN",
            "rate_limits": "HTTP 403",
            "originates_from_another_dataset": True,
            "pilot_result": "rejected",
            "rejection_reason": "source_unavailable",
        },
        {
            "source": "WatchBase CDN via Timepiecepedia image URLs",
            "source_url": "https://cdn.watchbase.com/watch/lg/",
            "access_method": "Published dataset example points at a /watch/lg/ CDN path",
            "original_bytes_downloadable": False,
            "typical_dimensions": "UNKNOWN",
            "typical_file_size": "UNKNOWN",
            "license": "The dataset text is described as CC0. That dedication was not established for the photographs.",
            "provenance": "The example image URL is a sized catalog rendition. No photograph was downloaded.",
            "stable_ids": "UNKNOWN",
            "duplicate_detection_possible": False,
            "derivatives_identifiable": True,
            "expected_independent_photographs": "UNKNOWN",
            "automation_permitted": "Not used. The photograph license is not established.",
            "rate_limits": "UNKNOWN",
            "originates_from_another_dataset": True,
            "pilot_result": "rejected",
            "rejection_reason": "photograph_license_not_established",
        },
    ]


def render_report(status: dict) -> str:
    groups = status["coverage"]["groups"]
    order = (
        ("Unseen luxury", "unseen_luxury"),
        ("Ordinary", "ordinary_watch"),
        ("Smart/digital", "smart_digital"),
        ("Wrist/store", "wrist_store"),
    )
    coverage_lines = [
        "| Group | Target | Candidate | Admitted | Holdout target | Ready |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, key in order:
        item = groups[key]
        coverage_lines.append(
            f"| {label} | {item['target']} | {item['candidate']} | {item['admitted']} | {item['holdout_target']} | {'yes' if item['ready'] else 'no'} |"
        )
    source_lines = [
        "| Source | Category coverage | Brands | Candidate | Admitted | Rejected | Rejection reason | Provenance |",
        "|---|---|---|---:|---:|---:|---|---|",
        "| Pixnio | none of the four protocol groups | UNKNOWN | 1 | 0 | 1 | ambiguous_category | CC0 page, no source-declared brand |",
        "| Openverse | not collected | not collected | 0 | 0 | 0 | resized Flickr rendition | aggregator URL is not the original |",
        "| Flickr | not collected | uploader title only | 0 | 0 | 0 | original above the 1.5 MB cap | one CC BY-SA original was larger than the frozen cap |",
        "| Wikimedia Commons | ceiling only, not collected | category names | 0 | 0 | 0 | bulk originals are not dependable and the ceiling misses the targets | Phase 36 HTTP 429 record |",
        "| Internet Archive | not a watch-photo corpus | UNKNOWN | 0 | 0 | 0 | search hits are not brand-verified photographs | license not established for a watch corpus |",
        "| Met, Smithsonian, Europeana, Hippopx | unavailable | UNKNOWN | 0 | 0 | 0 | endpoint denied or gone | bytes and license were not established |",
        "| WatchBase CDN | not collected | not used | 0 | 0 | 0 | photograph license not established | CC0 claim covers the dataset text, not the shown CDN rendition |",
    ]
    ceiling = status["commons_rate_limit_ceiling"]
    counts = status["audit_counts"]
    lines = [
        "# Phase 37 acquisition report",
        "",
        "Phase 37 collected and audited sources. It did not score the frozen authenticity model and it did not change the Phase 36 protocol.",
        "",
        f"Protocol hash: `{status['protocol_sha256']}`",
        "",
        "## Source table",
        "",
        *source_lines,
        "",
        "## Independence table",
        "",
        "| Check | Count |",
        "|---|---:|",
        f"| Exact duplicates rejected | {counts['exact_duplicates_rejected']} |",
        f"| Near duplicates rejected | {counts['near_duplicates_rejected']} |",
        f"| Derivative files rejected | {counts['derivatives_rejected']} |",
        f"| Overlapping source IDs or URLs rejected | {counts['exact_duplicates_rejected']} |",
        "| HypeVault catalog overlap in the admitted set | 0 |",
        "",
        "The one stored pilot did not match a Phase 16, Phase 35, Phase 36, or Phase 25 SHA-256, source URL, filename, or dHash within Hamming 4. It was still rejected because the source does not verify a brand or a protocol category.",
        "",
        "Openverse and the WatchBase `/watch/lg/` example were refused before download because they are resized or license-unproven derivatives. Those refusals are source-level. They are not counted as downloaded derivative files.",
        "",
        "## Coverage table",
        "",
        *coverage_lines,
        "",
        "Unseen luxury brands with at least 8 holdout images: 0.",
        "",
        "Images per brand: none admitted.",
        "",
        "Independent source groups per brand: none admitted.",
        "",
        "## Commons ceiling that was not collected",
        "",
        "Phase 36 recorded 531 original downloads that failed with HTTP 429 after the size and title filters. Those files were not added to Phase 37. If every one of them had later passed license and content checks, the new-file ceiling would still be:",
        "",
        f"- unseen luxury: {ceiling['unseen_luxury']} (target 300)",
        f"- ordinary watches: {ceiling['ordinary_watch']} (target 200)",
        f"- smart/digital: {ceiling['smart_digital']} (target 100)",
        f"- wrist/store: {ceiling['wrist_store']} (target 50, useful holdout 25)",
        "",
        "The Phase 36 admitted files stay in the Phase 36 holdout and are not members of this corpus.",
        "",
        "## Stop",
        "",
        "Acquisition stopped because no qualified source can supply original, provenance-backed photographs at the registered targets. Wikimedia original delivery is rate-limited, and its recorded ceiling misses the unseen, ordinary, and wrist targets. The other pilots failed original-byte, license, or source-verified brand checks. The protocol was not relaxed.",
        "",
        f"`WATCH_OOD_V2_CORPUS_READY = {str(status['WATCH_OOD_V2_CORPUS_READY']).lower()}`",
        "",
    ]
    return "\n".join(lines)


def write_phase37(repo: Path) -> dict:
    protocol = load_protocol()
    if protocol_sha256() != PROTOCOL_SHA256:
        raise AcquisitionError("refusing to write because the Phase 36 protocol changed")
    pilot = repo / "ml_rtx5080" / "experiments" / "watch_like_ood_v2" / "pilot" / "pixnio_2017-06-22-20-53-09.jpg"
    _refuse_final_test(pilot)
    index = load_prior_index(repo)
    image_bytes = pilot.read_bytes()
    with Image.open(pilot) as image:
        width, height = image.size
        image_format = image.format or "UNKNOWN"
        perceptual = dhash_image(image)
    stamp = pilot.stat().st_mtime
    candidate = {
        "image_path": str(pilot.resolve()),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "source": "Pixnio",
        "source_url": "https://pixnio.com/free-images/2017/06/22/2017-06-22-20-53-09.jpg",
        "source_id": "pixnio:2017-06-22-20-53-09",
        "brand": "UNKNOWN",
        "category": "UNKNOWN",
        "subcategory": "UNKNOWN",
        "license": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "collection_timestamp": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        "image_width": width,
        "image_height": height,
        "file_size": len(image_bytes),
        "format": image_format,
        "is_original_or_derivative": "original_photo",
        "independent_source_group": "pixnio:2017-06-22-20-53-09",
        "phase": 37,
        "admission_status": "pending",
        "rejection_reason": "",
        "brand_verification": "unknown",
        "source_declared_brand": "UNKNOWN",
        "source_declared_category": "wristwatch",
    }
    audited = audit_rows([candidate], index, {candidate["sha256"]: perceptual})
    manifest = {
        "phase": 37,
        "methodology": "acquisition_and_provenance_only",
        "model_scored": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "sample_count": len(audited["rows"]),
        "samples": audited["rows"],
    }
    out = repo / "ml_rtx5080" / "experiments" / "watch_like_ood_v2"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    file_hash = file_sha256(manifest_path)
    (out / "acquisition_manifest.sha256").write_text(file_hash + "\n")
    evaluation_lock = out / "watch_like_ood_v2_manifest.sha256"
    if evaluation_lock.exists():
        evaluation_lock.unlink()
    summary = coverage(audited["rows"], protocol)
    status = {
        "WATCH_OOD_V2_CORPUS_READY": summary["corpus_ready"],
        "PROTOCOL_HASH_UNCHANGED": protocol_sha256() == PROTOCOL_SHA256,
        "PHASE36_HOLDOUT_MODIFIED": False,
        "FINAL_TEST_TOUCHED": False,
        "MODEL_TOUCHED": False,
        "THRESHOLD_TOUCHED": False,
        "CALIBRATION_TOUCHED": False,
        "protocol_sha256": protocol_sha256(),
        "acquisition_manifest_sha256": file_hash,
        "evaluation_corpus_locked": False,
        "audit_counts": audited["counts"],
        "coverage": summary,
        "commons_rate_limit_ceiling": commons_rate_limit_ceiling(repo),
        "phase36_images_added": 0,
    }
    (out / "source_qualification.json").write_text(
        json.dumps({"sources": qualification_sources()}, indent=2, sort_keys=True) + "\n"
    )
    (out / "data_audit.json").write_text(json.dumps(audited["counts"], indent=2, sort_keys=True) + "\n")
    (out / "acquisition_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    (out / "acquisition_report.md").write_text(render_report(status))
    return status
