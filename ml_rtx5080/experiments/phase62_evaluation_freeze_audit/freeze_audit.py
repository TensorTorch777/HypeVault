"""Synthetic freeze-audit helper.

This module accepts records that are already in memory. It does not open
images, walk directories, or read a protected manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1] / "phase59_manifest_validator"
sys.path.insert(0, str(VALIDATOR_ROOT))

from manifest_validator import validate_manifest

FORBIDDEN_KEYS = frozenset({"image_bytes", "image_path", "file_path"})


def _code(finding: dict) -> str:
    field = finding.get("field")
    reason = finding.get("reason") or ""
    if field == "product_id" and "partition" in reason:
        return "PRODUCT_PARTITION_LEAKAGE"
    if field == "image_sha256" and "partition" in reason:
        return "IMAGE_PARTITION_LEAKAGE"
    if field == "duplicate_group_id":
        return "DUPLICATE_GROUP_PARTITION_LEAKAGE"
    if "overlaps" in reason:
        return "MODEL_MEMBERSHIP_OVERLAP"
    if "not checked" in reason or "not supplied" in reason or "unresolved" in reason:
        return "MODEL_MEMBERSHIP_UNCHECKED"
    if field in {"seller_id", "marketplace_id"}:
        return "SELLER_OR_MARKETPLACE_REVIEW"
    if field == "verified_brand":
        return "BRAND_MISSING_CLASS"
    if field in {
        "license_status",
        "consent_status",
        "permitted_purposes",
        "retention_until",
        "deletion_status",
        "evidence_category_a",
        "evidence_category_b",
    }:
        return "RIGHTS_OR_RETENTION"
    return "PRIMARY_ELIGIBILITY_FAILED"


def audit_synthetic_manifest(records: list[dict], membership: dict | None) -> dict:
    """Summarize an in-memory fixture. The result is a demo, not a frozen dataset."""
    if isinstance(records, (str, bytes, Path)):
        raise TypeError("dataset paths are not accepted")
    if not isinstance(records, list):
        raise TypeError("records must be an in-memory list")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each record must be a dict")
        if FORBIDDEN_KEYS.intersection(record):
            raise ValueError("image content and file paths are out of scope")
    result = validate_manifest(records, membership)
    counts = Counter(item["eligibility"] for item in result["records"])
    body = {
        "report_class": "SYNTHETIC_DEMO",
        "dataset_status": "NOT_FROZEN",
        "real_cohort": "NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION",
        "overall": result["overall"],
        "record_count": len(records),
        "eligibility_counts": {
            key: counts.get(key, 0) for key in ("PASS", "FAIL", "UNKNOWN", "NOT_ELIGIBLE")
        },
        "finding_count": len(result["findings"]),
        "exclusion_reason_codes": sorted({_code(item) for item in result["findings"]}),
        "data_readiness": "NO_GO_FOR_AUTHENTICITY_CLAIMS",
        "acquisition_authorized": False,
        "inference_authorized": False,
        "minimum_counts_approved": False,
        "acceptance_criteria_approved": False,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["aggregate_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body
