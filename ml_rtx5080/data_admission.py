"""Admission check for candidate_image_record_v2.

A record is admitted only when every field the label depends on has real evidence.
UNKNOWN, missing, or forbidden evidence keeps it out. This module does not read
images and does not decide labels.
"""

from __future__ import annotations

from datetime import date

SCHEMA_ID = "candidate_image_record_v2"
ADMISSIBLE_LABELS = frozenset({"authentic", "counterfeit"})
ADMISSIBLE_SOURCES = frozenset({"QUALIFIED_EXAMINER", "OWNER_PERMISSION_PHOTOGRAPHY", "INSTITUTION"})
FORBIDDEN_METHODS = frozenset(
    {
        "directory_name",
        "filename",
        "seller_claim",
        "listing_label",
        "wordmark_only",
        "model_prediction",
        "image_geometry",
        "jpeg_quantization",
        "visual_suspicion_only",
        "photo_only",
    }
)
PURPOSES = ("evaluation", "training")


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and (not value.strip() or value.strip().upper() == "UNKNOWN"))


def admission_decision(record: dict, *, purpose: str = "evaluation", today: date | None = None) -> tuple[str, list[str]]:
    """Return ("ADMIT", []) or ("REJECT", reasons). A REJECT never changes the stored label."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    today = today or date.today()
    reasons: list[str] = []
    image = record.get("image") or {}
    source = record.get("source") or {}
    rights = record.get("rights") or {}
    brand = record.get("brand") or {}
    label = record.get("label") or {}
    disposition = record.get("disposition") or {}

    sha = str(image.get("sha256") or "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        reasons.append("IMAGE_SHA256_MISSING")
    if image.get("duplicate_status") != "UNIQUE":
        reasons.append("DUPLICATE_STATUS_NOT_UNIQUE")
    if image.get("exif_location_stripped") is not True:
        reasons.append("LOCATION_METADATA_NOT_STRIPPED")

    if source.get("source_type") not in ADMISSIBLE_SOURCES:
        reasons.append("SOURCE_NOT_INDEPENDENT")
    for field in ("source_reference", "product_or_listing_id", "source_group_id"):
        if _blank(source.get(field)):
            reasons.append(f"SOURCE_{field.upper()}_MISSING")
    if not source.get("chain_of_custody"):
        reasons.append("CHAIN_OF_CUSTODY_MISSING")

    if _blank(rights.get("permission_id")) or rights.get("permission_status") != "EFFECTIVE":
        reasons.append("PERMISSION_NOT_EFFECTIVE")
    if rights.get("research_use") != "GRANTED":
        reasons.append("RESEARCH_USE_NOT_GRANTED")
    if purpose == "training" and rights.get("training_use") != "GRANTED":
        reasons.append("TRAINING_USE_NOT_GRANTED")
    retention = rights.get("retention_until")
    if retention:
        try:
            if date.fromisoformat(retention) < today:
                reasons.append("RETENTION_EXPIRED")
        except ValueError:
            reasons.append("RETENTION_DATE_INVALID")

    if _blank(brand.get("value")) or _blank(brand.get("basis")) or _blank(brand.get("evidence_id")):
        reasons.append("BRAND_EVIDENCE_MISSING")
    elif str(brand.get("basis")).strip().lower() in FORBIDDEN_METHODS:
        reasons.append("BRAND_BASIS_NOT_ADMISSIBLE")

    if label.get("value") not in ADMISSIBLE_LABELS:
        reasons.append("LABEL_NOT_VERIFIED")
    methods = [str(m).strip().lower() for m in (label.get("method") or [])]
    if not methods:
        reasons.append("VERIFICATION_METHOD_MISSING")
    if any(m in FORBIDDEN_METHODS for m in methods):
        reasons.append("VERIFICATION_METHOD_NOT_ADMISSIBLE")
    for field in ("verifier_ref", "verifier_qualification_evidence_id", "evidence_id", "evidence_date"):
        if _blank(label.get(field)):
            reasons.append(f"LABEL_{field.upper()}_MISSING")
    if label.get("verifier_independent") is not True:
        reasons.append("VERIFIER_NOT_INDEPENDENT")
    if label.get("confidence") != "certain":
        reasons.append("LABEL_CONFIDENCE_NOT_CERTAIN")

    if disposition.get("status") in ("HOLD_DISPUTED", "EXCLUDE"):
        reasons.append(f"DISPOSITION_{disposition.get('status')}")

    return ("ADMIT", []) if not reasons else ("REJECT", reasons)
