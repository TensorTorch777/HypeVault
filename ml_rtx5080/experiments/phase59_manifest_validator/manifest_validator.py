"""In-memory checks for a future evaluation manifest.

This module does not read images, walk directories, or open locked-test files.
Unknown membership is never reported as a pass.
"""

from __future__ import annotations

from datetime import date

PARTITIONS = frozenset({"training", "calibration", "development", "locked_evaluation"})
VERIFIED_LABELS = frozenset({"AUTHENTIC", "COUNTERFEIT", "INDETERMINATE"})
REVIEWER_OUTCOMES = frozenset({"AUTHENTIC", "COUNTERFEIT", "INDETERMINATE", "NOT_REVIEWED"})
EVIDENCE_CATEGORIES = frozenset({
    "MANUFACTURER_PROVENANCE",
    "AUTHORIZED_SERVICE_RECORD",
    "QUALIFIED_PHYSICAL_EXAMINATION",
    "DOCUMENTED_CHAIN_OF_PROVENANCE",
    "OTHER_DOCUMENTED",
    "NOT_RECORDED",
})
ADJUDICATION_STATUS = frozenset({"NOT_STARTED", "PENDING", "AGREED", "ADJUDICATED", "UNRESOLVED"})
TRI_STATE = frozenset({"PASS", "FAIL", "UNKNOWN"})
MODELS = ("dinov2_legacy", "dinov3_experimental")
MEMBERSHIP_SPLITS = ("training", "calibration")


class MembershipView:
    """Explicit task-specific membership. checked=False means the comparison was not run."""

    def __init__(
        self,
        *,
        checked: bool,
        product_ids: set[str] | None = None,
        image_sha256: set[str] | None = None,
        source_ids: set[str] | None = None,
        seller_ids: set[str] | None = None,
        marketplace_ids: set[str] | None = None,
    ) -> None:
        self.checked = checked
        self.product_ids = set(product_ids or ())
        self.image_sha256 = set(image_sha256 or ())
        self.source_ids = set(source_ids or ())
        self.seller_ids = set(seller_ids or ())
        self.marketplace_ids = set(marketplace_ids or ())


def _finding(status: str, severity: str, field: str, reason: str, sample_id: str | None) -> dict:
    if status not in {"FAIL", "UNKNOWN"}:
        raise ValueError("findings are FAIL or UNKNOWN, never PASS")
    return {
        "status": status,
        "severity": severity,
        "field": field,
        "reason": reason,
        "sample_id": sample_id,
    }


def _known(value: object) -> bool:
    return isinstance(value, str) and value not in {"", "UNKNOWN"}


def _real_date(value: object) -> bool:
    if not isinstance(value, str) or value == "UNKNOWN":
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _adjudication_ok(record: dict) -> bool:
    status = record.get("adjudication_status")
    outcome = record.get("adjudication_outcome")
    label = record.get("verified_label")
    left = record.get("reviewer_a_outcome")
    right = record.get("reviewer_b_outcome")
    if status == "AGREED":
        return outcome == label and left == outcome and right == outcome and outcome in {"AUTHENTIC", "COUNTERFEIT"}
    if status == "ADJUDICATED":
        opposite = {left, right} == {"AUTHENTIC", "COUNTERFEIT"}
        return opposite and outcome == label and outcome in {"AUTHENTIC", "COUNTERFEIT"}
    return False


def _eligibility_findings(record: dict) -> list[dict]:
    sample_id = record.get("sample_id") if isinstance(record.get("sample_id"), str) else None
    findings = []
    if record.get("verified_label") not in VERIFIED_LABELS:
        findings.append(_finding("FAIL", "error", "verified_label", "verified_label is not an allowed value", sample_id))
    if record.get("reviewer_a_outcome") not in REVIEWER_OUTCOMES:
        findings.append(_finding("FAIL", "error", "reviewer_a_outcome", "reviewer outcome is not an allowed value", sample_id))
    if record.get("reviewer_b_outcome") not in REVIEWER_OUTCOMES:
        findings.append(_finding("FAIL", "error", "reviewer_b_outcome", "reviewer outcome is not an allowed value", sample_id))
    if record.get("evidence_category_a") not in EVIDENCE_CATEGORIES:
        findings.append(_finding("FAIL", "error", "evidence_category_a", "evidence category is not an allowed value", sample_id))
    if record.get("evidence_category_b") not in EVIDENCE_CATEGORIES:
        findings.append(_finding("FAIL", "error", "evidence_category_b", "evidence category is not an allowed value", sample_id))
    if record.get("partition") not in PARTITIONS:
        findings.append(_finding("FAIL", "error", "partition", "partition is not an allowed value", sample_id))
    if not record.get("eligible_primary_benchmark"):
        return findings
    checks = [
        (not _known(record.get("product_id")), "product_id", "primary-eligible record has no product identity"),
        (not _known(record.get("source_id")), "source_id", "primary-eligible record has no source identity"),
        (not _known(record.get("verified_brand")), "verified_brand", "primary-eligible record has no verified brand"),
        (record.get("verified_label") not in {"AUTHENTIC", "COUNTERFEIT"}, "verified_label", "primary-eligible record has no resolved label"),
        (not _known(record.get("evidence_record_id")), "evidence_record_id", "primary-eligible record has no evidence reference"),
        (record.get("evidence_category_a") == "NOT_RECORDED", "evidence_category_a", "primary-eligible record has no evidence category"),
        (record.get("evidence_category_b") == "NOT_RECORDED", "evidence_category_b", "primary-eligible record has no evidence category"),
        (record.get("license_status") != "APPROVED", "license_status", "primary-eligible record lacks approved image-use rights"),
        ("ML_EVALUATION" not in set(record.get("permitted_purposes") or []), "permitted_purposes", "primary-eligible record lacks ML_EVALUATION permission"),
        (record.get("consent_status") not in {"APPROVED", "NOT_REQUIRED"}, "consent_status", "primary-eligible record lacks a consent or legal basis"),
        (not _real_date(record.get("retention_until")), "retention_until", "primary-eligible record lacks a real retention date"),
        (record.get("deletion_status") != "RETENTION_ACTIVE", "deletion_status", "primary-eligible record is not in active retention"),
        (not _adjudication_ok(record), "adjudication_status", "primary-eligible adjudication does not match the resolved label"),
    ]
    for failed, field, reason in checks:
        if failed:
            findings.append(_finding("FAIL", "error", field, reason, sample_id))
    for model in MODELS:
        declared = (record.get("declared_disjointness") or {}).get(model)
        if declared not in TRI_STATE:
            findings.append(_finding("UNKNOWN", "review", "declared_disjointness", f"{model} membership status is unresolved", sample_id))
        elif declared == "UNKNOWN":
            findings.append(_finding("UNKNOWN", "review", "declared_disjointness", f"{model} membership status is unresolved", sample_id))
        elif declared == "FAIL":
            findings.append(_finding("FAIL", "error", "declared_disjointness", f"{model} declared disjointness failed", sample_id))
    return findings


def _cross_partition(records: list[dict], field: str, reason: str) -> list[dict]:
    grouped: dict[str, set[str]] = {}
    for record in records:
        value = record.get(field)
        if not _known(value):
            continue
        grouped.setdefault(value, set()).add(record.get("partition"))
    findings = []
    for value, partitions in grouped.items():
        del value
        if len(partitions) > 1:
            findings.append(_finding("FAIL", "error", field, reason, None))
    return findings


def _overlap(records: list[dict], membership: dict) -> list[dict]:
    findings = []
    for record in records:
        if not record.get("eligible_primary_benchmark"):
            continue
        sample_id = record.get("sample_id") if isinstance(record.get("sample_id"), str) else None
        for model in MODELS:
            model_sets = membership.get(model)
            if not isinstance(model_sets, dict):
                findings.append(_finding("UNKNOWN", "review", "declared_disjointness", f"{model} membership was not supplied", sample_id))
                continue
            for split in MEMBERSHIP_SPLITS:
                view = model_sets.get(split)
                if view is None or not view.checked:
                    findings.append(_finding("UNKNOWN", "review", split, f"{model} {split} membership was not checked", sample_id))
                    continue
                if _known(record.get("product_id")) and record["product_id"] in view.product_ids:
                    findings.append(_finding("FAIL", "error", "product_id", f"product overlaps {model} {split} membership", sample_id))
                if _known(record.get("image_sha256")) and record["image_sha256"] in view.image_sha256:
                    findings.append(_finding("FAIL", "error", "image_sha256", f"image hash overlaps {model} {split} membership", sample_id))
                if _known(record.get("source_id")) and record["source_id"] in view.source_ids:
                    findings.append(_finding("FAIL", "error", "source_id", f"source overlaps {model} {split} membership", sample_id))
                if _known(record.get("seller_id")) and record["seller_id"] in view.seller_ids:
                    findings.append(_finding("UNKNOWN", "review", "seller_id", f"seller overlap with {model} {split} requires review", sample_id))
                if _known(record.get("marketplace_id")) and record["marketplace_id"] in view.marketplace_ids:
                    findings.append(_finding("UNKNOWN", "review", "marketplace_id", f"marketplace overlap with {model} {split} requires review", sample_id))
    return findings


def _brand_gaps(records: list[dict], eligible_ids: set[str]) -> list[dict]:
    classes: dict[str, set[str]] = {}
    for record in records:
        if record.get("sample_id") not in eligible_ids:
            continue
        brand = record.get("verified_brand")
        label = record.get("verified_label")
        if not _known(brand) or label not in {"AUTHENTIC", "COUNTERFEIT"}:
            continue
        classes.setdefault(brand, set()).add(label)
    findings = []
    for brand, labels in sorted(classes.items()):
        del brand
        if labels != {"AUTHENTIC", "COUNTERFEIT"}:
            missing = "COUNTERFEIT" if "COUNTERFEIT" not in labels else "AUTHENTIC"
            findings.append(_finding("FAIL", "error", "verified_brand", f"brand has no qualifying {missing} examples", None))
    return findings


def _record_status(findings: list[dict], claims_eligibility: bool) -> str:
    statuses = {item["status"] for item in findings}
    if "FAIL" in statuses:
        return "FAIL"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    if claims_eligibility:
        return "PASS"
    return "NOT_ELIGIBLE"


def validate_manifest(records: list[dict], membership: dict | None) -> dict:
    """Validate records already in memory. membership=None means no model check was supplied."""
    if membership is None:
        membership = {}
    per_record = []
    findings: list[dict] = []
    for record in records:
        local = _eligibility_findings(record)
        findings.extend(local)
        per_record.append({
            "sample_id": record.get("sample_id") if isinstance(record.get("sample_id"), str) else None,
            "eligibility": _record_status(local, bool(record.get("eligible_primary_benchmark"))),
        })
    findings.extend(_cross_partition(records, "product_id", "one product_id is assigned to more than one partition"))
    findings.extend(_cross_partition(records, "image_sha256", "one image hash is assigned to more than one partition"))
    findings.extend(_cross_partition(records, "duplicate_group_id", "one duplicate group is assigned to more than one partition"))
    overlap = _overlap(records, membership)
    findings.extend(overlap)
    for item in per_record:
        related = [found for found in overlap if found["sample_id"] == item["sample_id"]]
        if any(found["status"] == "FAIL" for found in related):
            item["eligibility"] = "FAIL"
        elif any(found["status"] == "UNKNOWN" for found in related) and item["eligibility"] == "PASS":
            item["eligibility"] = "UNKNOWN"
    passed = {item["sample_id"] for item in per_record if item["eligibility"] == "PASS"}
    gaps = _brand_gaps(records, passed)
    findings.extend(gaps)
    if any(item["status"] == "FAIL" for item in findings) or any(item["eligibility"] == "FAIL" for item in per_record):
        overall = "FAIL"
    elif any(item["status"] == "UNKNOWN" for item in findings) or any(item["eligibility"] == "UNKNOWN" for item in per_record):
        overall = "UNKNOWN"
    else:
        overall = "PASS"
    return {
        "overall": overall,
        "records": per_record,
        "findings": findings,
        "data_readiness": "NO_GO_FOR_AUTHENTICITY_CLAIMS",
        "acquisition_authorized": False,
        "inference_authorized": False,
    }
