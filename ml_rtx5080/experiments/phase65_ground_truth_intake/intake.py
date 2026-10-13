"""Phase 65 offline ground-truth intake (metadata only).

Connects hypevault_scraper manifests to Phase 55 evidence-ledger eligibility and
ml_rtx5080.data_admission without opening images, mutating source manifests, or
changing production gates.

Importing this module never starts acquisition, generation, training, or inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ml_rtx5080") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ml_rtx5080"))

from hypevault_scraper.common import (  # noqa: E402
    AUTO_ASSIGNABLE_STATUSES,
    SCHEMA_VERSION as SCRAPER_SCHEMA_VERSION,
    STATUS_SYNTHETIC,
    STATUS_UNVERIFIED_WEB,
    STATUS_VERIFIED_AUTHENTIC,
    STATUS_VERIFIED_COUNTERFEIT,
    VERIFIED_STATUSES,
)
from data_admission import admission_decision  # noqa: E402

PHASE65_SCHEMA = "phase65_intake_report_v1"
REVIEW_LEDGER_SCHEMA = "phase65_review_ledger_v1"
QUEUE_SCHEMA = "phase65_review_queue_v1"

REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "record_id",
    "image_path",
    "stored_sha256",
    "original_sha256",
    "width",
    "height",
    "format",
    "byte_size",
    "family",
    "claimed_brand",
    "claimed_model",
    "search_provider",
    "source_image_url",
    "source_page_url",
    "provenance_completeness",
    "label_status",
    "target_label",
    "evidence_status",
    "rights_review_status",
    "intended_use_eligibility",
    "duplicate_status",
)

UNKNOWNISH = frozenset({"", "UNKNOWN", "NOT_RECORDED", "NOT_APPLICABLE"})

DATA_READINESS = "NO_GO_FOR_AUTHENTICITY_CLAIMS"
ACQUISITION_AUTHORIZED = False
INFERENCE_AUTHORIZED = False

PHASE55_SCHEMA_PATH = (
    REPO_ROOT
    / "ml_rtx5080"
    / "experiments"
    / "phase55_ground_truth_acquisition"
    / "evidence_ledger.schema.json"
)


class IntakeError(Exception):
    """Fail-closed intake error."""


class ImageAccessBlocked(IntakeError):
    """Raised if intake attempts to open an image or protected path."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_unknown(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip().upper() in UNKNOWNISH)


def assert_metadata_only_path(path: Path) -> None:
    """Refuse image binaries and locked evaluation paths."""
    lowered = {part.lower() for part in path.parts}
    banned = {
        "final_test",
        "final-test",
        "ood",
        "excluded_ood",
        "locked_evaluation",
        "label_0_watches",
        "label_1_watches",
        "label_0_sneakers",
        "label_1_sneakers",
    }
    if lowered & banned:
        raise ImageAccessBlocked(f"refusing protected or legacy training path: {path}")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        raise ImageAccessBlocked(f"intake is metadata-only; refusing image path open: {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    assert_metadata_only_path(path)
    if not path.is_file():
        raise IntakeError(f"manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntakeError(f"{path}:{line_no}: malformed JSONL ({exc})") from exc
        if not isinstance(payload, dict):
            raise IntakeError(f"{path}:{line_no}: record must be a JSON object")
        rows.append(payload)
    return rows


def validate_scraper_record(record: Mapping[str, Any], *, source_path: str, index: int) -> list[str]:
    """Return validation error codes. Empty list means structurally valid for intake."""
    errors: list[str] = []
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in record:
            errors.append(f"MISSING_FIELD:{field}")
    if record.get("schema_version") != SCRAPER_SCHEMA_VERSION:
        errors.append(f"UNSUPPORTED_SCHEMA:{record.get('schema_version')!r}")
    if not isinstance(record.get("record_id"), str) or not record.get("record_id"):
        errors.append("RECORD_ID_INVALID")
    if not _sha256_hex(record.get("stored_sha256")):
        errors.append("STORED_SHA256_INVALID")
    original = record.get("original_sha256")
    if original not in {"NOT_RECORDED"} and not _sha256_hex(original):
        errors.append("ORIGINAL_SHA256_INVALID")
    label_status = record.get("label_status")
    if label_status not in AUTO_ASSIGNABLE_STATUSES:
        if label_status in VERIFIED_STATUSES:
            errors.append("VERIFIED_LABEL_NOT_ALLOWED_IN_SCRAPER_MANIFEST")
        else:
            errors.append(f"LABEL_STATUS_INVALID:{label_status!r}")
    if record.get("target_label") is not None:
        errors.append("TARGET_LABEL_MUST_BE_NULL")
    eligibility = record.get("intended_use_eligibility") or {}
    if not isinstance(eligibility, dict):
        errors.append("INTENDED_USE_ELIGIBILITY_INVALID")
    else:
        if eligibility.get("authenticity_benchmark") is not False:
            errors.append("AUTHENTICITY_BENCHMARK_MUST_BE_FALSE")
        if eligibility.get("model_training_as_verified_class") is not False:
            errors.append("TRAINING_AS_VERIFIED_CLASS_MUST_BE_FALSE")
    if record.get("provenance_completeness") not in {"COMPLETE", "PARTIAL", "INCOMPLETE"}:
        # scraper emits PARTIAL/INCOMPLETE; COMPLETE reserved if ever filled later
        if record.get("provenance_completeness") not in {"PARTIAL", "INCOMPLETE"}:
            errors.append(f"PROVENANCE_COMPLETENESS_INVALID:{record.get('provenance_completeness')!r}")
    for dim in ("width", "height", "byte_size"):
        value = record.get(dim)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{dim.upper()}_INVALID")
    image_path = record.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        errors.append("IMAGE_PATH_INVALID")
    # Never open the referenced image; only check the string is not a protected absolute path claim.
    if isinstance(image_path, str):
        try:
            assert_metadata_only_path(Path(image_path))
        except ImageAccessBlocked:
            # Relative artifact paths ending in .jpg are expected references; do not open them.
            # Block only when the path claims a protected cohort directory.
            parts = {part.lower() for part in Path(image_path).parts}
            banned = {"final_test", "final-test", "excluded_ood", "locked_evaluation"}
            if parts & banned:
                errors.append("PROTECTED_PATH_REFERENCE")
    _ = source_path, index
    return errors


def build_queue_row(
    record: Mapping[str, Any],
    *,
    manifest_path: str,
    validation_errors: list[str],
    ledger_entries: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ledger_entries = ledger_entries or []
    latest = ledger_entries[-1] if ledger_entries else None
    product_id = "UNKNOWN"
    evidence_refs: list[str] = []
    reviewer_status = "NOT_STARTED"
    adjudication_status = "NOT_STARTED"
    rights_status = record.get("rights_review_status") or "UNREVIEWED"
    evidence_status = record.get("evidence_status") or "UNREVIEWED"
    eligibility = "INELIGIBLE"
    eligibility_reason = "awaiting_evidence_backed_human_review"
    if validation_errors:
        eligibility = "INVALID"
        eligibility_reason = ",".join(validation_errors)
    elif latest:
        product_id = latest.get("product_id") or "UNKNOWN"
        evidence_refs = list(latest.get("evidence_references") or [])
        reviewer_status = latest.get("reviewer_status") or reviewer_status
        adjudication_status = latest.get("adjudication_status") or adjudication_status
        rights_status = latest.get("rights_review_status") or rights_status
        evidence_status = latest.get("evidence_status") or evidence_status
        eligibility = latest.get("eligibility_decision") or "INELIGIBLE"
        eligibility_reason = latest.get("eligibility_reason") or eligibility_reason

    provenance = record.get("provenance_completeness") or "INCOMPLETE"
    missing_ids = []
    for field in ("claimed_brand", "claimed_model", "source_image_url", "source_page_url"):
        if _is_unknown(record.get(field)):
            missing_ids.append(field)

    return {
        "schema_version": QUEUE_SCHEMA,
        "record_id": record.get("record_id"),
        "manifest_path": manifest_path,
        "image_path": record.get("image_path"),
        "stored_sha256": record.get("stored_sha256"),
        "original_sha256": record.get("original_sha256"),
        "claimed_brand": record.get("claimed_brand"),
        "claimed_model": record.get("claimed_model"),
        "product_id": product_id,
        "source_image_url": record.get("source_image_url"),
        "source_page_url": record.get("source_page_url"),
        "retrieved_at_utc": record.get("retrieved_at_utc") or "NOT_RECORDED",
        "search_provider": record.get("search_provider"),
        "provenance_completeness": provenance,
        "label_status": record.get("label_status"),
        "target_label": None,
        "evidence_status": evidence_status,
        "evidence_references": evidence_refs or ["NOT_RECORDED"],
        "rights_review_status": rights_status,
        "intended_use_eligibility": deepcopy(record.get("intended_use_eligibility") or {}),
        "reviewer_status": reviewer_status,
        "adjudication_status": adjudication_status,
        "duplicate_status": record.get("duplicate_status"),
        "near_duplicate_of": record.get("near_duplicate_of") or "NOT_APPLICABLE",
        "duplicate_group_id": record.get("stored_sha256"),
        "missing_or_unknown_fields": missing_ids,
        "validation_errors": validation_errors,
        "eligibility_decision": eligibility,
        "eligibility_reason": eligibility_reason,
        "ledger_entry_count": len(ledger_entries),
    }


def load_review_ledger(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    assert_metadata_only_path(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        if row.get("schema_version") != REVIEW_LEDGER_SCHEMA:
            raise IntakeError(f"unsupported review ledger schema: {row.get('schema_version')!r}")
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            raise IntakeError("review ledger row missing record_id")
        grouped.setdefault(record_id, []).append(row)
    return grouped


def append_review_decision(ledger_path: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    """Append one human decision. Never rewrites prior rows or source manifests."""
    assert_metadata_only_path(ledger_path)
    payload = dict(decision)
    payload["schema_version"] = REVIEW_LEDGER_SCHEMA
    payload.setdefault("decision_id", hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32])
    payload.setdefault("recorded_at_utc", utc_now_iso())
    if payload.get("authenticity_decision") in VERIFIED_STATUSES:
        # Ledger may propose verified outcomes only with Phase 55 evidence attached.
        if not payload.get("phase55_evidence_record"):
            raise IntakeError("verified authenticity decisions require phase55_evidence_record")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return payload


def evaluate_phase55_eligibility(evidence: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate a proposed Phase 55 evidence-ledger record with the authoritative schema."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - exercised in CI with jsonschema installed
        raise IntakeError("jsonschema is required to evaluate Phase 55 eligibility") from exc
    schema = json.loads(PHASE55_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [e.message for e in validator.iter_errors(dict(evidence))]
    if errors:
        return False, errors
    if evidence.get("eligible_primary_benchmark") is not True:
        return False, ["eligible_primary_benchmark must be true for verified admission"]
    if evidence.get("verified_label") not in {"AUTHENTIC", "COUNTERFEIT"}:
        return False, ["verified_label must be AUTHENTIC or COUNTERFEIT"]
    return True, []


def evaluate_data_admission(candidate: Mapping[str, Any], *, purpose: str = "evaluation") -> tuple[str, list[str]]:
    return admission_decision(dict(candidate), purpose=purpose, today=date(2026, 10, 14))


def assess_review_against_contracts(
    scraper_record: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed assessment: scraper status alone never verifies a class."""
    blockers: list[str] = []
    label_status = scraper_record.get("label_status")
    if label_status in AUTO_ASSIGNABLE_STATUSES:
        # Expected starting state.
        pass
    else:
        blockers.append(f"unexpected_scraper_label_status:{label_status}")

    authenticity = decision.get("authenticity_decision") or "UNREVIEWED"
    rights_ok = decision.get("rights_review_status") == "APPROVED"
    proposing_verified = authenticity in {
        STATUS_VERIFIED_AUTHENTIC,
        STATUS_VERIFIED_COUNTERFEIT,
        "AUTHENTIC",
        "COUNTERFEIT",
    }

    if label_status == STATUS_SYNTHETIC and proposing_verified:
        blockers.append("synthetic_not_physical_ground_truth")

    if proposing_verified:
        evidence = decision.get("phase55_evidence_record")
        if not isinstance(evidence, dict):
            blockers.append("missing_phase55_evidence_record")
        else:
            ok, errs = evaluate_phase55_eligibility(evidence)
            if not ok:
                blockers.extend(f"phase55:{e}" for e in errs)
            if evidence.get("image_sha256") != scraper_record.get("stored_sha256"):
                blockers.append("phase55_image_sha256_mismatch")
        if not rights_ok:
            blockers.append("rights_not_approved")
    elif authenticity in {"UNREVIEWED", "INSUFFICIENT_EVIDENCE", "REJECTED", "NOT_STARTED"}:
        blockers.append(f"authenticity_{str(authenticity).lower()}")
    else:
        blockers.append(f"authenticity_decision_unsupported:{authenticity!r}")

    candidate = decision.get("candidate_image_record_v2")
    admission_status = "NOT_EVALUATED"
    admission_reasons: list[str] = []
    if isinstance(candidate, dict):
        admission_status, admission_reasons = evaluate_data_admission(candidate)
        if admission_status != "ADMIT":
            blockers.append("admission_rejected")

    disputed = decision.get("adjudication_status") in {"UNRESOLVED", "PENDING", "NOT_STARTED"}
    if disputed and proposing_verified:
        blockers.append("adjudication_incomplete")

    # Authenticity may be evidence-backed while rights/admission still block ML use.
    authenticity_hard_blockers = {
        "missing_phase55_evidence_record",
        "phase55_image_sha256_mismatch",
        "synthetic_not_physical_ground_truth",
        "adjudication_incomplete",
    }
    authenticity_verified = proposing_verified and not (
        authenticity_hard_blockers.intersection(blockers)
        or any(b.startswith("phase55:") for b in blockers)
    )

    ml_eligible = authenticity_verified and rights_ok and (
        admission_status == "ADMIT" if isinstance(candidate, dict) else True
    )

    return {
        "authenticity_verified": authenticity_verified,
        "ml_use_eligible": ml_eligible,
        "blockers": blockers,
        "admission_status": admission_status,
        "admission_reasons": admission_reasons,
        "scraper_label_status_unchanged": label_status,
    }


def ingest_manifests(
    manifest_paths: Iterable[Path],
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    ledger = load_review_ledger(ledger_path)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, list[str]] = {}
    source_fingerprints: list[str] = []

    for path in manifest_paths:
        before = path.read_bytes()
        source_fingerprints.append(hashlib.sha256(before).hexdigest())
        rows = load_jsonl(path)
        for index, record in enumerate(rows):
            errors = validate_scraper_record(record, source_path=str(path), index=index)
            record_id = str(record.get("record_id") or f"missing-{path.name}-{index}")
            if record_id in seen_ids:
                errors.append(f"DUPLICATE_RECORD_ID:{seen_ids[record_id]}")
            else:
                seen_ids[record_id] = str(path)
            stored = record.get("stored_sha256")
            if isinstance(stored, str):
                seen_hashes.setdefault(stored, []).append(record_id)
            entries = ledger.get(record_id, [])
            row = build_queue_row(
                record,
                manifest_path=str(path),
                validation_errors=errors,
                ledger_entries=entries,
            )
            if not errors and entries:
                assessment = assess_review_against_contracts(record, entries[-1])
                if assessment["authenticity_verified"]:
                    row["eligibility_decision"] = "ADMITTED" if assessment["ml_use_eligible"] else "VERIFIED_LABEL_RIGHTS_BLOCKED"
                    label = (entries[-1].get("phase55_evidence_record") or {}).get("verified_label")
                    row["verified_label"] = label or "UNKNOWN"
                    row["eligibility_reason"] = (
                        f"phase55_verified:{label};ml_use_eligible={assessment['ml_use_eligible']}"
                    )
                else:
                    row["eligibility_decision"] = "BLOCKED"
                    row["eligibility_reason"] = ",".join(assessment["blockers"]) or "blocked"
                row["admission_status"] = assessment["admission_status"]
                row["assessment_blockers"] = assessment["blockers"]
            queue.append(row)
            bucket = invalid if errors else valid
            bucket.append({"record": record, "queue": row, "manifest_path": str(path)})

        after = path.read_bytes()
        if after != before:
            raise IntakeError(f"source manifest mutated during intake: {path}")

    duplicate_hashes = {sha: ids for sha, ids in seen_hashes.items() if len(ids) > 1}
    return {
        "schema_version": PHASE65_SCHEMA,
        "generated_at_utc": utc_now_iso(),
        "manifest_paths": [str(p) for p in manifest_paths],
        "manifest_sha256": source_fingerprints,
        "valid": valid,
        "invalid": invalid,
        "queue": queue,
        "duplicate_record_ids": [],  # encoded in invalid errors
        "duplicate_hashes": duplicate_hashes,
        "ledger_path": str(ledger_path) if ledger_path else None,
    }


def build_readiness_report(intake: Mapping[str, Any]) -> dict[str, Any]:
    queue = intake.get("queue") or []
    label_counts = Counter(row.get("label_status") for row in queue)
    provenance_counts = Counter(row.get("provenance_completeness") for row in queue)
    evidence_counts = Counter(row.get("evidence_status") for row in queue)
    rights_counts = Counter(row.get("rights_review_status") for row in queue)
    eligibility_counts = Counter(row.get("eligibility_decision") for row in queue)

    unknown_product = sum(1 for row in queue if _is_unknown(row.get("product_id")))
    incomplete_provenance = sum(
        1 for row in queue if row.get("provenance_completeness") in {"INCOMPLETE", "PARTIAL"}
    )
    near_dups = sum(1 for row in queue if row.get("duplicate_status") == "NEAR_DUPLICATE_CANDIDATE")
    exact_dup_groups = len(intake.get("duplicate_hashes") or {})

    verified_authentic = 0
    verified_counterfeit = 0
    admitted = 0
    for row in queue:
        decision = row.get("eligibility_decision")
        verified_label = row.get("verified_label")
        if decision in {"ADMITTED", "VERIFIED_LABEL_RIGHTS_BLOCKED"} and verified_label == "AUTHENTIC":
            verified_authentic += 1
        elif decision in {"ADMITTED", "VERIFIED_LABEL_RIGHTS_BLOCKED"} and verified_label == "COUNTERFEIT":
            verified_counterfeit += 1
        if decision == "ADMITTED":
            admitted += 1

    both_classes = verified_authentic > 0 and verified_counterfeit > 0
    blockers = [
        "independently_verified_labels_remain_zero" if verified_authentic + verified_counterfeit == 0 else None,
        "acquisition_authorized_false",
        "inference_authorized_false",
        "no_minimum_sample_plan_approved",
        "dataset_not_frozen",
    ]
    blockers = [b for b in blockers if b]

    return {
        "schema_version": "phase65_readiness_report_v1",
        "generated_at_utc": utc_now_iso(),
        "records_processed": len(queue),
        "valid_records": len(intake.get("valid") or []),
        "invalid_records": len(intake.get("invalid") or []),
        "label_status_counts": dict(label_counts),
        "provenance_counts": dict(provenance_counts),
        "evidence_status_counts": dict(evidence_counts),
        "rights_review_status_counts": dict(rights_counts),
        "eligibility_counts": dict(eligibility_counts),
        "unknown_product_identity_count": unknown_product,
        "incomplete_or_partial_provenance_count": incomplete_provenance,
        "exact_duplicate_hash_groups": exact_dup_groups,
        "near_duplicate_candidates": near_dups,
        "awaiting_evidence_review": evidence_counts.get("UNREVIEWED", 0),
        "awaiting_rights_review": rights_counts.get("UNREVIEWED", 0),
        "verified_authentic_count": verified_authentic,
        "verified_counterfeit_count": verified_counterfeit,
        "admitted_under_existing_contracts": admitted,
        "both_classes_have_admissible_support": both_classes,
        "data_readiness": DATA_READINESS,
        "acquisition_authorized": ACQUISITION_AUTHORIZED,
        "inference_authorized": INFERENCE_AUTHORIZED,
        "dataset_frozen": False,
        "benchmark_ready": False,
        "blockers": blockers,
        "recommended_next_actions": [
            "Complete Phase 60/61 authorized evidence collection for physical products.",
            "Append Phase 65 review-ledger decisions with Phase 55 evidence records.",
            "Keep scraper manifests unchanged; never map into Label_0_* / Label_1_*.",
            "Do not freeze or score until both classes have admissible evidence under an approved plan.",
        ],
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def write_queue_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "manifest_path",
        "image_path",
        "stored_sha256",
        "claimed_brand",
        "claimed_model",
        "product_id",
        "label_status",
        "provenance_completeness",
        "evidence_status",
        "rights_review_status",
        "eligibility_decision",
        "eligibility_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 65 offline ground-truth intake. Validates scraper manifests, "
            "builds a human review queue, and writes a readiness report. Metadata only."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a scraper manifest.jsonl (repeatable). Metadata only.",
    )
    parser.add_argument(
        "--review-ledger",
        default=None,
        help="Optional existing append-only review ledger JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for intake report, review queue, and readiness report.",
    )
    parser.add_argument(
        "--append-decision",
        default=None,
        help="Optional JSON file with one human review decision to append to the ledger.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifests = [Path(p) for p in args.manifest]
    ledger = Path(args.review_ledger) if args.review_ledger else None
    output_dir = Path(args.output_dir)

    if args.append_decision:
        if ledger is None:
            raise SystemExit("--append-decision requires --review-ledger")
        decision = json.loads(Path(args.append_decision).read_text(encoding="utf-8"))
        append_review_decision(ledger, decision)

    intake = ingest_manifests(manifests, ledger_path=ledger)
    report = build_readiness_report(intake)

    write_json(output_dir / "intake_report.json", {
        "schema_version": intake["schema_version"],
        "generated_at_utc": intake["generated_at_utc"],
        "manifest_paths": intake["manifest_paths"],
        "manifest_sha256": intake["manifest_sha256"],
        "valid_count": len(intake["valid"]),
        "invalid_count": len(intake["invalid"]),
        "duplicate_hashes": intake["duplicate_hashes"],
        "invalid": [
            {
                "record_id": item["queue"]["record_id"],
                "manifest_path": item["manifest_path"],
                "errors": item["queue"]["validation_errors"],
            }
            for item in intake["invalid"]
        ],
        "data_readiness": DATA_READINESS,
        "acquisition_authorized": ACQUISITION_AUTHORIZED,
        "inference_authorized": INFERENCE_AUTHORIZED,
    })
    write_jsonl(output_dir / "review_queue.jsonl", intake["queue"])
    write_queue_csv(output_dir / "review_queue.csv", intake["queue"])
    write_json(output_dir / "readiness_report.json", report)

    print(json.dumps({
        "valid": len(intake["valid"]),
        "invalid": len(intake["invalid"]),
        "verified_authentic": report["verified_authentic_count"],
        "verified_counterfeit": report["verified_counterfeit_count"],
        "data_readiness": report["data_readiness"],
        "output_dir": str(output_dir),
    }, indent=2))
    return 0 if not intake["invalid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
