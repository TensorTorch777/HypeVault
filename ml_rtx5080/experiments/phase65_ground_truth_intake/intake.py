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

PROTECTED_PATH_COMPONENTS = frozenset({
    "final_test",
    "final-test",
    "ood",
    "excluded_ood",
    "locked_evaluation",
    "label_0_watches",
    "label_1_watches",
    "label_0_sneakers",
    "label_1_sneakers",
})

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"})

AUTHENTICITY_DECISIONS = frozenset({
    "UNREVIEWED",
    "INSUFFICIENT_EVIDENCE",
    "REJECTED",
    "AUTHENTIC",
    "COUNTERFEIT",
    "NOT_STARTED",
})
VERIFIED_DECISIONS = frozenset({"AUTHENTIC", "COUNTERFEIT"})

# Matches hypevault_scraper.common assignment contract (UNIQUE or near-dup flag).
SCRAPER_DUPLICATE_STATUSES = frozenset({"UNIQUE", "NEAR_DUPLICATE_CANDIDATE"})

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
LEDGER_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "review_ledger.schema.json"


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


def path_has_protected_component(path: Path) -> bool:
    """True when any path component is a protected cohort or legacy training directory."""
    return bool({part.lower() for part in path.parts} & PROTECTED_PATH_COMPONENTS)


def resolve_input_path(path: Path | str) -> Path:
    """Resolve symlinks and refuse protected components before any read."""
    candidate = Path(path)
    # Inspect unresolved parts first (covers string-only claims).
    if path_has_protected_component(candidate):
        raise ImageAccessBlocked(f"refusing protected or legacy training path: {candidate}")
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise IntakeError(f"cannot resolve path {candidate}: {exc}") from exc
    if path_has_protected_component(resolved):
        raise ImageAccessBlocked(f"refusing protected or legacy training path: {resolved}")
    return resolved


def assert_readable_metadata_path(path: Path | str) -> Path:
    """Guard CLI/input metadata files before read_bytes/read_text."""
    resolved = resolve_input_path(path)
    if resolved.suffix.lower() in IMAGE_EXTENSIONS:
        raise ImageAccessBlocked(f"intake is metadata-only; refusing image path open: {resolved}")
    return resolved


# Back-compat alias used by tests and callers that mean "guard before open".
assert_metadata_only_path = assert_readable_metadata_path


def _resolved_set(paths: Iterable[Path | str]) -> set[Path]:
    return {resolve_input_path(p) for p in paths}


def assert_writable_output_path(path: Path | str, *, protected_inputs: Iterable[Path]) -> Path:
    """Refuse writes that would overwrite inputs or land in protected locations."""
    resolved = resolve_input_path(path)
    if resolved.suffix.lower() in IMAGE_EXTENSIONS:
        raise ImageAccessBlocked(f"refusing image output path: {resolved}")
    protected = _resolved_set(protected_inputs)
    if resolved in protected:
        raise IntakeError(f"refusing to overwrite input path: {resolved}")
    # Also refuse writing inside a protected directory component tree already checked by resolve.
    return resolved


def assert_ledger_append_path(
    ledger_path: Path | str,
    *,
    protected_inputs: Iterable[Path | str],
) -> Path:
    """Resolve ledger destination and refuse collisions with protected inputs.

    Callers must pass every path that must not be appended to or overwritten
    (manifests, --append-decision JSON, existing decision files, images).
    Comparisons use symlink-resolved paths.
    """
    resolved = resolve_input_path(ledger_path)
    if resolved.suffix.lower() in IMAGE_EXTENSIONS:
        raise ImageAccessBlocked(f"refusing image path as review ledger: {resolved}")
    protected = _resolved_set(protected_inputs)
    if resolved in protected:
        raise IntakeError(
            f"refusing review-ledger path colliding with a protected input: {resolved}"
        )
    return resolved


def image_path_reference_errors(image_path: str) -> list[str]:
    """String-only checks for manifest image_path. Never opens the file."""
    errors: list[str] = []
    if not isinstance(image_path, str) or not image_path.strip():
        return ["IMAGE_PATH_INVALID"]
    candidate = Path(image_path)
    if path_has_protected_component(candidate):
        errors.append("PROTECTED_PATH_REFERENCE")
    # Image extensions are expected as references; they are not opened here.
    return errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = assert_readable_metadata_path(path)
    if not resolved.is_file():
        raise IntakeError(f"manifest not found: {resolved}")
    rows: list[dict[str, Any]] = []
    text = resolved.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntakeError(f"{resolved}:{line_no}: malformed JSONL ({exc})") from exc
        if not isinstance(payload, dict):
            raise IntakeError(f"{resolved}:{line_no}: record must be a JSON object")
        rows.append(payload)
    return rows


def load_json_object(path: Path) -> dict[str, Any]:
    resolved = assert_readable_metadata_path(path)
    if not resolved.is_file():
        raise IntakeError(f"json file not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntakeError(f"malformed JSON in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntakeError(f"{resolved}: expected a JSON object")
    return payload


def normalize_authenticity_decision(value: object) -> str:
    """Map review decisions to AUTHENTIC/COUNTERFEIT or pass through ledger enums."""
    if not isinstance(value, str) or not value.strip():
        return "UNREVIEWED"
    raw = value.strip().upper()
    aliases = {
        STATUS_VERIFIED_AUTHENTIC: "AUTHENTIC",
        STATUS_VERIFIED_COUNTERFEIT: "COUNTERFEIT",
        "VERIFIED_AUTHENTIC": "AUTHENTIC",
        "VERIFIED_COUNTERFEIT": "COUNTERFEIT",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in AUTHENTICITY_DECISIONS:
        raise IntakeError(f"unsupported authenticity_decision: {value!r}")
    return normalized


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
        if record.get("provenance_completeness") not in {"PARTIAL", "INCOMPLETE"}:
            errors.append(f"PROVENANCE_COMPLETENESS_INVALID:{record.get('provenance_completeness')!r}")
    for dim in ("width", "height", "byte_size"):
        value = record.get(dim)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{dim.upper()}_INVALID")
    image_path = record.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        errors.append("IMAGE_PATH_INVALID")
    else:
        errors.extend(image_path_reference_errors(image_path))
    duplicate_status = record.get("duplicate_status")
    if duplicate_status not in SCRAPER_DUPLICATE_STATUSES:
        errors.append(f"DUPLICATE_STATUS_INVALID:{duplicate_status!r}")
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


def _ledger_schema_validator():
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover
        raise IntakeError("jsonschema is required to validate review ledger rows") from exc
    schema = json.loads(LEDGER_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_ledger_row(row: Mapping[str, Any]) -> None:
    """Fail closed on malformed ledger rows using the Phase 65 schema.

    Verified authenticity proposals must embed a Phase 55 evidence record that
    validates against evidence_ledger.schema.json. Invalid evidence is rejected;
    historical rows are never rewritten.
    """
    validator = _ledger_schema_validator()
    errors = sorted(validator.iter_errors(dict(row)), key=lambda e: list(e.path))
    if errors:
        messages = "; ".join(e.message for e in errors)
        raise IntakeError(f"review ledger schema validation failed: {messages}")
    decision = normalize_authenticity_decision(row.get("authenticity_decision"))
    if decision not in VERIFIED_DECISIONS:
        return
    evidence = row.get("phase55_evidence_record")
    if not isinstance(evidence, dict):
        raise IntakeError("verified authenticity decisions require phase55_evidence_record")
    ok, errs = evaluate_phase55_eligibility(evidence)
    if not ok:
        raise IntakeError(
            "embedded phase55_evidence_record failed schema/eligibility validation: "
            + "; ".join(errs)
        )
    verified_label = evidence.get("verified_label")
    if verified_label != decision:
        raise IntakeError(
            f"phase55 verified_label {verified_label!r} does not match "
            f"authenticity_decision {decision!r}"
        )
    adjudication_outcome = evidence.get("adjudication_outcome")
    if adjudication_outcome != decision:
        raise IntakeError(
            f"phase55 adjudication_outcome {adjudication_outcome!r} does not match "
            f"authenticity_decision {decision!r}"
        )


def load_review_ledger(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    resolved = assert_readable_metadata_path(path)
    if not resolved.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(resolved):
        validate_ledger_row(row)
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            raise IntakeError("review ledger row missing record_id")
        grouped.setdefault(record_id, []).append(dict(row))
    return grouped


def scan_manifest_inputs(
    manifest_paths: Iterable[Path],
    *,
    fingerprint: bool = True,
) -> dict[str, Any]:
    """Shared no-write scan of scraper manifests with cross-record uniqueness.

    Returns resolved paths, per-row structural errors, duplicate record IDs,
    duplicate hash groups, and optional content fingerprints. Never opens images.
    """
    resolved_manifests = [assert_readable_metadata_path(path) for path in manifest_paths]
    seen_ids: dict[str, list[str]] = {}
    seen_hashes: dict[str, list[str]] = {}
    source_fingerprints: list[str] = []
    items: list[dict[str, Any]] = []

    for path in resolved_manifests:
        before = path.read_bytes() if fingerprint else b""
        if fingerprint:
            source_fingerprints.append(hashlib.sha256(before).hexdigest())
        rows = load_jsonl(path)
        for index, record in enumerate(rows):
            errors = validate_scraper_record(record, source_path=str(path), index=index)
            record_id = str(record.get("record_id") or f"missing-{path.name}-{index}")
            seen_ids.setdefault(record_id, []).append(str(path))
            stored = record.get("stored_sha256")
            if isinstance(stored, str) and _sha256_hex(stored):
                seen_hashes.setdefault(stored, []).append(record_id)
            items.append(
                {
                    "record": record,
                    "manifest_path": str(path),
                    "index": index,
                    "record_id": record_id,
                    "errors": list(errors),
                }
            )
        if fingerprint:
            after = path.read_bytes()
            if after != before:
                raise IntakeError(f"source manifest mutated during intake: {path}")

    duplicate_record_ids = sorted({rid for rid, paths in seen_ids.items() if len(paths) > 1})
    duplicate_hashes = {sha: ids for sha, ids in seen_hashes.items() if len(ids) > 1}
    return {
        "resolved_manifests": resolved_manifests,
        "items": items,
        "seen_ids": seen_ids,
        "seen_hashes": seen_hashes,
        "duplicate_record_ids": duplicate_record_ids,
        "duplicate_hashes": duplicate_hashes,
        "manifest_sha256": source_fingerprints,
    }


def preflight_inputs_before_append(
    manifest_paths: Iterable[Path],
    ledger_path: Path | None,
) -> None:
    """No-write validation of manifests and the existing ledger before any append.

    Raises IntakeError if any manifest is unreadable/malformed, any scraper
    record fails structural validation, any existing ledger row fails
    schema/Phase 55 validation, or cross-manifest duplicate record IDs / hashes
    violate uniqueness. Never opens images and never writes.
    """
    # Validate existing ledger rows first (fail closed on bad history).
    load_review_ledger(ledger_path)
    scan = scan_manifest_inputs(manifest_paths, fingerprint=False)
    invalid: list[str] = []
    for item in scan["items"]:
        if item["errors"]:
            invalid.append(
                f"{item['manifest_path']}:{item['record_id']}:{','.join(item['errors'])}"
            )
    for record_id in scan["duplicate_record_ids"]:
        first_path = scan["seen_ids"][record_id][0]
        invalid.append(f"DUPLICATE_RECORD_ID:{record_id}:{first_path}")
    for sha, ids in sorted(scan["duplicate_hashes"].items()):
        invalid.append(f"DUPLICATE_HASH_GROUP:{sha}:{','.join(ids)}")
    if invalid:
        raise IntakeError(
            "append preflight failed; refusing to mutate review ledger while "
            f"inputs are invalid: {'; '.join(invalid[:5])}"
            + (" ..." if len(invalid) > 5 else "")
        )


def append_review_decision(
    ledger_path: Path,
    decision: Mapping[str, Any],
    *,
    protected_inputs: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Append one human decision. Never rewrites prior rows or source manifests.

    ``protected_inputs`` must include every path that must not be appended to
    (manifests, decision JSON, images). An empty/omitted set only applies the
    generic protected-component and image-extension guards; call sites that
    have other inputs MUST pass them.
    """
    resolved_ledger = assert_ledger_append_path(
        ledger_path,
        protected_inputs=list(protected_inputs or ()),
    )
    payload = dict(decision)
    payload["schema_version"] = REVIEW_LEDGER_SCHEMA
    payload["authenticity_decision"] = normalize_authenticity_decision(
        payload.get("authenticity_decision")
    )
    payload.setdefault("recorded_at_utc", utc_now_iso())
    payload.setdefault(
        "decision_id",
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32],
    )
    validate_ledger_row(payload)
    resolved_ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
    with resolved_ledger.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return payload


def evaluate_phase55_eligibility(evidence: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate a proposed Phase 55 evidence-ledger record with the authoritative schema."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover
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


def evaluate_data_admission(
    candidate: Mapping[str, Any],
    *,
    purpose: str = "evaluation",
    today: date | None = None,
) -> tuple[str, list[str]]:
    return admission_decision(dict(candidate), purpose=purpose, today=today or date.today())


def assess_review_against_contracts(
    scraper_record: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    today: date | None = None,
    duplicate_hash_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Fail-closed assessment: scraper status alone never verifies a class."""
    blockers: list[str] = []
    label_status = scraper_record.get("label_status")
    if label_status not in AUTO_ASSIGNABLE_STATUSES:
        blockers.append(f"unexpected_scraper_label_status:{label_status}")

    try:
        authenticity = normalize_authenticity_decision(decision.get("authenticity_decision"))
    except IntakeError as exc:
        return {
            "authenticity_verified": False,
            "ml_use_eligible": False,
            "blockers": [str(exc)],
            "admission_status": "NOT_EVALUATED",
            "admission_reasons": [],
            "normalized_authenticity_decision": "UNREVIEWED",
            "scraper_label_status_unchanged": label_status,
        }

    rights_ok = decision.get("rights_review_status") == "APPROVED"
    proposing_verified = authenticity in VERIFIED_DECISIONS

    if label_status == STATUS_SYNTHETIC and proposing_verified:
        blockers.append("synthetic_not_physical_ground_truth")

    if duplicate_hash_ids and len(duplicate_hash_ids) > 1:
        blockers.append("duplicate_hash_group")

    evidence = decision.get("phase55_evidence_record")
    if proposing_verified:
        if not isinstance(evidence, dict):
            blockers.append("missing_phase55_evidence_record")
        else:
            ok, errs = evaluate_phase55_eligibility(evidence)
            if not ok:
                blockers.extend(f"phase55:{e}" for e in errs)
            if evidence.get("image_sha256") != scraper_record.get("stored_sha256"):
                blockers.append("phase55_image_sha256_mismatch")
            verified_label = evidence.get("verified_label")
            if verified_label != authenticity:
                blockers.append(
                    f"decision_evidence_label_mismatch:{authenticity}!={verified_label}"
                )
            adjudication_outcome = evidence.get("adjudication_outcome")
            if adjudication_outcome != authenticity:
                blockers.append(
                    f"decision_adjudication_outcome_mismatch:{authenticity}!={adjudication_outcome}"
                )
        if not rights_ok:
            blockers.append("rights_not_approved")
    elif authenticity in {"UNREVIEWED", "INSUFFICIENT_EVIDENCE", "REJECTED", "NOT_STARTED"}:
        blockers.append(f"authenticity_{authenticity.lower()}")
    else:
        blockers.append(f"authenticity_decision_unsupported:{authenticity!r}")

    disputed = decision.get("adjudication_status") in {"UNRESOLVED", "PENDING", "NOT_STARTED"}
    if disputed and proposing_verified:
        blockers.append("adjudication_incomplete")

    # Source-manifest uniqueness is authoritative for ML eligibility. A candidate
    # payload claiming UNIQUE cannot override NEAR_DUPLICATE_CANDIDATE (or other
    # non-unique) status on the scraper record.
    source_duplicate_status = scraper_record.get("duplicate_status")
    if source_duplicate_status != "UNIQUE":
        blockers.append(f"source_duplicate_status_not_unique:{source_duplicate_status}")

    candidate = decision.get("candidate_image_record_v2")
    admission_status = "NOT_EVALUATED"
    admission_reasons: list[str] = []
    if proposing_verified:
        if not isinstance(candidate, dict):
            blockers.append("missing_candidate_image_record_v2")
            admission_status = "MISSING"
        else:
            candidate_sha = ((candidate.get("image") or {}).get("sha256"))
            if candidate_sha != scraper_record.get("stored_sha256"):
                blockers.append("candidate_image_sha256_mismatch")
            candidate_label = ((candidate.get("label") or {}).get("value") or "").lower()
            expected = "authentic" if authenticity == "AUTHENTIC" else "counterfeit"
            if candidate_label != expected:
                blockers.append(f"candidate_label_mismatch:{candidate_label}!={expected}")
            candidate_dup = (candidate.get("image") or {}).get("duplicate_status")
            if candidate_dup != "UNIQUE":
                blockers.append("candidate_duplicate_status_not_unique")
            if candidate_dup == "UNIQUE" and source_duplicate_status != "UNIQUE":
                blockers.append("candidate_cannot_override_source_duplicate_status")
            admission_status, admission_reasons = evaluate_data_admission(
                candidate, today=today
            )
            if admission_status != "ADMIT":
                blockers.append("admission_rejected")
                blockers.extend(f"admission:{r}" for r in admission_reasons)

    authenticity_hard_blockers = {
        "missing_phase55_evidence_record",
        "phase55_image_sha256_mismatch",
        "synthetic_not_physical_ground_truth",
        "adjudication_incomplete",
        "duplicate_hash_group",
    }
    authenticity_verified = proposing_verified and not (
        authenticity_hard_blockers.intersection(blockers)
        or any(
            b.startswith("phase55:")
            or b.startswith("decision_evidence_label_mismatch")
            or b.startswith("decision_adjudication_outcome_mismatch")
            for b in blockers
        )
    )

    ml_eligible = (
        authenticity_verified
        and rights_ok
        and admission_status == "ADMIT"
        and source_duplicate_status == "UNIQUE"
        and "missing_candidate_image_record_v2" not in blockers
        and "candidate_image_sha256_mismatch" not in blockers
        and not any(b.startswith("candidate_label_mismatch") for b in blockers)
        and "candidate_duplicate_status_not_unique" not in blockers
        and "candidate_cannot_override_source_duplicate_status" not in blockers
        and not any(b.startswith("source_duplicate_status_not_unique") for b in blockers)
        and "admission_rejected" not in blockers
    )

    return {
        "authenticity_verified": authenticity_verified,
        "ml_use_eligible": ml_eligible,
        "blockers": blockers,
        "admission_status": admission_status,
        "admission_reasons": admission_reasons,
        "normalized_authenticity_decision": authenticity,
        "scraper_label_status_unchanged": label_status,
    }


def _block_from_verified_admitted(
    row: dict[str, Any],
    *,
    reason: str,
    as_validation_error: bool = False,
) -> None:
    """Block a queue row from verified/admitted counts with an explicit reason.

    Duplicate record IDs are structural validation failures (as_validation_error=True).
    Exact duplicate hashes remain distinct: they demote eligibility without that flag.
    """
    if as_validation_error and reason not in (row.get("validation_errors") or []):
        row.setdefault("validation_errors", []).append(reason)
    row["eligibility_decision"] = "BLOCKED"
    existing = row.get("eligibility_reason") or ""
    if reason not in existing.split(","):
        row["eligibility_reason"] = (
            f"{existing},{reason}" if existing and existing != reason else reason
        )
    blockers = row.setdefault("assessment_blockers", [])
    if reason not in blockers:
        blockers.append(reason)
    row.pop("verified_label", None)
    if row.get("admission_status") == "ADMIT":
        row["admission_status"] = "BLOCKED_DUPLICATE"


def ingest_manifests(
    manifest_paths: Iterable[Path],
    *,
    ledger_path: Path | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    scan = scan_manifest_inputs(manifest_paths, fingerprint=True)
    resolved_manifests = scan["resolved_manifests"]
    seen_ids = scan["seen_ids"]
    seen_hashes = scan["seen_hashes"]
    duplicate_record_ids = scan["duplicate_record_ids"]
    duplicate_hashes = scan["duplicate_hashes"]
    ledger = load_review_ledger(ledger_path)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    pending_items: list[dict[str, Any]] = []

    for item in scan["items"]:
        record = item["record"]
        record_id = item["record_id"]
        errors = list(item["errors"])
        path = item["manifest_path"]
        stored = record.get("stored_sha256")
        entries = ledger.get(record_id, [])
        row = build_queue_row(
            record,
            manifest_path=path,
            validation_errors=list(errors),
            ledger_entries=entries,
        )
        # Provisional assessment; duplicate-ID/hash demotion happens in a second pass
        # so the first conflicting row cannot remain verified/admitted.
        if not errors and entries:
            dup_ids = seen_hashes.get(str(stored), [record_id])
            assessment = assess_review_against_contracts(
                record,
                entries[-1],
                today=today,
                duplicate_hash_ids=dup_ids,
            )
            if assessment["authenticity_verified"]:
                row["eligibility_decision"] = (
                    "ADMITTED" if assessment["ml_use_eligible"] else "VERIFIED_NOT_ML_ELIGIBLE"
                )
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
        pending_items.append({"record": record, "queue": row, "manifest_path": path})

    duplicate_id_set = set(duplicate_record_ids)
    for item in pending_items:
        row = item["queue"]
        record_id = str(row.get("record_id") or "")
        if record_id in duplicate_id_set:
            first_path = seen_ids[record_id][0]
            _block_from_verified_admitted(
                row,
                reason=f"DUPLICATE_RECORD_ID:{first_path}",
                as_validation_error=True,
            )
        stored = item["record"].get("stored_sha256")
        if stored in duplicate_hashes:
            _block_from_verified_admitted(
                row,
                reason="duplicate_hash_group",
                as_validation_error=False,
            )

        if row.get("validation_errors"):
            if row.get("eligibility_decision") not in {"BLOCKED", "INVALID"}:
                row["eligibility_decision"] = "INVALID"
            invalid.append(item)
        else:
            valid.append(item)
        queue.append(row)

    return {
        "schema_version": PHASE65_SCHEMA,
        "generated_at_utc": utc_now_iso(),
        "manifest_paths": [str(p) for p in resolved_manifests],
        "manifest_sha256": scan["manifest_sha256"],
        "valid": valid,
        "invalid": invalid,
        "queue": queue,
        "duplicate_record_ids": duplicate_record_ids,
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
    admitted_authentic = 0
    admitted_counterfeit = 0
    for row in queue:
        decision = row.get("eligibility_decision")
        verified_label = row.get("verified_label")
        # Only count verified authenticity when assessment succeeded (ADMITTED or
        # verified-but-not-ML-eligible). Duplicates demoted to BLOCKED are excluded.
        if decision in {"ADMITTED", "VERIFIED_NOT_ML_ELIGIBLE"} and verified_label == "AUTHENTIC":
            verified_authentic += 1
        elif decision in {"ADMITTED", "VERIFIED_NOT_ML_ELIGIBLE"} and verified_label == "COUNTERFEIT":
            verified_counterfeit += 1
        # Admitted requires actual data_admission ADMIT (eligibility_decision ADMITTED).
        # VERIFIED_NOT_ML_ELIGIBLE never counts as ML-admissible class support.
        if decision == "ADMITTED" and row.get("admission_status") == "ADMIT":
            admitted += 1
            if verified_label == "AUTHENTIC":
                admitted_authentic += 1
            elif verified_label == "COUNTERFEIT":
                admitted_counterfeit += 1

    both_classes = admitted_authentic > 0 and admitted_counterfeit > 0
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
        "duplicate_record_ids": list(intake.get("duplicate_record_ids") or []),
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
    # Guard every input before any read.
    manifests = [assert_readable_metadata_path(Path(p)) for p in args.manifest]
    decision_path = (
        assert_readable_metadata_path(Path(args.append_decision)) if args.append_decision else None
    )

    # Ledger destination must not collide with manifests, decision JSON, images,
    # or protected paths. Check before reading decision content or appending.
    ledger: Path | None = None
    if args.review_ledger:
        collision_inputs: list[Path] = list(manifests)
        if decision_path is not None:
            collision_inputs.append(decision_path)
        if args.append_decision:
            # Enforce collision before any append that could modify a file.
            ledger = assert_ledger_append_path(
                Path(args.review_ledger),
                protected_inputs=collision_inputs,
            )
        else:
            ledger = assert_readable_metadata_path(Path(args.review_ledger))
            # Still refuse a ledger path that is the same resolved file as a manifest.
            assert_ledger_append_path(ledger, protected_inputs=collision_inputs)

    protected_inputs = list(manifests)
    if ledger is not None:
        protected_inputs.append(ledger)
    if decision_path is not None:
        protected_inputs.append(decision_path)

    output_dir = resolve_input_path(Path(args.output_dir))
    for name in ("intake_report.json", "review_queue.jsonl", "review_queue.csv", "readiness_report.json"):
        assert_writable_output_path(output_dir / name, protected_inputs=protected_inputs)

    if decision_path is not None:
        if ledger is None:
            raise SystemExit("--append-decision requires --review-ledger")
        # Re-check collision immediately before write (resolved paths, incl. symlinks).
        assert_ledger_append_path(ledger, protected_inputs=[*manifests, decision_path])
        # No-write preflight: abort before mutating the ledger if manifests or
        # the existing ledger are malformed/invalid. Leaves all inputs and any
        # pre-existing outputs untouched.
        preflight_inputs_before_append(manifests, ledger)
        decision = load_json_object(decision_path)
        append_review_decision(
            ledger,
            decision,
            protected_inputs=[*manifests, decision_path],
        )

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
        "duplicate_record_ids": intake["duplicate_record_ids"],
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
        "admitted": report["admitted_under_existing_contracts"],
        "data_readiness": report["data_readiness"],
        "output_dir": str(output_dir),
    }, indent=2))
    return 0 if not intake["invalid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
