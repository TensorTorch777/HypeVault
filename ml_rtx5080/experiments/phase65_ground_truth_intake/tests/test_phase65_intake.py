"""Offline Phase 65 intake tests. Synthetic metadata only; no images or network."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml_rtx5080"))

from intake import (  # noqa: E402
    ACQUISITION_AUTHORIZED,
    DATA_READINESS,
    INFERENCE_AUTHORIZED,
    ImageAccessBlocked,
    IntakeError,
    append_review_decision,
    assess_review_against_contracts,
    assert_ledger_append_path,
    assert_metadata_only_path,
    assert_readable_metadata_path,
    assert_writable_output_path,
    build_readiness_report,
    image_path_reference_errors,
    ingest_manifests,
    load_json_object,
    load_review_ledger,
    main,
    normalize_authenticity_decision,
)
from hypevault_scraper.common import (  # noqa: E402
    SCHEMA_VERSION,
    STATUS_SYNTHETIC,
    STATUS_UNVERIFIED_WEB,
    STATUS_VERIFIED_AUTHENTIC,
)


def _sha(n: int = 1) -> str:
    return f"{n:064x}"


def scraper_row(**overrides) -> dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "record_id": "rec_web_001",
        "created_at_utc": "2026-10-14T00:00:00Z",
        "image_path": "unverified_web_images/watches/Demo/aaaa.jpg",
        "stored_sha256": _sha(1),
        "original_sha256": _sha(2),
        "width": 96,
        "height": 96,
        "format": "JPEG",
        "byte_size": 1234,
        "family": "watches",
        "claimed_brand": "Demo Brand",
        "claimed_model": "UNKNOWN",
        "search_query": "demo watch",
        "retrieved_at_utc": "2026-10-14T00:00:00Z",
        "search_provider": "test",
        "source_image_url": "UNKNOWN",
        "source_page_url": "UNKNOWN",
        "provenance_completeness": "INCOMPLETE",
        "provenance_note": "test",
        "label_status": STATUS_UNVERIFIED_WEB,
        "target_label": None,
        "evidence_status": "UNREVIEWED",
        "rights_review_status": "UNREVIEWED",
        "intended_use_eligibility": {
            "authenticity_benchmark": False,
            "model_training_as_verified_class": False,
            "reason": "test",
        },
        "duplicate_status": "UNIQUE",
        "near_duplicate_of": "NOT_APPLICABLE",
        "perceptual_hash": "NOT_RECORDED",
        "generation": "NOT_APPLICABLE",
    }
    row.update(overrides)
    return row


def phase55_eligible(*, sha: str, label: str = "AUTHENTIC") -> dict:
    return {
        "sample_id": "sample_SYNTHETIC_ELIG",
        "image_sha256": sha,
        "product_id": "product_SYNTHETIC_001",
        "declared_brand": "Demo Brand",
        "verified_brand": "Demo Brand",
        "reference_family": "Synthetic family",
        "source_id": "source_SYNTHETIC_001",
        "seller_id": "seller_SYNTHETIC_001",
        "marketplace_id": "market_SYNTHETIC_001",
        "historical_directory_label": "UNKNOWN",
        "reviewer_a_outcome": label,
        "reviewer_b_outcome": label,
        "evidence_category_a": "QUALIFIED_PHYSICAL_EXAMINATION",
        "evidence_category_b": "MANUFACTURER_PROVENANCE",
        "adjudication_status": "AGREED",
        "adjudication_outcome": label,
        "verified_label": label,
        "license_status": "APPROVED",
        "consent_status": "APPROVED",
        "permitted_purposes": ["ML_EVALUATION"],
        "retention_until": "2027-12-31",
        "deletion_status": "RETENTION_ACTIVE",
        "source_product_disjointness": {
            "dinov2_legacy": "PASS",
            "dinov3_experimental": "PASS",
        },
        "eligible_primary_benchmark": True,
    }


def candidate_admissible(*, sha: str, label: str = "authentic", retention_until: str = "2028-12-31") -> dict:
    return {
        "record_id": "synthetic-candidate-1",
        "image": {
            "sha256": sha,
            "uri": "store://synthetic/1.jpg",
            "width": 2000,
            "height": 1500,
            "format": "JPEG",
            "file_size_bytes": 812345,
            "exif_location_stripped": True,
            "derivative_of": None,
            "duplicate_status": "UNIQUE",
            "duplicate_of": None,
        },
        "source": {
            "source_type": "QUALIFIED_EXAMINER",
            "source_reference": "private:partner-ref-1",
            "product_or_listing_id": "item-1",
            "source_group_id": "group-1",
            "chain_of_custody": [
                {"step": "captured", "actor_ref": "examiner-ref-1", "date": "2026-09-01"}
            ],
        },
        "rights": {
            "permission_id": "perm-1",
            "permission_status": "EFFECTIVE",
            "research_use": "GRANTED",
            "training_use": "GRANTED",
            "public_display": "DENIED",
            "retention_until": retention_until,
        },
        "brand": {
            "value": "Demo Brand",
            "basis": "case-back engraving and movement calibre",
            "evidence_id": "ev-1",
        },
        "label": {
            "value": label,
            "method": ["movement_inspection", "serial_verification"],
            "verifier_ref": "examiner-ref-1",
            "verifier_qualification_evidence_id": "qual-1",
            "verifier_independent": True,
            "confidence": "certain",
            "evidence_id": "ev-1",
            "evidence_date": "2026-09-01",
        },
        "disposition": {"status": "PENDING", "reason": "awaiting admission check"},
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Phase65IntakeTests(unittest.TestCase):
    def test_valid_scraper_records_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row(), scraper_row(record_id="rec_web_002", stored_sha256=_sha(3))])
            before = manifest.read_bytes()
            result = ingest_manifests([manifest])
            self.assertEqual(len(result["valid"]), 2)
            self.assertEqual(len(result["invalid"]), 0)
            self.assertEqual(manifest.read_bytes(), before)
            report = build_readiness_report(result)
            self.assertEqual(report["verified_authentic_count"], 0)
            self.assertEqual(report["verified_counterfeit_count"], 0)
            self.assertEqual(report["data_readiness"], DATA_READINESS)
            self.assertFalse(report["acquisition_authorized"])
            self.assertFalse(report["inference_authorized"])

    def test_malformed_and_unsupported_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("{not-json\n", encoding="utf-8")
            with self.assertRaises(Exception):
                ingest_manifests([bad])
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row(schema_version="nope")])
            result = ingest_manifests([manifest])
            self.assertEqual(len(result["invalid"]), 1)
            self.assertTrue(
                any("UNSUPPORTED_SCHEMA" in e for e in result["invalid"][0]["queue"]["validation_errors"])
            )

    def test_unverified_and_synthetic_never_auto_verify(self) -> None:
        web = scraper_row()
        synthetic = scraper_row(record_id="rec_syn", label_status=STATUS_SYNTHETIC, stored_sha256=_sha(9))
        for record in (web, synthetic):
            assessment = assess_review_against_contracts(
                record,
                {"authenticity_decision": "UNREVIEWED", "rights_review_status": "UNREVIEWED"},
            )
            self.assertFalse(assessment["authenticity_verified"])
            self.assertFalse(assessment["ml_use_eligible"])
        forced = assess_review_against_contracts(
            synthetic,
            {
                "authenticity_decision": "COUNTERFEIT",
                "rights_review_status": "APPROVED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(9), label="COUNTERFEIT"),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(9), label="counterfeit"),
            },
        )
        self.assertFalse(forced["authenticity_verified"])
        self.assertIn("synthetic_not_physical_ground_truth", forced["blockers"])

    def test_missing_provenance_stays_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row()])
            row = ingest_manifests([manifest])["queue"][0]
            self.assertEqual(row["source_image_url"], "UNKNOWN")
            self.assertEqual(row["provenance_completeness"], "INCOMPLETE")
            self.assertIn("source_image_url", row["missing_or_unknown_fields"])

    def test_duplicate_ids_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            write_manifest(a, [scraper_row(record_id="dup")])
            write_manifest(b, [scraper_row(record_id="dup", stored_sha256=_sha(1))])
            result = ingest_manifests([a, b])
            self.assertEqual(len(result["invalid"]), 2)
            self.assertEqual(result["duplicate_record_ids"], ["dup"])
            for item in result["invalid"]:
                self.assertTrue(
                    any("DUPLICATE_RECORD_ID" in e for e in item["queue"]["validation_errors"])
                )
            write_manifest(a, [scraper_row(record_id="one", stored_sha256=_sha(7))])
            write_manifest(b, [scraper_row(record_id="two", stored_sha256=_sha(7))])
            result = ingest_manifests([a, b])
            self.assertIn(_sha(7), result["duplicate_hashes"])
            self.assertEqual(sorted(result["duplicate_hashes"][_sha(7)]), ["one", "two"])
            self.assertEqual(result["duplicate_record_ids"], [])

    def test_human_review_cannot_bypass_evidence_and_rights(self) -> None:
        record = scraper_row()
        weak = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "UNREVIEWED",
                "adjudication_status": "AGREED",
            },
        )
        self.assertFalse(weak["authenticity_verified"])
        self.assertIn("missing_phase55_evidence_record", weak["blockers"])
        strong = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "APPROVED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(1)),
            },
        )
        self.assertTrue(strong["authenticity_verified"])
        self.assertTrue(strong["ml_use_eligible"])
        self.assertEqual(strong["admission_status"], "ADMIT")

    def test_verified_label_does_not_imply_ml_eligibility_without_rights(self) -> None:
        record = scraper_row()
        assessment = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "UNREVIEWED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(1)),
            },
        )
        self.assertTrue(assessment["authenticity_verified"])
        self.assertFalse(assessment["ml_use_eligible"])
        self.assertIn("rights_not_approved", assessment["blockers"])

    def test_disputed_adjudication_remains_ineligible(self) -> None:
        record = scraper_row()
        assessment = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "APPROVED",
                "adjudication_status": "UNRESOLVED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(1)),
            },
        )
        self.assertFalse(assessment["authenticity_verified"])
        self.assertIn("adjudication_incomplete", assessment["blockers"])

    def test_ledger_append_does_not_mutate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            ledger = Path(tmp) / "ledger.jsonl"
            write_manifest(manifest, [scraper_row()])
            before = manifest.read_bytes()
            append_review_decision(
                ledger,
                {
                    "record_id": "rec_web_001",
                    "authenticity_decision": "INSUFFICIENT_EVIDENCE",
                    "rights_review_status": "UNREVIEWED",
                    "reviewer_ref": "reviewer_role_a",
                },
            )
            result = ingest_manifests([manifest], ledger_path=ledger)
            self.assertEqual(manifest.read_bytes(), before)
            self.assertEqual(result["queue"][0]["ledger_entry_count"], 1)
            self.assertEqual(result["queue"][0]["eligibility_decision"], "BLOCKED")

    def test_readiness_counts_and_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(
                manifest,
                [
                    scraper_row(),
                    scraper_row(record_id="syn", label_status=STATUS_SYNTHETIC, stored_sha256=_sha(8)),
                ],
            )
            report = build_readiness_report(ingest_manifests([manifest]))
            self.assertEqual(report["records_processed"], 2)
            self.assertEqual(report["label_status_counts"][STATUS_UNVERIFIED_WEB], 1)
            self.assertEqual(report["label_status_counts"][STATUS_SYNTHETIC], 1)
            self.assertEqual(report["verified_authentic_count"], 0)
            self.assertEqual(report["verified_counterfeit_count"], 0)
            self.assertFalse(report["both_classes_have_admissible_support"])
            self.assertEqual(report["data_readiness"], "NO_GO_FOR_AUTHENTICITY_CLAIMS")
            self.assertFalse(report["dataset_frozen"])
            self.assertFalse(report["benchmark_ready"])

    def test_does_not_read_image_files_or_protected_paths(self) -> None:
        with self.assertRaises(ImageAccessBlocked):
            assert_metadata_only_path(Path("final_test/secret.jpg"))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row(image_path="final_test/locked/x.jpg")])
            result = ingest_manifests([manifest])
            self.assertTrue(
                any("PROTECTED_PATH_REFERENCE" in e for e in result["invalid"][0]["queue"]["validation_errors"])
            )
            image = Path(tmp) / "fake.jpg"
            image.write_bytes(b"not-opened")
            write_manifest(manifest, [scraper_row(image_path=str(image))])
            opened = []
            real_open = Path.open

            def tracking_open(self, *args, **kwargs):
                if self.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    opened.append(str(self))
                return real_open(self, *args, **kwargs)

            with mock.patch.object(Path, "open", tracking_open):
                ingest_manifests([manifest])
            self.assertEqual(opened, [])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            out = Path(tmp) / "out"
            write_manifest(manifest, [scraper_row()])
            code = main(["--manifest", str(manifest), "--output-dir", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue((out / "intake_report.json").exists())
            self.assertTrue((out / "review_queue.jsonl").exists())
            self.assertTrue((out / "review_queue.csv").exists())
            self.assertTrue((out / "readiness_report.json").exists())
            readiness = json.loads((out / "readiness_report.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["data_readiness"], DATA_READINESS)

    def test_invariants_unchanged(self) -> None:
        self.assertEqual(DATA_READINESS, "NO_GO_FOR_AUTHENTICITY_CLAIMS")
        self.assertFalse(ACQUISITION_AUTHORIZED)
        self.assertFalse(INFERENCE_AUTHORIZED)
        policy = json.loads(
            (REPO / "ml_rtx5080/experiments/publication_safety_v1/publication_policy.json").read_text()
        )
        self.assertFalse(policy["AUTHENTICITY_MODEL_PRODUCTION_APPROVED"])
        scope = json.loads((REPO / "ml_rtx5080/experiments/scope_lock_v1/scope_contract.json").read_text())
        self.assertEqual(scope["PRODUCTION_PROMOTION"], "FORBIDDEN")
        self.assertNotEqual(STATUS_VERIFIED_AUTHENTIC, STATUS_UNVERIFIED_WEB)

    def test_verified_scraper_label_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row(label_status=STATUS_VERIFIED_AUTHENTIC)])
            result = ingest_manifests([manifest])
            self.assertEqual(len(result["invalid"]), 1)
            self.assertTrue(
                any(
                    "VERIFIED_LABEL_NOT_ALLOWED_IN_SCRAPER_MANIFEST" in e
                    for e in result["invalid"][0]["queue"]["validation_errors"]
                )
            )

    # --- Review finding regressions (A–F) ---

    def test_protected_manifest_path_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            protected = Path(tmp) / "final_test" / "manifest.jsonl"
            protected.parent.mkdir(parents=True)
            write_manifest(protected, [scraper_row()])
            opened: list[str] = []
            real_read_bytes = Path.read_bytes

            def tracking_read_bytes(self):
                opened.append(str(self))
                return real_read_bytes(self)

            with mock.patch.object(Path, "read_bytes", tracking_read_bytes):
                with self.assertRaises(ImageAccessBlocked):
                    ingest_manifests([protected])
            self.assertEqual(opened, [])

    def test_protected_append_decision_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decision = Path(tmp) / "ood" / "decision.json"
            decision.parent.mkdir(parents=True)
            decision.write_text("{}", encoding="utf-8")
            opened: list[str] = []
            real_read_text = Path.read_text

            def tracking_read_text(self, *args, **kwargs):
                opened.append(str(self))
                return real_read_text(self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", tracking_read_text):
                with self.assertRaises(ImageAccessBlocked):
                    assert_readable_metadata_path(decision)
                with self.assertRaises(ImageAccessBlocked):
                    load_json_object(decision)
            self.assertEqual(opened, [])

    def test_ood_and_legacy_image_path_flagged_without_open(self) -> None:
        for ref in (
            "excluded_ood/watch.jpg",
            "label_0_watches/x.png",
            "datasets/locked_evaluation/y.jpg",
            "final-test/z.webp",
        ):
            errors = image_path_reference_errors(ref)
            self.assertIn("PROTECTED_PATH_REFERENCE", errors, ref)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row(image_path="ood/never_open.jpg")])
            opened: list[str] = []
            real_open = Path.open

            def tracking_open(self, *args, **kwargs):
                if "never_open" in str(self):
                    opened.append(str(self))
                return real_open(self, *args, **kwargs)

            with mock.patch.object(Path, "open", tracking_open):
                result = ingest_manifests([manifest])
            self.assertEqual(opened, [])
            self.assertTrue(
                any("PROTECTED_PATH_REFERENCE" in e for e in result["invalid"][0]["queue"]["validation_errors"])
            )

    def test_decision_conflicts_with_phase55_verified_label(self) -> None:
        record = scraper_row()
        assessment = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "APPROVED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1), label="COUNTERFEIT"),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(1), label="authentic"),
            },
        )
        self.assertFalse(assessment["authenticity_verified"])
        self.assertFalse(assessment["ml_use_eligible"])
        self.assertTrue(
            any(b.startswith("decision_evidence_label_mismatch") for b in assessment["blockers"])
        )

    def test_verified_without_candidate_not_ml_eligible_or_admitted(self) -> None:
        record = scraper_row()
        assessment = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "APPROVED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
            },
        )
        self.assertTrue(assessment["authenticity_verified"])
        self.assertFalse(assessment["ml_use_eligible"])
        self.assertIn("missing_candidate_image_record_v2", assessment["blockers"])
        self.assertEqual(assessment["admission_status"], "MISSING")

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            ledger = Path(tmp) / "ledger.jsonl"
            write_manifest(manifest, [scraper_row()])
            append_review_decision(
                ledger,
                {
                    "record_id": "rec_web_001",
                    "authenticity_decision": "AUTHENTIC",
                    "rights_review_status": "APPROVED",
                    "adjudication_status": "AGREED",
                    "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
                },
            )
            result = ingest_manifests([manifest], ledger_path=ledger)
            report = build_readiness_report(result)
            self.assertEqual(result["queue"][0]["eligibility_decision"], "VERIFIED_NOT_ML_ELIGIBLE")
            self.assertEqual(report["verified_authentic_count"], 1)
            self.assertEqual(report["admitted_under_existing_contracts"], 0)

    def test_candidate_hash_and_label_mismatches_blocked(self) -> None:
        record = scraper_row()
        hash_mismatch = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "APPROVED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(99)),
            },
        )
        self.assertFalse(hash_mismatch["ml_use_eligible"])
        self.assertIn("candidate_image_sha256_mismatch", hash_mismatch["blockers"])

        label_mismatch = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "APPROVED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
                "candidate_image_record_v2": candidate_admissible(sha=_sha(1), label="counterfeit"),
            },
        )
        self.assertFalse(label_mismatch["ml_use_eligible"])
        self.assertTrue(
            any(b.startswith("candidate_label_mismatch") for b in label_mismatch["blockers"])
        )

    def test_malformed_ledger_rows_and_invalid_enums_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "schema_version": "phase65_review_ledger_v1",
                        "record_id": "rec_web_001",
                        "authenticity_decision": "TOTALLY_FAKE",
                        "rights_review_status": "APPROVED",
                        "recorded_at_utc": "2026-10-14T00:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(IntakeError):
                load_review_ledger(ledger)

            with self.assertRaises(IntakeError):
                append_review_decision(
                    Path(tmp) / "ledger2.jsonl",
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "MAYBE",
                        "rights_review_status": "APPROVED",
                    },
                )

            with self.assertRaises(IntakeError):
                append_review_decision(
                    Path(tmp) / "ledger3.jsonl",
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "AUTHENTIC",
                        "rights_review_status": "APPROVED",
                        # verified decision without phase55 evidence
                    },
                )

            self.assertEqual(normalize_authenticity_decision(STATUS_VERIFIED_AUTHENTIC), "AUTHENTIC")

    def test_duplicate_hashes_cannot_inflate_admitted_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            ledger = Path(tmp) / "ledger.jsonl"
            write_manifest(a, [scraper_row(record_id="one", stored_sha256=_sha(7))])
            write_manifest(b, [scraper_row(record_id="two", stored_sha256=_sha(7))])
            for rid in ("one", "two"):
                append_review_decision(
                    ledger,
                    {
                        "record_id": rid,
                        "authenticity_decision": "AUTHENTIC",
                        "rights_review_status": "APPROVED",
                        "adjudication_status": "AGREED",
                        "phase55_evidence_record": phase55_eligible(sha=_sha(7)),
                        "candidate_image_record_v2": candidate_admissible(sha=_sha(7)),
                    },
                )
            result = ingest_manifests([a, b], ledger_path=ledger)
            report = build_readiness_report(result)
            self.assertEqual(report["verified_authentic_count"], 0)
            self.assertEqual(report["admitted_under_existing_contracts"], 0)
            for row in result["queue"]:
                self.assertEqual(row["eligibility_decision"], "BLOCKED")
                self.assertIn("duplicate_hash_group", row.get("assessment_blockers") or [row["eligibility_reason"]])

    def test_retention_uses_injected_or_current_date(self) -> None:
        record = scraper_row()
        decision = {
            "authenticity_decision": "AUTHENTIC",
            "rights_review_status": "APPROVED",
            "adjudication_status": "AGREED",
            "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
            "candidate_image_record_v2": candidate_admissible(
                sha=_sha(1), retention_until="2020-01-01"
            ),
        }
        expired = assess_review_against_contracts(record, decision, today=date(2026, 10, 14))
        self.assertFalse(expired["ml_use_eligible"])
        self.assertEqual(expired["admission_status"], "REJECT")
        self.assertIn("admission:RETENTION_EXPIRED", expired["blockers"])

        still_valid = assess_review_against_contracts(
            record,
            {
                **decision,
                "candidate_image_record_v2": candidate_admissible(
                    sha=_sha(1), retention_until="2099-01-01"
                ),
            },
            today=date.today(),
        )
        self.assertTrue(still_valid["ml_use_eligible"])
        self.assertEqual(still_valid["admission_status"], "ADMIT")

    def test_output_cannot_overwrite_input_manifest_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row()])
            with self.assertRaises(IntakeError):
                assert_writable_output_path(manifest, protected_inputs=[manifest])
            ledger = Path(tmp) / "ledger.jsonl"
            ledger.write_text("", encoding="utf-8")
            with self.assertRaises(IntakeError):
                assert_writable_output_path(ledger, protected_inputs=[manifest, ledger])
            # CLI: output-dir whose intake_report.json would overwrite the manifest
            collide_dir = Path(tmp)
            collide_manifest = collide_dir / "intake_report.json"
            write_manifest(collide_manifest, [scraper_row()])
            with self.assertRaises(IntakeError):
                main(["--manifest", str(collide_manifest), "--output-dir", str(collide_dir)])

    def test_review_ledger_colliding_with_manifest_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row()])
            before = manifest.read_bytes()
            out = Path(tmp) / "out"
            with self.assertRaises(IntakeError):
                main(
                    [
                        "--manifest",
                        str(manifest),
                        "--review-ledger",
                        str(manifest),
                        "--append-decision",
                        str(Path(tmp) / "decision.json"),
                        "--output-dir",
                        str(out),
                    ]
                )
            self.assertEqual(manifest.read_bytes(), before)
            self.assertFalse(out.exists())

            with self.assertRaises(IntakeError):
                append_review_decision(
                    manifest,
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "INSUFFICIENT_EVIDENCE",
                        "rights_review_status": "UNREVIEWED",
                    },
                    protected_inputs=[manifest],
                )
            self.assertEqual(manifest.read_bytes(), before)

    def test_review_ledger_colliding_with_append_decision_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            decision = Path(tmp) / "decision.json"
            write_manifest(manifest, [scraper_row()])
            decision.write_text(
                json.dumps(
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "INSUFFICIENT_EVIDENCE",
                        "rights_review_status": "UNREVIEWED",
                    }
                ),
                encoding="utf-8",
            )
            before_m = manifest.read_bytes()
            before_d = decision.read_bytes()
            out = Path(tmp) / "out"
            with self.assertRaises(IntakeError):
                main(
                    [
                        "--manifest",
                        str(manifest),
                        "--review-ledger",
                        str(decision),
                        "--append-decision",
                        str(decision),
                        "--output-dir",
                        str(out),
                    ]
                )
            self.assertEqual(manifest.read_bytes(), before_m)
            self.assertEqual(decision.read_bytes(), before_d)

            with self.assertRaises(IntakeError):
                assert_ledger_append_path(decision, protected_inputs=[manifest, decision])
            self.assertEqual(decision.read_bytes(), before_d)

    def test_review_ledger_symlink_alias_collisions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row()])
            before_m = manifest.read_bytes()
            alias = Path(tmp) / "ledger_alias.jsonl"
            alias.symlink_to(manifest)
            decision = Path(tmp) / "decision.json"
            decision.write_text(
                json.dumps(
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "INSUFFICIENT_EVIDENCE",
                        "rights_review_status": "UNREVIEWED",
                    }
                ),
                encoding="utf-8",
            )
            before_d = decision.read_bytes()
            out = Path(tmp) / "out"
            with self.assertRaises(IntakeError):
                main(
                    [
                        "--manifest",
                        str(manifest),
                        "--review-ledger",
                        str(alias),
                        "--append-decision",
                        str(decision),
                        "--output-dir",
                        str(out),
                    ]
                )
            self.assertEqual(manifest.read_bytes(), before_m)
            self.assertEqual(decision.read_bytes(), before_d)

            # Symlink of decision used as ledger target.
            decision_alias = Path(tmp) / "decision_as_ledger.json"
            decision_alias.symlink_to(decision)
            with self.assertRaises(IntakeError):
                append_review_decision(
                    decision_alias,
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "INSUFFICIENT_EVIDENCE",
                        "rights_review_status": "UNREVIEWED",
                    },
                    protected_inputs=[manifest, decision],
                )
            self.assertEqual(decision.read_bytes(), before_d)

    def test_duplicate_record_ids_block_all_verified_admitted_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            b = Path(tmp) / "b.jsonl"
            ledger = Path(tmp) / "ledger.jsonl"
            write_manifest(
                a,
                [scraper_row(record_id="shared", stored_sha256=_sha(11))],
            )
            write_manifest(
                b,
                [scraper_row(record_id="shared", stored_sha256=_sha(22))],
            )
            # Evidence/candidate pairs that would otherwise pass for each hash.
            for sha in (_sha(11), _sha(22)):
                append_review_decision(
                    ledger,
                    {
                        "record_id": "shared",
                        "authenticity_decision": "AUTHENTIC",
                        "rights_review_status": "APPROVED",
                        "adjudication_status": "AGREED",
                        "phase55_evidence_record": phase55_eligible(sha=sha),
                        "candidate_image_record_v2": candidate_admissible(sha=sha),
                    },
                )
            result = ingest_manifests([a, b], ledger_path=ledger)
            report = build_readiness_report(result)
            self.assertEqual(result["duplicate_record_ids"], ["shared"])
            self.assertEqual(len(result["invalid"]), 2)
            self.assertEqual(report["verified_authentic_count"], 0)
            self.assertEqual(report["verified_counterfeit_count"], 0)
            self.assertEqual(report["admitted_under_existing_contracts"], 0)
            for row in result["queue"]:
                self.assertEqual(row["eligibility_decision"], "BLOCKED")
                self.assertTrue(
                    any("DUPLICATE_RECORD_ID" in e for e in row["validation_errors"])
                )
                self.assertNotIn(row.get("verified_label"), {"AUTHENTIC", "COUNTERFEIT"})

    def test_malformed_phase55_evidence_rejected_on_append_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            malformed = {
                "sample_id": "bad",
                "image_sha256": _sha(1),
                # missing required Phase 55 fields
            }
            with self.assertRaises(IntakeError) as ctx:
                append_review_decision(
                    ledger,
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "AUTHENTIC",
                        "rights_review_status": "APPROVED",
                        "adjudication_status": "AGREED",
                        "phase55_evidence_record": malformed,
                    },
                )
            self.assertIn("phase55_evidence_record", str(ctx.exception).lower())
            self.assertFalse(ledger.exists())

            # Pre-written malformed verified row must fail on load (no silent repair).
            ledger.write_text(
                json.dumps(
                    {
                        "schema_version": "phase65_review_ledger_v1",
                        "decision_id": "deadbeef",
                        "record_id": "rec_web_001",
                        "authenticity_decision": "AUTHENTIC",
                        "rights_review_status": "APPROVED",
                        "recorded_at_utc": "2026-10-14T00:00:00Z",
                        "phase55_evidence_record": malformed,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            before = ledger.read_bytes()
            with self.assertRaises(IntakeError):
                load_review_ledger(ledger)
            self.assertEqual(ledger.read_bytes(), before)

            # Conflicting label in otherwise schema-shaped evidence rejected on append.
            conflict = phase55_eligible(sha=_sha(1), label="COUNTERFEIT")
            with self.assertRaises(IntakeError):
                append_review_decision(
                    Path(tmp) / "ledger2.jsonl",
                    {
                        "record_id": "rec_web_001",
                        "authenticity_decision": "AUTHENTIC",
                        "rights_review_status": "APPROVED",
                        "adjudication_status": "AGREED",
                        "phase55_evidence_record": conflict,
                    },
                )

            # Malformed evidence must never contribute to verified/admitted counts.
            manifest = Path(tmp) / "manifest.jsonl"
            write_manifest(manifest, [scraper_row()])
            with self.assertRaises(IntakeError):
                ingest_manifests([manifest], ledger_path=ledger)


if __name__ == "__main__":
    unittest.main()
