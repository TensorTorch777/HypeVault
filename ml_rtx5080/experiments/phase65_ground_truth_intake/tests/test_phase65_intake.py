"""Offline Phase 65 intake tests. Synthetic metadata only; no images or network."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
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
    append_review_decision,
    assess_review_against_contracts,
    assert_metadata_only_path,
    build_readiness_report,
    ingest_manifests,
    main,
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
            self.assertTrue(any("DUPLICATE_RECORD_ID" in e for e in result["invalid"][0]["queue"]["validation_errors"]))
            write_manifest(a, [scraper_row(record_id="one", stored_sha256=_sha(7))])
            write_manifest(b, [scraper_row(record_id="two", stored_sha256=_sha(7))])
            result = ingest_manifests([a, b])
            self.assertIn(_sha(7), result["duplicate_hashes"])
            self.assertEqual(sorted(result["duplicate_hashes"][_sha(7)]), ["one", "two"])

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
            },
        )
        self.assertTrue(strong["authenticity_verified"])
        self.assertTrue(strong["ml_use_eligible"])

    def test_verified_label_does_not_imply_ml_eligibility_without_rights(self) -> None:
        record = scraper_row()
        assessment = assess_review_against_contracts(
            record,
            {
                "authenticity_decision": "AUTHENTIC",
                "rights_review_status": "UNREVIEWED",
                "adjudication_status": "AGREED",
                "phase55_evidence_record": phase55_eligible(sha=_sha(1)),
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
            # Ensure intake never opens a referenced .jpg during a normal run.
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


if __name__ == "__main__":
    unittest.main()
