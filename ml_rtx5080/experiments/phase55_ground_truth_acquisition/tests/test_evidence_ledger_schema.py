"""Schema contract tests using synthetic metadata only; no images or private evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "evidence_ledger.schema.json"


def pending_record() -> dict:
    return {
        "sample_id": "sample_SYNTHETIC_001",
        "image_sha256": "a" * 64,
        "product_id": "UNKNOWN",
        "declared_brand": "Patek Philippe",
        "verified_brand": "UNKNOWN",
        "reference_family": "UNKNOWN",
        "source_id": "UNKNOWN",
        "seller_id": "UNKNOWN",
        "marketplace_id": "UNKNOWN",
        "historical_directory_label": "Label_0_Watches",
        "reviewer_a_outcome": "NOT_REVIEWED",
        "reviewer_b_outcome": "NOT_REVIEWED",
        "evidence_category_a": "NOT_RECORDED",
        "evidence_category_b": "NOT_RECORDED",
        "adjudication_status": "NOT_STARTED",
        "adjudication_outcome": "NOT_ADJUDICATED",
        "verified_label": "INDETERMINATE",
        "license_status": "NOT_RECORDED",
        "consent_status": "NOT_RECORDED",
        "permitted_purposes": [],
        "retention_until": "UNKNOWN",
        "deletion_status": "NOT_RECORDED",
        "source_product_disjointness": {
            "dinov2_legacy": "UNKNOWN",
            "dinov3_experimental": "UNKNOWN",
        },
        "eligible_primary_benchmark": False,
    }


def eligible_record() -> dict:
    row = pending_record()
    row.update({
        "sample_id": "sample_SYNTHETIC_002",
        "product_id": "product_SYNTHETIC_001",
        "verified_brand": "Patek Philippe",
        "reference_family": "Synthetic reference family",
        "source_id": "source_SYNTHETIC_001",
        "reviewer_a_outcome": "AUTHENTIC",
        "reviewer_b_outcome": "AUTHENTIC",
        "evidence_category_a": "QUALIFIED_PHYSICAL_EXAMINATION",
        "evidence_category_b": "MANUFACTURER_PROVENANCE",
        "adjudication_status": "AGREED",
        "adjudication_outcome": "AUTHENTIC",
        "verified_label": "AUTHENTIC",
        "license_status": "APPROVED",
        "consent_status": "APPROVED",
        "permitted_purposes": ["ML_EVALUATION", "NONPUBLIC_RESEARCH"],
        "retention_until": "2027-12-31",
        "deletion_status": "RETENTION_ACTIVE",
        "source_product_disjointness": {
            "dinov2_legacy": "PASS",
            "dinov3_experimental": "PASS",
        },
        "eligible_primary_benchmark": True,
    })
    return row


class EvidenceLedgerSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def assert_valid(self, row: dict) -> None:
        errors = list(self.validator.iter_errors(row))
        self.assertEqual(errors, [], "\n".join(error.message for error in errors))

    def assert_invalid(self, row: dict) -> None:
        self.assertTrue(list(self.validator.iter_errors(row)))

    def test_synthetic_pending_metadata_is_valid_and_not_eligible(self) -> None:
        row = pending_record()
        self.assertFalse(row["eligible_primary_benchmark"])
        self.assert_valid(row)

    def test_required_fields_are_enforced(self) -> None:
        row = pending_record()
        del row["verified_label"]
        self.assert_invalid(row)

    def test_unknown_or_invalid_outcome_cannot_be_verified_label(self) -> None:
        row = pending_record()
        row["verified_label"] = "FAKE"
        self.assert_invalid(row)

    def test_directory_label_alone_cannot_make_record_eligible(self) -> None:
        row = pending_record()
        row["eligible_primary_benchmark"] = True
        self.assert_invalid(row)

    def test_eligible_record_requires_independent_evidence_rights_and_disjointness(self) -> None:
        self.assert_valid(eligible_record())

    def test_eligibility_fails_without_model_disjointness(self) -> None:
        row = eligible_record()
        row["source_product_disjointness"]["dinov3_experimental"] = "UNKNOWN"
        self.assert_invalid(row)

    def test_eligibility_fails_without_ml_evaluation_license_purpose(self) -> None:
        row = eligible_record()
        row["permitted_purposes"] = ["NONPUBLIC_RESEARCH"]
        self.assert_invalid(row)

    def test_historical_directory_label_does_not_override_verified_label(self) -> None:
        row = eligible_record()
        row["historical_directory_label"] = "Label_1_Watches"
        # Changing the historical label must not change the explicit evidence-backed outcome.
        self.assertEqual(row["verified_label"], "AUTHENTIC")
        self.assert_valid(row)

    def test_reviewer_identity_or_image_bytes_are_not_allowed(self) -> None:
        row = pending_record()
        row["reviewer_name"] = "Synthetic Person"
        self.assert_invalid(row)
        row = pending_record()
        row["image_bytes"] = "not real bytes"
        self.assert_invalid(row)

    def test_hash_format_is_enforced(self) -> None:
        row = pending_record()
        row["image_sha256"] = "not-a-hash"
        self.assert_invalid(row)

    def test_retention_date_must_be_a_real_date(self) -> None:
        row = pending_record()
        row["retention_until"] = "2027-99-77"
        self.assert_invalid(row)


if __name__ == "__main__":
    unittest.main()
