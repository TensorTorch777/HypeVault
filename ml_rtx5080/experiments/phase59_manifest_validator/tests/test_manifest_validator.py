"""Synthetic in-memory tests. No image directory or real manifest is read."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifest_validator import MembershipView, validate_manifest

SCHEMA = json.loads((ROOT / "manifest.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def _checked(**kwargs) -> MembershipView:
    return MembershipView(checked=True, **kwargs)


def _empty_membership() -> dict:
    return {
        model: {"training": _checked(), "calibration": _checked()}
        for model in ("dinov2_legacy", "dinov3_experimental")
    }


def _base(**overrides) -> dict:
    row = {
        "sample_id": "sample_SYNTH001",
        "image_sha256": "a" * 64,
        "duplicate_group_id": "group_SYNTH_001",
        "product_id": "product_SYNTH_001",
        "verified_brand": "Synthetic Brand",
        "reference_family": "Synthetic family",
        "verified_label": "AUTHENTIC",
        "evidence_record_id": "evidence_SYNTH_001",
        "source_id": "source_SYNTH_001",
        "seller_id": "seller_SYNTH_001",
        "marketplace_id": "market_SYNTH_001",
        "license_status": "APPROVED",
        "permitted_purposes": ["ML_EVALUATION"],
        "consent_status": "APPROVED",
        "retention_until": "2027-12-31",
        "deletion_status": "RETENTION_ACTIVE",
        "partition": "locked_evaluation",
        "protocol_version": "phase56_validation_protocol_v1",
        "frozen_model_ids": {
            "dinov2_legacy": "fe1daa0bf71c5e9b",
            "dinov3_experimental": "5a38c93fd442b036",
        },
        "reviewer_a_outcome": "AUTHENTIC",
        "reviewer_b_outcome": "AUTHENTIC",
        "evidence_category_a": "QUALIFIED_PHYSICAL_EXAMINATION",
        "evidence_category_b": "MANUFACTURER_PROVENANCE",
        "adjudication_status": "AGREED",
        "adjudication_outcome": "AUTHENTIC",
        "declared_disjointness": {"dinov2_legacy": "PASS", "dinov3_experimental": "PASS"},
        "eligible_primary_benchmark": True,
    }
    row.update(overrides)
    return row


def _pair() -> list[dict]:
    authentic = _base()
    counterfeit = _base(
        sample_id="sample_SYNTH002",
        image_sha256="b" * 64,
        duplicate_group_id="group_SYNTH_002",
        product_id="product_SYNTH_002",
        verified_label="COUNTERFEIT",
        evidence_record_id="evidence_SYNTH_002",
        reviewer_a_outcome="COUNTERFEIT",
        reviewer_b_outcome="COUNTERFEIT",
        adjudication_outcome="COUNTERFEIT",
    )
    return [authentic, counterfeit]


def _pending() -> dict:
    return _base(
        sample_id="sample_SYNTHPEND",
        product_id="UNKNOWN",
        verified_brand="UNKNOWN",
        verified_label="INDETERMINATE",
        evidence_record_id="UNKNOWN",
        source_id="UNKNOWN",
        seller_id="UNKNOWN",
        marketplace_id="UNKNOWN",
        license_status="NOT_RECORDED",
        permitted_purposes=[],
        consent_status="NOT_RECORDED",
        retention_until="UNKNOWN",
        deletion_status="NOT_RECORDED",
        reviewer_a_outcome="NOT_REVIEWED",
        reviewer_b_outcome="NOT_REVIEWED",
        evidence_category_a="NOT_RECORDED",
        evidence_category_b="NOT_RECORDED",
        adjudication_status="NOT_STARTED",
        adjudication_outcome="NOT_ADJUDICATED",
        declared_disjointness={"dinov2_legacy": "UNKNOWN", "dinov3_experimental": "UNKNOWN"},
        eligible_primary_benchmark=False,
    )


class ManifestValidatorTests(unittest.TestCase):
    def test_valid_synthetic_manifest_passes_and_is_schema_valid(self) -> None:
        records = _pair()
        for record in records:
            self.assertEqual(list(VALIDATOR.iter_errors(record)), [])
        result = validate_manifest(records, _empty_membership())
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual({item["eligibility"] for item in result["records"]}, {"PASS"})
        self.assertFalse(result["acquisition_authorized"])
        self.assertFalse(result["inference_authorized"])

    def test_pending_record_is_valid_and_not_eligible(self) -> None:
        record = _pending()
        self.assertEqual(list(VALIDATOR.iter_errors(record)), [])
        result = validate_manifest([record], _empty_membership())
        self.assertEqual(result["records"][0]["eligibility"], "NOT_ELIGIBLE")
        self.assertNotEqual(result["records"][0]["eligibility"], "PASS")
        self.assertFalse(any(item["status"] == "FAIL" for item in result["findings"]))

    def test_product_and_duplicate_leakage_between_partitions(self) -> None:
        first, second = _pair()
        second["partition"] = "training"
        second["product_id"] = first["product_id"]
        product = validate_manifest([first, second], _empty_membership())
        self.assertTrue(any(item["field"] == "product_id" and item["status"] == "FAIL" for item in product["findings"]))
        copy = _base(sample_id="sample_SYNTH003", product_id="product_SYNTH_003", duplicate_group_id="group_SYNTH_003")
        copy["partition"] = "training"
        copy["image_sha256"] = first["image_sha256"]
        hashed = validate_manifest([first, copy], _empty_membership())
        self.assertTrue(any(item["field"] == "image_sha256" and item["status"] == "FAIL" for item in hashed["findings"]))
        grouped = _base(sample_id="sample_SYNTH004", product_id="product_SYNTH_004", image_sha256="c" * 64)
        grouped["partition"] = "calibration"
        grouped["duplicate_group_id"] = first["duplicate_group_id"]
        groups = validate_manifest([first, grouped], _empty_membership())
        self.assertTrue(any(item["field"] == "duplicate_group_id" and item["status"] == "FAIL" for item in groups["findings"]))

    def test_model_overlaps_and_unknown_membership(self) -> None:
        records = _pair()
        dinov2 = _empty_membership()
        dinov2["dinov2_legacy"]["training"] = _checked(product_ids={"product_SYNTH_001"})
        dinov2_result = validate_manifest(records, dinov2)
        self.assertTrue(any("dinov2_legacy training" in item["reason"] and item["status"] == "FAIL" for item in dinov2_result["findings"]))
        dinov3 = _empty_membership()
        dinov3["dinov3_experimental"]["calibration"] = _checked(source_ids={"source_SYNTH_001"})
        dinov3_result = validate_manifest(records, dinov3)
        self.assertTrue(any("dinov3_experimental calibration" in item["reason"] and item["status"] == "FAIL" for item in dinov3_result["findings"]))
        unknown = _empty_membership()
        unknown["dinov2_legacy"]["training"] = MembershipView(checked=False)
        unknown_result = validate_manifest(records, unknown)
        self.assertEqual(unknown_result["overall"], "UNKNOWN")
        self.assertNotEqual(unknown_result["overall"], "PASS")

    def test_invalid_label_reviewer_rights_and_missing_identity(self) -> None:
        bad_label = _base(verified_label="FAKE")
        self.assertTrue(list(VALIDATOR.iter_errors(bad_label)))
        self.assertEqual(validate_manifest([bad_label], _empty_membership())["records"][0]["eligibility"], "FAIL")
        bad_reviewer = _base(reviewer_a_outcome="FAKE")
        self.assertEqual(validate_manifest([bad_reviewer], _empty_membership())["records"][0]["eligibility"], "FAIL")
        missing_product = _base(product_id="UNKNOWN")
        missing = validate_manifest([missing_product], _empty_membership())
        self.assertTrue(any(item["field"] == "product_id" and item["status"] == "FAIL" for item in missing["findings"]))
        bad_rights = _base(deletion_status="NOT_RECORDED", retention_until="UNKNOWN", license_status="NOT_RECORDED")
        rights = validate_manifest([bad_rights], _empty_membership())
        fields = {item["field"] for item in rights["findings"] if item["status"] == "FAIL"}
        self.assertIn("deletion_status", fields)
        self.assertIn("retention_until", fields)
        self.assertIn("license_status", fields)

    def test_seller_overlap_is_review_not_pass_and_brand_class_gap_fails(self) -> None:
        membership = _empty_membership()
        membership["dinov2_legacy"]["training"] = _checked(seller_ids={"seller_SYNTH_001"})
        overlap = validate_manifest(_pair(), membership)
        self.assertTrue(any(item["field"] == "seller_id" and item["status"] == "UNKNOWN" for item in overlap["findings"]))
        self.assertEqual(overlap["overall"], "UNKNOWN")
        only_authentic = [_base(), _base(sample_id="sample_SYNTH005", product_id="product_SYNTH_005", image_sha256="d" * 64, duplicate_group_id="group_SYNTH_005")]
        gap = validate_manifest(only_authentic, _empty_membership())
        self.assertTrue(any("COUNTERFEIT" in item["reason"] and item["status"] == "FAIL" for item in gap["findings"]))
        self.assertEqual(gap["overall"], "FAIL")


if __name__ == "__main__":
    unittest.main()
