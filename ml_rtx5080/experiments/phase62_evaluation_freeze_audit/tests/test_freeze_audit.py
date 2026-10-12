"""Synthetic freeze-audit tests. No image or protected manifest is opened."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE59 = ROOT.parent / "phase59_manifest_validator"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHASE59))

from freeze_audit import audit_synthetic_manifest
from manifest_validator import MembershipView

CHECKSUMS = ROOT / "manifest_checksums.json"


def _checked(**kwargs) -> MembershipView:
    return MembershipView(checked=True, **kwargs)


def _empty_membership() -> dict:
    return {
        model: {"training": _checked(), "calibration": _checked()}
        for model in ("dinov2_legacy", "dinov3_experimental")
    }


def _row(**overrides) -> dict:
    row = {
        "sample_id": "sample_SYNTH001",
        "image_sha256": "a" * 64,
        "duplicate_group_id": "group_SYNTH_001",
        "product_id": "product_SYNTH_001",
        "verified_brand": "Synthetic Brand",
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
    return [
        _row(),
        _row(
            sample_id="sample_SYNTH002",
            image_sha256="b" * 64,
            duplicate_group_id="group_SYNTH_002",
            product_id="product_SYNTH_002",
            verified_label="COUNTERFEIT",
            evidence_record_id="evidence_SYNTH_002",
            reviewer_a_outcome="COUNTERFEIT",
            reviewer_b_outcome="COUNTERFEIT",
            adjudication_outcome="COUNTERFEIT",
        ),
    ]


class FreezeAuditTests(unittest.TestCase):
    def test_synthetic_pair_is_not_a_frozen_dataset(self) -> None:
        report = audit_synthetic_manifest(_pair(), _empty_membership())
        self.assertEqual(report["report_class"], "SYNTHETIC_DEMO")
        self.assertEqual(report["dataset_status"], "NOT_FROZEN")
        self.assertEqual(report["overall"], "PASS")
        self.assertEqual(report["eligibility_counts"]["PASS"], 2)
        self.assertEqual(report["data_readiness"], "NO_GO_FOR_AUTHENTICITY_CLAIMS")
        self.assertFalse(report["acquisition_authorized"])
        self.assertFalse(report["inference_authorized"])
        self.assertFalse(report["minimum_counts_approved"])
        self.assertEqual(report, json.loads(CHECKSUMS.read_text(encoding="utf-8")))

    def test_unchecked_membership_stays_unknown(self) -> None:
        membership = {
            "dinov2_legacy": {"training": MembershipView(checked=False), "calibration": _checked()},
            "dinov3_experimental": {"training": _checked(), "calibration": _checked()},
        }
        report = audit_synthetic_manifest(_pair(), membership)
        self.assertEqual(report["overall"], "UNKNOWN")
        self.assertIn("MODEL_MEMBERSHIP_UNCHECKED", report["exclusion_reason_codes"])
        self.assertEqual(report["real_cohort"], "NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION")

    def test_brand_gap_and_path_refusal(self) -> None:
        report = audit_synthetic_manifest([_row()], _empty_membership())
        self.assertEqual(report["overall"], "FAIL")
        self.assertIn("BRAND_MISSING_CLASS", report["exclusion_reason_codes"])
        with self.assertRaises(TypeError):
            audit_synthetic_manifest("ml_rtx5080/data", _empty_membership())
        with self.assertRaises(ValueError):
            audit_synthetic_manifest([_row(image_path="/tmp/watch.jpg")], _empty_membership())
