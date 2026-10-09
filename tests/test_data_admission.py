"""candidate_image_record_v2 admission. Synthetic records only; no images are read."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))

SCHEMA = REPO / "docs" / "data_governance" / "candidate_image_record_v2.schema.json"
TODAY = date(2026, 10, 9)


def _record() -> dict:
    return {
        "record_id": "synthetic-1",
        "image": {
            "sha256": "a" * 64,
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
            "chain_of_custody": [{"step": "captured", "actor_ref": "examiner-ref-1", "date": "2026-09-01"}],
        },
        "rights": {
            "permission_id": "perm-1",
            "permission_status": "EFFECTIVE",
            "research_use": "GRANTED",
            "training_use": "GRANTED",
            "public_display": "DENIED",
            "retention_until": "2028-12-31",
        },
        "brand": {"value": "Patek Philippe", "basis": "case-back engraving and movement calibre", "evidence_id": "ev-1"},
        "label": {
            "value": "authentic",
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


class DataAdmissionTests(unittest.TestCase):
    def test_schema_requires_every_e3_field(self) -> None:
        schema = json.loads(SCHEMA.read_text())
        props = schema["properties"]
        self.assertEqual(set(schema["required"]), {"record_id", "image", "source", "rights", "brand", "label", "disposition"})
        self.assertTrue({"sha256", "uri", "width", "height", "derivative_of", "duplicate_status"} <= set(props["image"]["required"]))
        self.assertTrue({"source_reference", "product_or_listing_id", "source_group_id", "chain_of_custody"} <= set(props["source"]["required"]))
        self.assertTrue({"permission_id", "research_use", "training_use"} <= set(props["rights"]["required"]))
        self.assertTrue({"method", "verifier_ref", "evidence_id", "evidence_date"} <= set(props["label"]["required"]))
        self.assertIn("UNKNOWN", props["label"]["properties"]["value"]["enum"])
        self.assertNotIn("verifier_name", props["label"]["properties"])

    def test_complete_record_is_admitted(self) -> None:
        from data_admission import admission_decision

        self.assertEqual(admission_decision(_record(), purpose="training", today=TODAY), ("ADMIT", []))

    def test_unknown_or_forbidden_evidence_is_never_admitted(self) -> None:
        from data_admission import admission_decision

        cases = {
            "LABEL_NOT_VERIFIED": lambda r: r["label"].update(value="UNKNOWN"),
            "VERIFICATION_METHOD_NOT_ADMISSIBLE": lambda r: r["label"].update(method=["directory_name"]),
            "SOURCE_NOT_INDEPENDENT": lambda r: r["source"].update(source_type="UNKNOWN"),
            "PERMISSION_NOT_EFFECTIVE": lambda r: r["rights"].update(permission_status="NOT_EFFECTIVE"),
            "RESEARCH_USE_NOT_GRANTED": lambda r: r["rights"].update(research_use="UNKNOWN"),
            "BRAND_BASIS_NOT_ADMISSIBLE": lambda r: r["brand"].update(basis="wordmark_only"),
            "VERIFIER_NOT_INDEPENDENT": lambda r: r["label"].update(verifier_independent=False),
            "LABEL_CONFIDENCE_NOT_CERTAIN": lambda r: r["label"].update(confidence="probable"),
            "DUPLICATE_STATUS_NOT_UNIQUE": lambda r: r["image"].update(duplicate_status="UNKNOWN"),
            "SOURCE_SOURCE_GROUP_ID_MISSING": lambda r: r["source"].update(source_group_id="UNKNOWN"),
            "RETENTION_EXPIRED": lambda r: r["rights"].update(retention_until="2026-01-01"),
            "DISPOSITION_HOLD_DISPUTED": lambda r: r["disposition"].update(status="HOLD_DISPUTED"),
        }
        for expected, mutate in cases.items():
            record = copy.deepcopy(_record())
            mutate(record)
            decision, reasons = admission_decision(record, today=TODAY)
            self.assertEqual(decision, "REJECT", expected)
            self.assertIn(expected, reasons)
        for method in ("model_prediction", "seller_claim", "listing_label", "jpeg_quantization", "image_geometry", "photo_only"):
            record = copy.deepcopy(_record())
            record["label"]["method"] = ["movement_inspection", method]
            self.assertEqual(admission_decision(record, today=TODAY)[0], "REJECT", method)

    def test_training_needs_training_permission(self) -> None:
        from data_admission import admission_decision

        record = _record()
        record["rights"]["training_use"] = "UNKNOWN"
        self.assertEqual(admission_decision(record, purpose="evaluation", today=TODAY), ("ADMIT", []))
        decision, reasons = admission_decision(record, purpose="training", today=TODAY)
        self.assertEqual(decision, "REJECT")
        self.assertEqual(reasons, ["TRAINING_USE_NOT_GRANTED"])

    def test_admission_does_not_rewrite_the_label(self) -> None:
        from data_admission import admission_decision

        record = _record()
        record["label"]["value"] = "undetermined"
        before = copy.deepcopy(record)
        admission_decision(record, today=TODAY)
        self.assertEqual(record, before)


if __name__ == "__main__":
    unittest.main()
