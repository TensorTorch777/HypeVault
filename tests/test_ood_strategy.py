"""Phase 38 feasibility review. The protocol file and the model stay frozen."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "ml_rtx5080" / "experiments" / "ood_strategy_v1"
PROTOCOL = REPO / "ml_rtx5080" / "experiments" / "watch_like_ood_v1" / "watch_like_ood_protocol_v1.json"
PROTOCOL_SHA = "babaaf2b5c30d252c39765886ee5db032c09c4e3f25cf0e9eacc63f17128dde9"


class OODStrategyTests(unittest.TestCase):
    def test_protocol_file_is_unchanged(self) -> None:
        digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        self.assertEqual(digest, PROTOCOL_SHA)

    def test_recommendation_blocks_promotion(self) -> None:
        payload = json.loads((REVIEW / "recommendation.json").read_text())
        self.assertEqual(payload["PROTOCOL_DECISION"], "SCOPE_REDUCTION_RECOMMENDED")
        self.assertTrue(payload["SCIENTIFIC_FEASIBILITY"])
        self.assertFalse(payload["DATA_GOVERNANCE_FEASIBILITY"])
        self.assertEqual(payload["RECOMMENDED_SCOPE"], "five_brand_authenticity_research_prototype")
        self.assertEqual(payload["PHASE35_STATUS"], "OOD_FAIL")
        self.assertEqual(payload["PHASE36_STATUS"], "WATCH_OOD_UNRESOLVED")
        self.assertIsNone(payload["PRODUCTION_OOD_THRESHOLD"])
        self.assertFalse(payload["PROMOTION_ALLOWED"])
        self.assertFalse(payload["MODEL_CHANGED"])
        self.assertFalse(payload["CALIBRATION_CHANGED"])
        self.assertFalse(payload["FINAL_TEST_TOUCHED"])
        self.assertEqual(payload["protocol_sha256"], PROTOCOL_SHA)
        self.assertFalse(payload["protocol_v1_edited"])

    def test_review_does_not_adopt_a_silent_protocol_edit(self) -> None:
        review = json.loads((REVIEW / "protocol_feasibility_review_v1.json").read_text())
        self.assertEqual(review["current_protocol_sha256"], PROTOCOL_SHA)
        self.assertFalse(review["protocol_file_edited"])
        self.assertEqual(review["byte_cap_classification"], "weakly_justified")
        self.assertEqual(review["PROTOCOL_DECISION"], "SCOPE_REDUCTION_RECOMMENDED")
        self.assertTrue(all(item["adopted"] is False for item in review["proposed_changes_not_adopted"]))
        route_c = review["routes"]["C_copyrighted_research_only"]
        self.assertEqual(route_c["LEGAL_BASIS"], "UNKNOWN")
        self.assertEqual(route_c["classification"], "REQUIRES_INSTITUTIONAL_APPROVAL")
        self.assertFalse(route_c["proceed"])

    def test_image_licenses_are_not_upgraded_from_unknown(self) -> None:
        matrix = json.loads((REVIEW / "license_matrix.json").read_text())
        allowed = {"explicit", "inherited", "unknown", "restricted", "unavailable"}
        self.assertTrue(matrix["sources"])
        for source in matrix["sources"]:
            self.assertIn(source["image_license"], allowed)
            self.assertIn(source["metadata_license"], allowed)
            if source["image_license"] == "unknown":
                self.assertNotIn("CC0", source["image_license_name"])

    def test_ranking_does_not_treat_quantity_as_sufficient(self) -> None:
        lines = (REVIEW / "source_ranking.csv").read_text().splitlines()
        header = lines[0].split(",")
        self.assertIn("meets_registered_package", header)
        package_index = header.index("meets_registered_package")
        self.assertGreater(len(lines), 1)
        for line in lines[1:]:
            self.assertEqual(line.split(",")[package_index], "false")

    def test_sample_size_review_keeps_phase36_underpowered(self) -> None:
        analysis = json.loads((REVIEW / "sample_size_analysis.json").read_text())
        self.assertTrue(analysis["do_not_lower_minimums"])
        by_name = {row["name"]: row for row in analysis["observed"]}
        phase36 = by_name["phase36_unseen_holdout"]
        self.assertEqual(phase36["n"], 14)
        self.assertGreater(phase36["wilson_high"] - phase36["wilson_low"], 0.3)
        phase35 = by_name["phase35_unseen_luxury"]
        self.assertEqual(phase35["n"], 194)
        self.assertGreater(phase35["wilson_low"], 0.8)


if __name__ == "__main__":
    unittest.main()
