"""Phase 46 sixth-brand protocol. No images and no checkpoints are opened."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "ml_rtx5080" / "experiments" / "sixth_brand_validation_v1"
FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


class SixthBrandProtocolTests(unittest.TestCase):
    def test_protocol_hash_matches_the_file(self):
        raw = (ROOT / "protocol_v1.json").read_bytes()
        recorded = (ROOT / "protocol_v1.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recorded)

    def test_manifest_hash_matches_and_contains_no_images(self):
        raw = (ROOT / "external_brand_manifest.json").read_bytes()
        recorded = (ROOT / "external_brand_manifest.sha256").read_text().split()[0]
        manifest = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recorded)
        self.assertEqual(manifest["samples"], [])
        self.assertFalse(manifest["inference_performed"])
        self.assertEqual(manifest["protocol_sha256"], (ROOT / "protocol_v1.sha256").read_text().split()[0])

    def test_frozen_checkpoint_identity_and_threshold_stay_fixed(self):
        protocol = _load("protocol_v1.json")
        model = protocol["frozen_model"]
        self.assertEqual(model["sha256"], FROZEN_SHA)
        self.assertEqual(protocol["checkpoint_verified_sha256"], FROZEN_SHA)
        self.assertEqual(model["temperature"], 0.24038200410185356)
        self.assertEqual(model["decision_threshold"], 0.5)
        self.assertFalse(model["retrain"])
        self.assertFalse(model["recalibrate"])
        self.assertFalse(model["new_threshold"])

    def test_excluded_brands_cover_the_catalog_and_prior_ood_sets(self):
        excluded = _load("protocol_v1.json")["excluded_brands"]
        for brand in ("A. Lange & Söhne", "Audemars Piguet", "Patek Philippe", "Richard Mille", "Vacheron Constantin"):
            self.assertIn(brand, excluded["training_catalog"])
        for brand in ("Rolex", "Omega", "Cartier"):
            self.assertIn(brand, excluded["phase35_scored_manifest"])
            self.assertIn(brand, excluded["phase36_protocol"])
        for brand in ("Jaeger-LeCoultre", "IWC", "Tudor", "Breitling"):
            self.assertIn(brand, excluded["phase36_protocol"])

    def test_blocked_status_does_not_invent_scores(self):
        status = _load("status.json")
        metrics = _load("external_brand_metrics.json")
        self.assertEqual(status["EXTERNAL_CLASSIFICATION_VALIDATION"], "BLOCKED")
        self.assertEqual(status["INDEPENDENT_IMAGES"], 0)
        self.assertIsNone(status["SIXTH_BRAND_PR_AUC"])
        self.assertIsNone(status["SIXTH_BRAND_FALSE_AUTHENTIC_RATE"])
        self.assertFalse(metrics["scoring_performed"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertIsNone(metrics["false_authentic_rate"])
        self.assertEqual((ROOT / "predictions.jsonl").read_text(), "")

    def test_richard_mille_forensics_do_not_change_the_threshold(self):
        analysis = _load("rm_false_authentic_analysis.json")
        self.assertEqual(analysis["cases_inspected"], 15)
        self.assertEqual(analysis["aggregate_false_authentic_count"], 20)
        self.assertFalse(analysis["threshold_changed"])
        self.assertFalse(analysis["checkpoint_changed"])
        self.assertFalse(analysis["duplicate_leakage_found"])
        below = [row for row in analysis["cases"] if row["p_fake"] < 0.01]
        self.assertGreaterEqual(len(below), 3)
        for row in analysis["cases"]:
            self.assertEqual(row["true_label"], 1)
            self.assertEqual(row["predicted_decision"], "AUTHENTIC")
            self.assertEqual(row["decision_threshold"], 0.5)
            self.assertIsNone(row["mahalanobis"])

    def test_final_test_isolation_flags(self):
        status = _load("status.json")
        audit = _load("data_audit.json")
        self.assertFalse(status["FINAL_TEST_TOUCHED"])
        self.assertFalse(status["MODEL_CHANGED"])
        self.assertFalse(status["CALIBRATION_CHANGED"])
        self.assertFalse(status["THRESHOLD_CHANGED"])
        self.assertFalse(status["PRODUCTION_PROMOTION_ALLOWED"])
        self.assertFalse(audit["final_test_images_opened"])
        self.assertFalse(_load("protocol_v1.json")["final_test_touched"])


if __name__ == "__main__":
    unittest.main()
