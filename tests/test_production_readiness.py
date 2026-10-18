"""Phase 34 readiness rules. The audit reads artifacts and does not open images."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))


def _ready_evidence() -> dict:
    return {
        "final_test_complete": True,
        "calibration_complete": True,
        "robustness_audit_complete": True,
        "real_ood_validation_complete": True,
        "production_runtime_parity_complete": True,
        "production_serving_infrastructure_available": True,
        "no_unresolved_critical_failure": True,
        "quality_review_workload_accepted": True,
        "policy_formally_approved": True,
        "ood_explicitly_not_required": False,
        "model_quality_strong": True,
        "shadow_path_stable": True,
        "limitations_explicit": True,
        "model_parity_failed": False,
        "checkpoint_mismatch": False,
        "safety_constraint_failed": False,
        "production_runtime_unsafe": False,
        "fail_closed_failed": False,
    }


class ProductionReadinessTests(unittest.TestCase):
    def test_complete_evidence_is_ready(self) -> None:
        from production_readiness import READY, classify_readiness

        decision = classify_readiness(_ready_evidence())
        self.assertEqual(decision["status"], READY)
        self.assertEqual(decision["missing_requirements"], [])

    def test_pending_ood_blocks_ready_when_required(self) -> None:
        from production_readiness import CONDITIONAL, READY, classify_readiness

        evidence = _ready_evidence()
        evidence["real_ood_validation_complete"] = False
        decision = classify_readiness(evidence)
        self.assertNotEqual(decision["status"], READY)
        self.assertEqual(decision["status"], CONDITIONAL)
        self.assertIn("real_ood_validation_complete", decision["missing_requirements"])

    def test_missing_runtime_blocks_ready(self) -> None:
        from production_readiness import READY, classify_readiness

        evidence = _ready_evidence()
        evidence["production_serving_infrastructure_available"] = False
        evidence["production_runtime_parity_complete"] = False
        decision = classify_readiness(evidence)
        self.assertNotEqual(decision["status"], READY)
        self.assertIn("production_serving_infrastructure_available", decision["missing_requirements"])

    def test_missing_final_test_or_calibration_is_not_ready(self) -> None:
        from production_readiness import NOT_READY, classify_readiness

        for field in ("final_test_complete", "calibration_complete"):
            evidence = _ready_evidence()
            evidence[field] = False
            decision = classify_readiness(evidence)
            self.assertEqual(decision["status"], NOT_READY, field)

    def test_unsafe_runtime_is_not_ready(self) -> None:
        from production_readiness import NOT_READY, classify_readiness

        evidence = _ready_evidence()
        evidence["production_runtime_unsafe"] = True
        evidence["fail_closed_failed"] = True
        decision = classify_readiness(evidence)
        self.assertEqual(decision["status"], NOT_READY)
        self.assertIn("production_runtime_unsafe", decision["unsafe"])

    def test_checkpoint_and_policy_status(self) -> None:
        from production_readiness import CHECKPOINT_SHA256, TEMPERATURE

        self.assertEqual(CHECKPOINT_SHA256, "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(TEMPERATURE, 0.24038200410185356)
        report_path = REPO / "ml_rtx5080" / "experiments" / "production_readiness_v1" / "readiness_report.json"
        if not report_path.is_file():
            from production_readiness import run_phase34

            run_phase34()
        report = json.loads(report_path.read_text())
        self.assertEqual(report["checkpoint_sha256"], CHECKPOINT_SHA256)
        self.assertEqual(report["membership_hash"], "78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678")
        self.assertEqual(report["manifest_file_sha256"], "4d0d20c17d12ac29cb299b7b4f369ebcce3c5acd7b2fd5430d61f952b9852eea")
        self.assertNotEqual(report["membership_hash"], report["manifest_file_sha256"])
        self.assertEqual(report["current_production_status"], "not_promoted")
        self.assertFalse(report["promotion_performed"])
        self.assertEqual(report["final_test_samples_used"], 0)

    def test_risk_register_schema_and_workload(self) -> None:
        from production_readiness import REVIEW_RATE, RISK_FIELDS, expected_review_volume

        self.assertAlmostEqual(expected_review_volume(100), 100 * REVIEW_RATE)
        self.assertAlmostEqual(expected_review_volume(1000), 1000 * REVIEW_RATE)
        self.assertAlmostEqual(expected_review_volume(10000), 10000 * REVIEW_RATE)
        self.assertAlmostEqual(REVIEW_RATE, 1855 / 26994)
        path = REPO / "ml_rtx5080" / "experiments" / "production_readiness_v1" / "risk_register.csv"
        if not path.is_file():
            from production_readiness import run_phase34

            run_phase34()
        header = path.read_text().splitlines()[0].split(",")
        self.assertEqual(tuple(header), RISK_FIELDS)
        self.assertGreaterEqual(len(path.read_text().splitlines()), 11)

    def test_audit_does_not_open_images_and_status_is_calculated(self) -> None:
        from PIL import Image
        from production_readiness import CONDITIONAL, classify_readiness, load_evidence

        opened = {"called": False}
        original = Image.open

        def guard(*_args, **_kwargs):
            opened["called"] = True
            raise AssertionError("an image was opened during the audit")

        Image.open = guard
        try:
            evidence = load_evidence()
        finally:
            Image.open = original
        self.assertFalse(opened["called"])
        decision = classify_readiness(evidence)
        self.assertEqual(decision["status"], CONDITIONAL)
        self.assertIn("real_ood_validation_complete", decision["missing_requirements"])
        self.assertFalse(evidence["production_runtime_ready"])
        self.assertTrue(evidence["shadow_ready"])
        self.assertIsNone(evidence["production_ood_threshold"])


if __name__ == "__main__":
    unittest.main()
