"""Publication safety.

Research results never publish. Listing photos are screened (size, wristwatch present,
original photograph) and either listed or rejected with a reason; nothing is ever
presented as authenticity-verified.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))


class PublicationSafetyTests(unittest.TestCase):
    def test_authenticity_publication_stays_blocked(self) -> None:
        from inference.publication_gate import AUTHENTICITY_MODEL_PRODUCTION_APPROVED, publication_decision

        self.assertFalse(AUTHENTICITY_MODEL_PRODUCTION_APPROVED)
        decision = publication_decision(
            model="LEGACY_DINOV2",
            model_status="LEGACY",
            production_validation="NOT_ESTABLISHED",
            model_decision="AUTHENTIC",
            policy_status="OK",
            listing_state="pending",
        )
        self.assertEqual(decision, "BLOCKED")
        self.assertNotIn(decision, {"AUTHENTIC", "FAKE", "LIVE"})
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        self.assertNotIn("ListingStatus.live", routes)
        self.assertIn("check_content", routes)
        self.assertLess(routes.index("check_content"), routes.index("classify_listing_photo_with_hold(pil)"))

    def test_fake_error_and_research_results_block_publication(self) -> None:
        from inference.publication_gate import publication_decision
        from inference.research_access import research_success_body
        from inference.scope_gate import failure_body, unsupported_scope_body

        cases = [
            {"model": "LEGACY_DINOV2", "model_status": "LEGACY", "production_validation": "NOT_ESTABLISHED", "model_decision": "FAKE", "policy_status": "OK", "listing_state": "pending"},
            {"model": "LEGACY_DINOV2", "model_status": "LEGACY", "production_validation": "NOT_ESTABLISHED", "model_decision": None, "policy_status": "MODEL_ERROR", "listing_state": "pending"},
            {"model": "LEGACY_DINOV2", "model_status": "LEGACY", "production_validation": "NOT_ESTABLISHED", "model_decision": None, "policy_status": "POLICY_ERROR", "listing_state": "pending"},
            {"model": "DINOV3_RESEARCH_PROTOTYPE", "model_status": "RESEARCH", "production_validation": "NOT_ESTABLISHED", "model_decision": "AUTHENTIC", "policy_status": "OK", "listing_state": "pending"},
            {"model": "LEGACY_DINOV2", "model_status": "LEGACY", "production_validation": "NOT_ESTABLISHED", "model_decision": None, "policy_status": "UNSUPPORTED_SCOPE", "listing_state": "pending"},
            {"model": None, "model_status": None, "production_validation": None, "model_decision": "AUTHENTIC", "policy_status": "OK", "listing_state": "pending"},
        ]
        for case in cases:
            self.assertEqual(publication_decision(**case), "BLOCKED")

        for body in (
            failure_body("MODEL_ERROR", "The authenticity model did not return a result."),
            failure_body("POLICY_ERROR", "blocked"),
            unsupported_scope_body(),
        ):
            self.assertIsNone(body["decision"])
            self.assertEqual(body["publication_decision"], "BLOCKED")
            self.assertNotIn(body["status"], {"AUTHENTIC", "FAKE"})

        research = research_success_body(
            decision="AUTHENTIC",
            declared_brand="Patek Philippe",
            checkpoint_sha="5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
            policy_version="shadow_v1",
        )
        self.assertEqual(research["publication_decision"], "BLOCKED")
        self.assertNotIn("5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f", (REPO / "backend" / "inference" / "routes.py").read_text())

    def test_seller_update_cannot_set_publication_status(self) -> None:
        from listings.models import ListingUpdate

        self.assertNotIn("status", ListingUpdate.model_fields)
        self.assertNotIn("verdict", ListingUpdate.model_fields)
        research = (REPO / "backend" / "inference" / "research_routes.py").read_text()
        self.assertNotIn("ListingStatus", research)
        for name in ("scripts/seed_database.py", "scripts/import_chrono24_csv.py"):
            text = (REPO / name).read_text()
            self.assertNotIn("ListingStatus.live", text)
            self.assertNotIn('verdict="AUTHENTIC"', text)

    def test_publication_artifacts_exist(self) -> None:
        root = REPO / "ml_rtx5080" / "experiments" / "publication_safety_v1"
        for name in (
            "publication_flow_audit.json",
            "publication_policy.json",
            "listing_state_machine.json",
            "publication_gate_report.md",
            "legacy_verdict_risk_register.json",
            "publication_safety_test_results.json",
        ):
            self.assertTrue((root / name).is_file(), name)

    def test_live_snapshot_hash_matches_the_file(self) -> None:
        import hashlib

        root = REPO / "ml_rtx5080" / "experiments" / "publication_safety_v1"
        raw = (root / "legacy_live_listing_snapshot.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        recorded = (root / "legacy_live_listing_snapshot.sha256").read_text().split()[0]
        self.assertEqual(recorded, digest)


if __name__ == "__main__":
    unittest.main()
