"""Phase 44 customer labels. Live is publication state, not authenticity verification."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

LEGACY = "Legacy screening — not verified"
DEMO = "Demo listing — not verified"
FORBIDDEN = ("Verified Authentic", "Authenticated", "Certified", "Guaranteed Authentic")


class CustomerLabelTests(unittest.TestCase):
    def test_legacy_image_backed_listing_stays_live_and_unverified(self) -> None:
        from listings.customer_label import customer_label, evidence_role

        label = customer_label(status="live", verdict="AUTHENTIC", confidence=0.992304, s3_url="listings/example.jpg")
        self.assertEqual(label, LEGACY)
        self.assertEqual(
            evidence_role(verdict="AUTHENTIC", confidence=0.992304, s3_url="listings/example.jpg"),
            "HISTORICAL_MODEL_EVIDENCE",
        )

    def test_seed_listing_stays_live_and_is_not_a_model_prediction(self) -> None:
        from database import ListingCategory, ListingStatus
        from listings.customer_label import customer_label, evidence_role
        from listings.models import ListingRead

        label = customer_label(status="live", verdict="AUTHENTIC", confidence=0.965, s3_url=None)
        self.assertEqual(label, DEMO)
        self.assertEqual(evidence_role(verdict="AUTHENTIC", confidence=0.965, s3_url=None), "SEED_CONSTANT")
        row = ListingRead(
            id=uuid4(),
            seller_id=uuid4(),
            product_name="Patek Philippe Nautilus 5711",
            category=ListingCategory.watch,
            brand="Patek Philippe",
            condition="Excellent",
            size="40mm",
            s3_url=None,
            verdict="AUTHENTIC",
            confidence=0.965,
            status=ListingStatus.live,
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(row.status.value, "live")
        self.assertEqual(row.customer_label, DEMO)
        self.assertEqual(row.evidence_role, "SEED_CONSTANT")
        self.assertEqual(row.verdict, "AUTHENTIC")

    def test_new_authentic_result_stays_pending_and_blocked(self) -> None:
        from inference.publication_gate import (
            AUTHENTICITY_MODEL_PRODUCTION_APPROVED,
            legacy_publication_status,
            publication_decision,
        )
        from listings.customer_label import customer_label

        self.assertFalse(AUTHENTICITY_MODEL_PRODUCTION_APPROVED)
        self.assertEqual(legacy_publication_status("AUTHENTIC"), "pending")
        self.assertEqual(
            publication_decision(
                model="LEGACY_DINOV2",
                model_status="LEGACY",
                production_validation="NOT_ESTABLISHED",
                model_decision="AUTHENTIC",
                policy_status="OK",
                listing_state="pending",
            ),
            "BLOCKED",
        )
        self.assertEqual(
            customer_label(status="pending", verdict="AUTHENTIC", confidence=0.99, s3_url="listings/new.jpg"),
            "Pending",
        )

    def test_research_fake_and_model_error_do_not_certify(self) -> None:
        from inference.publication_gate import legacy_publication_status, publication_decision
        from inference.research_access import research_success_body
        from listings.customer_label import customer_label

        research = research_success_body(
            decision="AUTHENTIC",
            declared_brand="Patek Philippe",
            checkpoint_sha="5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
            policy_version="shadow_v1",
        )
        self.assertEqual(research["publication_decision"], "BLOCKED")
        self.assertEqual(legacy_publication_status("FAKE"), "rejected")
        self.assertEqual(
            customer_label(status="rejected", verdict="FAKE", confidence=0.91, s3_url="listings/fake.jpg"),
            "Rejected",
        )
        self.assertEqual(legacy_publication_status(None), "pending")
        self.assertEqual(
            publication_decision(
                model="LEGACY_DINOV2",
                model_status="LEGACY",
                production_validation="NOT_ESTABLISHED",
                model_decision=None,
                policy_status="MODEL_ERROR",
                listing_state="pending",
            ),
            "BLOCKED",
        )

    def test_no_customer_label_is_an_authenticity_badge(self) -> None:
        from listings.customer_label import customer_label

        samples = [
            customer_label(status="live", verdict="AUTHENTIC", confidence=0.99, s3_url="a.jpg"),
            customer_label(status="live", verdict="AUTHENTIC", confidence=0.965, s3_url=None),
            customer_label(status="pending", verdict="AUTHENTIC", confidence=0.99, s3_url="a.jpg"),
            customer_label(status="rejected", verdict="FAKE", confidence=0.2, s3_url="a.jpg"),
        ]
        for label in samples:
            for phrase in FORBIDDEN:
                self.assertNotIn(phrase.casefold(), label.casefold())
            self.assertNotEqual(label.casefold(), "authentic")

    def test_listing_displays_do_not_render_a_certification_badge(self) -> None:
        paths = [
            "frontend/src/components/AuthBadge.tsx",
            "frontend/src/components/product/ProductVisualPanel.tsx",
            "frontend/src/app/page.tsx",
            "frontend/src/app/product/[id]/page.tsx",
            "frontend/src/app/seller/dashboard/page.tsx",
        ]
        badge = (REPO / "frontend/src/components/AuthBadge.tsx").read_text()
        self.assertIn("Pending", badge)
        self.assertIn("Rejected", badge)
        self.assertIn("Research result — not verified", badge)
        self.assertNotIn('? "AUTHENTIC"', badge)
        for relative in paths:
            text = (REPO / relative).read_text()
            for phrase in FORBIDDEN + ("Live · Authentic",):
                self.assertNotIn(phrase, text, relative)


if __name__ == "__main__":
    unittest.main()
