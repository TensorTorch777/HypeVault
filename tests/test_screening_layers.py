"""Known-AI copy matching, holding borderline photos for a proof photo, and the admin review of held listings."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from database import ListingStatus  # noqa: E402
from inference import known_ai  # noqa: E402
from inference.content_gate import ContentCheck  # noqa: E402
from listings.models import ReviewDecision  # noqa: E402
from listings.routes import review_listing  # noqa: E402  (at import: a scraper test later drops backend/ from sys.path)
from inference.screening import (  # noqa: E402
    KNOWN_AI_MATCH,
    NEEDS_PROOF,
    PASSED,
    PROOF_SUBMITTED,
    REVIEW_APPROVED,
    REVIEW_REJECTED,
    check_known_ai,
    check_origin,
)

WATCH = ContentCheck(is_watch=True, watch_score=0.99, looks_like=None, model="clip")


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class KnownAiTests(unittest.TestCase):
    def setUp(self) -> None:
        known_ai.reset_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = patch.object(known_ai.settings, "known_ai_store", self.tmp.name)
        self.store.start()
        known_ai._state.update(vectors=np.stack([unit(1, 0, 0), unit(0, 1, 0)]), sources=["seed:a", "seed:b"])

    def tearDown(self) -> None:
        self.store.stop()
        self.tmp.cleanup()
        known_ai.reset_for_tests()

    def test_a_copy_is_rejected_with_its_similarity(self) -> None:
        with patch.object(known_ai, "describe", return_value=unit(0.9, 0.1, 0)):
            match = known_ai.check(object())
        self.assertEqual(match.source, "seed:a")
        self.assertTrue(match.matched)
        out = check_known_ai(match.similarity, 0.65, match.source)
        assert out is not None
        self.assertEqual((out.status, out.code), ("rejected", KNOWN_AI_MATCH))
        self.assertIn("screenshotted", out.reason)

    def test_an_unrelated_photo_passes(self) -> None:
        with patch.object(known_ai, "describe", return_value=unit(0, 0, 1)):
            match = known_ai.check(object())
        self.assertFalse(match.matched)
        self.assertIsNone(check_known_ai(match.similarity, 0.65, match.source))

    def test_a_rejected_upload_is_remembered_once_and_kept_on_disk(self) -> None:
        v = unit(0, 0, 1)
        known_ai.remember(v, "upload:LIKELY_GENERATED:x")
        known_ai.remember(v, "upload:LIKELY_GENERATED:x")
        self.assertEqual(len(known_ai._state["sources"]), 3)
        self.assertEqual(np.load(Path(self.tmp.name) / "vectors.npy").shape, (1, 3))
        with patch.object(known_ai, "describe", return_value=unit(0, 0.1, 1)):
            self.assertTrue(known_ai.check(object()).matched)

    def test_the_seed_library_and_model_hash_are_pinned(self) -> None:
        z = np.load(known_ai.SEED)
        self.assertEqual(z["vectors"].shape[1], 512)
        self.assertIn("testtest.jpeg", z["source"].tolist())
        self.assertEqual(len(known_ai.MODEL_SHA256), 64)


class HoldTests(unittest.TestCase):
    def _origin(self, p_fake: float, hold: float | None = 0.2):
        verdict = "FAKE" if p_fake >= 0.5 else "AUTHENTIC"
        conf = p_fake if verdict == "FAKE" else 1 - p_fake
        return check_origin(raw_verdict=verdict, raw_confidence=conf, min_authentic_confidence=0.5, content=WATCH,
                            origin_model="PHASE76_DINOV3_PE", hold_from_p_fake=hold)

    def test_a_borderline_photo_is_held_for_a_proof_photo(self) -> None:
        out = self._origin(0.3)
        self.assertEqual((out.status, out.code), ("pending", NEEDS_PROOF))
        self.assertIn("today's date", out.reason)
        self.assertEqual(out.details["p_ai"], 0.3)

    def test_a_clear_photo_is_listed_and_an_ai_one_rejected(self) -> None:
        self.assertEqual(self._origin(0.05).code, PASSED)
        self.assertEqual(self._origin(0.7).status, "rejected")

    def test_no_hold_band_lists_as_before(self) -> None:
        self.assertEqual(self._origin(0.3, hold=None).code, PASSED)

    def test_the_edited_copy_model_holds_but_never_rejects(self) -> None:
        def origin(p_fake, margin):
            verdict = "FAKE" if p_fake >= 0.5 else "AUTHENTIC"
            conf = p_fake if verdict == "FAKE" else 1 - p_fake
            return check_origin(raw_verdict=verdict, raw_confidence=conf, min_authentic_confidence=0.5, content=WATCH,
                                origin_model="PHASE76_DINOV3_PE", hold_from_p_fake=0.2, edited_copy_margin=margin)

        held = origin(0.05, 0.4)
        self.assertEqual((held.status, held.code, held.details["held_by"]), ("pending", NEEDS_PROOF, "edited_copy_model"))
        self.assertEqual(origin(0.05, -0.4).code, PASSED)
        self.assertEqual(origin(0.05, None).code, PASSED)
        self.assertEqual(origin(0.7, -5.0).status, "rejected")  # Phase 76 still decides rejections
        self.assertEqual(origin(0.3, 0.4).details["held_by"], "borderline_score")


class Phase78HeadTests(unittest.TestCase):
    def test_the_frozen_arrays_load_hash_checked_on_phase76_features(self) -> None:
        sys.path.insert(0, str(REPO / "ml_rtx5080"))
        import phase78

        arrays, cut = phase78.load_head()
        self.assertEqual(arrays["w"].shape, (5632,))
        self.assertEqual(arrays["mu"].shape, arrays["sd"].shape)
        self.assertLess(cut, 0)
        rule = __import__("json").loads(phase78.RULE.read_text())
        self.assertEqual(rule["training"]["copies_per_image"], 2)


class _Db:
    def __init__(self, listing, seller):
        self.listing, self.seller, self.commits = listing, seller, 0

    async def get(self, model, key):
        return self.listing if key == self.listing.id else self.seller if key == self.seller.id else None

    async def commit(self):
        self.commits += 1

    async def refresh(self, _obj):
        return None


def _listing(code: str, status=ListingStatus.pending):
    from database import ListingCategory

    return SimpleNamespace(
        id=uuid.uuid4(), seller_id=uuid.uuid4(), product_name="Patek", category=ListingCategory.watch, brand="Patek Philippe",
        condition=None, size=None, s3_url=None, verdict="AUTHENTIC", confidence=0.7, status=status, screening_code=code,
        screening_reason="held", screening_details={"p_ai": 0.3}, created_at=__import__("datetime").datetime.now(),
    )


class ReviewTests(unittest.TestCase):
    def _review(self, listing, decision, note=None):
        seller = SimpleNamespace(id=listing.seller_id, email="seller@example.com")
        db = _Db(listing, seller)
        admin = SimpleNamespace(id=uuid.uuid4())
        return asyncio.run(review_listing(listing.id, ReviewDecision(decision=decision, note=note), db, admin)), db

    def test_approval_needs_a_proof_photo(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            self._review(_listing(NEEDS_PROOF), "approve")
        self.assertEqual(caught.exception.status_code, 409)

    def test_approve_lists_and_reject_explains(self) -> None:
        listing = _listing(PROOF_SUBMITTED)
        out, db = self._review(listing, "approve")
        self.assertEqual((listing.status, listing.screening_code), (ListingStatus.live, REVIEW_APPROVED))
        self.assertEqual(out.seller_email, "seller@example.com")
        self.assertEqual(out.customer_label, "Listed — photo screened, authenticity not verified")
        self.assertEqual(db.commits, 1)

        listing = _listing(NEEDS_PROOF)
        self._review(listing, "reject", "different watch")
        self.assertEqual((listing.status, listing.screening_code), (ListingStatus.rejected, REVIEW_REJECTED))
        self.assertIn("different watch", listing.screening_reason)

    def test_only_held_listings_can_be_reviewed(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            self._review(_listing(PASSED, ListingStatus.live), "reject")


if __name__ == "__main__":
    unittest.main()
