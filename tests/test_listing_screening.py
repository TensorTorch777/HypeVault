"""Listing screening: a photo is listed only if it is a watch and looks original; otherwise rejected with a reason."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from inference.content_gate import ContentCheck  # noqa: E402
from inference.screening import (  # noqa: E402
    IMAGE_TOO_SMALL,
    LIKELY_GENERATED,
    NOT_A_WATCH,
    NOT_CONFIDENTLY_ORIGINAL,
    PASSED,
    check_content,
    check_origin,
    check_size,
)

WATCH = ContentCheck(is_watch=True, watch_score=0.99, looks_like=None, model="clip")
SELFIE = ContentCheck(is_watch=False, watch_score=0.002, looks_like="a person", model="clip")


class ListingScreeningTests(unittest.TestCase):
    def test_a_selfie_is_rejected_as_not_a_watch_with_a_reason(self) -> None:
        out = check_content(SELFIE)
        assert out is not None
        self.assertEqual(out.status, "rejected")
        self.assertEqual(out.code, NOT_A_WATCH)
        self.assertIn("a person", out.reason)
        self.assertIn("wristwatch", out.reason)

    def test_a_watch_passes_the_content_check(self) -> None:
        self.assertIsNone(check_content(WATCH))

    def test_small_photos_are_rejected_with_their_size(self) -> None:
        out = check_size(300, 220, 400)
        assert out is not None
        self.assertEqual((out.status, out.code), ("rejected", IMAGE_TOO_SMALL))
        self.assertIn("300×220", out.reason)
        self.assertIsNone(check_size(800, 600, 400))

    def test_generated_and_unsure_photos_are_rejected(self) -> None:
        fake = check_origin(raw_verdict="FAKE", raw_confidence=0.93, min_authentic_confidence=0.88, content=WATCH)
        self.assertEqual((fake.status, fake.code), ("rejected", LIKELY_GENERATED))
        self.assertIn("93%", fake.reason)
        unsure = check_origin(raw_verdict="AUTHENTIC", raw_confidence=0.7, min_authentic_confidence=0.88, content=WATCH)
        self.assertEqual((unsure.status, unsure.code), ("rejected", NOT_CONFIDENTLY_ORIGINAL))
        self.assertIn("70%", unsure.reason)

    def test_an_original_watch_photo_is_listed_without_an_authenticity_claim(self) -> None:
        out = check_origin(raw_verdict="AUTHENTIC", raw_confidence=0.97, min_authentic_confidence=0.88, content=WATCH)
        self.assertEqual((out.status, out.code), ("live", PASSED))
        self.assertIn("not a certificate of authenticity", out.reason)

    def test_unlisted_listings_are_private_to_their_seller(self) -> None:
        routes = (REPO / "backend" / "listings" / "routes.py").read_text()
        self.assertIn("def _visible", routes)
        self.assertGreaterEqual(routes.count("if not _visible(listing, viewer)"), 2)


if __name__ == "__main__":
    unittest.main()
