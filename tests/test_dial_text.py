"""Dial OCR seller hint: brand matching, fail-safe behaviour, and no effect on listing status."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from PIL import Image  # noqa: E402

from inference import dial_text  # noqa: E402


class _Result:
    def __init__(self, lines):
        self.txts = tuple(t for t, _ in lines)
        self.scores = tuple(s for _, s in lines)


class DialTextTests(unittest.TestCase):
    def test_brand_words_match_inside_run_together_text(self) -> None:
        self.assertTrue(dial_text.brand_in_lines("Vacheron Constantin", [("VACHERONCONSTANTIN", 0.9)]))
        self.assertTrue(dial_text.brand_in_lines("A. Lange & Söhne", [("A. LANGE & SÖHNE", 0.8)]))
        self.assertFalse(dial_text.brand_in_lines("Patek Philippe", [("PATILUE", 0.9), ("GENEVE", 0.95)]))
        self.assertFalse(dial_text.brand_in_lines("Patek Philippe", [("PATEK PHILIPPE", 0.3)]))

    def test_found_and_not_found(self) -> None:
        image = Image.new("RGB", (800, 800), "white")
        reader = mock.Mock(side_effect=[_Result([]), _Result([("PATEK PHILIPPE", 0.97)])])
        with mock.patch.object(dial_text, "_ocr", return_value=reader):
            self.assertEqual(dial_text.read_dial_brand(image, "Patek Philippe"), "FOUND")
        reader = mock.Mock(return_value=_Result([("SWISS MADE", 0.99)]))
        with mock.patch.object(dial_text, "_ocr", return_value=reader):
            self.assertEqual(dial_text.read_dial_brand(image, "Patek Philippe"), "NOT_FOUND")

    def test_missing_ocr_or_brand_is_unknown(self) -> None:
        image = Image.new("RGB", (800, 800), "white")
        with mock.patch.object(dial_text, "_ocr", side_effect=ImportError("no rapidocr")):
            self.assertEqual(dial_text.read_dial_brand(image, "Patek Philippe"), "UNKNOWN")
        self.assertEqual(dial_text.read_dial_brand(image, "Rolex"), "UNKNOWN")
        self.assertEqual(dial_text.read_dial_brand(image, None), "UNKNOWN")

    def test_hint_never_changes_the_listing_status(self) -> None:
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        # The dial reading comes after the screening outcome is fixed and only adds details and a hint.
        self.assertLess(routes.index("assert outcome is not None"), routes.index("read_dial_brand, pil"))
        between = routes[routes.index("read_dial_brand, pil"):routes.index("listing.status = ListingStatus(outcome.status)")]
        self.assertNotIn("outcome =", between)


if __name__ == "__main__":
    unittest.main()
