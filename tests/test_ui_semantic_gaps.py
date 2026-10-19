"""Customer-facing semantic copy. Does not load a model or open the final test."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONT = REPO / "frontend" / "src"


class UiSemanticGapTests(unittest.TestCase):
    def test_homepage_shows_no_hard_coded_market_figures(self) -> None:
        page = (FRONT / "app/page.tsx").read_text()
        figures = (FRONT / "components/home/HomeFigures.tsx").read_text()
        self.assertNotIn("Demo example", page)
        self.assertNotRegex(page, r"\$\d")
        self.assertIn("fetchMarketStatus", figures)
        self.assertIn("never a certificate of authenticity", page)

    def test_no_illustrative_prices_remain_anywhere(self) -> None:
        for path in FRONT.rglob("*.ts*"):
            text = path.read_text()
            self.assertNotIn("Illustrative sample data", text, path)
            self.assertNotIn("buildMockComparisonPayload", text, path)
            self.assertNotIn("$33,900", text, path)
        self.assertFalse((FRONT / "lib/mockComparison.ts").exists())

    def test_research_serving_status_is_not_validity(self) -> None:
        page = (FRONT / "app/research/page.tsx").read_text()
        copy = (FRONT / "lib/semanticCopy.ts").read_text()
        self.assertIn("SERVING_STATUS_LABEL", page)
        self.assertIn('export const SERVING_STATUS_LABEL = "Serving status"', copy)
        self.assertIn("VALIDITY_NOT_ESTABLISHED", page)
        self.assertIn("Real-world authenticity validity: not established.", copy)
        self.assertIn("EXPERIMENTAL — NOT APPROVED FOR PRODUCTION", page)
        self.assertIn("data.model !== EXPECTED_MODEL[selected]", page)
        self.assertIn("data.model !== EXPECTED_MODEL[id]", page)
        self.assertIn("Compare both models", page)
        self.assertIn("neither one is treated as more accurate", page)
        self.assertIn("cannot publish a listing", page)
        self.assertIn("DeclaredBrandSelect", page)
        self.assertIn("Declared brand", page)
        self.assertIn("Research access is not enabled", copy)
        self.assertIn("Outside current research scope — no result produced", copy)
        self.assertNotIn("Fake", copy)

    def test_market_failure_is_not_rendered_as_estimated_prices(self) -> None:
        panel = (FRONT / "components/market/MarketPanel.tsx").read_text()
        self.assertIn("Market data could not be loaded", panel)
        self.assertIn("Nothing is estimated in the meantime", panel)
        self.assertIn("No prices are shown rather than stale or estimated ones", panel)
        self.assertIn("Snapshot", panel)
        recent = (FRONT / "lib/api.ts").read_text()
        self.assertIn('state: "error"', recent)
        self.assertIn("Could not load recent listings.", recent)

    def test_seller_retry_reuses_the_listing_id(self) -> None:
        page = (FRONT / "app/seller/upload/page.tsx").read_text()
        self.assertIn("Retry screening", page)
        self.assertIn("Creating listing.", page)
        self.assertIn("Screening in progress.", page)
        self.assertIn("was not created again", page)
        self.assertIn("fd.append(\"listing_id\", id)", page)
        self.assertIn("DeclaredBrandSelect", page)
        brands = (FRONT / "lib/supportedBrands.ts").read_text()
        gate = (REPO / "backend/inference/scope_gate.py").read_text()
        for name in (
            "A. Lange & Söhne",
            "Audemars Piguet",
            "Patek Philippe",
            "Richard Mille",
            "Vacheron Constantin",
        ):
            self.assertIn(f'"{name}"', brands)
            self.assertIn(f'"{name}"', gate)

    def test_visible_listing_copy_is_not_an_authenticity_certificate(self) -> None:
        labels = (FRONT / "lib/customerLabel.ts").read_text()
        badge = (FRONT / "components/AuthBadge.tsx").read_text()
        product = (FRONT / "app/product/[id]/page.tsx").read_text()
        self.assertIn("Visible in marketplace — authenticity not verified", labels)
        self.assertIn("not independent proof that the item is counterfeit", labels)
        self.assertIn("Model classification:", badge)
        self.assertNotIn("Authenticity confirmed", badge)
        self.assertIn("customerStatusCopy", product)
        self.assertNotIn("Live market snapshot", product)
        gate = (REPO / "backend/inference/publication_gate.py").read_text()
        self.assertIn("AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False", gate)


if __name__ == "__main__":
    unittest.main()
