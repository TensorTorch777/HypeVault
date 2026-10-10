"""Customer-facing semantic copy. Does not load a model or open the final test."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONT = REPO / "frontend" / "src"


class UiSemanticGapTests(unittest.TestCase):
    def test_homepage_demo_cards_are_not_framed_as_live_inventory(self) -> None:
        page = (FRONT / "app/page.tsx").read_text()
        self.assertNotIn("Inventory · live", page)
        self.assertNotIn("FRESH IN THE VAULT", page)
        self.assertNotIn("FIVE EVALUATED BRANDS", page)
        self.assertIn("Demo examples — not live inventory", page)
        self.assertIn("Demo example — not live inventory", page)
        self.assertNotIn("href={`/product/", page)

    def test_static_visuals_are_labelled_illustrative(self) -> None:
        how = (FRONT / "components/home/HowItWorks.tsx").read_text()
        self.assertIn("Illustration only — not a live scan or model-generated localization.", how)
        self.assertIn("Illustrative sample data — not live market prices", how)
        self.assertNotIn("We pull real-time prices", how)
        stats = (FRONT / "components/campaign/HomeVaultStats.tsx").read_text()
        self.assertIn("Illustrative figures — not live operations or live market prices.", stats)
        philosophy = (FRONT / "components/home/PhilosophyBento.tsx").read_text()
        self.assertIn("not a live price feed", philosophy)
        self.assertNotIn("Live now", how)
        promo = (FRONT / "components/home/AppleBentoPromo.tsx").read_text()
        self.assertIn("Illustrative sample data — not live market prices.", promo)
        chart = (FRONT / "components/PriceTrendChart.tsx").read_text()
        self.assertIn("Illustrative sample data — not live market prices.", chart)

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

    def test_comparison_failure_is_not_rendered_as_mock_prices(self) -> None:
        table = (FRONT / "components/ComparisonTable.tsx").read_text()
        self.assertNotIn("buildMockComparisonPayload", table)
        copy = (FRONT / "lib/semanticCopy.ts").read_text()
        self.assertIn('typeof data.error === "string"', copy)
        self.assertIn('view === "error"', table)
        self.assertIn('view === "empty"', table)
        self.assertIn("Could not load market data", table)
        self.assertIn("No market observations returned", table)
        recent = (FRONT / "lib/api.ts").read_text()
        self.assertIn('state: "error"', recent)
        self.assertIn("Could not load recent listings.", recent)

    def test_seller_retry_reuses_the_listing_id(self) -> None:
        page = (FRONT / "app/seller/upload/page.tsx").read_text()
        self.assertIn("Retry screening", page)
        self.assertIn("Creating listing.", page)
        self.assertIn("Legacy screening in progress.", page)
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
