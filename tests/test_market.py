"""Market data: parsing, pseudonyms, accessory filter, eBay Browse client, Chrono24 dataset reader."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from market.chrono24_dataset import read_items  # noqa: E402
from market.ebay_browse import EbayApiError, EbayBrowseClient, EbayNotConfigured  # noqa: E402
from market.types import canonical_brand, excluded_listing, is_accessory, is_not_wristwatch, parse_price, price_stats, seller_pseudonym  # noqa: E402

SEARCH_FIXTURE = {
    "total": 4,
    "itemSummaries": [
        {
            "itemId": "v1|1111|0",
            "title": "Patek Philippe Nautilus 5711/1A Blue Dial Steel 2018 Box Papers",
            "price": {"value": "98500.00", "currency": "USD"},
            "image": {"imageUrl": "https://i.ebayimg.com/images/g/a/s-l1600.jpg"},
            "itemWebUrl": "https://www.ebay.com/itm/1111",
            "seller": {"username": "SomeDealer", "feedbackPercentage": "99.8", "feedbackScore": 5120},
            "condition": "Pre-owned",
        },
        {
            "itemId": "v1|2222|0",
            "title": "Green Dial For Nautilus 5711 aftermarket part",
            "price": {"value": "900.00", "currency": "USD"},
            "seller": {"username": "parts"},
        },
        {"itemId": "v1|3333|0", "title": "Rolex Submariner 124060", "price": {"value": "11000", "currency": "USD"}},
        {"itemId": "v1|4444|0", "title": "Patek Philippe Calatrava 5196", "price": {"value": "0", "currency": "USD"}},
    ],
}


def _transport(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(200, json={"access_token": "tok-123", "expires_in": 7200})
        if request.url.path.endswith("/item_summary/search"):
            if request.headers.get("Authorization") != "Bearer tok-123":
                return httpx.Response(401, json={})
            return httpx.Response(200, json=SEARCH_FIXTURE)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


class HelperTests(unittest.TestCase):
    def test_parse_price(self):
        self.assertEqual(parse_price("$71,968"), 71968.0)
        self.assertEqual(parse_price("98500.00"), 98500.0)
        self.assertEqual(parse_price("$10,000 – $12,000"), 10000.0)
        self.assertIsNone(parse_price("Price on request"))
        self.assertIsNone(parse_price("0"))
        self.assertIsNone(parse_price(None))

    def test_brand_canonicalisation(self):
        self.assertEqual(canonical_brand("A. Lange & Sohne"), "A. Lange & Söhne")
        self.assertEqual(canonical_brand("patek philippe"), "Patek Philippe")
        self.assertIsNone(canonical_brand("Rolex"))

    def test_seller_pseudonym_is_salted_and_stable(self):
        a = seller_pseudonym("SomeDealer", "salt-1")
        self.assertEqual(a, seller_pseudonym("somedealer ", "salt-1"))
        self.assertNotEqual(a, seller_pseudonym("SomeDealer", "salt-2"))
        self.assertNotIn("dealer", a.casefold())
        self.assertEqual(len(a), 16)
        self.assertIsNone(seller_pseudonym(None, "salt"))

    def test_accessory_filter_keeps_complete_watches(self):
        self.assertTrue(is_accessory("Rare Factory Green Dial For Nautilus 5711/1a"))
        self.assertTrue(is_accessory("Royal Oak strap only 26331"))
        self.assertTrue(is_accessory("Patek replica 5711"))
        self.assertFalse(is_accessory("Royal Oak Dual Time 18k Rose Gold Folding Clasp Box Papers"))
        self.assertFalse(is_accessory("Pagoda Ladies Ref 4900 Yellow Gold on Strap"))
        self.assertFalse(is_accessory("A. Lange & Söhne 1815 Homage To Walter Lange"))

    def test_pocket_watches_and_clocks_are_not_wristwatch_comparables(self) -> None:
        for title in (
            "Patek Philippe Naviquartz Marine Clock",
            "A. Lange & Söhne DUF Savonette Taschenuhr von 1911",
            "Patek Philippe Gondolo Pocket bolso Red Dial Museum",
            "Audemars Piguet Royal Oak Alarm Table Clock",
            "Vacheron Constantin NOS Open Face Deck Watch In Silver",
        ):
            self.assertTrue(is_not_wristwatch(title), title)
            self.assertTrue(excluded_listing(title), title)
        for title in (
            "Patek Philippe Calatrava Ref. 3919-020J with Chain-link Style bracelet",
            "Patek Philippe Nautilus 5711/1A-010",
            "Audemars Piguet Royal Oak Offshore 26470ST",
        ):
            self.assertFalse(is_not_wristwatch(title), title)

    def test_price_stats(self):
        self.assertEqual(price_stats([3, 1, 2, 4]), {"count": 4, "min": 1, "median": 2.5, "max": 4})
        self.assertIsNone(price_stats([]))


class EbayClientTests(unittest.TestCase):
    def _client(self, calls):
        return EbayBrowseClient(
            "app-id", "cert-id", environment="sandbox", pseudonym_salt="pepper",
            http=httpx.AsyncClient(transport=_transport(calls)),
        )

    def test_requires_keys(self):
        with self.assertRaises(EbayNotConfigured):
            EbayBrowseClient("", "")

    def test_search_maps_and_filters(self):
        calls: list = []

        async def go():
            client = self._client(calls)
            items, total = await client.search("Nautilus 5711", brand="Patek Philippe", limit=500)
            again, _ = await client.search("Nautilus 5711", brand="Patek Philippe")
            return items, total, again

        items, total, again = asyncio.run(go())
        self.assertEqual(total, 4)
        # Part listing, out-of-scope brand handled by brand filter, and zero price are dropped.
        self.assertEqual([i.external_id for i in items], ["v1|1111|0"])
        item = items[0]
        self.assertEqual(item.brand, "Patek Philippe")
        self.assertEqual(item.price, 98500.0)
        self.assertEqual(item.seller_feedback_pct, 99.8)
        self.assertEqual(item.seller_ref, seller_pseudonym("SomeDealer", "pepper"))
        self.assertNotIn("SomeDealer", json.dumps(item.as_row(), default=str))
        self.assertEqual(item.provenance["environment"], "sandbox")
        token_calls = [c for c in calls if c.url.path.endswith("/oauth2/token")]
        self.assertEqual(len(token_calls), 1, "token is reused between searches")
        search = [c for c in calls if c.url.path.endswith("/item_summary/search")][0]
        self.assertTrue(str(search.url).startswith("https://api.sandbox.ebay.com/"))
        self.assertEqual(search.url.params["category_ids"], "31387")
        self.assertEqual(search.url.params["limit"], "200")
        self.assertIn("Brand:{Patek Philippe}", search.url.params["aspect_filter"])
        self.assertEqual(search.headers["X-EBAY-C-MARKETPLACE-ID"], "EBAY_US")

    def test_rejected_token_is_reported(self):
        def handler(request):
            if request.url.path.endswith("/oauth2/token"):
                return httpx.Response(401, json={"error": "invalid_client"})
            return httpx.Response(500)

        async def go():
            client = EbayBrowseClient("a", "b", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            await client.search("x")

        with self.assertRaises(EbayApiError):
            asyncio.run(go())


class Chrono24DatasetTests(unittest.TestCase):
    def test_reads_supported_complete_watches_only(self):
        rows = (
            "source,brand,model,reference,listing_title,price_example,year,condition_chrono\n"
            'Chrono24,A. Lange & Söhne,Lange 1,101.001,A. Lange & Söhne Lange 1 101.001,"$71,968",1994,Very good\n'
            "Chrono24,Patek Philippe,Nautilus,5711,Patek Philippe Nautilus 5711,Price on request,2015,\n"
            'Chrono24,Patek Philippe,Nautilus,5711,Green Dial For Nautilus 5711,"$39,995",,\n'
            'Chrono24,Rolex,Submariner,124060,Rolex Submariner,"$11,000",2021,\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c24.csv"
            path.write_text(rows, encoding="utf-8")
            items = list(read_items(path, date(2023, 9, 16)))
            again = list(read_items(path, date(2023, 9, 16)))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual((item.brand, item.reference, item.price, item.year), ("A. Lange & Söhne", "101.001", 71968.0, "1994"))
        self.assertIsNone(item.observed_at)
        self.assertEqual(item.observed_on, date(2023, 9, 16))
        self.assertEqual(item.external_id, again[0].external_id)
        self.assertIn("not scraped", item.provenance["terms"])


if __name__ == "__main__":
    unittest.main()


class EbayImageSearchTests(unittest.TestCase):
    def test_search_by_image_posts_base64_and_filters(self):
        calls: list = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if request.url.path.endswith("/oauth2/token"):
                return httpx.Response(200, json={"access_token": "tok-123", "expires_in": 7200})
            if request.url.path.endswith("/item_summary/search_by_image"):
                return httpx.Response(200, json=SEARCH_FIXTURE)
            return httpx.Response(404)

        async def go():
            client = EbayBrowseClient("app-id", "cert-id", environment="sandbox", pseudonym_salt="pepper",
                                      http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            return await client.search_by_image("aGVsbG8=", limit=500)

        items = asyncio.run(go())
        self.assertEqual([i.external_id for i in items], ["v1|1111|0"])  # parts, other brands and zero prices dropped
        req = calls[-1]
        self.assertEqual(req.method, "POST")
        self.assertEqual(json.loads(req.content), {"image": "aGVsbG8="})
        self.assertEqual(req.url.params["category_ids"], "31387")
        self.assertEqual(req.url.params["limit"], "200")
