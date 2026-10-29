"""Live Chrono24 search: Apify mapping, HTML parsing, filtering, export and provider selection. No network."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from market.chrono24_live import (  # noqa: E402
    ApifyChrono24Provider,
    Chrono24Error,
    Chrono24NotConfigured,
    _display_price,
    build_query,
    parse_apify_item,
    parse_search_html,
    resolve_provider_name,
    search_live,
)

APIFY_ROWS = [
    {
        "listingId": "38291001",
        "title": "Patek Philippe Nautilus",
        "subtitle": "5711/1A-010 Blue Dial",
        "price": 129500,
        "currency": "USD",
        "listingUrl": "https://www.chrono24.com/patekphilippe/nautilus--id38291001.htm",
        "mainImage": "https://img.chrono24.com/images/uhren/1-a.jpg",
        "imageUrls": ["https://img.chrono24.com/images/uhren/1-a.jpg", "https://img.chrono24.com/images/uhren/1-b.jpg"],
        "location": {"city": "Geneva", "countryCode": "CH", "countryName": "Switzerland"},
        "sellerType": "Dealer",
        "detail": {
            "referenceNumber": "5711/1A-010",
            "manufacturer": "Patek Philippe",
            "model": "Nautilus",
            "condition": "Very good",
            "yearOfProduction": "2019",
            "fullImageUrls": ["https://img.chrono24.com/images/uhren/1-full.jpg"],
            "seller": {"name": "Swiss Dealer SA", "rating": 4.9, "reviewCount": 812},
        },
    },
    {
        "listingId": "38291002",
        "title": "Blue dial for Patek Philippe Nautilus 5711",
        "price": 2400,
        "currency": "USD",
        "listingUrl": "https://www.chrono24.com/patekphilippe/dial--id38291002.htm",
    },
    {
        "listingId": "38291003",
        "title": "Rolex Submariner Date",
        "price": 14000,
        "currency": "USD",
        "listingUrl": "https://www.chrono24.com/rolex/submariner--id38291003.htm",
        "detail": {"manufacturer": "Rolex", "yearOfProduction": "Unknown"},
    },
    {"title": "no url row"},
]

SEARCH_HTML = """
<html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"ItemList","itemListElement":[
 {"@type":"Product","name":"Audemars Piguet Royal Oak 15500ST","brand":{"@type":"Brand","name":"Audemars Piguet"},
  "mpn":"15500ST.OO.1220ST.01","image":"https://img.chrono24.com/ap-ld.jpg",
  "offers":{"@type":"Offer","price":"41250","priceCurrency":"USD","url":"https://www.chrono24.com/audemarspiguet/royal-oak--id111.htm"}}
]}
</script></head><body>
<div class="wt-x"><a href="/audemarspiguet/royal-oak--id111.htm?pos=1"><img data-src="https://img.chrono24.com/ap-card.jpg">
  <div>Audemars Piguet Royal Oak 15500ST</div><div>2021</div><div>$41,250</div><div>Dubai, UAE</div></a></div>
<div><a href="/audemarspiguet/royal-oak--id222.htm"><img src="https://img.chrono24.com/ap2.jpg"><span>Royal Oak 15202ST Jumbo</span>
  <span>1998</span><span>98.500 €</span></a></div>
<a href="/search/index.htm?page=2">Next</a>
</body></html>
"""


def _run(coro):
    return asyncio.run(coro)


class ApifyMappingTests(unittest.TestCase):
    def test_full_row(self):
        item = parse_apify_item(APIFY_ROWS[0])
        self.assertEqual(item.listing_id, "38291001")
        self.assertEqual(item.title, "Patek Philippe Nautilus — 5711/1A-010 Blue Dial")
        self.assertEqual(item.price, 129500.0)
        self.assertEqual((item.brand, item.model, item.reference), ("Patek Philippe", "Nautilus", "5711/1A-010"))
        self.assertEqual((item.condition, item.year), ("Very good", "2019"))
        self.assertEqual(item.location, "Geneva, Switzerland")
        self.assertEqual((item.seller_name, item.seller_rating), ("Swiss Dealer SA", "4.9"))
        self.assertEqual(item.image_url, "https://img.chrono24.com/images/uhren/1-a.jpg")
        self.assertEqual(len(item.image_urls), 3)  # deduplicated, full-size appended

    def test_unknown_year_and_missing_url(self):
        self.assertIsNone(parse_apify_item(APIFY_ROWS[2]).year)
        self.assertIsNone(parse_apify_item(APIFY_ROWS[3]))

    def test_automation_lab_schema(self):
        item = parse_apify_item(
            {"listingId": "9", "url": "https://www.chrono24.com/x--id9.htm", "title": "Rolex GMT", "brand": "Rolex",
             "referenceNumber": "126710BLRO", "price": 23000, "imageUrl": "https://img/x.jpg", "sellerLocation": "Germany", "year": 2022}
        )
        self.assertEqual((item.brand, item.reference, item.year, item.location), ("Rolex", "126710BLRO", "2022", "Germany"))


class ApifyProviderTests(unittest.TestCase):
    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_request_and_filtering(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=APIFY_ROWS)

        async def go():
            async with self._client(handler) as http:
                provider = ApifyChrono24Provider("tok", http=http)
                return await search_live(provider, "Patek Philippe Nautilus 5711", brand="Patek Philippe", limit=5)

        section = _run(go())
        self.assertIn("/acts/epicscrapers~chrono24-scraper/run-sync-get-dataset-items", seen["url"])
        self.assertEqual(seen["auth"], "Bearer tok")
        self.assertEqual(seen["body"]["searchQuery"], "Patek Philippe Nautilus 5711")
        self.assertTrue(seen["body"]["enrichDetails"])
        # the dial (accessory) and the Rolex (other brand) are dropped
        self.assertEqual([i["listing_id"] for i in section["items"]], ["38291001"])
        self.assertEqual(section["status"], "ok")
        self.assertEqual(section["price_range"]["min"], 129500.0)

    def test_errors(self):
        async def go(status):
            async with self._client(lambda r: httpx.Response(status, text="nope")) as http:
                return await ApifyChrono24Provider("tok", http=http).search("x", 3)

        with self.assertRaises(Chrono24NotConfigured):
            _run(go(401))
        with self.assertRaises(Chrono24Error):
            _run(go(500))
        with self.assertRaises(Chrono24NotConfigured):
            ApifyChrono24Provider("")


class HtmlParsingTests(unittest.TestCase):
    def test_parse_search_page(self):
        items = {i.listing_id: i for i in parse_search_html(SEARCH_HTML)}
        self.assertEqual(set(items), {"111", "222"})
        ap = items["111"]
        self.assertEqual(ap.listing_url, "https://www.chrono24.com/audemarspiguet/royal-oak--id111.htm")
        self.assertEqual((ap.price, ap.currency, ap.brand), (41250.0, "USD", "Audemars Piguet"))
        self.assertEqual(ap.reference, "15500ST.OO.1220ST.01")
        self.assertEqual(ap.year, "2021")
        self.assertEqual(ap.image_url, "https://img.chrono24.com/ap-ld.jpg")
        self.assertIn("https://img.chrono24.com/ap-card.jpg", ap.image_urls)
        jumbo = items["222"]
        self.assertEqual((jumbo.price, jumbo.currency, jumbo.year), (98500.0, "EUR", "1998"))

    def test_display_price(self):
        self.assertEqual(_display_price("$129,500"), 129500.0)
        self.assertEqual(_display_price("98.500 €"), 98500.0)
        self.assertEqual(_display_price("€12.345,50"), 12345.5)


class ExportAndConfigTests(unittest.TestCase):
    def test_export_csv_and_jsonl(self):
        async def go(tmp):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=APIFY_ROWS))) as http:
                p = ApifyChrono24Provider("tok", http=http)
                await search_live(p, "Patek Philippe Nautilus", brand="Patek Philippe", export_dir=tmp)
                await search_live(p, "Rolex Submariner", brand="Rolex", export_dir=tmp)

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _run(go(tmp))
            lines = (tmp / "listings.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(x)["listing_id"] for x in lines], ["38291001", "38291003"])
            with (tmp / "listings.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)  # one header only
            self.assertEqual(rows[0]["reference"], "5711/1A-010")
            self.assertEqual(rows[0]["provider"], "apify")

    def test_provider_resolution_and_query(self):
        self.assertEqual(resolve_provider_name("auto", "tok"), "apify")
        self.assertEqual(resolve_provider_name("auto", ""), "dataset")
        self.assertEqual(resolve_provider_name("Playwright", ""), "playwright")
        with self.assertRaises(ValueError):
            resolve_provider_name("bing", "")
        self.assertEqual(build_query("Nautilus 5711", "Patek Philippe"), "Patek Philippe Nautilus 5711")
        self.assertEqual(build_query("Patek Philippe Nautilus", "Patek Philippe"), "Patek Philippe Nautilus")
        self.assertEqual(build_query("Untitled listing", "Richard Mille"), "Richard Mille")


class BulkCliTests(unittest.TestCase):
    def test_dry_run_plan(self):
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "scraper"))
        import chrono24_scraper as cli

        args = cli.parse_args(["--supported-only", "--queries", "Nautilus 5711, Royal Oak 15500"])
        jobs = cli.plan(args)
        self.assertEqual(len(jobs), 7)
        self.assertEqual(jobs[-1], (None, "Royal Oak 15500"))
        self.assertGreater(len(cli.plan(cli.parse_args([]))), 20)
        with self.assertRaises(SystemExit):
            cli.main(["--brands", "Rolex"])  # no --allow-network
        self.assertEqual(cli.main(["--brands", "Rolex", "--dry-run"]), 0)


class ServiceTests(unittest.TestCase):
    """search_chrono24: cache, live result, and fallback to the dataset when the live provider fails."""

    def setUp(self):
        from market import chrono24_service as svc

        self.svc = svc
        self._saved = (svc.settings.chrono24_provider, svc.settings.apify_token, svc.settings.chrono24_export_dir,
                       svc.make_provider, svc.repository.find_listings)
        svc.settings.chrono24_provider = "apify"
        svc.settings.apify_token = "tok"
        svc.settings.chrono24_export_dir = ""

        async def no_rows(*a, **k):
            return [], None

        svc.repository.find_listings = no_rows

    def tearDown(self):
        s = self.svc
        (s.settings.chrono24_provider, s.settings.apify_token, s.settings.chrono24_export_dir,
         s.make_provider, s.repository.find_listings) = self._saved

    def test_live_then_cached(self):
        calls = []

        class Fake:
            name = "apify"

            async def search(self, q, limit):
                calls.append(q)
                return [parse_apify_item(APIFY_ROWS[0])]

        class Redis(dict):
            async def get(self, k):
                return dict.get(self, k)

            async def set(self, k, v, ex=None):
                self[k] = v

        self.svc.make_provider = lambda name, settings: Fake()
        r = Redis()
        first = _run(self.svc.search_chrono24(None, r, "Nautilus 5711", "Patek Philippe"))
        second = _run(self.svc.search_chrono24(None, r, "Nautilus 5711", "Patek Philippe"))
        self.assertEqual(first["count"], 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(calls, ["Patek Philippe Nautilus 5711"])

    def test_fallback_on_error(self):
        class Broken:
            name = "apify"

            async def search(self, q, limit):
                raise Chrono24Error("blocked")

        self.svc.make_provider = lambda name, settings: Broken()
        out = _run(self.svc.search_chrono24(None, None, "Royal Oak", "Audemars Piguet"))
        self.assertEqual(out["provider"], "dataset")
        self.assertIn("blocked", out["note"])


if __name__ == "__main__":
    unittest.main()
