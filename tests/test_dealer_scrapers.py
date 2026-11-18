"""Dealer crawlers (The 1916 Company, Chronext): parsing, discovery, politeness, export. No network.

PWF is a generic schema.org Product fixture used to exercise JSON-LD parsing and the crawl runner; it
does not imply Watchfinder is a supported source (it is blocked by an AWS WAF challenge and excluded).
WF_WAF guards that we detect such a challenge.
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from market.dealers.crawl import crawl_site, export_listings, to_market_items  # noqa: E402
from market.dealers.extract import (  # noqa: E402
    jsonld_product,
    label_values,
    normalize_reference,
    parse_money,
    parse_sitemap,
    reference_from_title,
    slug_candidates,
    visible_lines,
    year_of,
)
from market.dealers.http import BlockedError, DisallowedError, PoliteFetcher  # noqa: E402
from market.dealers.sites import Chronext, The1916Company  # noqa: E402

# Shaped after the live pages: SFCC page with og tags and spec rows, no JSON-LD.
P1916 = """<html><head>
<meta property="og:title" content="Pre-Owned Patek Philippe Nautilus 7118/1R-001 | The 1916 Company">
<meta property="og:image" content="https://watchbox-cdn.imgix.net/posted-product-images/639_pate_5078257_35-65.png?w=1200">
<script>window.dataLayer=[{"price":"1"}]</script></head><body>
<h1><span>Patek Philippe</span> <span>Nautilus</span> <span>7118/1R-001</span></h1>
<div class="price"><span>Current Price:</span> <span>$149,950</span></div>
<p>Or $4,166/mo with financing</p>
<dl><dt>Reference Number</dt><dd>7118/1R-001</dd><dt>SKU</dt><dd>5078257</dd><dt>Year</dt><dd>2023</dd>
<dt>Box</dt><dd>Yes</dd><dt>Papers</dt><dd>Yes</dd><dt>Case Size</dt><dd>35mm</dd>
<dt>Case Material</dt><dd>Rose Gold</dd><dt>Dial Color</dt><dd>Silver</dd><dt>Movement</dt><dd>Automatic</dd></dl>
</body></html>"""

# Magento page with schema.org Product.
PWF = """<html><head><title>Patek Philippe Nautilus 5726/1A-001 | Watchfinder &amp; Co.</title>
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"BreadcrumbList"},
{"@type":"Product","name":"Patek Philippe Nautilus 5726/1A-001","brand":{"@type":"Brand","name":"Patek Philippe"},
"mpn":"5726/1A-001","image":["https://images.watchfinder.co.uk/imgv2/stock/455474/a.jpg","https://images.watchfinder.co.uk/imgv2/stock/455474/b.jpg"],
"offers":{"@type":"Offer","price":"88500","priceCurrency":"GBP","availability":"https://schema.org/InStock","itemCondition":"https://schema.org/UsedCondition"}}]}
</script></head><body><h1>Patek Philippe Nautilus</h1>
<table><tr><th>Year</th><td>2016</td></tr><tr><th>Box and papers</th><td>Box and papers</td></tr>
<tr><th>Case size</th><td>40.5mm</td></tr><tr><th>Case material</th><td>Steel</td></tr><tr><th>Series</th><td>Nautilus</td></tr></table>
</body></html>"""

# Chronext: price in visitor currency with a struck-through old price, no JSON-LD.
PCX = """<html><head><meta property="og:title" content="Patek Philippe Nautilus 5711/1A-010 | CHRONEXT">
<meta property="og:image" content="https://img.chronext.com/5711.jpg"></head><body>
<h1>Patek Philippe Nautilus 5711/1A-010</h1>
<div>In time for Xmas</div><div><s>Was USD 123,160</s></div><div>USD 119,740</div><div>Save USD 3,420</div>
<ul><li>Condition: Like New</li><li>Year: 2019</li><li>Scope of delivery: Box and papers</li><li>Case diameter: 40 mm</li></ul>
</body></html>"""



# The 1916 Company embeds the product as Salesforce Commerce Cloud c_* fields; booleans for box/papers.
P1916_EMBEDDED = (
    P1916
    + '<script>window.__d={"c_baseRefNum":"5146/1G-010","c_condition":"Pre-Owned","c_caseSize":"39mm",'
    '"c_displayedMaterial":"White Gold","c_dialColor":"Gray","c_hasBox":false,"c_hasPapers":false,"c_brandFamily":"Complications"}</script>'
)

# Watchfinder's category/product pages come back as an AWS WAF JavaScript challenge.
WF_WAF = """<!DOCTYPE html><html><head><title></title>
<script>window.awsWafCookieDomainList=[];window.gokuProps={"key":"AQI","iv":"x","context":"y"};</script>
<script src="https://c0a.token.awswaf.com/x/challenge.js"></script></head>
<body><div id="challenge-container"></div></body></html>"""


def _run(coro):
    return asyncio.run(coro)


class ExtractTests(unittest.TestCase):
    def test_money(self):
        self.assertEqual(parse_money("Current Price: $149,950"), (149950.0, "USD"))
        self.assertEqual(parse_money("USD 119,740"), (119740.0, "USD"))
        self.assertEqual(parse_money("£88,500"), (88500.0, "GBP"))
        self.assertEqual(parse_money("98.500 €"), (98500.0, "EUR"))
        self.assertEqual(parse_money("CHF 1'250'000"), (1250000.0, "CHF"))
        self.assertEqual(parse_money("no price"), (None, None))

    def test_reference_and_year(self):
        self.assertEqual(reference_from_title("Patek Philippe Nautilus 5711/1A-010 2019 40mm"), "5711/1A-010")
        self.assertEqual(normalize_reference("5711/1A-010"), normalize_reference("57111a010"))
        self.assertEqual(year_of("Circa 1998"), "c. 1998")
        self.assertEqual(year_of("2023"), "2023")

    def test_slugs(self):
        self.assertIn("a-lange-sohne", slug_candidates("A. Lange & Söhne"))
        self.assertIn("a-lange-and-sohne", slug_candidates("A. Lange & Söhne"))
        self.assertEqual(slug_candidates("Patek Philippe"), ["patek-philippe"])

    def test_sitemaps(self):
        idx = "<sitemapindex><sitemap><loc>https://x.com/sm-products-1.xml</loc></sitemap></sitemapindex>"
        self.assertEqual(parse_sitemap(idx), (["https://x.com/sm-products-1.xml"], []))
        urlset = "<urlset><url><loc><![CDATA[https://x.com/a?b=1&amp;c=2]]></loc></url></urlset>"
        self.assertEqual(parse_sitemap(urlset), ([], ["https://x.com/a?b=1&c=2"]))

    def test_label_values_forms(self):
        lines = visible_lines("<div>Year</div><div>2016</div><p>Box: Yes</p><p>Papers No</p>")
        got = label_values(lines, {"year": ("Year",), "box": ("Box",), "papers": ("Papers",)})
        self.assertEqual(got, {"year": "2016", "box": "Yes", "papers": "No"})

    def test_jsonld_graph(self):
        self.assertEqual(jsonld_product(PWF)["mpn"], "5726/1A-001")


class SiteParseTests(unittest.TestCase):
    def test_1916(self):
        url = "https://www.the1916company.com/pre-owned/patek-philippe/nautilus/5078257/"
        item = The1916Company().parse(url, P1916, "Patek Philippe")
        self.assertEqual((item.price, item.currency), (149950.0, "USD"))  # not the $4,166/mo financing line
        self.assertEqual(item.title, "Patek Philippe Nautilus 7118/1R-001")
        self.assertEqual((item.reference, item.listing_id, item.year), ("7118/1R-001", "5078257", "2023"))
        self.assertEqual((item.box, item.papers), (True, True))
        self.assertEqual((item.case_size, item.case_material, item.dial), ("35mm", "Rose Gold", "Silver"))
        self.assertEqual(item.model, "Nautilus")
        self.assertTrue(item.image_url.startswith("https://watchbox-cdn.imgix.net/"))
        self.assertEqual(item.condition, "Pre-owned")

    def test_chronext(self):
        url = "https://www.chronext.com/patek-philippe/nautilus/5711-1a-010-c123456"
        item = Chronext().parse(url, PCX, "Patek Philippe")
        self.assertEqual((item.price, item.currency), (119740.0, "USD"))  # not the "Was" price
        self.assertEqual((item.condition, item.year, item.box, item.papers), ("Like New", "2019", True, True))
        self.assertEqual((item.reference, item.case_size), ("5711/1A-010", "40 mm"))

    def test_1916_embedded_overrides_rows(self):
        url = "https://www.the1916company.com/pre-owned/patek-philippe/complications/5100103/"
        item = The1916Company().parse(url, P1916_EMBEDDED, "Patek Philippe")
        # Embedded c_hasBox/c_hasPapers=false win; on this watch both are absent in the body.
        self.assertEqual((item.box, item.papers), (False, False))
        self.assertEqual((item.case_material, item.dial), ("White Gold", "Gray"))

    def test_watchfinder_waf_is_a_challenge(self):
        from market.dealers.http import looks_like_challenge

        self.assertTrue(looks_like_challenge(WF_WAF))

    def test_chronext_prefers_product_image(self):
        html = PCX.replace(
            "</body>", '<img src="https://cdn-products.chronext.com/V/6/V68563/V68563_1_det.png"></body>'
        )
        item = Chronext().parse("https://www.chronext.com/patek-philippe/nautilus/5711-1a-010-c1", html, "Patek Philippe")
        self.assertTrue(item.image_url.startswith("https://cdn-products.chronext.com/"))

    def test_product_url_rules(self):
        s = The1916Company()
        self.assertTrue(s.is_product_url("https://www.the1916company.com/pre-owned/patek-philippe/nautilus/5078257/", "patek-philippe"))
        self.assertTrue(s.is_product_url("https://www.the1916company.com/pre-owned/patek-philippe/nautilus/nautilus-3800-1j-011", "patek-philippe"))
        self.assertFalse(s.is_product_url("https://www.the1916company.com/pre-owned/patek-philippe/nautilus/ref-5723/", "patek-philippe"))
        self.assertFalse(s.is_product_url("https://www.the1916company.com/pre-owned/rolex/submariner/123/", "patek-philippe"))
        c = Chronext()
        self.assertTrue(c.is_product_url("https://www.chronext.com/patek-philippe/nautilus/5711-1a-010-c123456", "patek-philippe"))
        self.assertFalse(c.is_product_url("https://www.chronext.com/patek-philippe/nautilus-gold", "patek-philippe"))
        self.assertFalse(c.is_product_url("https://www.chronext.com/journal/news/patek-philippe-5711", "patek-philippe"))


def _site_transport(pages: dict[str, tuple[int, str]], log: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        log.append(url)
        status, body = pages.get(url, (404, "not found"))
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


class CrawlTests(unittest.TestCase):
    BASE = "https://www.the1916company.com"

    def _pages(self):
        p1 = f"{self.BASE}/pre-owned/patek-philippe/nautilus/5078257/"
        p2 = f"{self.BASE}/pre-owned/patek-philippe/aquanaut/5000001/"
        return {
            f"{self.BASE}/robots.txt": (200, f"User-agent: *\nDisallow: /cart\nDisallow: /pre-owned/patek-philippe/aquanaut/\nSitemap: {self.BASE}/sitemap_index.xml\n"),
            f"{self.BASE}/sitemap_index.xml": (200, f"<sitemapindex><sitemap><loc>{self.BASE}/sitemap_blog.xml</loc></sitemap><sitemap><loc>{self.BASE}/sitemap_products.xml</loc></sitemap></sitemapindex>"),
            f"{self.BASE}/sitemap_products.xml": (200, f"<urlset><url><loc>{p1}</loc></url><url><loc>{p2}</loc></url><url><loc>{self.BASE}/pre-owned/rolex/sub/1/</loc></url></urlset>"),
            p1: (200, P1916),
        }

    def test_crawl_sitemap_robots_export_db(self):
        log: list[str] = []

        async def go():
            async with httpx.AsyncClient(transport=_site_transport(self._pages(), log)) as client:
                fetcher = PoliteFetcher(min_delay=0, client=client)
                return await crawl_site(The1916Company(), fetcher, ["Patek Philippe"], 10)

        report = _run(go())
        self.assertEqual([l.listing_id for l in report.listings], ["5078257"])
        self.assertEqual(report.discovered, 2)
        self.assertEqual(report.skipped, 1)  # aquanaut page is disallowed by robots.txt and never requested
        self.assertNotIn(f"{self.BASE}/pre-owned/patek-philippe/aquanaut/5000001/", log)
        self.assertNotIn(f"{self.BASE}/sitemap_blog.xml", log)

        with tempfile.TemporaryDirectory() as d:
            export_listings(Path(d), report.listings)
            export_listings(Path(d), report.listings)
            with (Path(d) / "listings.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reference"], "7118/1R-001")
            self.assertEqual(json.loads((Path(d) / "listings.jsonl").read_text().splitlines()[0])["price"], 149950.0)

        items = to_market_items(report.listings, datetime(2026, 11, 13, tzinfo=timezone.utc))
        self.assertEqual((items[0].source, items[0].brand, items[0].external_id), ("the1916company", "Patek Philippe", "5078257"))
        self.assertTrue(items[0].provenance["box"])

    def test_challenge_stops_site(self):
        pages = self._pages()
        pages[f"{self.BASE}/pre-owned/patek-philippe/nautilus/5078257/"] = (200, "<html><title>Just a moment</title><div id='cf-challenge'></div></html>")

        async def go():
            async with httpx.AsyncClient(transport=_site_transport(pages, [])) as client:
                return await crawl_site(The1916Company(), PoliteFetcher(min_delay=0, client=client), ["Patek Philippe"], 10)

        report = _run(go())
        self.assertIn("bot challenge", report.blocked)
        self.assertEqual(report.listings, [])

    def test_403_and_disallow(self):
        async def go(pages, url):
            async with httpx.AsyncClient(transport=_site_transport(pages, [])) as client:
                return await PoliteFetcher(min_delay=0, client=client).get_text(url)

        with self.assertRaises(BlockedError):
            _run(go({"https://a.test/robots.txt": (200, ""), "https://a.test/x": (403, "no")}, "https://a.test/x"))
        with self.assertRaises(DisallowedError):
            _run(go({"https://a.test/robots.txt": (200, "User-agent: *\nDisallow: /")}, "https://a.test/x"))

    def test_category_fallback(self):
        # Chronext lists no sitemap in robots.txt, so discovery falls back to paging the category page.
        base = "https://www.chronext.com"
        listing = f"{base}/patek-philippe/nautilus/5711-1a-010-c123456"
        pages = {
            f"{base}/robots.txt": (200, "User-agent: *\nAllow: /\n"),
            f"{base}/patek-philippe": (200, f'<a href="/patek-philippe/nautilus/5711-1a-010-c123456">x</a><a href="/patek-philippe/nautilus-gold">cat</a>'),
            listing: (200, PCX),
        }

        async def go():
            async with httpx.AsyncClient(transport=_site_transport(pages, [])) as client:
                return await crawl_site(Chronext(), PoliteFetcher(min_delay=0, client=client), ["Patek Philippe"], 5)

        report = _run(go())
        self.assertEqual([(l.source, l.reference, l.currency) for l in report.listings], [("chronext", "5711/1A-010", "USD")])


class CliTests(unittest.TestCase):
    def test_plan_and_guards(self):
        sys.path.insert(0, str(ROOT / "scraper"))
        import dealer_scraper as cli

        sites, brands = cli.plan(cli.parse_args([]))
        self.assertIn("the1916company", sites)
        self.assertIn("chronext", sites)
        self.assertIn("europeanwatch", sites)
        self.assertEqual(len(brands), 5)
        self.assertEqual(len(cli.plan(cli.parse_args(["--more-brands"]))[1]), 15)
        self.assertEqual(cli.parse_args(["--delay", "0.1"]).delay, 1.0)  # never faster than 1 request/s per site
        with self.assertRaises(SystemExit):
            cli.main(["--sites", "chronext"])  # no --allow-network
        with self.assertRaises(SystemExit):
            cli.plan(cli.parse_args(["--sites", "ebay"]))
        self.assertEqual(cli.main(["--dry-run"]), 0)


# A Shopify /products.json product (shape taken from a real watch-dealer feed).
SHOPIFY_PRODUCT = {
    "id": 7788990011,
    "title": "Patek Philippe Nautilus 5711/1A-010 Blue Dial",
    "handle": "patek-philippe-nautilus-5711-1a-010",
    "vendor": "Patek Philippe",
    "product_type": "Watches",
    "body_html": "<p>Stainless steel, 40mm. Circa 2019.</p>",
    "tags": ["Year:2019", "Reference:5711/1A-010", "Full Set", "Condition:Pre-owned", "Case Material:Steel"],
    "variants": [{"price": "145000.00", "available": True}],
    "images": [{"src": "https://cdn.shopify.com/s/files/1/x/5711-a.jpg?v=1"}, {"src": "https://cdn.shopify.com/s/files/1/x/5711-b.jpg"}],
}

STRAP_PRODUCT = {
    "id": 1, "title": "Alligator Strap for Nautilus", "handle": "strap", "vendor": "Patek Philippe",
    "product_type": "Straps", "tags": ["Accessory"], "variants": [{"price": "450.00", "available": True}], "images": [],
}


class ShopifyTests(unittest.TestCase):
    def setUp(self):
        from market.dealers.shopify import WatchCollectorsUK

        self.site = WatchCollectorsUK()

    def test_parse_product(self):
        item = self.site.parse_product(SHOPIFY_PRODUCT, "Patek Philippe")
        self.assertEqual((item.price, item.currency), (145000.0, "GBP"))
        self.assertEqual(item.reference, "5711/1A-010")
        self.assertEqual(item.year, "2019")
        self.assertEqual((item.box, item.papers), (True, True))  # "Full Set"
        self.assertEqual(item.case_size, "40mm")
        self.assertEqual(item.case_material, "Steel")
        self.assertEqual(item.condition, "Pre-owned")
        self.assertEqual(len(item.image_urls), 2)
        self.assertTrue(item.listing_url.endswith("/products/patek-philippe-nautilus-5711-1a-010"))

    def test_brand_and_watch_filters(self):
        self.assertTrue(self.site._matches_brand(SHOPIFY_PRODUCT, "Patek Philippe"))
        self.assertFalse(self.site._matches_brand(SHOPIFY_PRODUCT, "Rolex"))
        self.assertTrue(self.site._is_watch(SHOPIFY_PRODUCT))
        self.assertFalse(self.site._is_watch(STRAP_PRODUCT))  # straps rejected

    def test_discover_and_crawl_from_feed(self):
        feed = {"products": [SHOPIFY_PRODUCT, STRAP_PRODUCT]}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/robots.txt" in url:
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            if "/products.json" in url and "page=1" in url:
                return httpx.Response(200, json=feed)
            if "/products.json" in url:
                return httpx.Response(200, json={"products": []})
            return httpx.Response(404, text="x")

        async def go():
            from market.dealers.crawl import crawl_site

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await crawl_site(self.site, PoliteFetcher(min_delay=0, client=client), ["Patek Philippe"], 10)

        report = _run(go())
        # The strap is filtered out; only the watch remains.
        self.assertEqual([l.reference for l in report.listings], ["5711/1A-010"])
        self.assertEqual(report.listings[0].source, "watchcollectors")

    def test_brand_from_title_when_vendor_is_the_store(self):
        # Watch Collectors UK sets vendor to its own name; the brand has to come from the title.
        product = {**SHOPIFY_PRODUCT, "vendor": "Watch Collectors UK"}
        item = self.site.parse_product(product, "Patek Philippe")
        self.assertEqual(item.brand, "Patek Philippe")
        self.assertEqual([m.brand for m in to_market_items([item])], ["Patek Philippe"])  # reaches market_listings
        # A real brand in vendor still wins, and an unrelated title does not borrow the hint.
        self.assertEqual(self.site.parse_product(SHOPIFY_PRODUCT, "Rolex").brand, "Patek Philippe")
        other = {**SHOPIFY_PRODUCT, "vendor": "Watch Collectors UK", "title": "Rolex Submariner 124060"}
        self.assertEqual(self.site.parse_product(other, "Patek Philippe").brand, "Watch Collectors UK")

    def _crawl_feed(self, served_currency: str | None):
        seen_cookies: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/robots.txt" in url:
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            if "/products.json" in url:
                seen_cookies.append(request.headers.get("cookie", ""))
                headers = {"set-cookie": f"cart_currency={served_currency}; path=/"} if served_currency else {}
                products = [SHOPIFY_PRODUCT] if "page=1" in url else []
                return httpx.Response(200, json={"products": products}, headers=headers)
            return httpx.Response(404, text="x")

        async def go():
            from market.dealers.crawl import crawl_site

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await crawl_site(self.site, PoliteFetcher(min_delay=0, client=client), ["Patek Philippe", "Audemars Piguet"], 10)

        return _run(go()), seen_cookies

    def test_feed_asks_for_the_store_market(self):
        report, cookies = self._crawl_feed("GBP")
        self.assertIn("localization=GB", cookies[0])
        self.assertIn("cart_currency=GBP", cookies[0])
        self.assertEqual([(l.price, l.currency) for l in report.listings], [(145000.0, "GBP")])

    def test_converted_prices_are_refused_not_relabelled(self):
        # Shopify Markets priced the feed in rupees (visitor IP in India): skip the store, don't call it GBP.
        report, cookies = self._crawl_feed("INR")
        self.assertEqual(report.listings, [])
        self.assertTrue(any("INR" in e for e in report.errors))
        self.assertEqual(len(cookies), 1)  # the feed is not refetched for the second brand


class MarketSearchDealerTests(unittest.TestCase):
    """Dealers ride along in search_market without changing what eBay and the dataset return."""

    EBAY_HIT = {"status": "ok", "items": [{"price": 1.0}], "stats": None, "fetched_at": None, "cached": False}
    EBAY_EMPTY = {"status": "ok", "items": [], "stats": None, "fetched_at": None, "cached": False}
    C24_EMPTY = {"status": "no_match", "items": [], "stats": None}
    DEALERS_HIT = {"status": "ok", "max_age_days": 30, "searched": ["Chronext"], "dealers": [{"label": "Chronext", "items": [{}], "count": 1}], "note": "n"}

    def _search(self, ebay, c24, dealers, **kw):
        from unittest import mock

        import market.service as svc

        ebay_mock = mock.AsyncMock(side_effect=ebay if isinstance(ebay, list) else [ebay, ebay])
        c24_mock = mock.AsyncMock(side_effect=c24 if isinstance(c24, list) else [c24, c24])
        dealer_mock = mock.AsyncMock(side_effect=dealers) if isinstance(dealers, Exception) else mock.AsyncMock(return_value=dealers)
        session = mock.AsyncMock()
        with mock.patch.object(svc, "_ebay_section", ebay_mock), mock.patch.object(svc, "_chrono24_section", c24_mock), mock.patch.object(
            svc, "dealer_section", dealer_mock
        ):
            out = _run(svc.search_market(session, None, kw.pop("q", "Nautilus 5711/1A"), "Patek Philippe", **kw))
        return out, ebay_mock, dealer_mock, session

    def test_dealers_follow_ebay_in_the_same_response(self):
        out, _, dealer_mock, _ = self._search(self.EBAY_HIT, self.C24_EMPTY, self.DEALERS_HIT)
        self.assertEqual(list(out["sources"]), ["ebay", "chrono24_dataset", "dealers"])
        self.assertIs(out["sources"]["ebay"], self.EBAY_HIT)  # eBay section passed through untouched
        self.assertEqual(out["found"], {"ebay": True, "chrono24_dataset": False})  # unchanged contract
        self.assertEqual(out["sources"]["dealers"]["status"], "ok")
        dealer_mock.assert_awaited_once_with(mock_any(), "Nautilus 5711/1A", "Patek Philippe")

    def test_dealer_matches_do_not_stop_the_ebay_fallback(self):
        # Nothing on eBay or in the dataset: the shorter query runs, even though dealers had matches.
        out, ebay_mock, dealer_mock, _ = self._search(
            [self.EBAY_EMPTY, self.EBAY_HIT], [self.C24_EMPTY, self.C24_EMPTY], self.DEALERS_HIT, fallback_query="Nautilus"
        )
        self.assertEqual(out["tried"], ["Nautilus 5711/1A", "Nautilus"])
        self.assertEqual(ebay_mock.await_count, 2)
        self.assertEqual(dealer_mock.await_args.args[1], "Nautilus")  # dealers searched with the query that was used

    def test_dealer_failure_leaves_ebay_intact(self):
        out, _, _, session = self._search(self.EBAY_HIT, self.C24_EMPTY, RuntimeError("db down"))
        self.assertEqual(out["sources"]["dealers"]["status"], "error")
        self.assertEqual(out["found"], {"ebay": True, "chrono24_dataset": False})
        self.assertIs(out["sources"]["ebay"], self.EBAY_HIT)
        session.rollback.assert_awaited()

    def test_card_images_are_resized_on_imgix_only(self):
        from market.dealers.service import _card_image

        self.assertEqual(_card_image("https://watchbox-cdn.imgix.net/a.png"), "https://watchbox-cdn.imgix.net/a.png?w=640&auto=format")
        self.assertEqual(_card_image("https://cdn-products.chronext.com/a.png"), "https://cdn-products.chronext.com/a.png")
        self.assertIsNone(_card_image(None))


def mock_any():
    from unittest import mock

    return mock.ANY


class EuropeanWatchTests(unittest.TestCase):
    def test_product_url_rules(self):
        from market.dealers.sites import EuropeanWatchCo

        s = EuropeanWatchCo()
        self.assertTrue(s.is_product_url("https://www.europeanwatch.com/watch/patek-philippe-5726a-001-nautilus-73399", "patek-philippe"))
        self.assertFalse(s.is_product_url("https://www.europeanwatch.com/brand/patek-philippe", "patek-philippe"))
        self.assertEqual(s.listing_id("https://www.europeanwatch.com/watch/patek-philippe-5726a-001-nautilus-73399", {}), "73399")


if __name__ == "__main__":
    unittest.main()
