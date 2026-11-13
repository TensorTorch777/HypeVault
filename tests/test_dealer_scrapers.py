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
        self.assertEqual(sites, ["the1916company", "chronext"])
        self.assertEqual(len(brands), 5)
        self.assertEqual(len(cli.plan(cli.parse_args(["--more-brands"]))[1]), 15)
        self.assertEqual(cli.parse_args(["--delay", "0.1"]).delay, 1.0)  # never faster than 1 request/s per site
        with self.assertRaises(SystemExit):
            cli.main(["--sites", "chronext"])  # no --allow-network
        with self.assertRaises(SystemExit):
            cli.plan(cli.parse_args(["--sites", "ebay"]))
        self.assertEqual(cli.main(["--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
