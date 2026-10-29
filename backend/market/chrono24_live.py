"""Live Chrono24 search: listing data and photos for a watch, on demand.

Used after a listing passes screening: the declared brand and product name are searched on Chrono24
and the matching listings are shown on the site.

Providers (setting ``CHRONO24_PROVIDER``):

- ``apify``: a hosted Chrono24 scraper on Apify (default actor ``epicscrapers/chrono24-scraper``),
  called through Apify's run-sync REST endpoint. Needs ``APIFY_TOKEN``. Recommended: the provider runs
  the browsers and proxies, and the repo never contacts chrono24.com itself.
- ``playwright``: a headless browser on this machine. Checks chrono24 robots.txt first and refuses
  paths it disallows. Selector-free parsing (listing links ``--id<n>.htm`` and JSON-LD), so it
  survives most layout changes, but expect blocking under load.
- ``dataset``: no live search; callers fall back to the imported Chrono24 dataset.
- ``auto`` (default): ``apify`` when ``APIFY_TOKEN`` is set, otherwise ``dataset``.

Every search can be appended to ``local_data/chrono24/listings.jsonl`` and ``listings.csv``.
Chrono24's terms restrict automated access; this module is for the HypeVault research prototype.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import threading
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus, urljoin

import httpx

from market.types import excluded_listing, parse_price

_log = logging.getLogger(__name__)

PROVIDER_APIFY = "apify"
PROVIDER_PLAYWRIGHT = "playwright"
PROVIDER_DATASET = "dataset"
PROVIDERS = (PROVIDER_APIFY, PROVIDER_PLAYWRIGHT, PROVIDER_DATASET)

DEFAULT_APIFY_ACTOR = "epicscrapers~chrono24-scraper"
APIFY_BASE = "https://api.apify.com/v2"

CSV_FIELDS = (
    "searched_at", "query", "provider", "listing_id", "title", "brand", "model", "reference", "price",
    "currency", "condition", "year", "location", "seller_type", "seller_name", "seller_rating",
    "listing_url", "image_url", "image_urls",
)

_ID_RE = re.compile(r"--id(\d+)\.htm", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"\b(19[2-9]\d|20[0-4]\d)\b")
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "CHF": "CHF", "¥": "JPY"}


class Chrono24Error(RuntimeError):
    """A provider failed (network, auth, blocked, bad response)."""


class Chrono24NotConfigured(Chrono24Error):
    """The selected provider is missing its configuration."""


@dataclass
class Chrono24Listing:
    listing_id: str
    title: str
    listing_url: str
    price: float | None = None
    currency: str | None = None
    brand: str | None = None
    model: str | None = None
    reference: str | None = None
    condition: str | None = None
    year: str | None = None
    location: str | None = None
    seller_type: str | None = None
    seller_name: str | None = None
    seller_rating: str | None = None
    image_url: str | None = None
    image_urls: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ── helpers ──────────────────────────────────────────────────────────────────


def _clean(text: Any) -> str | None:
    if text is None:
        return None
    value = _WS_RE.sub(" ", unescape(_TAG_RE.sub(" ", str(text)))).strip()
    return value or None


def _as_str(value: Any) -> str | None:
    if value is None or value == "" or str(value).lower() == "unknown":
        return None
    return str(value).strip() or None


def _clean_year(value: Any) -> str | None:
    """'2023 (Approximation)' -> 'c. 2023'; 'Unknown' -> None."""
    text = _as_str(value)
    if not text:
        return None
    match = _YEAR_RE.search(text)
    if not match:
        return None
    return f"c. {match.group(1)}" if "approx" in text.lower() else match.group(1)


def _fallback_id(url: str, title: str) -> str:
    match = _ID_RE.search(url or "")
    if match:
        return match.group(1)
    return hashlib.sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:16]


def _uniq(urls: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for u in urls:
        if isinstance(u, str) and u.startswith("http") and u not in out:
            out.append(u)
    return out


def _currency_from_text(text: str) -> str | None:
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return None


def _display_price(text: str) -> float | None:
    """'$12,345' -> 12345; '12.345 €' and '€12.345,50' (European grouping) -> 12345 / 12345.5."""
    digits = re.sub(r"[^\d.,]", "", text)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?", digits):
        digits = digits.replace(".", "").replace(",", ".")
    return parse_price(digits)


def _query_matches(listing: Chrono24Listing, brand: str | None) -> bool:
    """Drop parts, copies and pocket watches; keep only the declared brand when one is given."""
    if excluded_listing(listing.title):
        return False
    if brand:
        haystack = f"{listing.brand or ''} {listing.title}".casefold()
        tokens = [t for t in re.split(r"[^\w]+", brand.casefold()) if len(t) > 2]
        if tokens and not any(t in haystack for t in tokens):
            return False
    return True


def build_query(product_name: str | None, brand: str | None) -> str:
    """'Nautilus 5711' + 'Patek Philippe' -> 'Patek Philippe Nautilus 5711' (brand not repeated)."""
    name = " ".join((product_name or "").split())
    brand = " ".join((brand or "").split())
    if name.lower() in {"", "untitled listing"}:
        return brand
    if brand and brand.casefold() not in name.casefold():
        return f"{brand} {name}"
    return name


# ── Apify provider ───────────────────────────────────────────────────────────


def parse_apify_item(row: dict) -> Chrono24Listing | None:
    """Map one Apify dataset row to a listing. Handles the epicscrapers and automation-lab schemas."""
    detail = row.get("detail") or {}
    url = row.get("listingUrl") or row.get("url") or detail.get("detailUrl") or ""
    title = _clean(row.get("title")) or ""
    if not url or not title:
        return None
    subtitle = _clean(row.get("subtitle"))
    loc = row.get("location")
    if isinstance(loc, dict):
        location = ", ".join(x for x in (loc.get("city"), loc.get("countryName") or loc.get("countryCode")) if x) or None
    else:
        location = _as_str(loc) or _as_str(row.get("sellerLocation"))
    seller = detail.get("seller") or {}
    rating = seller.get("rating")
    # Full-size detail photos first: mainImage/imageUrls are 210px search thumbnails.
    images = _uniq(
        [*(detail.get("fullImageUrls") or []), row.get("mainImage"), row.get("imageUrl"), *(row.get("imageUrls") or [])]
    )
    price = row.get("price") if row.get("price") is not None else detail.get("price")
    return Chrono24Listing(
        listing_id=str(row.get("listingId") or _fallback_id(url, title)),
        title=title if not subtitle else f"{title} — {subtitle}",
        listing_url=url,
        price=parse_price(price),
        currency=_as_str(row.get("currency")),
        brand=_as_str(detail.get("manufacturer") or row.get("brand")),
        model=_as_str(detail.get("model") or row.get("model")),
        reference=_as_str(detail.get("referenceNumber") or row.get("referenceNumber")),
        condition=_as_str(detail.get("condition") or row.get("condition")),
        year=_clean_year(detail.get("yearOfProduction") or row.get("year")),
        location=location,
        seller_type=_as_str(row.get("sellerType")),
        seller_name=_as_str(seller.get("name")),
        seller_rating=None if rating is None else str(rating),
        image_url=images[0] if images else None,
        image_urls=images,
    )


class ApifyChrono24Provider:
    name = PROVIDER_APIFY

    def __init__(
        self,
        token: str,
        *,
        actor: str = DEFAULT_APIFY_ACTOR,
        currency: str = "USD",
        enrich_details: bool = True,
        timeout_sec: float = 120.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if not token:
            raise Chrono24NotConfigured("APIFY_TOKEN is not set")
        self._token = token
        self.actor = actor.replace("/", "~")
        self.currency = currency
        self.enrich_details = enrich_details
        self.timeout_sec = timeout_sec
        self._http = http

    def _input(self, query: str, limit: int) -> dict:
        if self.actor.startswith("automation-lab"):
            url = f"https://www.chrono24.com/search/index.htm?query={quote_plus(query)}&sortorder=1"
            return {"startUrls": [{"url": url}], "maxItems": limit, "maxPagesPerUrl": 1, "includeDetails": self.enrich_details}
        return {
            "searchQuery": query,
            "maxResults": limit,
            "currency": self.currency,
            "sortOrder": "0",  # relevance: the closest match to the screened watch first
            "enrichDetails": self.enrich_details,
        }

    async def search(self, query: str, limit: int) -> list[Chrono24Listing]:
        url = f"{APIFY_BASE}/acts/{self.actor}/run-sync-get-dataset-items"
        params = {"format": "json", "clean": "true", "timeout": str(int(self.timeout_sec))}
        headers = {"Authorization": f"Bearer {self._token}"}
        owns = self._http is None
        http = self._http or httpx.AsyncClient(timeout=self.timeout_sec + 15)
        try:
            resp = await http.post(url, params=params, headers=headers, json=self._input(query, limit))
        except httpx.HTTPError as exc:
            raise Chrono24Error(f"Apify request failed: {exc}") from exc
        finally:
            if owns:
                await http.aclose()
        if resp.status_code == 401:
            raise Chrono24NotConfigured("APIFY_TOKEN was rejected")
        if resp.status_code >= 400:
            raise Chrono24Error(f"Apify {resp.status_code}: {resp.text[:300]}")
        try:
            rows = resp.json()
        except ValueError as exc:
            raise Chrono24Error("Apify returned non-JSON") from exc
        if not isinstance(rows, list):
            raise Chrono24Error("Apify returned an unexpected payload")
        listings = [item for item in (parse_apify_item(r) for r in rows if isinstance(r, dict)) if item]
        return listings[:limit]


# ── Playwright provider ──────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(r"<a\b[^>]*href=\"([^\"]*--id\d+\.htm[^\"]*)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_IMG_RE = re.compile(r"<img\b[^>]*?(?:data-lazy-sweet-spot-master-src|data-src|src)=\"([^\"]+)\"", re.IGNORECASE)
_LDJSON_RE = re.compile(r"<script[^>]*type=\"application/ld\+json\"[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def _ldjson_offers(html: str, base_url: str) -> dict[str, dict]:
    """Collect Product/Offer data from JSON-LD blocks, keyed by listing id."""
    found: dict[str, dict] = {}

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for n in node:
                visit(n)
            return
        if not isinstance(node, dict):
            return
        offers = node.get("offers") if isinstance(node.get("offers"), dict) else {}
        url = node.get("url") or offers.get("url")
        if isinstance(url, str) and _ID_RE.search(url) and node.get("name") and _ID_RE.search(url).group(1) not in found:
            lid = _ID_RE.search(url).group(1)
            brand = node.get("brand")
            found[lid] = {
                "title": _clean(node.get("name")),
                "price": parse_price(offers.get("price") or node.get("price")),
                "currency": offers.get("priceCurrency") or node.get("priceCurrency"),
                "brand": brand.get("name") if isinstance(brand, dict) else _as_str(brand),
                "model": _as_str(node.get("model")),
                "reference": _as_str(node.get("mpn") or node.get("sku")),
                "image": node.get("image") if isinstance(node.get("image"), str) else None,
                "url": urljoin(base_url, url),
            }
        for value in node.values():
            visit(value)

    for block in _LDJSON_RE.findall(html):
        try:
            visit(json.loads(unescape(block.strip())))
        except ValueError:
            continue
    return found


def parse_search_html(html: str, base_url: str = "https://www.chrono24.com") -> list[Chrono24Listing]:
    """Parse a Chrono24 search results page without depending on CSS class names.

    Each listing is a link to ``/<brand>/<model>--id<number>.htm``. The text inside that link holds
    title, price, year and location; JSON-LD (when present) fills brand, reference and price.
    """
    ld = _ldjson_offers(html, base_url)
    by_id: dict[str, Chrono24Listing] = {}
    for href, inner in _ANCHOR_RE.findall(html):
        lid = _ID_RE.search(href).group(1)
        text = _clean(inner) or ""
        images = _uniq(urljoin(base_url, u) for u in _IMG_RE.findall(inner))
        existing = by_id.get(lid)
        if existing:
            existing.image_urls = _uniq([*existing.image_urls, *images])
            existing.image_url = existing.image_url or (existing.image_urls[0] if existing.image_urls else None)
            if len(text) > len(existing.title):
                existing.title = text[:300]
            continue
        if not text and not images:
            continue
        price_match = re.search(r"(?:[$€£₹¥]|CHF)\s?[\d.,]+|[\d.,]+\s?(?:[$€£₹¥]|CHF)", text)
        by_id[lid] = Chrono24Listing(
            listing_id=lid,
            title=text[:300] or lid,
            listing_url=urljoin(base_url, href.split("?")[0]),
            price=_display_price(price_match.group(0)) if price_match else None,
            currency=_currency_from_text(price_match.group(0)) if price_match else None,
            year=(_YEAR_RE.search(text).group(1) if _YEAR_RE.search(text) else None),
            image_url=images[0] if images else None,
            image_urls=images,
        )
    for lid, data in ld.items():
        item = by_id.get(lid)
        if item is None:
            if not data.get("title"):
                continue
            item = by_id[lid] = Chrono24Listing(listing_id=lid, title=data["title"], listing_url=data["url"])
        item.title = data.get("title") or item.title
        item.price = data.get("price") or item.price
        item.currency = data.get("currency") or item.currency
        item.brand = data.get("brand") or item.brand
        item.model = data.get("model") or item.model
        item.reference = data.get("reference") or item.reference
        if data.get("image"):
            item.image_urls = _uniq([data["image"], *item.image_urls])
            item.image_url = item.image_urls[0]
    return list(by_id.values())


class PlaywrightChrono24Provider:
    name = PROVIDER_PLAYWRIGHT
    user_agent = "Mozilla/5.0 (HypeVault research prototype; +https://github.com/TensorTorch777/HypeVault)"

    def __init__(self, base_url: str = "https://www.chrono24.com", *, respect_robots: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.respect_robots = respect_robots
        self._robots: urllib.robotparser.RobotFileParser | None = None

    async def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        if self._robots is None:
            parser = urllib.robotparser.RobotFileParser()
            try:
                async with httpx.AsyncClient(timeout=15, headers={"User-Agent": self.user_agent}) as http:
                    resp = await http.get(f"{self.base_url}/robots.txt")
                parser.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            except httpx.HTTPError as exc:
                raise Chrono24Error(f"could not read robots.txt: {exc}") from exc
            self._robots = parser
        return self._robots.can_fetch(self.user_agent, url)

    async def search(self, query: str, limit: int) -> list[Chrono24Listing]:
        url = f"{self.base_url}/search/index.htm?query={quote_plus(query)}&dosearch=true"
        if not await self._allowed(url):
            raise Chrono24Error("chrono24 robots.txt disallows the search page; use the apify provider")
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise Chrono24NotConfigured("playwright is not installed (pip install playwright && playwright install chromium)") from exc
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await (await browser.new_context(user_agent=self.user_agent, locale="en-US")).new_page()
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                if resp is not None and resp.status in (403, 429):
                    raise Chrono24Error(f"chrono24 answered {resp.status} (blocked)")
                try:
                    await page.wait_for_selector("a[href*='--id']", timeout=20000)
                except Exception:
                    pass
                html = await page.content()
            finally:
                await browser.close()
        listings = parse_search_html(html, self.base_url)
        if not listings and ("captcha" in html.lower() or "challenge" in html.lower()):
            raise Chrono24Error("chrono24 showed a bot challenge; use the apify provider")
        return listings[:limit]


# ── export ───────────────────────────────────────────────────────────────────

_export_lock = threading.Lock()


def export_results(directory: Path, *, query: str, provider: str, searched_at: str, listings: list[Chrono24Listing]) -> None:
    """Append one search's listings to listings.jsonl and listings.csv in ``directory``."""
    if not listings:
        return
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / "listings.jsonl"
    csv_path = directory / "listings.csv"
    with _export_lock:
        with jsonl.open("a", encoding="utf-8") as fh:
            for item in listings:
                fh.write(json.dumps({"searched_at": searched_at, "query": query, "provider": provider, **item.as_dict()}, ensure_ascii=False) + "\n")
        new_file = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            for item in listings:
                row = item.as_dict()
                row["image_urls"] = " ".join(item.image_urls)
                writer.writerow({"searched_at": searched_at, "query": query, "provider": provider, **row})


# ── entry point ──────────────────────────────────────────────────────────────


def resolve_provider_name(configured: str, apify_token: str) -> str:
    configured = (configured or "auto").strip().lower()
    if configured == "auto":
        return PROVIDER_APIFY if apify_token else PROVIDER_DATASET
    if configured not in PROVIDERS:
        raise ValueError(f"unknown CHRONO24_PROVIDER {configured!r}; use one of auto, {', '.join(PROVIDERS)}")
    return configured


def make_provider(name: str, settings) -> ApifyChrono24Provider | PlaywrightChrono24Provider:
    if name == PROVIDER_APIFY:
        return ApifyChrono24Provider(
            settings.apify_token,
            actor=settings.chrono24_apify_actor,
            currency=settings.chrono24_currency,
            enrich_details=settings.chrono24_enrich_details,
        )
    if name == PROVIDER_PLAYWRIGHT:
        return PlaywrightChrono24Provider(settings.chrono24_search_base)
    raise Chrono24NotConfigured("live Chrono24 search is off (CHRONO24_PROVIDER=dataset)")


async def search_live(
    provider, query: str, *, brand: str | None = None, limit: int = 12, export_dir: Path | None = None
) -> dict:
    """Run one live search and return a JSON-ready section."""
    searched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # Ask for a few extra rows: parts and other brands are filtered out afterwards.
    raw = await provider.search(query, min(limit * 2, 100))
    listings = [item for item in raw if _query_matches(item, brand)][:limit]
    if export_dir is not None:
        try:
            export_results(export_dir, query=query, provider=provider.name, searched_at=searched_at, listings=listings)
        except OSError as exc:
            _log.warning("chrono24_export_failed: %s", exc)
    prices = sorted(i.price for i in listings if i.price)
    return {
        "status": "ok" if listings else "no_match",
        "provider": provider.name,
        "query": query,
        "brand": brand,
        "fetched_at": searched_at,
        "count": len(listings),
        "price_range": {"min": prices[0], "max": prices[-1], "currency": listings[0].currency} if prices else None,
        "items": [i.as_dict() for i in listings],
    }
