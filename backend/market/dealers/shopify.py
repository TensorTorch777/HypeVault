"""Generic crawler for Shopify watch dealers via the public products.json feed.

Every Shopify store serves /products.json (and /collections/<handle>/products.json), paginated, with clean
structured data: title, vendor, product_type, tags, variants (price), images. That is far more reliable than
scraping HTML and is the same data the storefront renders, so one class covers any number of Shopify dealers.

Brand, reference, year and box/papers are read from the title, tags and options. robots.txt and request
spacing are still honoured by the shared PoliteFetcher; a bot challenge or 403 stops the site.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit

from market.dealers.extract import (
    normalize_reference,
    parse_money,
    reference_from_title,
    year_of,
)
from market.dealers.http import BlockedError, FetchError, PoliteFetcher
from market.dealers.sites import DealerListing, DealerSite, _CONDITION_SCHEMA
from market.types import canonical_brand

_log = logging.getLogger(__name__)

_BOX_TAG = re.compile(r"\b(full set|box (?:and|&|/) papers|complete set)\b", re.IGNORECASE)
_BOX_ONLY = re.compile(r"\bbox only\b", re.IGNORECASE)
_PAPERS_ONLY = re.compile(r"\bpapers only\b", re.IGNORECASE)
_WATCH_ONLY = re.compile(r"\b(watch only|head only|no box|no papers)\b", re.IGNORECASE)
_SIZE_RE = re.compile(r"\b(\d{2}(?:\.\d)?)\s?mm\b", re.IGNORECASE)
_TAG_SPLIT = re.compile(r"[;,]")
_HTML_TAG = re.compile(r"<[^>]+>")


def _tag_value(tags: list[str], *prefixes: str) -> str | None:
    """Shopify dealers often tag specs as 'Year:2019', 'Reference_5711'. Pull the value for a prefix."""
    for tag in tags:
        for pre in prefixes:
            m = re.match(rf"{re.escape(pre)}\s*[:_=-]\s*(.+)", tag.strip(), re.IGNORECASE)
            if m and m.group(1).strip():
                return m.group(1).strip()
    return None


class ShopifyDealerSite(DealerSite):
    """A Shopify storefront. Discovery and parsing use products.json, not HTML."""

    uses_feed = True  # crawl.py parses these from the cached feed, no per-page fetch

    #: Product types (whole words) or whole tags that are not wristwatches we want. Whole words only, so a
    #: "Serviced" or "Service History" tag on a pre-owned watch does not count as a "service" product.
    reject_types = (
        "strap", "band", "accessory", "accessories", "winder", "jewel", "jewellery", "jewelry", "ring",
        "necklace", "bracelet only", "service",
    )
    per_page = 250
    max_feed_pages = 20
    #: Shopify throttles products.json harder than pages; wait at least this long between feed requests.
    feed_delay = 8.0

    # Set when the store tags or names its brands differently from canonical.
    vendor_aliases: dict[str, str] = {}
    #: The store's home market. Shopify Markets converts prices into the visitor's local currency, picked
    #: from the IP address (from India it served rupees), so the crawl asks for this market explicitly.
    country: str | None = None

    def _brand_terms(self, brand: str) -> list[str]:
        base = brand.casefold()
        extra = [k for k, v in self.vendor_aliases.items() if v.casefold() == base]
        words = [w for w in re.split(r"[^\w]+", base) if len(w) > 2]
        return list(dict.fromkeys([base, *[a.casefold() for a in extra], *words]))

    def _matches_brand(self, product: dict, brand: str) -> bool:
        hay = " ".join(
            str(product.get(k, "")) for k in ("vendor", "title", "product_type")
        ).casefold() + " " + " ".join(str(t) for t in product.get("tags", [])).casefold()
        terms = self._brand_terms(brand)
        # Full brand name present, or (for multi-word brands) all significant words present.
        if brand.casefold() in hay:
            return True
        words = [w for w in terms if " " not in w]
        return len(words) >= 2 and all(w in hay for w in words[:2])

    def _is_watch(self, product: dict) -> bool:
        reject = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in self.reject_types) + r")s?\b", re.IGNORECASE)
        if reject.search(str(product.get("product_type") or "")):
            return False
        tags = product.get("tags") or []
        if isinstance(tags, str):
            tags = _TAG_SPLIT.split(tags)
        return not any(reject.fullmatch(str(t).strip()) for t in tags)

    async def _feed(self, fetcher: PoliteFetcher) -> list[dict]:
        """All products from the store feed (cached per run)."""
        cached = getattr(self, "_feed_cache", None)
        if cached is not None:
            return cached
        products: list[dict] = []
        # Give the feed extra breathing room so stricter stores don't 429 us.
        if fetcher.min_delay < self.feed_delay:
            fetcher._last[f"delay:{urlsplit(self.base_url).scheme}://{urlsplit(self.base_url).netloc}"] = self.feed_delay
        if self.country and self.default_currency:
            fetcher.set_cookies(self.base_url, {"localization": self.country, "cart_currency": self.default_currency})
        for page in range(1, self.max_feed_pages + 1):
            url = f"{self.base_url}/products.json?limit={self.per_page}&page={page}"
            try:
                data = await fetcher.get_json(url)
            except BlockedError:
                # A strict store that rate-limits the feed: keep what we have, skip the rest.
                if products:
                    _log.warning("%s: feed rate-limited after %d products; using what we have", self.source, len(products))
                    break
                raise
            try:
                self._check_currency(fetcher)
            except FetchError:
                self._feed_cache = []  # don't refetch the feed for every brand in this run
                raise
            batch = (data or {}).get("products") if isinstance(data, dict) else None
            if not batch:
                break
            products.extend(batch)
            if len(batch) < self.per_page:
                break
        self._feed_cache = products
        _log.info("%s: %d products in feed", self.source, len(products))
        return products

    def _check_currency(self, fetcher: PoliteFetcher) -> None:
        """The feed has no currency field. Shopify reports the one it priced in via the cart_currency cookie;
        if that is not the store's own currency, stop rather than label converted prices as GBP."""
        expected = (self.default_currency or "").upper()
        served = {v.upper() for v in fetcher.cookie_values(self.base_url, "cart_currency") if v}
        wrong = served - {expected}
        if expected and wrong:
            raise FetchError(f"{self.source}: store priced the feed in {', '.join(sorted(wrong))}, not {expected}; skipping")

    def _brand(self, product: dict, title: str, brand_hint: str | None) -> str | None:
        """The vendor field is the watch brand on some stores and the store's own name on others."""
        vendor = str(product.get("vendor") or "")
        if vendor in self.vendor_aliases:
            return self.vendor_aliases[vendor]
        if canonical_brand(vendor):
            return vendor
        if brand_hint and self._matches_brand({"title": title}, brand_hint):
            return brand_hint
        return vendor or brand_hint

    async def discover(self, fetcher: PoliteFetcher, brand: str, limit: int) -> list[str]:
        out: list[str] = []
        for product in await self._feed(fetcher):
            if self._matches_brand(product, brand) and self._is_watch(product):
                handle = product.get("handle")
                if handle:
                    out.append(f"{self.base_url}/products/{handle}")
                    if len(out) >= limit:
                        break
        return out

    def _product_by_handle(self, handle: str) -> dict | None:
        for p in getattr(self, "_feed_cache", []) or []:
            if p.get("handle") == handle:
                return p
        return None

    async def fetch_and_parse(self, fetcher: PoliteFetcher, url: str, brand_hint: str | None) -> DealerListing | None:
        """Parse from the cached feed product (no extra request needed)."""
        handle = url.rstrip("/").rsplit("/", 1)[-1]
        product = self._product_by_handle(handle)
        if product is None:
            data = await fetcher.get_json(f"{self.base_url}/products/{handle}.json")
            product = (data or {}).get("product") if isinstance(data, dict) else None
        if product is None:
            return None
        return self.parse_product(product, brand_hint)

    def parse_product(self, product: dict, brand_hint: str | None = None) -> DealerListing | None:
        title = " ".join(str(product.get("title", "")).split())
        if not title:
            return None
        variants = product.get("variants") or [{}]
        v0 = variants[0] if variants else {}
        price_raw = v0.get("price")
        # Shopify prices are bare decimal strings ("145000.00"); currency comes from the store default.
        price = None
        if price_raw not in (None, "", "0", "0.00"):
            try:
                price = float(str(price_raw).replace(",", ""))
            except ValueError:
                price, _ = parse_money(str(price_raw))
        currency = product.get("currency") or self.default_currency
        if not price or price <= 0:
            return None
        available = any(v.get("available") for v in variants) if variants else True
        tags = [str(t) for t in (product.get("tags") or [])]
        if isinstance(product.get("tags"), str):
            tags = [t.strip() for t in _TAG_SPLIT.split(product["tags"])]

        body = _HTML_TAG.sub(" ", product.get("body_html") or "")
        reference = _tag_value(tags, "ref", "reference", "model number") or reference_from_title(title)
        year = year_of(_tag_value(tags, "year") or title) or year_of(body)
        size = _tag_value(tags, "case size", "size", "diameter")
        if not size:
            m = _SIZE_RE.search(title) or _SIZE_RE.search(body)
            size = f"{m.group(1)}mm" if m else None
        condition = _tag_value(tags, "condition")
        tag_blob = " ".join(tags) + " " + title
        box = papers = None
        if _BOX_TAG.search(tag_blob):
            box = papers = True
        elif _WATCH_ONLY.search(tag_blob):
            box = papers = False
        if _BOX_ONLY.search(tag_blob):
            box, papers = True, False
        if _PAPERS_ONLY.search(tag_blob):
            box, papers = False, True

        images = []
        for img in product.get("images") or []:
            src = img.get("src") if isinstance(img, dict) else img
            if isinstance(src, str) and src.startswith("http"):
                images.append(src.split("?")[0])

        return DealerListing(
            source=self.source,
            listing_id=str(product.get("id") or product.get("handle")),
            title=title[:300],
            listing_url=f"{self.base_url}/products/{product.get('handle')}",
            price=price,
            currency=(currency or self.default_currency or "").upper() or None,
            brand=self._brand(product, title, brand_hint),
            model=_tag_value(tags, "model", "collection") or (product.get("product_type") or None),
            reference=reference,
            condition=_CONDITION_SCHEMA.get((condition or "").casefold(), condition) or self.default_condition,
            year=year,
            box=box,
            papers=papers,
            case_size=size,
            case_material=_tag_value(tags, "case material", "metal", "material"),
            dial=_tag_value(tags, "dial", "dial colour", "dial color"),
            movement=_tag_value(tags, "movement", "calibre", "caliber"),
            available=available,
            image_url=images[0] if images else None,
            image_urls=images[:8],
        )

    # HTML hooks are unused for Shopify, but keep the base interface satisfiable.
    def is_product_url(self, url: str, slug: str) -> bool:
        return "/products/" in url and urlsplit(self.base_url).netloc in url

    def category_url(self, slug: str, page: int) -> str:
        return f"{self.base_url}/collections/all"


class WatchCollectorsUK(ShopifyDealerSite):
    source = "watchcollectors"
    label = "Watch Collectors UK"
    base_url = "https://watchcollectors.co.uk"
    default_currency = "GBP"
    country = "GB"
    default_condition = "Pre-owned"


#: Candidate Shopify dealers. The --probe run on the live network keeps the ones that respond with a feed.
SHOPIFY_SITES: list[type[ShopifyDealerSite]] = [WatchCollectorsUK]
