"""The three dealers: where their product pages are and how to read them.

Discovery uses the site's XML sitemaps (listed in robots.txt) first, then falls back to paging through the
brand's category pages. Product pages are read from schema.org JSON-LD and OpenGraph tags when present,
then from the labelled spec rows shown on the page ("Reference Number", "Year", "Box", ...).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from market.dealers.extract import (
    clean_title,
    first_money,
    jsonld_product,
    label_values,
    links,
    meta_tags,
    parse_money,
    parse_sitemap,
    path_segments,
    reference_from_title,
    slug_candidates,
    visible_lines,
    year_of,
    yes_no,
)
from market.dealers.http import BlockedError, DisallowedError, FetchError, PoliteFetcher
from market.types import parse_price

_log = logging.getLogger(__name__)
_H1_RE = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class DealerListing:
    source: str
    listing_id: str
    title: str
    listing_url: str
    price: float | None
    currency: str | None
    brand: str | None = None
    model: str | None = None
    reference: str | None = None
    condition: str | None = None
    year: str | None = None
    box: bool | None = None
    papers: bool | None = None
    case_size: str | None = None
    case_material: str | None = None
    dial: str | None = None
    movement: str | None = None
    available: bool = True
    image_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())

    def as_dict(self) -> dict:
        return asdict(self)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _images(value: Any) -> list[str]:
    out: list[str] = []
    for v in value if isinstance(value, list) else [value]:
        url = v.get("url") or v.get("contentUrl") if isinstance(v, dict) else v
        if isinstance(url, str) and url.startswith("http") and url not in out:
            out.append(url)
    return out


_CONDITION_SCHEMA = {
    "newcondition": "New",
    "usedcondition": "Pre-owned",
    "refurbishedcondition": "Refurbished",
    "damagedcondition": "Damaged",
}


class DealerSite:
    source: str = ""
    label: str = ""
    base_url: str = ""
    default_currency: str | None = None
    default_condition: str | None = None
    site_names: tuple[str, ...] = ()
    price_prefer: tuple[str, ...] = ()
    max_sitemaps: int = 40
    max_category_pages: int = 10
    spec_labels: dict[str, tuple[str, ...]] = {}

    def __init__(self) -> None:
        self._sitemap_urls: list[str] | None = None

    # ── per-site hooks ──────────────────────────────────────────────────────

    def is_product_url(self, url: str, slug: str) -> bool:
        raise NotImplementedError

    def category_url(self, slug: str, page: int) -> str:
        raise NotImplementedError

    def listing_id(self, url: str, specs: dict[str, str]) -> str:
        return path_segments(url)[-1]

    def model_from_url(self, url: str) -> str | None:
        return None

    def embedded(self, html: str) -> dict[str, str]:
        """Extra spec values pulled from the page's own embedded JSON. Overrides scraped rows."""
        return {}

    def extra_images(self, html: str) -> list[str]:
        """Image URLs from embedded JSON, tried before og:image."""
        return []

    # ── discovery ───────────────────────────────────────────────────────────

    async def _all_sitemap_urls(self, fetcher: PoliteFetcher) -> list[str]:
        """Every page URL in the site's sitemaps (fetched once per run, product sitemaps first)."""
        if self._sitemap_urls is not None:
            return self._sitemap_urls
        queue = await fetcher.sitemaps(self.base_url) or [f"{self.base_url}/sitemap.xml", f"{self.base_url}/sitemap_index.xml"]
        seen: set[str] = set()
        urls: list[str] = []
        fetched = 0
        while queue and fetched < self.max_sitemaps:
            # Product sitemaps first; blog/CMS/image sitemaps are skipped.
            queue.sort(key=lambda u: (0 if re.search(r"product|item|watch|pdp", u, re.I) else 1))
            sm = queue.pop(0)
            if sm in seen or re.search(r"blog|journal|cms|content|image|video|news|page-sitemap|category|categories|static", sm, re.I):
                continue
            seen.add(sm)
            try:
                xml = await fetcher.get_text(sm)
            except DisallowedError:
                continue
            except BlockedError:
                raise
            except FetchError as exc:
                _log.info("%s: sitemap %s failed: %s", self.source, sm, exc)
                continue
            fetched += 1
            if not xml:
                continue
            children, pages = parse_sitemap(xml)
            queue.extend(c for c in children if c not in seen)
            urls.extend(pages)
        self._sitemap_urls = urls
        _log.info("%s: %d URLs from %d sitemaps", self.source, len(urls), fetched)
        return urls

    async def discover(self, fetcher: PoliteFetcher, brand: str, limit: int) -> list[str]:
        slugs = slug_candidates(brand)
        found: dict[str, None] = {}
        try:
            for url in await self._all_sitemap_urls(fetcher):
                if any(self.is_product_url(url, s) for s in slugs):
                    found.setdefault(url, None)
                    if len(found) >= limit:
                        return list(found)
        except BlockedError:
            raise
        if len(found) >= limit:
            return list(found)
        # Fallback: page through the brand's category listing.
        for slug in slugs:
            hit_any = False
            for page in range(1, self.max_category_pages + 1):
                try:
                    html = await fetcher.get_text(self.category_url(slug, page))
                except DisallowedError:
                    break
                if not html:
                    break
                new = [u for u in links(html, self.base_url) if self.is_product_url(u, slug) and u not in found]
                if not new:
                    break
                hit_any = True
                for u in new:
                    found.setdefault(u, None)
                if len(found) >= limit:
                    return list(found)[:limit]
            if hit_any:
                break
        return list(found)[:limit]

    # ── product page ────────────────────────────────────────────────────────

    def parse(self, url: str, html: str, brand_hint: str | None = None) -> DealerListing | None:
        product = jsonld_product(html) or {}
        meta = meta_tags(html)
        lines = visible_lines(html)
        specs = label_values(lines, self.spec_labels)
        # Values a site embeds in page JSON (more reliable than visible rows) win over scraped rows.
        specs = {**specs, **{k: v for k, v in self.embedded(html).items() if v not in (None, "")}}

        h1 = _H1_RE.search(html)
        h1_text = " ".join(_TAG_RE.sub(" ", h1.group(1)).split()) if h1 else None
        title = clean_title(
            product.get("name") or h1_text or meta.get("og:title") or meta.get("twitter:title") or meta.get("title"),
            self.site_names,
        )
        if not title:
            return None

        offers = _first(product.get("offers")) or {}
        if not isinstance(offers, dict):
            offers = {}
        price = parse_price(offers.get("price") or offers.get("lowPrice"))
        currency = offers.get("priceCurrency")
        if not price:
            price = parse_price(meta.get("product:price:amount") or meta.get("og:price:amount"))
            currency = currency or meta.get("product:price:currency") or meta.get("og:price:currency")
        if not price and specs.get("price"):
            price, currency = parse_money(specs["price"])
        if not price:
            title_at = next((i for i, l in enumerate(lines) if title.casefold()[:30] in l.casefold()), 0)
            price, cur = first_money(lines, after=title_at, prefer=self.price_prefer)
            currency = currency or cur
        if not price:
            return None
        currency = (currency or self.default_currency or "").upper() or None

        availability = str(offers.get("availability", "")).casefold()
        sold = any(w in availability for w in ("outofstock", "soldout", "discontinued")) or any(
            l.casefold() in ("sold", "sold out", "this watch has been sold") for l in lines[:200]
        )

        brand_node = product.get("brand")
        brand = (brand_node.get("name") if isinstance(brand_node, dict) else brand_node) or specs.get("brand") or brand_hint
        reference = specs.get("reference") or product.get("mpn") or reference_from_title(title)
        condition_raw = specs.get("condition") or str(offers.get("itemCondition", "")).rsplit("/", 1)[-1]
        condition = _CONDITION_SCHEMA.get(condition_raw.casefold(), condition_raw) or self.default_condition

        # Embedded booleans (e.g. 1916's c_hasBox) arrive as the strings "true"/"false".
        box = yes_no(specs.get("box"))
        papers = yes_no(specs.get("papers"))
        combined = (specs.get("box_papers") or "").casefold()
        if combined:
            if box is None:
                box = True if "box" in combined and not combined.startswith("no") else (False if combined.startswith("no") else None)
            if papers is None:
                papers = True if ("paper" in combined or "certificate" in combined) and "no paper" not in combined else (False if combined.startswith("no") else None)

        images = _images(product.get("image")) + self.extra_images(html) + _images(meta.get("og:image"))
        images = [u for u in dict.fromkeys(images) if "/chronext.jpg" not in u]  # drop the logo placeholder

        return DealerListing(
            source=self.source,
            listing_id=self.listing_id(url, specs),
            title=title[:300],
            listing_url=url.split("?")[0],
            price=price,
            currency=currency,
            brand=brand,
            model=specs.get("model") or (product.get("model") if isinstance(product.get("model"), str) else None) or self.model_from_url(url),
            reference=reference,
            condition=condition or None,
            year=year_of(specs.get("year") or product.get("productionDate") or product.get("releaseDate")),
            box=box,
            papers=papers,
            case_size=specs.get("case_size"),
            case_material=specs.get("case_material"),
            dial=specs.get("dial"),
            movement=specs.get("movement"),
            available=not sold,
            image_url=images[0] if images else None,
            image_urls=images,
        )


class The1916Company(DealerSite):
    source = "the1916company"
    label = "The 1916 Company"
    base_url = "https://www.the1916company.com"
    default_currency = "USD"
    default_condition = "Pre-owned"
    site_names = ("The 1916 Company", "1916 Company", "WatchBox")
    price_prefer = ("Current Price",)
    spec_labels = {
        "reference": ("Reference Number", "Reference No.", "Reference"),
        "sku": ("SKU",),
        "year": ("Year", "Year of Production"),
        "box": ("Box",),
        "papers": ("Papers",),
        "condition": ("Condition",),
        "case_size": ("Case Size",),
        "case_material": ("Case Material",),
        "dial": ("Dial Color", "Dial Colour"),
        "movement": ("Movement",),
        "model": ("Model", "Collection"),
        "brand": ("Brand",),
        "price": ("Current Price",),
    }

    def is_product_url(self, url: str, slug: str) -> bool:
        return bool(re.match(
            rf"^https?://(?:www\.)?the1916company\.com/pre-owned/{re.escape(slug)}/[a-z0-9-]+/(?!ref-)[a-z0-9-]*\d[a-z0-9-]*/?$",
            url.split("?")[0], re.IGNORECASE,
        ))

    def category_url(self, slug: str, page: int) -> str:
        # Salesforce Commerce Cloud paging: start = offset, sz = page size.
        return f"{self.base_url}/pre-owned/{slug}/?start={(page - 1) * 48}&sz=48"

    def listing_id(self, url: str, specs: dict[str, str]) -> str:
        sku = re.sub(r"\D", "", specs.get("sku", ""))
        return sku or path_segments(url)[-1]

    def model_from_url(self, url: str) -> str | None:
        segs = path_segments(url)
        return segs[2].replace("-", " ").title() if len(segs) >= 4 else None

    # 1916 embeds the product as a JS object with c_* keys (Salesforce Commerce Cloud).
    _C = {
        "year": "c_year",
        "condition": "c_condition",
        "case_size": "c_caseSize",
        "case_material": "c_displayedMaterial",
        "dial": "c_dialColor",
        "movement": "c_movement",
        "reference": "c_baseRefNum",
        "model": "c_brandFamily",
    }
    _BOOL = {"box": "c_hasBox", "papers": "c_hasPapers"}

    def embedded(self, html: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for field, key in self._C.items():
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', html)
            if m:
                out[field] = m.group(1).encode().decode("unicode_escape")
        for field, key in self._BOOL.items():
            m = re.search(rf'"{key}"\s*:\s*(true|false)', html)
            if m:
                out[field] = "Yes" if m.group(1) == "true" else "No"
        return out


class Chronext(DealerSite):
    source = "chronext"
    label = "Chronext"
    base_url = "https://www.chronext.com"
    default_currency = None  # Chronext prices in the visitor's currency; read it from the page
    site_names = ("CHRONEXT", "Chronext")
    spec_labels = {
        "reference": ("Reference number", "Reference", "Ref. No.", "Ref."),
        "year": ("Year", "Year of production", "Production year"),
        "condition": ("Condition",),
        "box_papers": ("Scope of delivery", "Delivery scope", "Box & papers", "Box and papers"),
        "box": ("Box",),
        "papers": ("Papers", "Certificate"),
        "case_size": ("Case diameter", "Case size"),
        "case_material": ("Case material",),
        "dial": ("Dial colour", "Dial color", "Dial"),
        "movement": ("Movement", "Calibre"),
        "model": ("Model", "Collection"),
        "brand": ("Brand",),
    }

    def is_product_url(self, url: str, slug: str) -> bool:
        clean = url.split("?")[0]
        if "/journal/" in clean:
            return False
        return bool(re.match(rf"^https?://(?:www\.)?chronext\.com/{re.escape(slug)}/[^/]+/[^/]*\d[^/]*/?$", clean, re.I))

    def category_url(self, slug: str, page: int) -> str:
        return f"{self.base_url}/{slug}" + (f"?page={page}" if page > 1 else "")

    def model_from_url(self, url: str) -> str | None:
        segs = path_segments(url)
        return segs[1].replace("-", " ").title() if len(segs) >= 3 else None

    _IMG_RE = re.compile(r"https://cdn-products\.chronext\.com/[A-Za-z0-9/_-]+\.(?:png|jpe?g|webp)")

    def extra_images(self, html: str) -> list[str]:
        # Real product shots live on cdn-products.chronext.com; og:image is the logo.
        out: list[str] = []
        for u in self._IMG_RE.findall(html):
            if u not in out:
                out.append(u)
        return out[:6]


class EuropeanWatchCo(DealerSite):
    source = "europeanwatch"
    label = "European Watch Co."
    base_url = "https://www.europeanwatch.com"
    default_currency = "USD"
    default_condition = "Pre-owned"
    site_names = ("European Watch Co.", "European Watch Company", "EWC")
    spec_labels = {
        "reference": ("Reference", "Ref.", "Ref"),
        "year": ("Year", "Circa"),
        "condition": ("Condition",),
        "box_papers": ("Box and Papers", "Box & Papers", "Accompaniments"),
        "box": ("Box",),
        "papers": ("Papers",),
        "case_size": ("Case Size", "Case Diameter", "Diameter"),
        "case_material": ("Case Material", "Metal", "Material"),
        "dial": ("Dial Color", "Dial Colour", "Dial Type"),  # bare "Dial" grabs a UI label, not the value
        "movement": ("Movement", "Caliber", "Calibre"),
        "model": ("Model", "Collection"),
        "brand": ("Brand",),
    }

    def is_product_url(self, url: str, slug: str) -> bool:
        clean = url.split("?")[0]
        # /watch/patek-philippe-5726a-001-...-73399  (trailing numeric id)
        return bool(re.match(rf"^https?://(?:www\.)?europeanwatch\.com/watch/{re.escape(slug)}-.+-\d+/?$", clean, re.I))

    def parse(self, url: str, html: str, brand_hint: str | None = None) -> DealerListing | None:
        listing = super().parse(url, html, brand_hint)
        # EWC puts the year in the title as "...Circa 1976" when there's no spec row for it.
        if listing and not listing.year:
            listing.year = year_of(listing.title)
        return listing

    def category_url(self, slug: str, page: int) -> str:
        # EWC paginates its brand pages with ?p=
        return f"{self.base_url}/brand/{slug}" + (f"?p={page}" if page > 1 else "")

    def listing_id(self, url: str, specs: dict[str, str]) -> str:
        m = re.search(r"-(\d+)/?$", url.split("?")[0])
        return m.group(1) if m else path_segments(url)[-1]

    def model_from_url(self, url: str) -> str | None:
        return None


_HTML_SITES = (The1916Company, Chronext, EuropeanWatchCo)


def _build_registry() -> dict[str, type[DealerSite]]:
    # shopify.py imports from this module, so it is imported here (after this module is loaded), not at top level.
    from market.dealers.shopify import SHOPIFY_SITES

    return {cls.source: cls for cls in (*_HTML_SITES, *SHOPIFY_SITES)}


class _LazyRegistry(dict):
    """The site registry, populated on first access so sites.py has no import-time dependency on shopify.py."""

    _ready = False

    def _ensure(self) -> None:
        if not self._ready:
            self.update(_build_registry())
            self._ready = True

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def __len__(self):
        self._ensure()
        return super().__len__()

    def values(self):
        self._ensure()
        return super().values()

    def items(self):
        self._ensure()
        return super().items()

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)


SITES: dict[str, type[DealerSite]] = _LazyRegistry()


class _LazyLabels(dict):
    def _ensure(self) -> None:
        if not self:
            self.update({k: v.label for k, v in SITES.items()})

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)

    def items(self):
        self._ensure()
        return super().items()


SOURCE_LABELS = _LazyLabels()


def site_for_url(url: str) -> DealerSite | None:
    host = urlsplit(url).netloc.casefold()
    for cls in SITES.values():
        if urlsplit(cls.base_url).netloc.removeprefix("www.") in host:
            return cls()
    return None


__all__ = ["DealerListing", "DealerSite", "SITES", "SOURCE_LABELS", "site_for_url"]
