"""Run a dealer crawl and save what it finds (CSV + JSONL, and ``market_listings`` rows for the site)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from market.dealers.http import BlockedError, DisallowedError, FetchError, PoliteFetcher
from market.dealers.sites import DealerListing, DealerSite
from market.types import MarketItem, canonical_brand, excluded_listing

_log = logging.getLogger(__name__)

CRAWLER_VERSION = "hypevault-dealer-crawler/1"

CSV_FIELDS = (
    "source", "listing_id", "brand", "model", "reference", "title", "price", "currency", "condition", "year",
    "box", "papers", "case_size", "case_material", "dial", "movement", "available", "listing_url", "image_url",
    "image_urls", "scraped_at",
)


@dataclass
class CrawlReport:
    source: str
    listings: list[DealerListing] = field(default_factory=list)
    discovered: int = 0
    pages_fetched: int = 0
    skipped: int = 0
    blocked: str | None = None
    errors: list[str] = field(default_factory=list)


def _brand_matches(listing: DealerListing, brand: str) -> bool:
    wanted = canonical_brand(brand) or brand
    got = canonical_brand(listing.brand) or listing.brand or ""
    if got.casefold() == wanted.casefold():
        return True
    # Same brand written differently ("PATEK PHILIPPE", "A. Lange & Sohne") or brand only in the title.
    key = [t for t in wanted.casefold().replace("&", " ").replace(".", " ").split() if len(t) > 2]
    haystack = f"{listing.brand or ''} {listing.title}".casefold()
    return bool(key) and all(t in haystack for t in key[:2])


async def crawl_site(
    site: DealerSite,
    fetcher: PoliteFetcher,
    brands: list[str],
    per_brand: int,
    *,
    include_sold: bool = False,
    save_html_dir: Path | None = None,
    on_listing: Callable[[DealerListing], None] | None = None,
) -> CrawlReport:
    report = CrawlReport(site.source)
    for brand in brands:
        try:
            urls = await site.discover(fetcher, brand, per_brand)
        except BlockedError as exc:
            report.blocked = str(exc)
            _log.warning("%s: %s", site.source, exc)
            return report
        except FetchError as exc:
            report.errors.append(f"discover {brand}: {exc}")
            continue
        report.discovered += len(urls)
        _log.info("%s: %s -> %d product pages", site.source, brand, len(urls))
        for url in urls:
            try:
                html = await fetcher.get_text(url)
            except BlockedError as exc:
                report.blocked = str(exc)
                _log.warning("%s: %s", site.source, exc)
                return report
            except DisallowedError:
                report.skipped += 1
                continue
            except FetchError as exc:
                report.errors.append(str(exc))
                continue
            report.pages_fetched += 1
            if not html:
                report.skipped += 1
                continue
            if save_html_dir is not None:
                save_html_dir.mkdir(parents=True, exist_ok=True)
                (save_html_dir / f"{site.source}-{site.listing_id(url, {})}.html").write_text(html, encoding="utf-8")
            listing = site.parse(url, html, brand)
            if (
                listing is None
                or not _brand_matches(listing, brand)
                or excluded_listing(listing.title)
                or (not listing.available and not include_sold)
            ):
                report.skipped += 1
                continue
            report.listings.append(listing)
            if on_listing:
                on_listing(listing)
    return report


def export_listings(directory: Path, listings: list[DealerListing]) -> None:
    """Append to <dir>/listings.jsonl and listings.csv (header written once)."""
    if not listings:
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "listings.jsonl").open("a", encoding="utf-8") as fh:
        for item in listings:
            fh.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")
    csv_path = directory / "listings.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for item in listings:
            row = item.as_dict()
            row["image_urls"] = " ".join(item.image_urls)
            writer.writerow(row)


def to_market_items(listings: list[DealerListing], observed: datetime | None = None) -> list[MarketItem]:
    """Rows for ``market_listings``. Only the five supported brands (the table's brand column is canonical)."""
    observed = observed or datetime.now(timezone.utc)
    items: list[MarketItem] = []
    for item in listings:
        brand = canonical_brand(item.brand)
        if not brand or not item.price or not item.currency:
            continue
        items.append(
            MarketItem(
                source=item.source,
                external_id=item.listing_id[:128],
                brand=brand,
                title=item.title[:512],
                price=float(item.price),
                currency=item.currency[:8],
                observed_on=observed.date() if isinstance(observed, datetime) else date.today(),
                observed_at=observed,
                model=(item.model or None) and item.model[:256],
                reference=(item.reference or None) and item.reference[:128],
                condition=(item.condition or None) and item.condition[:128],
                year=(item.year or None) and item.year[:16],
                item_url=item.listing_url,
                image_url=item.image_url,
                provenance={
                    "crawler": CRAWLER_VERSION,
                    "page_url": item.listing_url,
                    "box": item.box,
                    "papers": item.papers,
                    "case_size": item.case_size,
                    "case_material": item.case_material,
                    "dial": item.dial,
                    "image_urls": item.image_urls[:12],
                },
            )
        )
    return items
