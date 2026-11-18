"""The dealer section of a market search, read from ``market_listings`` (filled by scraper/dealer_scraper.py --save-db).

``market.service.search_market`` calls this with the same query and brand it sends to eBay, so every page
that shows the market (listing page, upload result, market search) gets dealers below eBay from one
request. Searching our own crawl snapshot keeps page loads fast and request volume to the dealers low:
the crawler runs on a schedule, never on a page view.
"""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlsplit

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import MarketListing
from market import repository
from market.dealers.sites import SITES, SOURCE_LABELS

NOTE = "Asking prices from dealer websites, collected by our crawler. Not sale prices, and not evidence of authenticity."

# Image CDNs that resize on request. The 1916 Company serves 3600px PNGs; cards are ~300px wide.
_RESIZING_HOSTS = ("imgix.net",)


def _card_image(url: str | None) -> str | None:
    if url and urlsplit(url).hostname and urlsplit(url).hostname.endswith(_RESIZING_HOSTS) and "?" not in url:
        return f"{url}?w=640&auto=format"
    return url


def _item(row) -> dict:
    d = repository.listing_dict(row)
    prov = row.provenance or {}
    d.update(
        {
            "image_url": _card_image(d["image_url"]),
            "box": prov.get("box"),
            "papers": prov.get("papers"),
            "case_size": prov.get("case_size"),
        }
    )
    return d


async def _dealer_matches(
    session: AsyncSession, source: str, query: str, brand: str | None, since: date, limit: int
) -> tuple[list[MarketListing], dict | None, str | None]:
    """One dealer's current listings matching the query: rows (closest to the median first), stats, currency.

    Each crawl adds one observation per watch per day, so only the newest row per listing counts; otherwise a
    watch seen in two weekly crawls would appear twice and skew the count and median. Cards and stats use one
    currency, the one most of the dealer's matches are priced in, so a median never mixes USD with GBP or EUR.
    """
    clauses = repository._match(source, brand, query) + [MarketListing.observed_on >= since]
    latest = (
        select(MarketListing.id)
        .where(and_(*clauses))
        .distinct(MarketListing.external_id)
        .order_by(MarketListing.external_id, MarketListing.observed_on.desc(), MarketListing.imported_at.desc())
    )
    current = MarketListing.id.in_(latest.scalar_subquery())
    top = (
        await session.execute(
            select(MarketListing.currency)
            .where(current)
            .group_by(MarketListing.currency)
            .order_by(func.count().desc(), MarketListing.currency)
            .limit(1)
        )
    ).first()
    if top is None:
        return [], None, None
    currency = top[0]
    where = and_(current, MarketListing.currency == currency)
    count, lo, median, hi = (
        await session.execute(
            select(
                func.count(),
                func.min(MarketListing.price),
                func.percentile_cont(0.5).within_group(MarketListing.price),
                func.max(MarketListing.price),
            ).where(where)
        )
    ).one()
    stats = {"count": int(count), "min": lo, "median": float(median), "max": hi} if count else None
    order = func.abs(MarketListing.price - stats["median"]) if stats else MarketListing.price
    rows = list((await session.execute(select(MarketListing).where(where).order_by(order, MarketListing.id).limit(limit))).scalars())
    return rows, stats, currency


async def dealer_section(session: AsyncSession, query: str, brand: str | None, *, limit: int = 8) -> dict:
    """Matches per dealer for `query` (already normalised) and the canonical `brand`, newest crawl window only."""
    since = date.today() - timedelta(days=settings.dealer_max_age_days)
    groups = []
    for source in SITES:
        rows, stats, currency = await _dealer_matches(session, source, query, brand, since, limit)
        if not rows:
            continue
        groups.append(
            {
                "source": source,
                "label": SOURCE_LABELS[source],
                "currency": currency,
                "count": stats["count"] if stats else len(rows),
                "stats": stats,
                "last_seen": max(r.observed_on for r in rows).isoformat(),
                "items": [_item(r) for r in rows],
            }
        )
    return {
        "status": "ok" if groups else "no_match",
        "max_age_days": settings.dealer_max_age_days,
        "searched": [SOURCE_LABELS[s] for s in SITES],
        "dealers": groups,
        "note": NOTE,
    }
