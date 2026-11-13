"""Dealer listings for the site, read from ``market_listings`` (filled by scraper/dealer_scraper.py --save-db).

Searching our own crawl snapshot keeps page loads fast and keeps request volume to the dealers low:
the crawler runs on a schedule, not on every page view.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from market import repository
from market.chrono24_live import build_query
from market.dealers.sites import SITES, SOURCE_LABELS
from market.types import canonical_brand


def _item(row) -> dict:
    d = repository.listing_dict(row)
    prov = row.provenance or {}
    d.update(
        {
            "box": prov.get("box"),
            "papers": prov.get("papers"),
            "case_size": prov.get("case_size"),
            "image_urls": prov.get("image_urls") or ([d["image_url"]] if d["image_url"] else []),
        }
    )
    return d


async def search_dealers(
    session: AsyncSession, product_name: str, brand: str | None = None, *, limit: int = 8
) -> dict:
    query = build_query(product_name, brand)[:200]
    canon = canonical_brand(brand) if brand else None
    since = date.today() - timedelta(days=settings.dealer_max_age_days)
    sections = []
    for source in SITES:
        rows, stats = await repository.find_listings(session, source, query, canon, limit=limit, since=since, sort="near_median")
        broadened = False
        if not rows:
            # No exact match: same brand and collection (first terms of the query), still useful context.
            terms = repository.search_terms(query)
            if len(terms) > 3:
                rows, stats = await repository.find_listings(
                    session, source, " ".join(terms[:3]), canon, limit=limit, since=since, sort="near_median"
                )
                broadened = bool(rows)
        sections.append(
            {
                "source": source,
                "label": SOURCE_LABELS[source],
                "count": stats["count"] if stats else len(rows),
                "stats": stats,
                "broadened": broadened,
                "last_seen": max((r.observed_on for r in rows), default=None),
                "items": [_item(r) for r in rows],
            }
        )
    for s in sections:
        s["last_seen"] = s["last_seen"].isoformat() if s["last_seen"] else None
    return {
        "query": query,
        "brand": canon or brand,
        "max_age_days": settings.dealer_max_age_days,
        "total": sum(len(s["items"]) for s in sections),
        "sections": sections,
        "note": "Asking prices from dealer websites, collected by our crawler. Not sale prices, and not evidence of authenticity.",
    }
