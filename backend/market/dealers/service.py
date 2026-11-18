"""The dealer section of a market search, read from ``market_listings`` (filled by scraper/dealer_scraper.py --save-db).

``market.service.search_market`` calls this with the same query and brand it sends to eBay, so every page
that shows the market (listing page, upload result, market search) gets dealers below eBay from one
request. Searching our own crawl snapshot keeps page loads fast and request volume to the dealers low:
the crawler runs on a schedule, never on a page view.
"""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlsplit

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
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


async def dealer_section(session: AsyncSession, query: str, brand: str | None, *, limit: int = 8) -> dict:
    """Matches per dealer for `query` (already normalised) and the canonical `brand`, newest crawl window only."""
    since = date.today() - timedelta(days=settings.dealer_max_age_days)
    groups = []
    for source, cls in SITES.items():
        # Count/median in each dealer's own currency, so GBP stores still get them. Never mixed with eBay's USD.
        currency = (getattr(cls, "default_currency", None) or "USD").upper()
        rows, stats = await repository.find_listings(
            session, source, query, brand, limit=limit, since=since, sort="near_median", stats_currency=currency
        )
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
