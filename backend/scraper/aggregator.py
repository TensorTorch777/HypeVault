"""Legacy price-bundle adapter over the market service.

Prices now come only from permitted sources: the eBay Browse API and the imported Chrono24
dataset. The old Playwright scrapers (StockX, Chrono24, eBay HTML) are no longer called, because
those sites' terms do not allow automated scraping. The bundle keeps its old keys so the SQS
worker that writes `price_snapshots` keeps working.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import settings
from database import AsyncSessionLocal
from market.service import search_market
from redis_client import get_redis

_log = logging.getLogger(__name__)
CACHE_TTL_SEC = settings.market_cache_ttl_sec


def _legacy_rows(section: dict, platform: str) -> list[dict]:
    return [
        {
            "platform": platform,
            "title": item["title"],
            "lowest_ask": item["price"],
            "currency": item["currency"],
            "estimated_delivery": None,
            "seller_rating": f"{item['seller_feedback_pct']:.1f}%" if item.get("seller_feedback_pct") else "N/A",
            "url": item.get("item_url"),
        }
        for item in section.get("items") or []
    ]


async def aggregate_prices(product_name: str, brand: str | None = None) -> dict:
    try:
        try:
            redis = await get_redis()
        except Exception:
            redis = None
        async with AsyncSessionLocal() as session:
            market = await search_market(session, redis, product_name, brand)
        sources = market["sources"]
        return {
            "stockx": [],
            "chrono24": _legacy_rows(sources["chrono24_dataset"], "chrono24"),
            "ebay": _legacy_rows(sources["ebay"], "ebay"),
            "scraped_at": market["generated_at"],
            "cache_ttl_sec": CACHE_TTL_SEC,
            "market": market,
        }
    except Exception as exc:
        _log.exception("aggregate_prices_failed: %s", exc)
        return {
            "stockx": [],
            "chrono24": [],
            "ebay": [],
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "cache_ttl_sec": CACHE_TTL_SEC,
            "error": "aggregation_failed",
        }
