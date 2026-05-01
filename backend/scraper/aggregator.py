"""Merge platform scrapers with Redis cache (30m TTL, refresh when ≤5m left)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from redis_client import get_redis
from scraper.readiness import check_scraper_readiness
from scraper.chrono24 import scrape_chrono24
from scraper.ebay import scrape_ebay
from scraper.stockx import scrape_stockx

_log = logging.getLogger(__name__)

CACHE_TTL_SEC = 30 * 60
MIN_REMAINING_SEC = 5 * 60


def _cache_key(product_name: str) -> str:
    return f"prices:{product_name.strip().lower()}"


async def aggregate_prices(product_name: str) -> dict:
    try:
        scraper_ready, scraper_state = check_scraper_readiness()
        if not scraper_ready:
            return {
                "stockx": [],
                "chrono24": [],
                "ebay": [],
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "cache_ttl_sec": CACHE_TTL_SEC,
                "cache_remaining_sec": 0,
                "error": "scraper_not_ready",
                "scraper_state": scraper_state,
            }

        redis = await get_redis()
        key = _cache_key(product_name)
        try:
            ttl = await redis.ttl(key)
            raw = await redis.get(key)
            if raw is not None and ttl > MIN_REMAINING_SEC:
                payload = json.loads(raw)
                payload["cache_ttl_sec"] = CACHE_TTL_SEC
                payload["cache_remaining_sec"] = max(0, int(ttl))
                return payload
        except Exception as exc:
            _log.warning("redis_cache_read_failed: %s", exc)

        stockx, chrono24, ebay = await asyncio.gather(
            scrape_stockx(product_name),
            scrape_chrono24(product_name),
            scrape_ebay(product_name),
        )
        payload = {
            "stockx": stockx,
            "chrono24": chrono24,
            "ebay": ebay,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "cache_ttl_sec": CACHE_TTL_SEC,
            "cache_remaining_sec": CACHE_TTL_SEC,
        }
        try:
            await redis.set(key, json.dumps(payload), ex=CACHE_TTL_SEC)
        except Exception as exc:
            _log.warning("redis_cache_write_failed: %s", exc)
        return payload
    except Exception as exc:
        _log.exception("aggregate_prices_failed: %s", exc)
        return {
            "stockx": [],
            "chrono24": [],
            "ebay": [],
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "cache_ttl_sec": CACHE_TTL_SEC,
            "cache_remaining_sec": 0,
            "error": "aggregation_failed",
        }
