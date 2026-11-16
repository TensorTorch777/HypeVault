"""Market search across permitted sources, with a Redis cache for live eBay results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from market import repository
from market.ebay_browse import EbayApiError, EbayBrowseClient, EbayNotConfigured
from market.types import SOURCE_CHRONO24_DATASET, SOURCE_EBAY, canonical_brand, price_stats

_log = logging.getLogger(__name__)
_TOKEN_KEY = "market:ebay:app_token"


def ebay_configured() -> bool:
    return bool(settings.ebay_client_id and settings.ebay_client_secret)


def _cache_key(query: str, brand: str | None) -> str:
    return f"market:v1:ebay:{(brand or '*').casefold()}:{' '.join(query.casefold().split())}"


def make_ebay_client(redis=None) -> EbayBrowseClient:
    async def token_get() -> str | None:
        return await redis.get(_TOKEN_KEY) if redis is not None else None

    async def token_set(token: str, ttl: int) -> None:
        if redis is not None:
            await redis.set(_TOKEN_KEY, token, ex=ttl)

    return EbayBrowseClient(
        settings.ebay_client_id,
        settings.ebay_client_secret,
        environment=settings.ebay_environment,
        marketplace_id=settings.ebay_marketplace_id,
        pseudonym_salt=settings.market_pseudonym_salt,
        token_get=token_get,
        token_set=token_set,
    )


async def _ebay_section(session: AsyncSession, redis, query: str, brand: str | None) -> dict:
    if not ebay_configured():
        return {"status": "not_configured", "items": [], "stats": None, "fetched_at": None, "cached": False}
    key = _cache_key(query, brand)
    if redis is not None:
        try:
            raw = await redis.get(key)
            if raw:
                payload = json.loads(raw)
                payload["cached"] = True
                return payload
        except Exception as exc:  # cache is an optimisation only
            _log.warning("market_cache_read_failed: %s", exc)
    client = make_ebay_client(redis)
    try:
        items, total = await client.search(query, brand=brand, limit=50)
    except EbayNotConfigured:
        return {"status": "not_configured", "items": [], "stats": None, "fetched_at": None, "cached": False}
    except (EbayApiError, Exception) as exc:
        _log.warning("market_ebay_failed: %s", exc)
        return {"status": "error", "items": [], "stats": None, "fetched_at": None, "cached": False}
    finally:
        await client.aclose()
    try:
        await repository.save_observations(session, items)
    except Exception as exc:
        _log.warning("market_ebay_persist_failed: %s", exc)
        await session.rollback()
    usd = [i for i in items if i.currency == "USD"]
    payload = {
        "status": "ok",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_reported": total,
        "stats": price_stats([i.price for i in usd]),
        "items": [
            {
                "source": i.source, "title": i.title, "brand": i.brand, "model": i.model,
                "reference": i.reference, "price": i.price, "currency": i.currency,
                "condition": i.condition, "year": i.year, "seller_feedback_pct": i.seller_feedback_pct,
                "item_url": i.item_url, "image_url": i.image_url,
                "observed_on": i.observed_on.isoformat(),
                "observed_at": i.observed_at.isoformat() if i.observed_at else None,
            }
            for i in sorted(items, key=lambda x: x.price)[:12]
        ],
        "cached": False,
    }
    if redis is not None:
        try:
            await redis.set(key, json.dumps(payload), ex=settings.market_cache_ttl_sec)
        except Exception as exc:
            _log.warning("market_cache_write_failed: %s", exc)
    return payload


async def _chrono24_section(session: AsyncSession, query: str, brand: str | None, sort: str = "lowest", limit: int = 12) -> dict:
    rows, stats = await repository.find_listings(session, SOURCE_CHRONO24_DATASET, query, brand, sort=sort, limit=limit)
    snapshot = rows[0].observed_on.isoformat() if rows else None
    return {
        "status": "ok" if rows else "no_match",
        "snapshot_date": snapshot,
        "note": "Dataset snapshot of public Chrono24 listings. Prices are not live.",
        "stats": stats,
        "sort": sort,
        "histogram": await repository.price_histogram(session, SOURCE_CHRONO24_DATASET, query, brand, stats),
        "items": [repository.listing_dict(r) for r in rows],
    }


def found_in(ebay: dict, chrono24: dict) -> dict:
    """Per source: True (has listings), False (searched, none), None (eBay not connected or failed)."""
    return {
        SOURCE_EBAY: bool(ebay["items"]) if ebay["status"] == "ok" else None,
        SOURCE_CHRONO24_DATASET: bool(chrono24["items"]),
    }


async def search_market(
    session: AsyncSession,
    redis,
    query: str,
    brand: str | None = None,
    *,
    sort: str = "lowest",
    limit: int = 12,
    fallback_query: str | None = None,
) -> dict:
    """Search both sources. When nothing matches anywhere and a shorter `fallback_query` is given
    ("Nautilus" for "Nautilus 5711/1A"), search again with it."""
    query = " ".join((query or "").split())[:200]
    fallback = " ".join((fallback_query or "").split())[:200]
    canonical = canonical_brand(brand) if brand else None
    sort = sort if sort in repository.SORTS else "lowest"
    limit = max(1, min(limit, 48))
    tried = [query]
    ebay = await _ebay_section(session, redis, query, canonical)
    chrono24 = await _chrono24_section(session, query, canonical, sort, limit)
    used = query
    if fallback and fallback.casefold() != query.casefold() and not any(found_in(ebay, chrono24).values()):
        tried.append(fallback)
        ebay = await _ebay_section(session, redis, fallback, canonical)
        chrono24 = await _chrono24_section(session, fallback, canonical, sort, limit)
        used = fallback
    return {
        "query": query,
        "query_used": used,
        "tried": tried,
        "brand": canonical,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "currency": "USD",
        "found": found_in(ebay, chrono24),
        "sources": {SOURCE_EBAY: ebay, SOURCE_CHRONO24_DATASET: chrono24},
    }


async def price_history(session: AsyncSession, query: str, brand: str | None = None, days: int = 365) -> dict:
    canonical = canonical_brand(brand) if brand else None
    series = await repository.daily_medians(session, SOURCE_EBAY, query, canonical, days=days)
    _rows, reference = await repository.find_listings(session, SOURCE_CHRONO24_DATASET, query, canonical, limit=1)
    snapshot = _rows[0].observed_on.isoformat() if _rows else None
    return {
        "query": query,
        "brand": canonical,
        "currency": "USD",
        "ebay_daily_median": series,
        "chrono24_reference": None if reference is None else {**reference, "snapshot_date": snapshot},
    }


async def market_status(session: AsyncSession) -> dict:
    return {
        "ebay": {"configured": ebay_configured(), "environment": settings.ebay_environment},
        "sources": await repository.source_summary(session),
    }
