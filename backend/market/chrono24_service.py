"""Chrono24 search for the site: live provider with a Redis cache, falling back to the imported dataset."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import AsyncSession

from config import REPO_ROOT, settings
from market import repository
from market.chrono24_live import (
    PROVIDER_DATASET,
    Chrono24Error,
    build_query,
    make_provider,
    resolve_provider_name,
    search_live,
)
from market.types import SOURCE_CHRONO24_DATASET, canonical_brand

_log = logging.getLogger(__name__)


def _cache_key(provider: str, query: str, brand: str | None, limit: int) -> str:
    raw = f"{provider}|{(brand or '*').casefold()}|{' '.join(query.casefold().split())}|{limit}"
    return f"market:v1:chrono24:{hashlib.sha1(raw.encode('utf-8')).hexdigest()}"


def _export_dir():
    if not settings.chrono24_export_dir:
        return None
    path = REPO_ROOT / settings.chrono24_export_dir
    return path if path.is_absolute() else path.resolve()


async def _dataset_section(session: AsyncSession, query: str, brand: str | None, limit: int, *, note: str) -> dict:
    """Matches from the imported Chrono24 snapshot, in the same shape as a live search."""
    rows, _stats = await repository.find_listings(
        session, SOURCE_CHRONO24_DATASET, query, canonical_brand(brand) if brand else None, sort="near_median", limit=limit
    )
    items = []
    for row in rows:
        d = repository.listing_dict(row)
        items.append(
            {
                "listing_id": hashlib.sha1(f"{d['title']}|{d['price']}".encode("utf-8")).hexdigest()[:16],
                "title": d["title"],
                "listing_url": d["item_url"] or f"https://www.chrono24.com/search/index.htm?query={quote_plus(d['title'][:80])}",
                "price": d["price"],
                "currency": d["currency"],
                "brand": d["brand"],
                "model": d["model"],
                "reference": d["reference"],
                "condition": d["condition"],
                "year": d["year"],
                "location": None,
                "seller_type": None,
                "seller_name": None,
                "seller_rating": None,
                "image_url": d["image_url"],
                "image_urls": [d["image_url"]] if d["image_url"] else [],
            }
        )
    prices = sorted(i["price"] for i in items if i["price"])
    return {
        "status": "ok" if items else "no_match",
        "provider": PROVIDER_DATASET,
        "query": query,
        "brand": brand,
        "fetched_at": rows[0].observed_on.isoformat() if rows else datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "price_range": {"min": prices[0], "max": prices[-1], "currency": "USD"} if prices else None,
        "items": items,
        "note": note,
    }


async def search_chrono24(
    session: AsyncSession, redis, product_name: str, brand: str | None = None, *, limit: int = 12, refresh: bool = False
) -> dict:
    query = build_query(product_name, brand)[:200]
    limit = max(1, min(limit, 48))
    name = resolve_provider_name(settings.chrono24_provider, settings.apify_token)
    if name == PROVIDER_DATASET:
        return await _dataset_section(session, query, brand, limit, note="Imported Chrono24 snapshot. Live search is off.")

    key = _cache_key(name, query, brand, limit)
    if redis is not None and not refresh:
        try:
            cached = await redis.get(key)
            if cached:
                return {**json.loads(cached), "cached": True}
        except Exception as exc:
            _log.warning("chrono24_cache_read_failed: %s", exc)

    try:
        section = await search_live(make_provider(name, settings), query, brand=brand, limit=limit, export_dir=_export_dir())
    except Chrono24Error as exc:
        _log.warning("chrono24_live_failed provider=%s: %s", name, exc)
        return await _dataset_section(
            session, query, brand, limit, note=f"Live Chrono24 search failed ({exc}). Showing the imported snapshot."
        )
    section["note"] = "Live Chrono24 listings. Prices are asking prices, not sale prices."
    section["cached"] = False
    if redis is not None and section["items"]:
        try:
            await redis.set(key, json.dumps(section), ex=settings.chrono24_cache_ttl_sec)
        except Exception as exc:
            _log.warning("chrono24_cache_write_failed: %s", exc)
    return section
