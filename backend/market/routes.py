"""Public market endpoints: search, price history and source status."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from market.chrono24_service import search_chrono24
from market.service import market_status, price_history, search_market
from redis_client import get_redis

_log = logging.getLogger(__name__)
router = APIRouter()


async def _redis_or_none():
    try:
        return await get_redis()
    except Exception as exc:
        _log.warning("market_redis_unavailable: %s", exc)
        return None


@router.get("/search")
async def market_search(
    db: Annotated[AsyncSession, Depends(get_db)],
    q: Annotated[str, Query(min_length=2, max_length=200)],
    brand: Annotated[str | None, Query(max_length=64)] = None,
    sort: Annotated[str, Query(pattern="^(near_median|lowest|highest)$")] = "lowest",
    limit: Annotated[int, Query(ge=1, le=48)] = 12,
) -> dict:
    try:
        return await search_market(db, await _redis_or_none(), q, brand, sort=sort, limit=limit)
    except Exception as exc:
        _log.exception("market_search_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Market data unavailable")


@router.get("/history")
async def market_history(
    db: Annotated[AsyncSession, Depends(get_db)],
    q: Annotated[str, Query(min_length=2, max_length=200)],
    brand: Annotated[str | None, Query(max_length=64)] = None,
    days: Annotated[int, Query(ge=7, le=730)] = 365,
) -> dict:
    try:
        return await price_history(db, q, brand, days)
    except Exception as exc:
        _log.exception("market_history_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Market history unavailable")


@router.get("/chrono24")
async def market_chrono24(
    db: Annotated[AsyncSession, Depends(get_db)],
    q: Annotated[str, Query(min_length=2, max_length=200)],
    brand: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=48)] = 12,
) -> dict:
    """Chrono24 listings for a watch: live when a provider is configured, else the imported snapshot."""
    try:
        return await search_chrono24(db, await _redis_or_none(), q, brand, limit=limit)
    except Exception as exc:
        _log.exception("market_chrono24_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chrono24 search unavailable")


@router.get("/status")
async def market_source_status(db: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    return await market_status(db)
