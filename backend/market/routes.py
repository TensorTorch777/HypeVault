"""Public market endpoints: search, identify a watch, price history and source status."""

from __future__ import annotations

import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db
from identify.service import identify_watch
from market.service import market_status, price_history, search_market
from redis_client import get_redis

ALLOWED_CT = {"image/jpeg", "image/png", "image/webp"}

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
    fallback: Annotated[str | None, Query(max_length=200)] = None,
) -> dict:
    try:
        return await search_market(db, await _redis_or_none(), q, brand, sort=sort, limit=limit, fallback_query=fallback)
    except Exception as exc:
        _log.exception("market_search_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Market data unavailable")


def _read_photo(content_type: str, raw: bytes) -> Image.Image:
    """Re-validate the upload on the server: type, size, a decodable image, a usable resolution."""
    if content_type not in ALLOWED_CT:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only JPEG, PNG or WebP photos are allowed")
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The photo is empty")
    if len(raw) > settings.upload_max_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"The photo is larger than {settings.upload_max_bytes // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(raw)) as handle:
            handle.verify()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This file is not a readable image")
    if min(pil.size) < settings.identify_min_side_px:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"The photo is too small ({pil.width}x{pil.height}); use one at least {settings.identify_min_side_px}px on each side",
        )
    return pil


@router.post("/identify")
async def market_identify(
    db: Annotated[AsyncSession, Depends(get_db)],
    image: Annotated[UploadFile | None, File()] = None,
    q: Annotated[str | None, Form(max_length=200)] = None,
) -> dict:
    """Name a watch from a photo, typed text, or both. Returns up to three guesses for the user to confirm."""
    text = (q or "").strip() or None
    if image is None and text is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Send a photo, a name, or both")
    pil = _read_photo((image.content_type or "").lower(), await image.read()) if image is not None else None
    try:
        return await identify_watch(db, pil, text)
    except Exception as exc:
        _log.exception("market_identify_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Watch identification unavailable")


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


@router.get("/status")
async def market_source_status(db: Annotated[AsyncSession, Depends(get_db)]) -> dict:
    return await market_status(db)
