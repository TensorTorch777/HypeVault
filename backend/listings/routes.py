"""Listing CRUD and price comparison."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import get_current_user, get_optional_user, require_seller
from database import Listing, ListingCategory, ListingStatus, User, UserRole, get_db
from listings.models import ListingCreate, ListingRead, ListingUpdate, PresignRequest, PresignResponse
from market.routes import _redis_or_none
from market.chrono24_service import search_chrono24
from market.service import search_market
from s3_client import generate_presigned_put_url

_log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=ListingRead,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "",
    response_model=ListingRead,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_listing(
    body: ListingCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_seller)],
) -> Listing:
    try:
        if body.category != ListingCategory.watch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only watch listings are supported",
            )
        listing = Listing(
            seller_id=user.id,
            product_name=body.product_name,
            category=body.category,
            brand=body.brand,
            condition=body.condition,
            size=body.size,
        )
        db.add(listing)
        await db.commit()
        await db.refresh(listing)
        return listing
    except Exception as exc:
        _log.exception("create_listing_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create listing")


@router.get("/", response_model=list[ListingRead])
@router.get("", response_model=list[ListingRead], include_in_schema=False)
async def list_my_listings(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> list[Listing]:
    try:
        q = select(Listing).order_by(Listing.created_at.desc())
        if user.role != UserRole.admin:  # admins review every seller's listings
            q = q.where(Listing.seller_id == user.id)
        result = await db.execute(q)
        return list(result.scalars().all())
    except Exception as exc:
        _log.exception("list_my_listings_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load listings")


@router.get("/recent", response_model=list[ListingRead])
async def recent_listings(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 6,
    category: ListingCategory | None = None,
    brand: str | None = None,
) -> list[Listing]:
    try:
        safe_limit = max(1, min(limit, 48))
        q = select(Listing).where(Listing.status == ListingStatus.live)
        if category is not None:
            q = q.where(Listing.category == category)
        if brand and brand.strip():
            q = q.where(Listing.brand.ilike(f"%{brand.strip()}%"))
        q = q.order_by(Listing.created_at.desc()).limit(safe_limit)
        result = await db.execute(q)
        return list(result.scalars().all())
    except Exception as exc:
        _log.exception("recent_listings_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load listings")


@router.get("/compare")
async def compare_prices(q: str, db: Annotated[AsyncSession, Depends(get_db)], brand: str | None = None) -> dict:
    try:
        name = (q or "").strip()
        if len(name) < 2:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query too short")
        return await search_market(db, await _redis_or_none(), name, brand)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("compare_prices_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Comparison unavailable")


def _visible(listing: Listing | None, viewer: User | None) -> bool:
    """Live listings are public. Rejected or held listings are visible only to their seller."""
    if listing is None:
        return False
    if listing.status == ListingStatus.live:
        return True
    return viewer is not None and (viewer.id == listing.seller_id or viewer.role == UserRole.admin)


@router.get("/{listing_id}/comparison")
async def price_comparison(
    listing_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    viewer: Annotated[User | None, Depends(get_optional_user)],
) -> dict:
    try:
        listing = await db.get(Listing, listing_id)
        if not _visible(listing, viewer):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
        return await search_market(db, await _redis_or_none(), listing.product_name, listing.brand)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("price_comparison_failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Price comparison unavailable. Please try again.",
        )


@router.get("/{listing_id}/chrono24")
async def chrono24_matches(
    listing_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    viewer: Annotated[User | None, Depends(get_optional_user)],
    limit: int = 12,
) -> dict:
    """Chrono24 listings for this watch. Only for listings that passed screening (status live)."""
    listing = await db.get(Listing, listing_id)
    if not _visible(listing, viewer):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
    if listing.status != ListingStatus.live:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Chrono24 search runs after the listing passes screening")
    try:
        return await search_chrono24(db, await _redis_or_none(), listing.product_name, listing.brand, limit=max(1, min(limit, 48)))
    except Exception as exc:
        _log.exception("chrono24_matches_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chrono24 search unavailable")


@router.get("/{listing_id}", response_model=ListingRead)
async def get_listing(
    listing_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    viewer: Annotated[User | None, Depends(get_optional_user)],
) -> Listing:
    try:
        listing = await db.get(Listing, listing_id)
        if not _visible(listing, viewer):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
        return listing
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("get_listing_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load listing")


@router.patch("/{listing_id}", response_model=ListingRead)
async def update_listing(
    listing_id: uuid.UUID,
    body: ListingUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_seller)],
) -> Listing:
    try:
        listing = await db.get(Listing, listing_id)
        if listing is None or listing.seller_id != user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
        data = body.model_dump(exclude_unset=True)
        for k, v in data.items():
            setattr(listing, k, v)
        await db.commit()
        await db.refresh(listing)
        return listing
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("update_listing_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not update listing")


@router.post("/presign", response_model=PresignResponse)
async def presign_upload(
    body: PresignRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_seller)],
) -> PresignResponse:
    try:
        listing = await db.get(Listing, body.listing_id)
        if listing is None or listing.seller_id != user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
        safe = body.filename.replace("..", "").replace("/", "")
        key = f"listings/{body.listing_id}/{uuid.uuid4()}_{safe}"
        out = await generate_presigned_put_url(key=key, content_type=body.content_type)
        return PresignResponse(url=out["url"], key=out["key"], dev=out.get("dev"))
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("presign_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not create upload URL")
