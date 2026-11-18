"""Listing CRUD and price comparison."""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import get_current_user, get_optional_user, require_admin, require_seller
from database import Listing, ListingCategory, ListingStatus, User, UserRole, get_db
from config import settings
from inference import screening
from listings.models import (
    AdminListingRead,
    ListingCreate,
    ListingRead,
    ListingUpdate,
    ReviewDecision,
)
from market.routes import _redis_or_none
from market.service import search_market
from storage import upload_file_bytes

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
        # Everyone, the admin included, sees only their own listings here; /all is the admin review queue.
        q = select(Listing).where(Listing.seller_id == user.id).order_by(Listing.created_at.desc())
        result = await db.execute(q)
        return list(result.scalars().all())
    except Exception as exc:
        _log.exception("list_my_listings_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load listings")


@router.get("/all", response_model=list[AdminListingRead])
async def list_all_listings(
    db: Annotated[AsyncSession, Depends(get_db)],
    _admin: Annotated[User, Depends(require_admin)],
) -> list[AdminListingRead]:
    try:
        q = select(Listing, User.email).join(User, User.id == Listing.seller_id).order_by(Listing.created_at.desc())
        result = await db.execute(q)
        return [AdminListingRead.model_validate(listing).model_copy(update={"seller_email": email}) for listing, email in result.all()]
    except Exception as exc:
        _log.exception("list_all_listings_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load listings")


@router.get("/recent", response_model=list[ListingRead])
async def recent_listings(
    db: Annotated[AsyncSession, Depends(get_db)],
    _admin: Annotated[User, Depends(require_admin)],
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


PROOF_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/{listing_id}/proof", response_model=ListingRead)
async def submit_proof_photo(
    listing_id: uuid.UUID,
    image: Annotated[UploadFile, File(...)],
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(require_seller)],
) -> Listing:
    """A held listing's proof photo: the watch beside today's date, handwritten. An admin then approves or rejects.

    The proof photo gets the AI-label and known-AI checks (either one rejects the listing) and an AI score for
    the admin; it never makes the listing live on its own.
    """
    from inference import known_ai
    from inference.live_screening import classify_listing_photo
    from inference.provenance import read_ai_label

    listing = await db.get(Listing, listing_id)
    if listing is None or listing.seller_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
    if listing.status != ListingStatus.pending or listing.screening_code not in screening.AWAITING_REVIEW:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This listing is not waiting for a proof photo")
    ct = (image.content_type or "").lower()
    raw = await image.read()
    if ct not in PROOF_TYPES or not raw or len(raw) > settings.upload_max_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a JPEG, PNG or WebP photo up to 10 MB")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file")

    url = await upload_file_bytes(key=f"listings/{listing_id}/proof_{uuid.uuid4()}_{image.filename or 'proof.jpg'}",
                                  body=raw, content_type=ct)
    label = await asyncio.to_thread(read_ai_label, raw, ct) if settings.provenance_check else None
    known = None
    if settings.known_ai_check:
        try:
            known = await asyncio.to_thread(known_ai.check, pil)
        except Exception as exc:
            _log.exception("proof_known_ai_failed: %s", exc)
    p_ai = None
    try:
        verdict, confidence, _model = await classify_listing_photo(pil)
        p_ai = round(confidence if verdict == "FAKE" else 1.0 - confidence, 4)
    except Exception as exc:
        _log.exception("proof_origin_failed: %s", exc)
    details = dict(listing.screening_details or {})
    details["proof"] = {
        "url": url, "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ai_label": label.maker if label else None,
        "known_ai_similarity": round(known.similarity, 4) if known else None, "p_ai": p_ai,
    }
    stopped = screening.check_label(label) or (
        screening.check_known_ai(known.similarity, settings.known_ai_min_similarity, known.source) if known else None)
    if stopped is not None:
        listing.status = ListingStatus.rejected
        listing.screening_code = stopped.code
        listing.screening_reason = "The proof photo was stopped: " + stopped.reason
        if known is not None:
            await asyncio.to_thread(known_ai.remember, known.vector, f"proof:{stopped.code}:{listing_id}")
    else:
        listing.screening_code = screening.PROOF_SUBMITTED
        listing.screening_reason = screening.PROOF_SUBMITTED_REASON
    listing.screening_details = details
    await db.commit()
    await db.refresh(listing)
    return listing


@router.post("/{listing_id}/review", response_model=AdminListingRead)
async def review_listing(
    listing_id: uuid.UUID,
    body: ReviewDecision,
    db: Annotated[AsyncSession, Depends(get_db)],
    admin: Annotated[User, Depends(require_admin)],
) -> AdminListingRead:
    """Admin decision on a held listing. Approval needs the seller's proof photo first."""
    listing = await db.get(Listing, listing_id)
    if listing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
    if listing.status != ListingStatus.pending or listing.screening_code not in screening.AWAITING_REVIEW:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This listing is not waiting for review")
    if body.decision == "approve" and listing.screening_code != screening.PROOF_SUBMITTED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The seller has not sent a proof photo yet")
    note = (body.note or "").strip() or None
    if body.decision == "approve":
        listing.status = ListingStatus.live
        listing.screening_code = screening.REVIEW_APPROVED
        listing.screening_reason = screening.REVIEW_APPROVED_REASON
    else:
        listing.status = ListingStatus.rejected
        listing.screening_code = screening.REVIEW_REJECTED
        listing.screening_reason = "An admin checked this listing and did not approve it" + (f": {note}" if note else ".")
    listing.screening_details = {**(listing.screening_details or {}), "review": {
        "decision": body.decision, "note": note, "admin_id": str(admin.id),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
    await db.commit()
    await db.refresh(listing)
    seller = await db.get(User, listing.seller_id)
    return AdminListingRead.model_validate(listing).model_copy(update={"seller_email": seller.email if seller else None})

