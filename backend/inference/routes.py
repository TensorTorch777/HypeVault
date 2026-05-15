"""POST /authenticate — image in, verdict + S3 out."""

from __future__ import annotations

import io
import logging
import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import get_current_user
from config import settings
from database import Listing, ListingCategory, ListingStatus, User, get_db
from inference.engine import classify_image
from inference.schemas import AuthenticateResponse
from inference.triton_client import preprocess_chw
from inference.verdict import apply_min_authentic_confidence
from s3_client import upload_file_bytes

_log = logging.getLogger(__name__)

router = APIRouter()
ALLOWED_CT = {"image/jpeg", "image/png", "image/webp"}

@router.post("/authenticate", response_model=AuthenticateResponse)
async def authenticate(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    image: Annotated[UploadFile, File(...)],
    product_name: Annotated[str | None, Form()] = None,
    category: Annotated[str | None, Form()] = None,
    listing_id: Annotated[str | None, Form()] = None,
) -> AuthenticateResponse:
    try:
        ct = (image.content_type or "application/octet-stream").lower()
        if ct not in ALLOWED_CT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG, PNG, or WebP images are allowed",
            )

        raw = await image.read()
        if len(raw) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image must be non-empty")
        if len(raw) > settings.upload_max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image exceeds maximum size of {settings.upload_max_bytes // (1024 * 1024)}MB",
            )

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file")
        arr = np.array(pil)  # HWC RGB uint8; resize + normalize in preprocess_chw
        nchw = preprocess_chw(arr)

        list_uuid: uuid.UUID
        listing: Listing | None = None
        if listing_id:
            try:
                list_uuid = uuid.UUID(listing_id)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid listing_id")
            listing = await db.get(Listing, list_uuid)
            if listing is None or listing.seller_id != current_user.id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
        else:
            list_uuid = uuid.uuid4()
            cat: ListingCategory = ListingCategory.sneaker
            if category:
                try:
                    cat = ListingCategory(category)
                except ValueError:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid category")
            listing = Listing(
                id=list_uuid,
                seller_id=current_user.id,
                product_name=(product_name or "New listing").strip()[:512],
                category=cat,
                status=ListingStatus.pending,
            )
            db.add(listing)
            await db.flush()

        fname = image.filename or "image.jpg"
        s3_key = f"listings/{list_uuid}/{uuid.uuid4()}_{fname}"
        s3_url = await upload_file_bytes(key=s3_key, body=raw, content_type=ct)

        try:
            raw_verdict, raw_confidence = await classify_image(nchw)
        except Exception as exc:
            _log.exception("inference_failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI verification service unavailable. Please try again.",
            )

        verdict, confidence = apply_min_authentic_confidence(
            raw_verdict,
            raw_confidence,
            settings.inference_min_authentic_confidence,
        )
        if raw_verdict == "AUTHENTIC" and verdict == "FAKE":
            _log.info(
                "authenticate_below_authentic_floor listing_id=%s raw_auth_conf=%.4f floor=%s",
                list_uuid,
                raw_confidence,
                settings.inference_min_authentic_confidence,
            )

        listing.s3_url = s3_url
        listing.verdict = verdict
        listing.confidence = float(confidence)
        if verdict == "FAKE":
            listing.status = ListingStatus.rejected
        else:
            listing.status = ListingStatus.live

        await db.commit()

        return AuthenticateResponse(
            verdict=verdict,  # type: ignore[arg-type]
            confidence=float(confidence),
            s3_url=s3_url,
            listing_id=str(list_uuid),
            listing_status=listing.status.value,  # type: ignore[arg-type]
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("authenticate_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Authentication request failed")
