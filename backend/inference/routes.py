"""POST /authenticate — legacy DINOv2 listing check.

This route stays on DINOv2. It is not the frozen DINOv3 research candidate
and it does not establish production authenticity validation.
"""

from __future__ import annotations

import io
import logging
import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import get_current_user
from config import settings
from database import Listing, ListingCategory, ListingStatus, User, get_db
from inference.engine import classify_image
from inference.publication_gate import legacy_publication_status
from inference.research_guard import deployment_mode, live_path_targets_dinov3
from inference.schemas import AuthenticateResponse
from inference.scope_gate import evaluate_declared_brand, failure_body, log_scope_request, unsupported_scope_body
from inference.triton_client import preprocess_chw
from inference.verdict import apply_min_authentic_confidence
from s3_client import upload_file_bytes

_log = logging.getLogger(__name__)

router = APIRouter()
ALLOWED_CT = {"image/jpeg", "image/png", "image/webp"}

@router.post(
    "/authenticate",
    response_model=AuthenticateResponse,
    summary="Legacy DINOv2 listing check",
    description=(
        "Legacy DINOv2 classifier for an existing listing flow. "
        "model is LEGACY_DINOV2, research_candidate is false, and "
        "production_validation is NOT_ESTABLISHED. "
        "The declared brand is not independently verified from the image. "
        "This is not the frozen DINOv3 research prototype and not a universal authenticity guarantee."
    ),
)
async def authenticate(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    image: Annotated[UploadFile, File(...)],
    product_name: Annotated[str | None, Form()] = None,
    category: Annotated[str | None, Form()] = None,
    listing_id: Annotated[str | None, Form()] = None,
    brand: Annotated[str | None, Form()] = None,
    logical_model: Annotated[str | None, Form()] = None,
) -> AuthenticateResponse | JSONResponse:
    if logical_model not in (None, "", "dinov2_legacy"):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=failure_body("POLICY_ERROR", "DINOv3 candidate is blocked on the live DINOv2 route."),
        )
    try:
        ct = (image.content_type or "application/octet-stream").lower()
        if ct not in ALLOWED_CT:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=failure_body("INVALID_INPUT", "Only JPEG, PNG, or WebP images are allowed"),
            )

        raw = await image.read()
        if len(raw) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=failure_body("INVALID_INPUT", "Image must be non-empty"),
            )
        if len(raw) > settings.upload_max_bytes:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=failure_body(
                    "INVALID_INPUT",
                    f"Image exceeds maximum size of {settings.upload_max_bytes // (1024 * 1024)}MB",
                ),
            )

        declared_brand = (brand or "").strip()
        if listing_id and not declared_brand:
            try:
                declared_listing_id = uuid.UUID(listing_id)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid listing_id")
            declared_listing = await db.get(Listing, declared_listing_id)
            if declared_listing is None or declared_listing.seller_id != current_user.id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Listing not found")
            declared_brand = (declared_listing.brand or "").strip()
        scope = evaluate_declared_brand(declared_brand)
        if scope.scope_status != "SUPPORTED":
            log_scope_request(
                _log,
                declared_brand=declared_brand or None,
                decision=None,
                mode=deployment_mode(),
                status="UNSUPPORTED_SCOPE",
            )
            return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=unsupported_scope_body())

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=failure_body("INVALID_INPUT", "Invalid image file"),
            )
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
            cat: ListingCategory = ListingCategory.watch
            if category:
                try:
                    cat = ListingCategory(category)
                except ValueError:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid category")
            if cat != ListingCategory.watch:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only watch listings are supported",
                )
            listing = Listing(
                id=list_uuid,
                seller_id=current_user.id,
                product_name=(product_name or "New listing").strip()[:512],
                category=cat,
                brand=scope.canonical_brand,
                status=ListingStatus.pending,
            )
            db.add(listing)
            await db.flush()

        fname = image.filename or "image.jpg"
        s3_key = f"listings/{list_uuid}/{uuid.uuid4()}_{fname}"
        s3_url = await upload_file_bytes(key=s3_key, body=raw, content_type=ct)

        if live_path_targets_dinov3(
            settings.local_model_path,
            settings.triton_model_name,
            settings.dinov2_model_name,
        ):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=failure_body("POLICY_ERROR", "DINOv3 candidate is blocked on the live DINOv2 route."),
            )

        try:
            raw_verdict, raw_confidence = await classify_image(nchw)
        except Exception as exc:
            _log.exception("inference_failed: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=failure_body("MODEL_ERROR", "The authenticity model did not return a result."),
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
        published = legacy_publication_status(verdict)
        if published == "live":
            raise RuntimeError("legacy DINOv2 cannot authorize publication")
        listing.status = ListingStatus(published)

        await db.commit()

        log_scope_request(
            _log,
            declared_brand=scope.canonical_brand,
            decision=verdict,
            mode=deployment_mode(),
            status=verdict,
        )
        return AuthenticateResponse(
            verdict=verdict,  # type: ignore[arg-type]
            confidence=float(confidence),
            s3_url=s3_url,
            listing_id=str(list_uuid),
            listing_status=listing.status.value,  # type: ignore[arg-type]
            status=verdict,  # type: ignore[arg-type]
            model_decision=verdict,  # type: ignore[arg-type]
            publication_decision="BLOCKED",
            declared_brand=scope.canonical_brand or "",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("authenticate_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Authentication request failed")
