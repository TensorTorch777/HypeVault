"""POST /authenticate — listing photo screening.

A photo must be large enough, carry no signed "made by AI" label (C2PA), show a wristwatch (zero-shot CLIP) and pass the legacy
DINOv2 origin check. Passing lists the watch; failing rejects it with a seller-facing
reason (inference.screening). This route stays on DINOv2, not the frozen DINOv3 research
candidate, and a listed watch is screened, not authenticity-certified.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import get_current_user
from config import settings
from database import Listing, ListingCategory, ListingStatus, User, get_db
from inference.live_screening import (
    PHASE76_FUSION,
    RESPONSE_MODEL,
    LiveModelConfigurationError,
    classify_listing_photo,
    min_authentic_confidence,
)
from inference.research_guard import deployment_mode
from inference.schemas import AuthenticateResponse
from inference.content_gate import score_watch
from inference.dial_text import NOT_FOUND_HINT, read_dial_brand
from inference.provenance import read_ai_label
from inference import known_ai
from inference.screening import AI_LABEL, LIKELY_GENERATED, check_content, check_known_ai, check_label, check_origin, check_size
from inference.scope_gate import evaluate_declared_brand, failure_body, log_scope_request, unsupported_scope_body
from inference.verdict import apply_min_authentic_confidence
from s3_client import upload_file_bytes

_log = logging.getLogger(__name__)

router = APIRouter()
ALLOWED_CT = {"image/jpeg", "image/png", "image/webp"}

@router.post(
    "/authenticate",
    response_model=AuthenticateResponse,
    summary="Listing photo screening",
    description=(
        "Screens a listing photo: size, wristwatch present (CLIP), and a real-vs-generated check "
        "by the model set in LIVE_SCREENING_MODEL (PHASE76_DINOV3_PE, DINOV3_FROZEN or LEGACY_DINOV2, named in `model`). "
        "Passing lists the watch; failing rejects it with a reason. research_candidate is false and "
        "production_validation is NOT_ESTABLISHED. "
        "The declared brand is not independently verified from the image. "
        "Screening checks the photo, not the watch; it is not an authenticity guarantee."
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
            content=failure_body("POLICY_ERROR", "The screening model is set by the server, not by the request."),
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

        # A new photo replaces any earlier evidence on this listing.
        listing.s3_url = s3_url
        listing.verdict = None
        listing.confidence = None

        # 1. size, 2. AI label in the file, 3. copy of a known AI image, 4. is it a watch, 5. is it an original photo.
        # Each can stop the listing; a borderline origin score holds it for a proof photo.
        outcome = check_size(pil.width, pil.height, settings.listing_min_short_side_px)
        if outcome is None and settings.provenance_check:
            outcome = check_label(await asyncio.to_thread(read_ai_label, raw, ct))
        known = None
        if settings.known_ai_check and (outcome is None or outcome.code == AI_LABEL):
            try:
                known = await asyncio.to_thread(known_ai.check, pil)
            except Exception as exc:
                _log.exception("known_ai_failed: %s", exc)
                listing.status = ListingStatus.pending
                await db.commit()
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=failure_body("MODEL_ERROR", "The known-image check did not run. The listing is held, not published."),
                )
        if outcome is None and known is not None:
            outcome = check_known_ai(known.similarity, settings.known_ai_min_similarity, known.source)
        content = None
        live_model_id = None
        if outcome is None:
            try:
                content = await asyncio.to_thread(score_watch, pil)
            except Exception as exc:
                _log.exception("content_gate_failed: %s", exc)
                listing.status = ListingStatus.pending
                await db.commit()
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=failure_body("MODEL_ERROR", "The watch check did not run. The listing is held, not published."),
                )
            outcome = check_content(content)
        if outcome is None and content is not None:
            try:
                raw_verdict, raw_confidence, live_model_id = await classify_listing_photo(pil)
            except LiveModelConfigurationError as exc:
                _log.error("live_model_misconfigured: %s", exc)
                listing.status = ListingStatus.pending
                await db.commit()
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=failure_body("POLICY_ERROR", "The screening model is misconfigured. The listing is held, not published."),
                )
            except Exception as exc:
                _log.exception("inference_failed: %s", exc)
                listing.status = ListingStatus.pending
                await db.commit()
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=failure_body("MODEL_ERROR", "The authenticity model did not return a result."),
                )
            verdict, confidence = apply_min_authentic_confidence(
                raw_verdict,
                raw_confidence,
                min_authentic_confidence(live_model_id),
            )
            listing.verdict = verdict
            listing.confidence = float(confidence)
            outcome = check_origin(
                raw_verdict=raw_verdict,
                raw_confidence=float(raw_confidence),
                min_authentic_confidence=min_authentic_confidence(live_model_id),
                content=content,
                origin_model=RESPONSE_MODEL[live_model_id],
                hold_from_p_fake=settings.proof_hold_min_p_fake if live_model_id == PHASE76_FUSION else None,
            )
        assert outcome is not None
        if known is not None and outcome.code in (AI_LABEL, LIKELY_GENERATED):
            # Remember it, so a crop or screenshot of this image is caught next time.
            await asyncio.to_thread(known_ai.remember, known.vector, f"upload:{outcome.code}:{list_uuid}")

        # Only photos that reached the origin check show a watch, so only they get a dial reading.
        dial_text = None
        if live_model_id is not None and settings.dial_text_check:
            dial_text = await asyncio.to_thread(read_dial_brand, pil, scope.canonical_brand)

        listing.status = ListingStatus(outcome.status)
        listing.screening_code = outcome.code
        listing.screening_reason = outcome.reason
        listing.screening_details = {
            **outcome.details,
            **({"dial_brand_text": dial_text} if dial_text else {}),
            **({"known_ai_similarity": round(known.similarity, 4)} if known is not None else {}),
        }
        await db.commit()

        log_scope_request(
            _log,
            declared_brand=scope.canonical_brand,
            decision=listing.verdict,
            mode=deployment_mode(),
            status=outcome.code,
        )
        return AuthenticateResponse(
            verdict=listing.verdict,  # type: ignore[arg-type]
            confidence=listing.confidence,
            s3_url=s3_url,
            listing_id=str(list_uuid),
            listing_status=outcome.status,
            status=listing.verdict,  # type: ignore[arg-type]
            model_decision=listing.verdict,  # type: ignore[arg-type]
            screening_code=outcome.code,
            screening_reason=outcome.reason,
            publication_decision={"live": "PUBLISHED", "rejected": "REJECTED"}.get(outcome.status, "BLOCKED"),
            declared_brand=scope.canonical_brand or "",
            model=RESPONSE_MODEL.get(live_model_id),
            dial_brand_text=dial_text,
            seller_hint=NOT_FOUND_HINT if dial_text == "NOT_FOUND" else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("authenticate_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Authentication request failed")
