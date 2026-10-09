"""POST /research/verify — allowlisted DINOv2 or DINOv3 research inference.

The selected logical id is resolved on the server. This route does not
create a listing and does not run when the deployment mode is production,
missing, or unknown. A missing Triton model is an error and is not
replaced by the other model.
"""

from __future__ import annotations

import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from auth.deps import get_current_user
from config import settings
from database import User
from inference.research_access import (
    checkpoint_block_body,
    log_research_event,
    mode_block_body,
    research_success_body,
    temperature_block_body,
)
from inference.model_router import (
    DINOV2_LEGACY,
    DINOV3_EXPERIMENTAL,
    ModelRoutingError,
    infer_allowlisted_model,
    model_readiness,
    resolve_model,
)
from inference.schemas import ResearchVerifyResponse
from inference.scope_gate import POLICY_VERSION, evaluate_declared_brand, failure_body, unsupported_scope_body

_log = logging.getLogger(__name__)

router = APIRouter()
ALLOWED_CT = {"image/jpeg", "image/png", "image/webp"}

_RESEARCH_DESCRIPTION = (
    "Frozen DINOv3 research classification within five user-declared watch brands. "
    "The brand is not independently verified from the image. "
    "This is not a production authenticity service. "
    "Production, missing, and unknown deployment modes return POLICY_ERROR and no verdict."
)


def frozen_checkpoint_digest() -> str:
    from inference.dinov3_model import CHECKPOINT
    from reference_inference import file_sha256

    return file_sha256(CHECKPOINT)


@router.post(
    "/verify",
    response_model=ResearchVerifyResponse,
    summary="DINOv3 five-brand research classification",
    description=_RESEARCH_DESCRIPTION,
)
async def research_verify(
    current_user: Annotated[User, Depends(get_current_user)],
    image: Annotated[UploadFile, File(...)],
    brand: Annotated[str | None, Form()] = None,
    logical_model: Annotated[str | None, Form()] = None,
) -> ResearchVerifyResponse | JSONResponse:
    del current_user
    selected = logical_model or DINOV3_EXPERIMENTAL
    try:
        spec = resolve_model(selected)
    except ModelRoutingError:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=failure_body("POLICY_ERROR", "Unknown model identifier."),
        )
    blocked = mode_block_body()
    if blocked is not None:
        log_research_event(
            _log,
            declared_brand=(brand or "").strip() or None,
            decision=None,
            status="POLICY_ERROR",
            checkpoint_sha=None,
            error=True,
        )
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=blocked)

    declared = (brand or "").strip()
    scope = evaluate_declared_brand(declared)
    if scope.scope_status != "SUPPORTED":
        log_research_event(
            _log,
            declared_brand=declared or None,
            decision=None,
            status="UNSUPPORTED_SCOPE",
            checkpoint_sha=None,
            error=True,
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=unsupported_scope_body())

    content_type = (image.content_type or "application/octet-stream").lower()
    raw = await image.read()
    invalid = _invalid_image_body(content_type, raw)
    if invalid is not None:
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=None,
            status="INVALID_INPUT",
            checkpoint_sha=None,
            error=True,
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=invalid)

    digest: str | None = None
    if spec["logical_id"] == DINOV3_EXPERIMENTAL:
        try:
            digest = frozen_checkpoint_digest()
        except Exception as exc:
            _log.exception("research_checkpoint_unreadable: %s", exc)
            log_research_event(
                _log,
                declared_brand=scope.canonical_brand,
                decision=None,
                status="POLICY_ERROR",
                checkpoint_sha=None,
                error=True,
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=failure_body("POLICY_ERROR", "DINOv3 checkpoint does not match the frozen artifact."),
            )
        mismatch = checkpoint_block_body(digest)
        if mismatch is not None:
            log_research_event(
                _log,
                declared_brand=scope.canonical_brand,
                decision=None,
                status="POLICY_ERROR",
                checkpoint_sha=digest,
                error=True,
            )
            return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=mismatch)

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        pil.load()
    except Exception:
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=None,
            status="INVALID_INPUT",
            checkpoint_sha=digest,
            error=True,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=failure_body("INVALID_INPUT", "Invalid image file"),
        )

    try:
        array = _array_for_model(spec, pil)
        routed = await infer_allowlisted_model(spec["logical_id"], array)
    except ModelRoutingError as exc:
        _log.exception("research_model_failed: %s", exc)
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=None,
            status="MODEL_ERROR",
            checkpoint_sha=digest,
            error=True,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=failure_body("MODEL_ERROR", "The selected model did not return a result."),
        )
    except Exception as exc:
        _log.exception("research_model_failed: %s", exc)
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=None,
            status="MODEL_ERROR",
            checkpoint_sha=digest,
            error=True,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=failure_body("MODEL_ERROR", "The selected model did not return a result."),
        )

    if spec["logical_id"] == DINOV2_LEGACY:
        from inference.verdict import logits_to_verdict
        import numpy as np

        verdict, _confidence = logits_to_verdict(np.asarray([routed["logit"]], dtype=np.float32))
        body = research_success_body(
            decision=verdict,
            declared_brand=scope.canonical_brand or "",
            checkpoint_sha="dinov2_classifier:1",
            policy_version="legacy_dinov2_logit_v1",
            model="LEGACY_DINOV2",
            model_version=str(routed["version"]),
            model_scope="LEGACY_DINOV2",
        )
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=verdict,
            status=verdict,
            checkpoint_sha="dinov2_classifier:1",
            error=False,
        )
        return ResearchVerifyResponse(**body)

    from inference.shadow import _policy_batch
    import torch

    result = _policy_batch([pil], torch.tensor([routed["logit"]], dtype=torch.float32), digest or "")[0]
    if result.system_status != "ok" or result.decision is None:
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=None,
            status="POLICY_ERROR",
            checkpoint_sha=digest,
            error=True,
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=failure_body("POLICY_ERROR", "The research model did not return an in-scope decision."),
        )
    if (
        checkpoint_block_body(result.checkpoint_sha256) is not None
        or temperature_block_body(result.temperature) is not None
        or result.policy_version != POLICY_VERSION
    ):
        log_research_event(
            _log,
            declared_brand=scope.canonical_brand,
            decision=None,
            status="POLICY_ERROR",
            checkpoint_sha=result.checkpoint_sha256,
            error=True,
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=failure_body("POLICY_ERROR", "DINOv3 checkpoint does not match the frozen artifact."),
        )

    body = research_success_body(
        decision=result.decision,
        declared_brand=scope.canonical_brand or "",
        checkpoint_sha=result.checkpoint_sha256,
        policy_version=result.policy_version,
        model="DINOV3_RESEARCH_PROTOTYPE",
        model_version=str(routed["version"]),
    )
    log_research_event(
        _log,
        declared_brand=scope.canonical_brand,
        decision=result.decision,
        status=result.decision,
        checkpoint_sha=result.checkpoint_sha256,
        error=False,
    )
    return ResearchVerifyResponse(**body)


def _array_for_model(spec: dict, pil: Image.Image):
    """Use the authoritative preprocess for the selected model only."""
    if spec["logical_id"] == DINOV3_EXPERIMENTAL:
        import sys
        from pathlib import Path

        ml_root = Path(__file__).resolve().parents[2] / "ml_rtx5080"
        if str(ml_root) not in sys.path:
            sys.path.insert(0, str(ml_root))
        from dinov3_serving import preprocess_serving_image

        tensor = preprocess_serving_image(pil)
        return tensor.unsqueeze(0).detach().cpu().numpy()
    import numpy as np
    from inference.triton_client import preprocess_chw

    return preprocess_chw(np.asarray(pil))


@router.get("/models")
async def research_model_readiness(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Per-model Triton readiness. This endpoint does not return a decision."""
    del current_user
    models = []
    for model_id in (DINOV2_LEGACY, DINOV3_EXPERIMENTAL):
        models.append(await model_readiness(model_id))
    return {"models": models, "publication_decision": "BLOCKED"}


def _invalid_image_body(content_type: str, raw: bytes) -> dict[str, str | None] | None:
    if content_type not in ALLOWED_CT:
        return failure_body("INVALID_INPUT", "Only JPEG, PNG, or WebP images are allowed")
    if len(raw) == 0:
        return failure_body("INVALID_INPUT", "Image must be non-empty")
    if len(raw) > settings.upload_max_bytes:
        return failure_body("INVALID_INPUT", f"Image exceeds maximum size of {settings.upload_max_bytes // (1024 * 1024)}MB")
    try:
        with Image.open(io.BytesIO(raw)) as handle:
            handle.verify()
    except Exception:
        return failure_body("INVALID_INPUT", "Invalid image file")
    return None
