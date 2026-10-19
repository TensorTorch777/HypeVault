"""Listing screening: every uploaded photo ends listed, rejected with a reason, or held.

Checks, in order, each of which can stop the listing:
  1. size    — short side at least `listing_min_short_side_px`, so dial details are readable
  2. watch   — zero-shot CLIP finds a wristwatch (inference.content_gate)
  3. origin  — the DINOv2 listing model does not flag the photo as AI-generated, and is
               confident (≥ `inference_min_authentic_confidence`) that it is an original photo

Passing all three lists the watch. That is a photo-screening result, not an authenticity
certificate: a real photograph of a counterfeit watch can pass. Customer copy says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from inference.content_gate import ContentCheck

Status = Literal["live", "rejected", "pending"]

PASSED = "PASSED"
IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
NOT_A_WATCH = "NOT_A_WATCH"
LIKELY_GENERATED = "LIKELY_GENERATED"
NOT_CONFIDENTLY_ORIGINAL = "NOT_CONFIDENTLY_ORIGINAL"

PASSED_REASON = (
    "Passed photo screening: a wristwatch was found and the photo looks like an original photograph. "
    "This is not a certificate of authenticity."
)


@dataclass(frozen=True)
class ScreeningOutcome:
    status: Status
    code: str
    reason: str
    details: dict = field(default_factory=dict)


def check_size(width: int, height: int, min_short_side: int) -> ScreeningOutcome | None:
    short = min(width, height)
    if short >= min_short_side:
        return None
    return ScreeningOutcome(
        status="rejected",
        code=IMAGE_TOO_SMALL,
        reason=(
            f"The photo is {width}×{height} px. Upload one at least {min_short_side} px on its shorter side "
            "so the dial and case details are readable."
        ),
        details={"width": width, "height": height, "min_short_side": min_short_side},
    )


def check_content(content: ContentCheck) -> ScreeningOutcome | None:
    details = {"watch_score": round(content.watch_score, 4), "content_model": content.model}
    if content.is_watch:
        return None
    return ScreeningOutcome(
        status="rejected",
        code=NOT_A_WATCH,
        reason=(
            f"We couldn't find a wristwatch in this photo — it looks more like {content.looks_like}. "
            "Upload a clear photo of the watch itself, with the dial and case in frame."
        ),
        details=details | {"looks_like": content.looks_like},
    )


def check_origin(
    *,
    raw_verdict: str,
    raw_confidence: float,
    min_authentic_confidence: float,
    content: ContentCheck,
) -> ScreeningOutcome:
    """Decide from the DINOv2 result once size and content have passed."""
    details = {
        "watch_score": round(content.watch_score, 4),
        "content_model": content.model,
        "origin_model": "LEGACY_DINOV2",
        "origin_verdict": raw_verdict,
        "origin_confidence": round(float(raw_confidence), 4),
        "min_original_confidence": min_authentic_confidence,
    }
    if raw_verdict == "FAKE":
        return ScreeningOutcome(
            status="rejected",
            code=LIKELY_GENERATED,
            reason=(
                f"The photo shows traces typical of AI-generated or heavily edited images "
                f"(model score {raw_confidence:.0%}). Upload an original, unedited photo you took of the watch."
            ),
            details=details,
        )
    if raw_confidence < min_authentic_confidence:
        return ScreeningOutcome(
            status="rejected",
            code=NOT_CONFIDENTLY_ORIGINAL,
            reason=(
                f"The screening model could not confirm this is an original photograph "
                f"(confidence {raw_confidence:.0%}; listings need {min_authentic_confidence:.0%}). "
                "Try a sharper, well-lit photo without filters, text overlays or collage edits."
            ),
            details=details,
        )
    return ScreeningOutcome(status="live", code=PASSED, reason=PASSED_REASON, details=details)
