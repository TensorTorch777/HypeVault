"""Listing screening: every uploaded photo ends listed, rejected with a reason, or held.

Checks, in order, each of which can stop the listing:
  1. size    — short side at least `listing_min_short_side_px`, so dial details are readable
  2. label   — the file carries no signed Content Credentials label saying an AI made it
               (inference.provenance; ChatGPT and Gemini sign their images)
  3. known   — the photo is not a copy (crop, rescale, screenshot) of an AI image already seen
               (inference.known_ai)
  4. watch   — zero-shot CLIP finds a wristwatch (inference.content_gate)
  5. origin  — the live model does not flag the photo as AI-generated, and is confident enough
               that it is an original photo. A borderline score holds the listing for a proof photo
               and an admin check instead of listing it.

Passing every check lists the watch. That is a photo-screening result, not an authenticity
certificate: a real photograph of a counterfeit watch can pass. Customer copy says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from inference.content_gate import ContentCheck
from inference.provenance import AiLabel

Status = Literal["live", "rejected", "pending"]

PASSED = "PASSED"
IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
AI_LABEL = "AI_LABEL"
NOT_A_WATCH = "NOT_A_WATCH"
LIKELY_GENERATED = "LIKELY_GENERATED"
NOT_CONFIDENTLY_ORIGINAL = "NOT_CONFIDENTLY_ORIGINAL"
KNOWN_AI_MATCH = "KNOWN_AI_MATCH"
NEEDS_PROOF = "NEEDS_PROOF"
PROOF_SUBMITTED = "PROOF_SUBMITTED"
REVIEW_APPROVED = "REVIEW_APPROVED"
REVIEW_REJECTED = "REVIEW_REJECTED"
# Codes that send a listing to the admin review queue
AWAITING_REVIEW = (NEEDS_PROOF, PROOF_SUBMITTED)

PASSED_REASON = (
    "Passed photo screening: a wristwatch was found and the photo looks like an original photograph. "
    "This is not a certificate of authenticity."
)
NEEDS_PROOF_REASON = (
    "Held for a quick check before it goes live: the photo screening was not sure enough. Add a second photo of the "
    "watch next to a piece of paper with today's date and your name handwritten on it. An admin compares it with "
    "your first photo, usually within a day."
)
PROOF_SUBMITTED_REASON = "Proof photo received. An admin will compare it with your first photo before the listing goes live."
REVIEW_APPROVED_REASON = (
    "Listed after an admin checked your proof photo. Screening checks the photo, not the watch: "
    "this is not a certificate of authenticity."
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


def check_label(label: AiLabel | None) -> ScreeningOutcome | None:
    if label is None:
        return None
    made_with = f" ({label.tool})" if label.tool else ""
    return ScreeningOutcome(
        status="rejected",
        code=AI_LABEL,
        reason=(
            f"This image carries a Content Credentials label signed by {label.maker} saying it was made "
            f"with AI{made_with}. Upload an original photo you took of the watch."
        ),
        details={"label_maker": label.maker, "label_tool": label.tool, "label_source_type": label.source_type,
                 "label_validation": label.validation_state},
    )


def check_known_ai(similarity: float, min_similarity: float, source: str | None) -> ScreeningOutcome | None:
    if similarity < min_similarity:
        return None
    return ScreeningOutcome(
        status="rejected",
        code=KNOWN_AI_MATCH,
        reason=(
            f"This photo is a copy of an AI-generated image we have already seen (match {similarity:.0%}), "
            "even if it was cropped, resized or screenshotted. Upload an original photo you took of the watch."
        ),
        details={"known_ai_similarity": round(similarity, 4), "known_ai_source": source},
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
    origin_model: str = "LEGACY_DINOV2",
    hold_from_p_fake: float | None = None,
    edited_copy_margin: float | None = None,
) -> ScreeningOutcome:
    """Decide from the origin model's result once size and content have passed.

    With `hold_from_p_fake`, a photo that passes but whose P(AI) is at least that value is held for a
    proof photo and an admin check instead of going live. So is one the Phase 78 head (trained on
    screenshot-style copies) calls AI: `edited_copy_margin` >= 0.
    """
    details = {
        "watch_score": round(content.watch_score, 4),
        "content_model": content.model,
        "origin_model": origin_model,
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
    p_fake = 1.0 - float(raw_confidence)
    if edited_copy_margin is not None:
        details = details | {"edited_copy_margin": round(edited_copy_margin, 4)}
    borderline = hold_from_p_fake is not None and p_fake >= hold_from_p_fake
    edited_copy = edited_copy_margin is not None and edited_copy_margin >= 0
    if borderline or edited_copy:
        return ScreeningOutcome(status="pending", code=NEEDS_PROOF, reason=NEEDS_PROOF_REASON,
                                details=details | {"p_ai": round(p_fake, 4), "hold_from_p_ai": hold_from_p_fake,
                                                   "held_by": "edited_copy_model" if edited_copy and not borderline else "borderline_score"})
    return ScreeningOutcome(status="live", code=PASSED, reason=PASSED_REASON, details=details)
