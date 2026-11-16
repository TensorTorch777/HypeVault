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


MODEL_NAMES = {
    "PHASE76_DINOV3_PE": "DINOv3 + Perception Encoder model",
    "DINOV3_FROZEN": "DINOv3 model",
    "LEGACY_DINOV2": "DINOv2 model",
}


def model_sentence(*, origin_model: str, p_fake: float, decision: str) -> str:
    """The model's own verdict in one sentence, shown first in every result that reached it."""
    name = MODEL_NAMES.get(origin_model, "screening model")
    verdicts = {
        LIKELY_GENERATED: "it calls the photo AI-generated",
        NOT_CONFIDENTLY_ORIGINAL: "it cannot confirm an original photograph",
        NEEDS_PROOF: "on its own it would hold the listing for a proof photo",
        PASSED: "it calls the photo an original photograph",
    }
    return f"Our {name} rates this photo {p_fake:.0%} likely AI-generated: {verdicts[decision]}."


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
    screenshot-style copies) calls AI: `edited_copy_margin` >= 0. Every reason starts with the model's
    own verdict (`details["model_says"]`).
    """
    p_fake = float(raw_confidence) if raw_verdict == "FAKE" else 1.0 - float(raw_confidence)
    details = {
        "watch_score": round(content.watch_score, 4),
        "content_model": content.model,
        "origin_model": origin_model,
        "origin_verdict": raw_verdict,
        "origin_confidence": round(float(raw_confidence), 4),
        "min_original_confidence": min_authentic_confidence,
        "p_ai": round(p_fake, 4),
    }

    def outcome(status: Status, code: str, advice: str) -> ScreeningOutcome:
        says = model_sentence(origin_model=origin_model, p_fake=p_fake, decision=code)
        return ScreeningOutcome(status=status, code=code, reason=f"{says} {advice}".strip(), details=details | {"model_says": says})

    if raw_verdict == "FAKE":
        return outcome("rejected", LIKELY_GENERATED, "Upload an original, unedited photo you took of the watch.")
    if raw_confidence < min_authentic_confidence:
        return outcome("rejected", NOT_CONFIDENTLY_ORIGINAL,
                       f"Its confidence in an original photo is {raw_confidence:.0%}; listings need {min_authentic_confidence:.0%}. "
                       "Try a sharper, well-lit photo without filters, text overlays or collage edits.")
    if edited_copy_margin is not None:
        details = details | {"edited_copy_margin": round(edited_copy_margin, 4)}
    borderline = hold_from_p_fake is not None and p_fake >= hold_from_p_fake
    edited_copy = edited_copy_margin is not None and edited_copy_margin >= 0
    if borderline or edited_copy:
        details = details | {"hold_from_p_ai": hold_from_p_fake,
                             "held_by": "edited_copy_model" if edited_copy and not borderline else "borderline_score"}
        if edited_copy and not borderline:
            return outcome("pending", NEEDS_PROOF,
                           "Its screenshot-trained check says the photo looks like an edited copy of an AI image, so it is held. "
                           + NEEDS_PROOF_REASON)
        return outcome("pending", NEEDS_PROOF, NEEDS_PROOF_REASON)
    return outcome("live", PASSED, PASSED_REASON)


def with_evidence(
    model: ScreeningOutcome, *, label: ScreeningOutcome | None = None, known: ScreeningOutcome | None = None
) -> ScreeningOutcome:
    """Combine the model's decision with a signed AI label or a known-AI match. Either one stops the listing,
    but the reason still says first what the model decided, and whether the evidence agrees with it."""
    if label is None and known is None:
        return model
    model_says = model.details.get("model_says")
    model_ai = model.code == LIKELY_GENERATED
    parts = [model_says] if model_says else []
    details = dict(model.details)
    if label is not None:
        tool = f" ({label.details['label_tool']})" if label.details.get("label_tool") else ""
        parts.append(
            f"{'The file’s signed Content Credentials label agrees' if model_ai else 'However, the file carries a signed Content Credentials label'}: "
            f"{label.details['label_maker']} says it was made with AI{tool}."
        )
        details |= label.details
    if known is not None:
        lead = "It is also" if model_ai or label is not None else "However, it is"
        parts.append(f"{lead} a copy of an AI-generated image we have already seen (match {known.details['known_ai_similarity']:.0%}).")
        details |= known.details
    if model.code == NOT_A_WATCH:
        parts.insert(0, model.reason)
    else:
        parts.append("It is not listed. Upload an original photo you took of the watch.")
    code = model.code if model.code in (LIKELY_GENERATED, NOT_A_WATCH) else (AI_LABEL if label is not None else KNOWN_AI_MATCH)
    return ScreeningOutcome(status="rejected", code=code, reason=" ".join(parts), details=details)
