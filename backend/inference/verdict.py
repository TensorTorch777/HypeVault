"""Shared binary verdict from a single logit (Triton or local PyTorch)."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

_raw_thresh = os.environ.get("INFERENCE_FAKE_THRESHOLD", "0.50")
FAKE_THRESHOLD: float = float(_raw_thresh)


def logits_to_verdict(
    logits: np.ndarray,
) -> tuple[Literal["AUTHENTIC", "FAKE"], float]:
    """
    Single logit → sigmoid(prob_fake). Same semantics as Triton export.
    """
    raw = float(logits.reshape(-1)[0])
    prob_fake = float(1.0 / (1.0 + np.exp(-raw)))

    if prob_fake >= FAKE_THRESHOLD:
        verdict: Literal["AUTHENTIC", "FAKE"] = "FAKE"
        confidence = prob_fake
    else:
        verdict = "AUTHENTIC"
        confidence = 1.0 - prob_fake

    return verdict, round(confidence, 6)


def apply_min_authentic_confidence(
    verdict: Literal["AUTHENTIC", "FAKE"],
    confidence: float,
    min_authentic: float,
) -> tuple[Literal["AUTHENTIC", "FAKE"], float]:
    """
    Policy: authentic confidence below `min_authentic` (e.g. 0.88) is treated as FAKE / deepfake for
    storage and API responses. Confidence after remap is P(fake) for consistency with logits_to_verdict.
    """
    c = float(confidence)
    if verdict == "AUTHENTIC" and c < float(min_authentic):
        return "FAKE", round(1.0 - c, 6)
    return verdict, round(c, 6)
