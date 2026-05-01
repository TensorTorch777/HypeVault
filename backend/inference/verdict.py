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
