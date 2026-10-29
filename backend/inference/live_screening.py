"""Origin check for listing photos: which model decides "original photograph or generated?".

`LIVE_SCREENING_MODEL` picks the model:

- `dinov3_frozen` (default, the current model): the frozen DINOv3 checkpoint `epoch_018.pt`, loaded
  through the research loader, which refuses any file whose SHA-256 differs from the frozen digest.
  The raw logit is divided by the frozen temperature before the sigmoid, so P(fake) is calibrated.
- `dinov2_legacy` (the earlier model): the DINOv2 route (`LOCAL_MODEL_PATH` / Triton), uncalibrated.

Both return the same (verdict, confidence) pair as `verdict.logits_to_verdict`: FAKE with
P(fake) when P(fake) >= 0.50, otherwise AUTHENTIC with 1 - P(fake). The 0.88 listing rule in
`verdict.apply_min_authentic_confidence` is applied afterwards by the route, for either model.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Literal

import numpy as np
from PIL import Image

from config import settings
from inference.research_guard import live_path_targets_dinov3

DINOV3_FROZEN = "dinov3_frozen"
DINOV2_LEGACY = "dinov2_legacy"
LIVE_MODELS = (DINOV3_FROZEN, DINOV2_LEGACY)
FAKE_THRESHOLD = 0.5

# Response labels for AuthenticateResponse.model
RESPONSE_MODEL = {DINOV3_FROZEN: "DINOV3_FROZEN", DINOV2_LEGACY: "LEGACY_DINOV2"}

_dinov3_lock = threading.Lock()


class LiveModelConfigurationError(RuntimeError):
    """The configured live model is unknown or points at the wrong checkpoint."""


def live_model() -> str:
    value = (settings.live_screening_model or "").strip().lower()
    if value not in LIVE_MODELS:
        raise LiveModelConfigurationError(f"LIVE_SCREENING_MODEL must be one of {LIVE_MODELS}, got {value!r}")
    return value


def verdict_from_probability(prob_fake: float) -> tuple[Literal["AUTHENTIC", "FAKE"], float]:
    if prob_fake >= FAKE_THRESHOLD:
        return "FAKE", round(prob_fake, 6)
    return "AUTHENTIC", round(1.0 - prob_fake, 6)


def _dinov3_probability(pil: Image.Image) -> float:
    from inference.dinov3_model import shared_backend

    backend = shared_backend()
    with _dinov3_lock:
        backend.load()
        batch = backend.preprocess(pil).unsqueeze(0)
        return float(backend.predict_probability(batch).reshape(-1)[0])


async def classify_listing_photo(pil: Image.Image) -> tuple[Literal["AUTHENTIC", "FAKE"], float, str]:
    """Return (verdict, confidence, model id) for an RGB listing photo."""
    model = live_model()
    if model == DINOV3_FROZEN:
        prob_fake = await asyncio.to_thread(_dinov3_probability, pil)
        verdict, confidence = verdict_from_probability(prob_fake)
        return verdict, confidence, model

    if live_path_targets_dinov3(settings.local_model_path, settings.triton_model_name, settings.dinov2_model_name):
        raise LiveModelConfigurationError("dinov2_legacy is selected but the legacy settings point at a DINOv3 checkpoint")
    from inference.engine import classify_image
    from inference.triton_client import preprocess_chw

    verdict, confidence = await classify_image(preprocess_chw(np.array(pil)))
    return verdict, confidence, model
