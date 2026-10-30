"""Origin check for listing photos: which model decides "original photograph or generated?".

`LIVE_SCREENING_MODEL` picks the model:

- `phase76_fusion` (default): the frozen Phase 76 head, DINOv3 (pretrained + fine-tuned) and
  Perception Encoder L/14 features with one logistic regression (ml_rtx5080/phase76.py, hash-checked).
  It passed the pre-registered benchmark_v6 (ChatGPT 30/30, Gemini 17/17, real eBay FAKE 2/102).
  P(fake) is sigmoid(score - t_fake), so 0.5 is exactly its frozen cut.
- `dinov3_frozen`: the frozen DINOv3 checkpoint `epoch_018.pt`, loaded through the research
  loader, which refuses any file whose SHA-256 differs from the frozen digest. The raw logit is
  divided by the frozen temperature before the sigmoid, so P(fake) is calibrated.
- `dinov2_legacy`: the DINOv2 route (`LOCAL_MODEL_PATH` / Triton), uncalibrated.

All return the same (verdict, confidence) pair as `verdict.logits_to_verdict`: FAKE with
P(fake) when P(fake) >= 0.50, otherwise AUTHENTIC with 1 - P(fake). The route then needs
`min_authentic_confidence(model)` in an original photo: 0.88 for the DINO models, and the frozen
cut alone for Phase 76, which is how benchmark_v6 measured it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import threading
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from config import settings
from inference.research_guard import live_path_targets_dinov3

PHASE76_FUSION = "phase76_fusion"
DINOV3_FROZEN = "dinov3_frozen"
DINOV2_LEGACY = "dinov2_legacy"
LIVE_MODELS = (PHASE76_FUSION, DINOV3_FROZEN, DINOV2_LEGACY)
FAKE_THRESHOLD = 0.5

# Response labels for AuthenticateResponse.model
RESPONSE_MODEL = {PHASE76_FUSION: "PHASE76_DINOV3_PE", DINOV3_FROZEN: "DINOV3_FROZEN", DINOV2_LEGACY: "LEGACY_DINOV2"}

_ML = Path(__file__).resolve().parents[2] / "ml_rtx5080"
_log = logging.getLogger(__name__)
_dinov3_lock = threading.Lock()
_phase76_lock = threading.Lock()
_phase76_head = None


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


def min_authentic_confidence(model: str) -> float:
    """Confidence in an original photo needed to list it.

    Phase 76 lists every photo below its frozen cut: benchmark_v6 measured that rule, and adding the
    0.88 margin would have turned away 2 more of its 102 real photos.
    """
    return FAKE_THRESHOLD if model == PHASE76_FUSION else settings.inference_min_authentic_confidence


def _phase76() -> object:
    global _phase76_head
    if _phase76_head is None:
        if str(_ML) not in sys.path:
            sys.path.insert(0, str(_ML))
        from phase76 import FrozenFusionHead

        _phase76_head = FrozenFusionHead()  # refuses a head whose SHA-256 differs from frozen_rule.json
    return _phase76_head


def phase76_margin(pil: Image.Image) -> float:
    """Phase 76 score minus its frozen cut: FAKE when >= 0. The research tab scores with this too."""
    with _phase76_lock:
        head = _phase76()
        return head.score_image(pil) - head.rule["t_fake"]


def phase76_head_sha256() -> str:
    with _phase76_lock:
        return _phase76().rule["model"]["sha256"]


def phase76_unavailable_reason() -> str | None:
    """None when the Phase 76 head can serve here: CUDA, and the frozen rule and head on disk."""
    if _phase76_head is not None:
        return None
    try:
        import torch
    except ImportError:
        return "CUDA_UNAVAILABLE"
    if not torch.cuda.is_available():
        return "CUDA_UNAVAILABLE"
    rule = _ML / "experiments" / "phase76_fusion" / "frozen_rule.json"
    head = _ML / "experiments" / "phase76_fusion" / "dinov3_pe_v6_head.joblib"
    if not (rule.exists() and head.exists()):
        return "HEAD_FILES_MISSING"
    return None


def _phase76_probability(pil: Image.Image) -> float:
    return 1.0 / (1.0 + math.exp(-phase76_margin(pil)))


def preload() -> None:
    """Load the Phase 76 backbones at startup so the first listing is not slow. Errors are logged only."""
    try:
        import torch

        if live_model() == PHASE76_FUSION and torch.cuda.is_available():
            with _phase76_lock:
                _phase76()
            _log.info("phase76_live_head_loaded")
    except Exception as exc:
        _log.exception("phase76_preload_failed: %s", exc)


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
    if model == PHASE76_FUSION:
        prob_fake = await asyncio.to_thread(_phase76_probability, pil)
        verdict, confidence = verdict_from_probability(prob_fake)
        return verdict, confidence, model

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
